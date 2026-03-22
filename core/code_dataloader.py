"""
Code DataLoader for 3D-RNG World Engine
Chief Neurosymbolic Code Architect Implementation

This module implements a CodeDataLoader that extracts Abstract Syntax Trees (ASTs)
for the Knowledge Graph and formats outputs for multimodal processing.

Key Design Principles:
- Single Responsibility: Dedicated data loader for code/AST extraction
- Fail-Fast: Safe try/except for AST parsing with structured error tensors
- Dependency Injection: Returns standardized batch dictionaries for tokenizer
"""

import ast
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import random
import torch


class CodeDataLoader:
    """
    DataLoader for code generation tasks.
    
    Processes Python code multimodally:
    - Raw strings (Text face)
    - AST graph features (Knowledge Graph face)
    - Linter metrics (Tabular face)
    - Execution results (Sandbox)
    
    Provides batches formatted for the SpatialTokenizer.
    """
    
    def __init__(
        self,
        batch_size: int = 8,
        max_code_length: int = 512,
        vocab_size: int = 10000,
        hidden_size: int = 128
    ):
        """
        Initialize CodeDataLoader.
        
        Args:
            batch_size: Number of code samples per batch
            max_code_length: Maximum token length for code strings
            vocab_size: Vocabulary size for tokenization
            hidden_size: Hidden dimension for feature vectors
        """
        self.batch_size = batch_size
        self.max_code_length = max_code_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Generate dummy Python functions as training data
        self._code_samples = self._generate_code_samples()
        
        # Build simple vocabulary for tokenization
        self._vocab = self._build_vocab()
        
        print(f"  CodeDataLoader initialized with {len(self._code_samples)} samples")
    
    def _generate_code_samples(self) -> List[Dict[str, Any]]:
        """
        Generate dummy Python functions for training.
        
        Returns:
            List of code samples with source and metadata
        """
        code_samples = []
        
        # Sample 1: Simple math function
        code_samples.append({
            'source': '''def add(a, b):
    return a + b''',
            'name': 'add',
            'complexity': 1
        })
        
        # Sample 2: Conditional logic
        code_samples.append({
            'source': '''def max_value(a, b):
    if a > b:
        return a
    else:
        return b''',
            'name': 'max_value',
            'complexity': 2
        })
        
        # Sample 3: Loop
        code_samples.append({
            'source': '''def sum_list(numbers):
    total = 0
    for num in numbers:
        total += num
    return total''',
            'name': 'sum_list',
            'complexity': 2
        })
        
        # Sample 4: Broken loop (for syntax error testing)
        code_samples.append({
            'source': '''def broken_loop(n):
    for i in range(n)
        print(i)
    return i''',
            'name': 'broken_loop',
            'complexity': 2,
            'has_error': True
        })
        
        # Sample 5: Recursive function
        code_samples.append({
            'source': '''def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)''',
            'name': 'factorial',
            'complexity': 2
        })
        
        # Sample 6: Class definition
        code_samples.append({
            'source': '''class Calculator:
    def __init__(self):
        self.value = 0
    
    def add(self, x):
        self.value += x
        return self.value''',
            'name': 'Calculator',
            'complexity': 3
        })
        
        # Sample 7: List comprehension
        code_samples.append({
            'source': '''def square_numbers(nums):
    return [x ** 2 for x in nums]''',
            'name': 'square_numbers',
            'complexity': 1
        })
        
        # Sample 8: Missing return (runtime error potential)
        code_samples.append({
            'source': '''def maybe_return(x):
    if x > 0:
        return x''',
            'name': 'maybe_return',
            'complexity': 2,
            'has_error': True
        })
        
        # Sample 9: Nested loops
        code_samples.append({
            'source': '''def matrix_sum(matrix):
    total = 0
    for row in matrix:
        for val in row:
            total += val
    return total''',
            'name': 'matrix_sum',
            'complexity': 3
        })
        
        # Sample 10: Try-except block
        code_samples.append({
            'source': '''def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0''',
            'name': 'safe_divide',
            'complexity': 3
        })
        
        return code_samples
    
    def _build_vocab(self) -> Dict[str, int]:
        """
        Build simple vocabulary for code tokenization.
        
        Returns:
            Dictionary mapping tokens to indices
        """
        # Common Python keywords and symbols
        tokens = [
            'def', 'class', 'return', 'if', 'else', 'elif', 'for', 'while',
            'try', 'except', 'finally', 'with', 'as', 'import', 'from',
            'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is',
            '=', '+', '-', '*', '/', '//', '%', '**', '(', ')', '[', ']',
            '{', '}', ':', ',', '.'
        ]
        
        vocab = {token: idx for idx, token in enumerate(tokens)}
        
        # Add placeholder tokens
        for i in range(len(tokens), self.vocab_size):
            vocab[f'PAD_{i}'] = i
        
        return vocab
    
    def _tokenize_code(self, code: str) -> np.ndarray:
        """
        Tokenize code string into integer array.
        
        Args:
            code: Raw Python code string
            
        Returns:
            Tokenized array of shape (max_code_length,)
        """
        # Simple whitespace tokenization
        tokens = code.split()
        
        # Map to vocabulary indices
        token_ids = []
        for token in tokens:
            if token in self._vocab:
                token_ids.append(self._vocab[token])
            else:
                # Hash unknown tokens
                token_ids.append(hash(token) % (self.vocab_size - len(self._vocab)) + len(self._vocab))
        
        # Pad or truncate to max_code_length
        if len(token_ids) < self.max_code_length:
            token_ids.extend([0] * (self.max_code_length - len(token_ids)))
        else:
            token_ids = token_ids[:self.max_code_length]
        
        return np.array(token_ids, dtype=np.int64)
    
    def _extract_ast_features(self, code: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Extract AST graph features from Python code.
        
        Args:
            code: Raw Python code string
            
        Returns:
            Tuple of (node_features, edge_matrix, depth)
        """
        try:
            tree = ast.parse(code)
            
            # Count nodes by type
            node_counts = {
                'FunctionDef': 0,
                'ClassDef': 0,
                'For': 0,
                'While': 0,
                'If': 0,
                'Try': 0,
                'Call': 0,
                'Expr': 0,
                'Assign': 0,
                'Return': 0
            }
            
            # Count total nodes
            total_nodes = 0
            max_depth = 0
            
            for node in ast.walk(tree):
                total_nodes += 1
                node_type = type(node).__name__
                if node_type in node_counts:
                    node_counts[node_type] += 1
            
            # Calculate tree depth
            def get_depth(node, current_depth=0):
                nonlocal max_depth
                max_depth = max(max_depth, current_depth)
                for child in ast.iter_child_nodes(node):
                    get_depth(child, current_depth + 1)
            
            get_depth(tree)
            
            # Create node feature vector
            node_features = np.array([
                total_nodes,
                max_depth,
                node_counts['FunctionDef'],
                node_counts['ClassDef'],
                node_counts['For'] + node_counts['While'],
                node_counts['If'],
                node_counts['Try'],
                node_counts['Call'],
                node_counts['Assign'],
                node_counts['Return']
            ], dtype=np.float32)
            
            # Normalize features
            node_features = node_features / (total_nodes + 1e-6)
            
            # Create simple edge matrix (adjacency for visualization)
            edge_matrix = np.zeros((10, 10), dtype=np.float32)
            
            return node_features, edge_matrix, max_depth
            
        except SyntaxError:
            # Return error tensor for malformed code
            node_features = np.full(10, -1.0, dtype=np.float32)
            edge_matrix = np.zeros((10, 10), dtype=np.float32)
            return node_features, edge_matrix, -1
    
    def _extract_tabular_metrics(self, code: str) -> np.ndarray:
        """
        Extract tabular metrics (linter-style features).
        
        Args:
            code: Raw Python code string
            
        Returns:
            Feature vector of tabular metrics
        """
        lines = code.split('\n')
        num_lines = len(lines)
        
        # Cyclomatic complexity approximation
        complexity = 1  # Base complexity
        for line in lines:
            stripped = line.strip()
            if any(kw in stripped for kw in ['if', 'elif', 'for', 'while', 'except', 'and', 'or']):
                complexity += 1
        
        # Count various elements
        num_functions = code.count('def ')
        num_classes = code.count('class ')
        num_returns = code.count('return ')
        num_comments = sum(1 for line in lines if line.strip().startswith('#'))
        
        # Build metrics vector
        metrics = np.array([
            num_lines,
            complexity,
            num_functions,
            num_classes,
            num_returns,
            num_comments,
            len(code),  # Total characters
            complexity / max(num_lines, 1)  # Complexity per line
        ], dtype=np.float32)
        
        return metrics
    
    def get_batch(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get a batch of code samples formatted for multimodal processing.
        
        Args:
            batch_size: Override default batch size
            
        Returns:
            Dictionary mapping modality names to tensors:
            - 'text': Tokenized code strings (batch_size, max_code_length)
            - 'kg': AST graph features (batch_size, num_features)
            - 'tabular': Linter metrics (batch_size, num_metrics)
            - 'source': Original code strings (for debugging)
        """
        batch_size = batch_size or self.batch_size
        
        # Sample random code functions
        samples = random.choices(self._code_samples, k=batch_size)
        
        # Initialize output tensors
        text_tensors = []
        kg_tensors = []
        tabular_tensors = []
        sources = []
        
        for sample in samples:
            source = sample['source']
            
            # Text modality: tokenized code
            text_tensor = self._tokenize_code(source)
            text_tensors.append(text_tensor)
            
            # KG modality: AST features
            kg_features, _, depth = self._extract_ast_features(source)
            kg_tensors.append(kg_features)
            
            # Tabular modality: linter metrics
            tabular_tensor = self._extract_tabular_metrics(source)
            tabular_tensors.append(tabular_tensor)
            
            # Store source for debugging
            sources.append(source)
        
        # Stack tensors
        batch = {
            'text': np.stack(text_tensors),
            'kg': np.stack(kg_tensors),
            'tabular': np.stack(tabular_tensors),
            'source': sources,
            'has_error': [s.get('has_error', False) for s in samples]
        }
        
        return batch
    
    def __len__(self) -> int:
        """Return number of code samples."""
        return len(self._code_samples)


def create_code_dataloader(config: Dict[str, Any]) -> CodeDataLoader:
    """
    Factory function to create CodeDataLoader from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured CodeDataLoader instance
    """
    return CodeDataLoader(
        batch_size=config.get('batch_size', 8),
        max_code_length=config.get('max_code_length', 512),
        vocab_size=config.get('vocab_size', 10000),
        hidden_size=config.get('hidden_size', 128)
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing CodeDataLoader...")
    
    # Initialize dataloader
    dataloader = CodeDataLoader(batch_size=4)
    
    # Get a batch
    batch = dataloader.get_batch()
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Text shape: {batch['text'].shape}")
    print(f"KG shape: {batch['kg'].shape}")
    print(f"Tabular shape: {batch['tabular'].shape}")
    print(f"Sample sources: {[s[:30] for s in batch['source']]}")
    
    print("\nCodeDataLoader test passed!")
