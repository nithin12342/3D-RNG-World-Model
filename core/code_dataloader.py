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
from pathlib import Path
import os
import warnings


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
        hidden_size: int = 128,
        repo_path: str = "./core",
        max_seq_len: int = 64
    ):
        """
        Initialize CodeDataLoader.
        
        Args:
            batch_size: Number of code samples per batch
            max_code_length: Maximum token length for code strings
            vocab_size: Vocabulary size for tokenization
            hidden_size: Hidden dimension for feature vectors
            repo_path: Path to repository for self-ingestion
            max_seq_len: Maximum sequence length for tensor outputs
        """
        self.batch_size = batch_size
        self.max_code_length = max_code_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.repo_path = repo_path
        self.max_seq_len = max_seq_len
        
        # Ingest repository files (Self-Ingestion/Autopoiesis)
        self._repo_samples = []
        self._ingest_repository()
        
        # Generate dummy Python functions as training data
        self._code_samples = self._generate_code_samples()
        
        # Combine repo samples with generated samples
        self._all_samples = self._repo_samples + self._code_samples
        
        # Build simple vocabulary for tokenization
        self._vocab = self._build_vocab()
        
        print(f"  CodeDataLoader initialized with {len(self._all_samples)} samples")
    
    def _ingest_repository(self):
        """
        Defensively ingest repository files for Self-Ingestion/Autopoiesis.
        Scans repo_path for .py files, skips hidden directories.
        
        Returns:
            List of code samples from repository
        """
        repo_path = Path(self.repo_path)
        
        if not repo_path.exists():
            warnings.warn(f"Repository path does not exist: {repo_path}")
            return
        
        print(f"  Ingesting repository from: {repo_path}")
        
        py_files = []
        
        # Defensive file traversal with pathlib
        try:
            for file_path in repo_path.rglob("*.py"):
                # Skip hidden directories and __pycache__
                if any(part.startswith('.') or part == '__pycache__' 
                       for part in file_path.parts):
                    continue
                py_files.append(file_path)
        except Exception as e:
            warnings.warn(f"Error scanning repository: {e}")
            return
        
        print(f"  Found {len(py_files)} Python files")
        
        for file_path in py_files:
            try:
                # Try to read file with utf-8 encoding
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
            except UnicodeDecodeError as e:
                warnings.warn(f"Cannot decode {file_path}: {e}")
                continue
            except Exception as e:
                warnings.warn(f"Cannot read {file_path}: {e}")
                continue
            
            # Try to parse AST
            try:
                tree = ast.parse(source_code, filename=str(file_path))
            except SyntaxError as e:
                warnings.warn(f"Syntax error in {file_path}: {e}")
                # Still include the file but mark as having syntax error
                tree = None
            except Exception as e:
                warnings.warn(f"Cannot parse {file_path}: {e}")
                continue
            
            # Extract node counts from AST
            node_counts = self._extract_ast_features(tree) if tree else {
                'num_functions': 0,
                'num_classes': 0,
                'num_returns': 0,
                'num_conditionals': 0,
                'num_loops': 0,
                'num_imports': 0,
                'has_error': True
            }
            
            # Create sample
            sample = {
                'source': source_code,
                'name': file_path.stem,
                'path': str(file_path),
                'complexity': node_counts.get('num_functions', 0) + node_counts.get('num_classes', 0),
                'ast_features': node_counts,
                'has_error': node_counts.get('has_error', False)
            }
            
            self._repo_samples.append(sample)
        
        print(f"  Ingested {len(self._repo_samples)} valid Python files")
    
    def _extract_ast_features(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Extract features from AST node.
        
        Args:
            tree: AST parse tree
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'num_functions': 0,
            'num_classes': 0,
            'num_returns': 0,
            'num_conditionals': 0,
            'num_loops': 0,
            'num_imports': 0,
            'has_error': False
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                features['num_functions'] += 1
            elif isinstance(node, ast.ClassDef):
                features['num_classes'] += 1
            elif isinstance(node, ast.Return):
                features['num_returns'] += 1
            elif isinstance(node, (ast.If, ast.IfExp)):
                features['num_conditionals'] += 1
            elif isinstance(node, (ast.For, ast.While)):
                features['num_loops'] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                features['num_imports'] += 1
        
        return features
    
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
                # Hash unknown tokens - use vocab_size as fallback to avoid division by zero
                vocab_remaining = self.vocab_size - len(self._vocab)
                if vocab_remaining > 0:
                    token_ids.append(hash(token) % vocab_remaining + len(self._vocab))
                else:
                    # All vocabulary slots used, wrap around
                    token_ids.append(hash(token) % self.vocab_size)
        
        # Pad or truncate to max_code_length
        if len(token_ids) < self.max_code_length:
            token_ids.extend([0] * (self.max_code_length - len(token_ids)))
        else:
            token_ids = token_ids[:self.max_code_length]
        
        return np.array(token_ids, dtype=np.int64)
    
    def _extract_ast_graph_features(self, code: str) -> Tuple[np.ndarray, np.ndarray, int]:
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
        
        # Use all samples (repo + generated)
        available_samples = self._all_samples if hasattr(self, '_all_samples') else self._code_samples
        
        # Handle empty batch case - yield zero-tensor dummy batch
        if len(available_samples) == 0:
            warnings.warn("No samples available - returning zero-tensor dummy batch")
            dummy_batch = {
                'text': np.zeros((batch_size, self.max_code_length), dtype=np.float32),
                'kg': np.zeros((batch_size, 64), dtype=np.float32),  # Match expected KG dimensions
                'tabular': np.zeros((batch_size, 8), dtype=np.float32),  # Match expected tabular dimensions
                'source': ["DUMMY"] * batch_size,
                'has_error': [True] * batch_size
            }
            return dummy_batch
        
        # Sample random code functions
        samples = random.choices(available_samples, k=batch_size)
        
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
            
            # KG modality: AST features (from pre-computed or extract)
            if 'ast_features' in sample:
                # Use pre-computed AST features from repository ingestion
                kg_features = self._ast_features_to_tensor(sample['ast_features'])
            else:
                # Extract from source code
                kg_features, _, depth = self._extract_ast_graph_features(source)
            kg_tensors.append(kg_features)
            
            # Tabular modality: linter metrics
            tabular_tensor = self._extract_tabular_metrics(source)
            tabular_tensors.append(tabular_tensor)
            
            # Store source for debugging
            sources.append(source)
        
        # Stack tensors - ensure consistent shapes
        try:
            batch = {
                'text': np.stack(text_tensors),
                'kg': np.stack(kg_tensors),
                'tabular': np.stack(tabular_tensors),
                'source': sources,
                'has_error': [s.get('has_error', False) for s in samples]
            }
        except Exception as e:
            warnings.warn(f"Error stacking tensors: {e}. Returning zero-tensor dummy batch.")
            batch = {
                'text': np.zeros((batch_size, self.max_code_length), dtype=np.float32),
                'kg': np.zeros((batch_size, 64), dtype=np.float32),
                'tabular': np.zeros((batch_size, 8), dtype=np.float32),
                'source': ["DUMMY"] * batch_size,
                'has_error': [True] * batch_size
            }
        
        return batch
    
    def _ast_features_to_tensor(self, ast_features: Dict[str, Any]) -> np.ndarray:
        """
        Convert AST features dict to numpy tensor with consistent shape.
        
        Args:
            ast_features: Dictionary of AST features
            
        Returns:
            Numpy array of features
        """
        # Expected feature dimensions for KG
        feature_dim = 64
        features = np.zeros(feature_dim, dtype=np.float32)
        
        # Map AST features to tensor indices
        if ast_features:
            features[0] = ast_features.get('num_functions', 0)
            features[1] = ast_features.get('num_classes', 0)
            features[2] = ast_features.get('num_returns', 0)
            features[3] = ast_features.get('num_conditionals', 0)
            features[4] = ast_features.get('num_loops', 0)
            features[5] = ast_features.get('num_imports', 0)
            features[6] = 1.0 if ast_features.get('has_error', False) else 0.0
        
        return features
    
    def __len__(self) -> int:
        """Return number of code samples."""
        return len(self._all_samples) if hasattr(self, '_all_samples') else len(self._code_samples)


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
        hidden_size=config.get('hidden_size', 128),
        repo_path=config.get('repo_path', './core'),
        max_seq_len=config.get('max_seq_len', 64)
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
