"""
Graph Community Tokenizer for 3D-RNG
Lead ML Systems Engineer & Data Scientist Implementation

This script implements a Graph Community Tokenizer that:
1. Ingests a sample text corpus
2. Builds a transition probability matrix mapping character co-occurrences as a directed graph
3. Implements Community Detection to cluster characters into logical sub-word tokens
4. Outputs a vocabulary dictionary mapping integers to semantic graph clusters

This enables parallel 'Spatial Chunk Decoding' in the 3D-RNG architecture.
"""

import numpy as np
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import json


class GraphCommunityTokenizer:
    """
    Builds a semantic vocabulary from text using graph community detection on character transitions.
    
    The tokenizer treats character sequences as a directed graph where:
    - Nodes = unique characters
    - Edges = transitions between characters with weights = transition probabilities
    - Communities = clusters of characters that frequently co-occur (potential sub-word units)
    """
    
    def __init__(self, min_char_freq: int = 2, min_transition_prob: float = 0.01):
        """
        Initialize the Graph Community Tokenizer.
        
        Args:
            min_char_freq: Minimum frequency for a character to be included in vocabulary
            min_transition_prob: Minimum transition probability to consider an edge
        """
        self.min_char_freq = min_char_freq
        self.min_transition_prob = min_transition_prob
        
        # Character frequency counts
        self.char_freq = Counter()
        
        # Transition counts: (char_from, char_to) -> count
        self.transition_counts = defaultdict(int)
        
        # Unique characters in vocabulary
        self.chars = set()
        
        # Character to index mapping
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # Transition probability matrix
        self.transition_matrix = None
        
        # Community assignments: char -> community_id
        self.char_communities = {}
        
        # Vocabulary: token_id -> list of characters (community)
        self.vocab = {}
        self.vocab_size = 0
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.UNK_TOKEN = 1
        self.START_TOKEN = 2
        self.END_TOKEN = 3
        
    def fit(self, texts: List[str]):
        """
        Build the tokenizer from a list of text samples.
        
        Args:
            texts: List of strings to train the tokenizer on
        """
        print("Building character frequency counts...")
        self._build_character_frequencies(texts)
        
        print("Building transition counts...")
        self._build_transition_counts(texts)
        
        print("Creating character mappings...")
        self._create_character_mappings()
        
        print("Building transition probability matrix...")
        self._build_transition_matrix()
        
        print("Detecting communities...")
        self._detect_communities()
        
        print("Building vocabulary from communities...")
        self._build_vocabulary()
        
        print(f"Tokenizer built with vocabulary size: {self.vocab_size}")
    
    def _build_character_frequencies(self, texts: List[str]):
        """Count frequency of each character in the corpus."""
        for text in texts:
            # Clean text: keep printable characters and spaces
            cleaned = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
            self.char_freq.update(cleaned.lower())
    
    def _build_transition_counts(self, texts: List[str]):
        """Count transitions between consecutive characters."""
        for text in texts:
            cleaned = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
            cleaned = cleaned.lower()
            
            # Add boundary tokens for start/end of text
            padded = '<START>' + cleaned + '<END>'
            
            for i in range(len(padded) - 1):
                char_from = padded[i]
                char_to = padded[i + 1]
                
                # Only count transitions between actual characters (not boundaries for freq)
                if char_from not in ['<START>', '<END>'] and char_to not in ['<START>', '<END>']:
                    self.transition_counts[(char_from, char_to)] += 1
                    
                # Always track character occurrences
                if char_from not in ['<START>', '<END>']:
                    self.chars.add(char_from)
                if char_to not in ['<START>', '<END>']:
                    self.chars.add(char_to)
    
    def _create_character_mappings(self):
        """Create mappings between characters and indices."""
        # Filter characters by minimum frequency
        filtered_chars = {char for char, freq in self.char_freq.items() 
                         if freq >= self.min_char_freq and char in self.chars}
        
        # Sort for consistent ordering
        sorted_chars = sorted(list(filtered_chars))
        
        # Create mappings (reserve indices for special tokens)
        self.char_to_idx = {char: idx + 4 for idx, char in enumerate(sorted_chars)}
        self.idx_to_char = {idx + 4: char for idx, char in enumerate(sorted_chars)}
        
        # Add special tokens
        self.char_to_idx['<PAD>'] = self.PAD_TOKEN
        self.char_to_idx['<UNK>'] = self.UNK_TOKEN
        self.char_to_idx['<START>'] = self.START_TOKEN
        self.char_to_idx['<END>'] = self.END_TOKEN
        
        self.idx_to_char[self.PAD_TOKEN] = '<PAD>'
        self.idx_to_char[self.UNK_TOKEN] = '<UNK>'
        self.idx_to_char[self.START_TOKEN] = '<START>'
        self.idx_to_char[self.END_TOKEN] = '<END>'
        
        self.chars = filtered_chars
    
    def _build_transition_matrix(self):
        """Build the transition probability matrix."""
        vocab_size = len(self.chars) + 4  # +4 for special tokens
        self.transition_matrix = np.zeros((vocab_size, vocab_size))
        
        # Fill in transition probabilities
        for (char_from, char_to), count in self.transition_counts.items():
            if char_from in self.char_to_idx and char_to in self.char_to_idx:
                from_idx = self.char_to_idx[char_from]
                to_idx = self.char_to_idx[char_to]
                self.transition_matrix[from_idx, to_idx] = count
        
        # Normalize rows to get probabilities (handle zero rows)
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        self.transition_matrix = self.transition_matrix / row_sums
    
    def _detect_communities(self):
        """
        Detect communities in the character transition graph.
        
        Uses a simple greedy modularity-based approach similar to Louvain algorithm.
        For simplicity and to avoid external dependencies, we implement a basic version.
        """
        num_chars = len(self.chars)
        if num_chars == 0:
            # Fallback: each character in its own community
            for i, char in enumerate(sorted(self.chars)):
                self.char_communities[char] = i
            return
        
        # Ensure transition matrix is built
        if self.transition_matrix is None:
            self._build_transition_matrix()
        
        # Create mapping from char to matrix index
        char_to_matrix_idx = {}
        matrix_idx_to_char = {}
        idx = 4  # Start after special tokens
        for char in sorted(self.chars):
            char_to_matrix_idx[char] = idx
            matrix_idx_to_char[idx] = char
            idx += 1
        
        # Initialize each character in its own community
        communities = {char: i for i, char in enumerate(sorted(self.chars))}
        num_communities = len(self.chars)
        
        # Calculate modularity gain for moving nodes between communities
        # Simplified version: iteratively merge communities that increase modularity
        improved = True
        iterations = 0
        max_iterations = 10
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try moving each character to a neighboring community
            for char in self.chars:
                current_community = communities[char]
                char_idx = char_to_matrix_idx[char]
                
                # Find best community to move to
                best_community = current_community
                best_gain = 0
                
                # Check communities of neighbors (characters that transition to/from this char)
                neighbor_communities = set()
                
                # Look at incoming transitions
                for from_char in self.chars:
                    from_idx = char_to_matrix_idx[from_char]
                    # Safety check for transition_matrix
                    if self.transition_matrix is not None and from_idx < self.transition_matrix.shape[0] and char_idx < self.transition_matrix.shape[1]:
                        weight = self.transition_matrix[from_idx, char_idx]
                        if weight > self.min_transition_prob:
                            neighbor_communities.add(communities[from_char])
                
                # Look at outgoing transitions
                for to_char in self.chars:
                    to_idx = char_to_matrix_idx[to_char]
                    # Safety check for transition_matrix
                    if self.transition_matrix is not None and char_idx < self.transition_matrix.shape[0] and to_idx < self.transition_matrix.shape[1]:
                        weight = self.transition_matrix[char_idx, to_idx]
                        if weight > self.min_transition_prob:
                            neighbor_communities.add(communities[to_char])
                
                # Evaluate moving to each neighboring community
                for target_community in neighbor_communities:
                    if target_community == current_community:
                        continue
                    
                    # Calculate modularity gain (simplified)
                    gain = self._calculate_modularity_gain(
                        char, current_community, target_community, 
                        communities, char_to_matrix_idx
                    )
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_community = target_community
                
                # Move character if it improves modularity
                if best_gain > 0:
                    communities[char] = best_community
                    improved = True
            
            # Reassign community IDs to be sequential
            unique_communities = list(set(communities.values()))
            community_map = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_communities))}
            communities = {char: community_map[comm] for char, comm in communities.items()}
            num_communities = len(unique_communities)
        
        self.char_communities = communities
    
    def _calculate_modularity_gain(self, char: str, current_comm: int, target_comm: int,
                                 communities: Dict[str, int], 
                                 char_to_matrix_idx: Dict[str, int]) -> float:
        """
        Calculate the gain in modularity from moving a character to a target community.
        This is a simplified version of the Louvain modularity gain calculation.
        """
        # Safety check for transition matrix
        if self.transition_matrix is None:
            return 0.0
            
        char_idx = char_to_matrix_idx[char]
        
        # Calculate current community's internal edge weight
        in_weight = 0.0
        out_weight = 0.0
        total_weight = 0.0
        
        # Sum weights of edges within current community
        for other_char in self.chars:
            other_idx = char_to_matrix_idx[other_char]
            # Safety checks for indices
            if (char_idx < self.transition_matrix.shape[0] and 
                other_idx < self.transition_matrix.shape[1]):
                if communities[other_char] == current_comm:
                    # Internal edges
                    in_weight += self.transition_matrix[char_idx, other_idx] + \
                                self.transition_matrix[other_idx, char_idx]
                else:
                    # External edges
                    out_weight += self.transition_matrix[char_idx, other_idx] + \
                                self.transition_matrix[other_idx, char_idx]
                
                total_weight += self.transition_matrix[char_idx, other_idx] + \
                              self.transition_matrix[other_idx, char_idx]
        
        # Calculate expected gain if moved to target community
        target_in_weight = 0.0
        for other_char in self.chars:
            other_idx = char_to_matrix_idx[other_char]
            # Safety checks for indices
            if (char_idx < self.transition_matrix.shape[0] and 
                other_idx < self.transition_matrix.shape[1]):
                if communities[other_char] == target_comm:
                    target_in_weight += self.transition_matrix[char_idx, other_idx] + \
                                      self.transition_matrix[other_idx, char_idx]
        
        # Simplified modularity gain formula
        if total_weight > 0:
            gain = (target_in_weight - in_weight) / total_weight
        else:
            gain = 0
            
        return max(0, gain)  # Only return positive gains
    
    def _build_vocabulary(self):
        """Build the final vocabulary from detected communities."""
        # Group characters by community
        community_chars = defaultdict(list)
        for char, community_id in self.char_communities.items():
            community_chars[community_id].append(char)
        
        # Sort communities by ID for consistent ordering
        sorted_communities = sorted(community_chars.items())
        
        # Build vocabulary: token_id -> list of characters
        self.vocab = {}
        for token_id, (community_id, chars) in enumerate(sorted_communities):
            # Sort characters within community for consistency
            sorted_chars = sorted(chars)
            self.vocab[token_id] = sorted_chars
        
        self.vocab_size = len(self.vocab)
        
        # Add special tokens to vocabulary (they remain as single-character tokens)
        # Actually, we'll keep special tokens separate and handle them in encoding
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs using the graph community vocabulary.
        
        Args:
            text: Input string to encode
            
        Returns:
            List of token IDs
        """
        # Clean and lowercase text
        cleaned = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        cleaned = cleaned.lower()
        
        tokens = []
        i = 0
        
        while i < len(cleaned):
            # Try to match the longest possible community token
            matched = False
            
            # Check from longest possible match down to single character
            for length in range(min(10, len(cleaned) - i), 0, -1):  # Max 10 char tokens
                substring = cleaned[i:i+length]
                
                # Check if this substring belongs to a single community
                if self._is_uniform_community(substring):
                    # Find the token ID for this community
                    token_id = self._get_token_for_chars(set(substring))
                    if token_id is not None:
                        tokens.append(token_id)
                        i += length
                        matched = True
                        break
            
            if not matched:
                # Fall back to single character (or unknown)
                char = cleaned[i]
                if char in self.char_to_idx:
                    tokens.append(self.char_to_idx[char])
                else:
                    tokens.append(self.UNK_TOKEN)
                i += 1
        
        return tokens
    
    def _is_uniform_community(self, substring: str) -> bool:
        """Check if all characters in substring belong to the same community."""
        if not substring:
            return False
        
        communities = set()
        for char in substring:
            if char in self.char_communities:
                communities.add(self.char_communities[char])
            else:
                # Character not in vocabulary
                return False
        
        return len(communities) == 1
    
    def _get_token_for_chars(self, char_set: Set[str]) -> Optional[int]:
        """Get the token ID for a set of characters (should all be in same community)."""
        if not char_set:
            return None
            
        # Get community of first character
        first_char = next(iter(char_set))
        if first_char not in self.char_communities:
            return None
            
        target_community = self.char_communities[first_char]
        
        # Verify all characters are in same community
        for char in char_set:
            if char not in self.char_communities or self.char_communities[char] != target_community:
                return None
        
        # Find token ID for this community
        for token_id, chars in self.vocab.items():
            if set(chars) == char_set:
                return token_id
        
        return None
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded string
        """
        chars = []
        for token_id in token_ids:
            if token_id in self.vocab:
                # Community token: output all characters (in practice, we might choose one)
                # For simplicity, we'll output the first character
                chars.append(self.vocab[token_id][0])
            elif token_id in self.idx_to_char:
                chars.append(self.idx_to_char[token_id])
            else:
                chars.append('<UNK>')
        
        return ''.join(chars)
    
    def get_vocab(self) -> Dict[int, List[str]]:
        """Get the vocabulary dictionary."""
        return self.vocab.copy()
    
    def save(self, filepath: str):
        """Save the tokenizer to a file."""
        data = {
            'vocab': self.vocab,
            'char_freq': dict(self.char_freq),
            'char_communities': self.char_communities,
            'min_char_freq': self.min_char_freq,
            'min_transition_prob': self.min_transition_prob
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load a tokenizer from a file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(
            min_char_freq=data['min_char_freq'],
            min_transition_prob=data['min_transition_prob']
        )
        
        tokenizer.vocab = {int(k): v for k, v in data['vocab'].items()}
        tokenizer.char_freq = Counter(data['char_freq'])
        tokenizer.char_communities = {k: int(v) for k, v in data['char_communities'].items()}
        tokenizer.vocab_size = len(tokenizer.vocab)
        
        # Rebuild mappings
        tokenizer._rebuild_mappings_from_vocab()
        
        return tokenizer
    
    def _rebuild_mappings_from_vocab(self):
        """Rebuild internal mappings from loaded vocabulary."""
        # Build char_to_idx and idx_to_char from vocab
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # Add special tokens
        self.char_to_idx['<PAD>'] = self.PAD_TOKEN
        self.char_to_idx['<UNK>'] = self.UNK_TOKEN
        self.char_to_idx['<START>'] = self.START_TOKEN
        self.char_to_idx['<END>'] = self.END_TOKEN
        
        self.idx_to_char[self.PAD_TOKEN] = '<PAD>'
        self.idx_to_char[self.UNK_TOKEN] = '<UNK>'
        self.idx_to_char[self.START_TOKEN] = '<START>'
        self.idx_to_char[self.END_TOKEN] = '<END>'
        
        # Add community tokens
        for token_id, chars in self.vocab.items():
            # For simplicity, map first char of community to token ID
            # In a real implementation, we'd need a more sophisticated approach
            if chars:
                char = chars[0]
                self.char_to_idx[char] = token_id
                self.idx_to_char[token_id] = char


def create_sample_corpus() -> List[str]:
    """Create a sample text corpus for testing."""
    return [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test sentence",
        "machine learning and artificial intelligence are fascinating fields",
        "natural language processing enables computers to understand human language",
        "graph neural networks represent a powerful approach to learning from structured data",
        "recursive neural graphs can learn complex patterns without backpropagation",
        "the three dimensional recursive neural graph uses spatially routed reinforcement learning",
        "pheromone trails guide signal propagation through the graph structure",
        "local learning rules enable emergent specialization in neural architectures",
        "sparse activation patterns reduce computational requirements significantly"
    ]


if __name__ == "__main__":
    # Example usage
    print("Creating Graph Community Tokenizer...")
    tokenizer = GraphCommunityTokenizer(min_char_freq=1, min_transition_prob=0.01)
    
    # Use sample corpus
    corpus = create_sample_corpus()
    print(f"Training on {len(corpus)} text samples...")
    
    tokenizer.fit(corpus)
    
    # Test encoding/decoding
    test_text = "hello world"
    print(f"\nOriginal text: '{test_text}'")
    
    encoded = tokenizer.encode(test_text)
    print(f"Encoded tokens: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded text: '{decoded}'")
    
    # Show vocabulary
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print("First 10 vocabulary entries:")
    for i, (token_id, chars) in enumerate(list(tokenizer.vocab.items())[:10]):
        print(f"  Token {token_id}: {chars}")
    
    # Save tokenizer
    tokenizer.save("graph_tokenizer.json")
    print("\nTokenizer saved to 'graph_tokenizer.json'")