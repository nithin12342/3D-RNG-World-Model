"""
Parallel Evaluation Harness for 3D-RNG
Lead ML Systems Engineer & Data Scientist Implementation

This script implements Phase 3: Evaluation Harness for the parallel 3D-RNG architecture.
It provides a training loop that feeds sequence chunks into the Input Face and implements
telemetry to track performance metrics.

Key Features:
- Training loop for sequence chunk processing
- Telemetry: Tracks 'Tokens Generated per Second' and 'Percentage of Inactive Nodes per Forward Pass'
- Baseline comparison targets for autoregressive vs parallel generation
- Support for both graph_tokenizer.py and parallel_graph_impl.py components
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional, Any
import json
import math


class ParallelRNGEvaluator:
    """
    Evaluation harness for training and testing the parallel 3D-RNG with spatial chunk decoding.
    """
    
    def __init__(self, graph_dimensions: Tuple[int, int, int] = (10, 8, 8),
                 hidden_size: int = 16,
                 input_face_size: Tuple[int, int] = (4, 4),
                 output_face_size: Tuple[int, int] = (4, 4)):
        """
        Initialize the evaluator with graph and tokenizer components.
        
        Args:
            graph_dimensions: Dimensions of the 3D grid (X, Y, Z)
            hidden_size: Dimensionality of hidden state vectors
            input_face_size: Size of input face grid
            output_face_size: Size of output face grid
        """
        self.graph_dimensions = graph_dimensions
        self.hidden_size = hidden_size
        self.input_face_size = input_face_size
        self.output_face_size = output_face_size
        
        # Initialize components (will be created when needed)
        self.graph = None
        self.tokenizer = None
        
        # Telemetry tracking
        self.training_history = []
        self.tokens_per_second_history = []
        self.inactive_percentage_history = []
        
    def setup_components(self, tokenizer_path: Optional[str] = None):
        """
        Initialize or load the graph and tokenizer components.
        
        Args:
            tokenizer_path: Path to saved tokenizer JSON file (if None, creates new tokenizer)
        """
        print("Setting up Parallel 3D-RNG components...")
        
        # Initialize the parallel graph
        from parallel_graph_impl import ParallelNeuralGraph3D
        self.graph = ParallelNeuralGraph3D(
            dim_x=self.graph_dimensions[0],
            dim_y=self.graph_dimensions[1], 
            dim_z=self.graph_dimensions[2],
            hidden_size=self.hidden_size,
            input_face_size=self.input_face_size,
            output_face_size=self.output_face_size
        )
        
        # Initialize or load tokenizer
        if tokenizer_path:
            try:
                from graph_tokenizer import GraphCommunityTokenizer
                self.tokenizer = GraphCommunityTokenizer.load(tokenizer_path)
                print(f"Loaded tokenizer from {tokenizer_path}")
            except Exception as e:
                print(f"Failed to load tokenizer: {e}")
                print("Creating new tokenizer instead...")
                self._create_new_tokenizer()
        else:
            self._create_new_tokenizer()
    
    def _create_new_tokenizer(self):
        """Create and train a new tokenizer on sample data."""
        from graph_tokenizer import GraphCommunityTokenizer, create_sample_corpus
        
        self.tokenizer = GraphCommunityTokenizer(
            min_char_freq=1, 
            min_transition_prob=0.01
        )
        
        # Train on sample corpus
        corpus = create_sample_corpus()
        print(f"Training tokenizer on {len(corpus)} samples...")
        self.tokenizer.fit(corpus)
        print(f"Tokenizer vocabulary size: {self.tokenizer.vocab_size}")
    
    def text_to_chunks(self, text: str, chunk_size: int = 16) -> List[List[int]]:
        """
        Convert text into chunks of token IDs for parallel processing.
        
        Args:
            text: Input text to tokenize
            chunk_size: Number of tokens per chunk (should match input face capacity)
            
        Returns:
            List of chunks, where each chunk is a list of token IDs
        """
        # Tokenize the text
        token_ids = self.tokenizer.encode(text)
        
        # Pad or truncate to make chunks of equal size
        chunks = []
        for i in range(0, len(token_ids), chunk_size):
            chunk = token_ids[i:i+chunk_size]
            # Pad chunk if necessary
            if len(chunk) < chunk_size:
                chunk.extend([self.tokenizer.PAD_TOKEN] * (chunk_size - len(chunk)))
            chunks.append(chunk)
        
        return chunks
    
    def chunks_to_input_tensor(self, chunks: List[List[int]]) -> np.ndarray:
        """
        Convert list of token ID chunks to input tensor for the graph.
        
        Args:
            chunks: List of chunks (each chunk is list of token IDs)
            
        Returns:
            Input tensor of shape [num_chunks, hidden_size]
        """
        # For simplicity, we'll map token IDs to random vectors
        # In a real implementation, you'd have an embedding layer
        num_chunks = len(chunks)
        input_tensor = np.zeros((num_chunks, self.hidden_size))
        
        for i, chunk in enumerate(chunks):
            # Create a vector representation of the chunk
            # Simple approach: average of one-hot encodings (would use embeddings in practice)
            chunk_vector = np.zeros(self.hidden_size)
            for token_id in chunk:
                if token_id < self.tokenizer.vocab_size + 4:  # +4 for special tokens
                    # Create a pseudo-random but deterministic vector for each token
                    np.random.seed(token_id + 42)  # Fixed seed for reproducibility
                    token_vector = np.random.randn(self.hidden_size) * 0.1
                    chunk_vector += token_vector
            
            # Normalize by chunk length (avoid division by zero)
            if len(chunk) > 0:
                chunk_vector /= len(chunk)
            
            input_tensor[i] = chunk_vector
        
        return input_tensor
    
    def compute_reward(self, output_tensor: np.ndarray, 
                      target_tensor: np.ndarray) -> np.ndarray:
        """
        Compute reward vector for each output channel.
        
        Args:
            output_tensor: Actual output from graph [num_outputs, hidden_size]
            target_tensor: Target output [num_outputs, hidden_size]
            
        Returns:
            Reward vector [num_outputs] with values in [-1, 1]
        """
        # Compute cosine similarity for each output channel
        rewards = []
        
        for i in range(output_tensor.shape[0]):
            output_vec = output_tensor[i]
            target_vec = target_tensor[i]
            
            # Compute cosine similarity
            dot_product = np.dot(output_vec, target_vec)
            norm_output = np.linalg.norm(output_vec)
            norm_target = np.linalg.norm(target_vec)
            
            if norm_output > 0 and norm_target > 0:
                similarity = dot_product / (norm_output * norm_target)
                # Map from [-1, 1] to reward range (already in that range)
                reward = similarity
            else:
                reward = 0.0
            
            rewards.append(reward)
        
        return np.array(rewards)
    
    def train_step(self, input_chunks: List[List[int]], 
                  target_chunks: List[List[int]]) -> Dict[str, float]:
        """
        Perform one training step with input and target chunks.
        
        Args:
            input_chunks: List of input token ID chunks
            target_chunks: List of target token ID chunks
            
        Returns:
            Dictionary of metrics for this training step
        """
        start_time = time.time()
        
        # Convert chunks to tensors
        input_tensor = self.chunks_to_input_tensor(input_chunks)
        target_tensor = self.chunks_to_input_tensor(target_chunks)
        
        # Forward pass through graph
        output_tensor, paths = self.graph.forward_probe(
            input_tensor, 
            max_steps=50
        )
        
        # Compute rewards based on output vs target
        if len(paths) > 0 and output_tensor.shape[0] == len(paths):
            rewards = self.compute_reward(output_tensor, target_tensor[:len(paths)])
        else:
            # If no paths or mismatch, use small negative reward
            rewards = np.full(len(input_chunks), -0.1)
        
        # Apply traceback reinforcement
        if len(paths) > 0:
            self.graph.traceback_reinforcement(paths, rewards)
        
        # Calculate metrics
        end_time = time.time()
        step_time = end_time - start_time
        
        # Tokens generated per second (based on output face size)
        tokens_generated = len(paths) * self.output_face_size[0] * self.output_face_size[1]
        tokens_per_second = tokens_generated / step_time if step_time > 0 else 0
        
        # Percentage of inactive nodes
        stats = self.graph.get_activation_statistics()
        inactive_percentage = (1 - stats['activation_ratio']) * 100
        
        # Face-specific stats
        face_stats = self.graph.get_face_activation_stats()
        
        # Store in history
        metrics = {
            'step_time': step_time,
            'tokens_per_second': tokens_per_second,
            'inactive_percentage': inactive_percentage,
            'input_face_active_pct': face_stats['input_face']['ratio'] * 100,
            'output_face_active_pct': face_stats['output_face']['ratio'] * 100,
            'avg_path_length': np.mean([len(p) for p in paths]) if paths else 0,
            'num_active_paths': len(paths),
            'total_possible_paths': len(self.graph.input_face_coords),
            'mean_reward': np.mean(rewards) if len(rewards) > 0 else 0,
            'reward_std': np.std(rewards) if len(rewards) > 0 else 0
        }
        
        self.training_history.append(metrics)
        self.tokens_per_second_history.append(tokens_per_second)
        self.inactive_percentage_history.append(inactive_percentage)
        
        return metrics
    
    def train_epoch(self, text_data: List[str], 
                   chunk_size: int = 16,
                   batch_size: int = 32) -> Dict[str, float]:
        """
        Train for one epoch on text data.
        
        Args:
            text_data: List of text strings to train on
            chunk_size: Number of tokens per chunk
            batch_size: Number of chunks per batch
            
        Returns:
            Dictionary of epoch-level metrics
        """
        print(f"Starting training epoch with {len(text_data)} text samples...")
        
        # Convert all text to chunks
        all_chunks = []
        for text in text_data:
            chunks = self.text_to_chunks(text, chunk_size)
            all_chunks.extend(chunks)
        
        print(f"Total chunks to process: {len(all_chunks)}")
        
        # Process in batches
        epoch_metrics = []
        num_batches = math.ceil(len(all_chunks) / batch_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_chunks))
            batch_chunks = all_chunks[start_idx:end_idx]
            
            # For autoregressive training, target is input shifted by one
            # But for chunk-based, we'll use the same chunks as target (next chunk prediction would be more complex)
            # For simplicity, we'll use identity mapping: try to reproduce the input
            target_chunks = batch_chunks.copy()
            
            # Train on batch
            batch_metrics = self.train_step(batch_chunks, target_chunks)
            epoch_metrics.append(batch_metrics)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{num_batches} completed")
        
        # Compute epoch averages
        if epoch_metrics:
            epoch_summary = {
                'avg_step_time': np.mean([m['step_time'] for m in epoch_metrics]),
                'avg_tokens_per_second': np.mean([m['tokens_per_second'] for m in epoch_metrics]),
                'avg_inactive_percentage': np.mean([m['inactive_percentage'] for m in epoch_metrics]),
                'avg_input_face_active': np.mean([m['input_face_active_pct'] for m in epoch_metrics]),
                'avg_output_face_active': np.mean([m['output_face_active_pct'] for m in epoch_metrics]),
                'avg_path_length': np.mean([m['avg_path_length'] for m in epoch_metrics]),
                'avg_num_active_paths': np.mean([m['num_active_paths'] for m in epoch_metrics]),
                'path_utilization': np.mean([m['num_active_paths'] / m['total_possible_paths'] 
                                           for m in epoch_metrics if m['total_possible_paths'] > 0]),
                'avg_reward': np.mean([m['mean_reward'] for m in epoch_metrics])
            }
        else:
            epoch_summary = {}
        
        print(f"Epoch completed. Avg tokens/sec: {epoch_summary.get('avg_tokens_per_second', 0):.2f}")
        return epoch_summary
    
    def evaluate_baseline_autoregressive(self, text_data: List[str],
                                       chunk_size: int = 16) -> Dict[str, float]:
        """
        Evaluate baseline autoregressive generation for comparison.
        This simulates what a standard autoregressive model would do.
        
        Args:
            text_data: List of text strings to evaluate on
            chunk_size: Number of tokens per chunk
            
        Returns:
            Dictionary of baseline metrics
        """
        print("Running autoregressive baseline evaluation...")
        
        # Simulate autoregressive processing (one token at a time)
        total_tokens = 0
        total_time = 0.0
        
        for text in text_data:
            tokens = self.tokenizer.encode(text)
            total_tokens += len(tokens)
            
            # Simulate processing time per token (much slower due to sequential nature)
            # In reality, this would involve multiple forward passes
            start_time = time.time()
            # Simulate work: for each token, do a minimal computation
            for _ in tokens:
                _ = np.random.randn(self.hidden_size)  # Dummy computation
            end_time = time.time()
            
            total_time += (end_time - start_time)
        
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        baseline_metrics = {
            'baseline_tokens_per_second': tokens_per_second,
            'total_tokens_processed': total_tokens,
            'total_evaluation_time': total_time,
            'estimated_speedup_vs_parallel': 0  # Will be calculated after parallel training
        }
        
        return baseline_metrics
    
    def get_telemetry_summary(self) -> Dict[str, any]:
        """
        Get summary of telemetry metrics collected during training.
        
        Returns:
            Dictionary summarizing training progress
        """
        if not self.training_history:
            return {"message": "No training data available"}
        
        # Recent performance (last 10 steps or all if less than 10)
        recent_history = self.training_history[-10:] if len(self.training_history) >= 10 else self.training_history
        
        summary = {
            'total_training_steps': len(self.training_history),
            'recent_avg_tokens_per_second': np.mean([m['tokens_per_second'] for m in recent_history]),
            'recent_avg_inactive_pct': np.mean([m['inactive_percentage'] for m in recent_history]),
            'recent_avg_input_face_active': np.mean([m['input_face_active_pct'] for m in recent_history]),
            'recent_avg_output_face_active': np.mean([m['output_face_active_pct'] for m in recent_history]),
            'recent_avg_path_length': np.mean([m['avg_path_length'] for m in recent_history]),
            'recent_path_utilization': np.mean([m['num_active_paths'] / m['total_possible_paths'] 
                                              for m in recent_history if m['total_possible_paths'] > 0]),
            'improvement_trend': self._calculate_improvement_trend()
        }
        
        return summary
    
    def _calculate_improvement_trend(self) -> str:
        """Calculate whether performance is improving, declining, or stable."""
        if len(self.tokens_per_second_history) < 5:
            return "insufficient_data"
        
        # Compare first third vs last third
        third = len(self.tokens_per_second_history) // 3
        if third == 0:
            return "insufficient_data"
        
        first_third = self.tokens_per_second_history[:third]
        last_third = self.tokens_per_second_history[-third:]
        
        first_avg = np.mean(first_third) if first_third else 0
        last_avg = np.mean(last_third) if last_third else 0
        
        if last_avg > first_avg * 1.1:
            return "improving"
        elif last_avg < first_avg * 0.9:
            return "declining"
        else:
            return "stable"
    
    def save_checkpoint(self, filepath_prefix: str):
        """
        Save model and tokenizer checkpoints.
        
        Args:
            filepath_prefix: Prefix for checkpoint files
        """
        # Save tokenizer
        if self.tokenizer:
            tokenizer_path = f"{filepath_prefix}_tokenizer.json"
            self.tokenizer.save(tokenizer_path)
            print(f"Tokenizer saved to {tokenizer_path}")
        
        # Save graph state (simplified - in practice you'd save weights)
        graph_path = f"{filepath_prefix}_graph_state.json"
        graph_state = {
            'dimensions': self.graph_dimensions,
            'hidden_size': self.hidden_size,
            'input_face_size': self.input_face_size,
            'output_face_size': self.output_face_size,
            'shared_weights_shape': self.graph.shared_weights.shape if self.graph else None,
            'pheromone_count': len(self.graph.pheromones) if self.graph else 0,
            'training_steps': len(self.training_history)
        }
        
        with open(graph_path, 'w') as f:
            json.dump(graph_state, f, indent=2)
        print(f"Graph state saved to {graph_path}")
        
        # Save telemetry
        telemetry_path = f"{filepath_prefix}_telemetry.json"
        telemetry_data = {
            'training_history': self.training_history,
            'tokens_per_second_history': self.tokens_per_second_history,
            'inactive_percentage_history': self.inactive_percentage_history,
            'telemetry_summary': self.get_telemetry_summary()
        }
        
        with open(telemetry_path, 'w') as f:
            # Convert numpy types to JSON-serializable types
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert(telemetry_data), f, indent=2)
        print(f"Telemetry saved to {telemetry_path}")
    
    def run_comparison_experiment(self, 
                                 train_texts: List[str],
                                 test_texts: List[str],
                                 epochs: int = 5,
                                 chunk_size: int = 16) -> Dict[str, any]:
        """
        Run a complete experiment comparing parallel 3D-RNG vs autoregressive baseline.
        
        Args:
            train_texts: Text data for training
            test_texts: Text data for testing/evaluation
            epochs: Number of training epochs
            chunk_size: Chunk size for spatial decoding
            
        Returns:
            Dictionary containing experiment results
        """
        print("=" * 60)
        print("PARALLEL 3D-RNG EXPERIMENT")
        print("=" * 60)
        
        # Setup components
        self.setup_components()
        
        # Run baseline evaluation first
        print("\n1. Running autoregressive baseline evaluation...")
        baseline_results = self.evaluate_baseline_autoregressive(test_texts, chunk_size)
        baseline_tps = baseline_results['baseline_tokens_per_second']
        print(f"   Baseline tokens/second: {baseline_tps:.2f}")
        
        # Training loop
        print(f"\n2. Training parallel 3D-RNG for {epochs} epochs...")
        epoch_results = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}:")
            epoch_metrics = self.train_epoch(train_texts, chunk_size=chunk_size)
            epoch_results.append(epoch_metrics)
            
            # Print progress
            if epoch_metrics:
                print(f"   Tokens/sec: {epoch_metrics.get('avg_tokens_per_second', 0):.2f}")
                print(f"   Inactive %: {epoch_metrics.get('avg_inactive_percentage', 0):.2f}")
                print(f"   Path utilization: {epoch_metrics.get('path_utilization', 0):.2%}")
        
        # Final evaluation
        print("\n3. Final evaluation...")
        final_telemetry = self.get_telemetry_summary()
        final_tps = final_telemetry.get('recent_avg_tokens_per_second', 0)
        
        # Calculate speedup
        speedup = final_tps / baseline_tps if baseline_tps > 0 else 0
        
        # Prepare results
        experiment_results = {
            'experiment_config': {
                'train_samples': len(train_texts),
                'test_samples': len(test_texts),
                'epochs': epochs,
                'chunk_size': chunk_size,
                'graph_dimensions': self.graph_dimensions,
                'hidden_size': self.hidden_size,
                'input_face_size': self.input_face_size,
                'output_face_size': self.output_face_size
            },
            'baseline_autoregressive': baseline_results,
            'parallel_3drng_training': epoch_results,
            'final_telemetry': final_telemetry,
            'performance_comparison': {
                'baseline_tokens_per_second': baseline_tps,
                'parallel_tokens_per_second': final_tps,
                'speedup_factor': speedup,
                'improvement_percentage': (speedup - 1) * 100 if speedup >= 1 else -(1 - speedup) * 100
            }
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 60)
        print(f"Baseline (Autoregressive): {baseline_tps:.2f} tokens/second")
        print(f"Parallel 3D-RNG:           {final_tps:.2f} tokens/second")
        print(f"Speedup Factor:            {speedup:.2f}x")
        print(f"Improvement:               {experiment_results['performance_comparison']['improvement_percentage']:+.2f}%")
        print(f"Final Inactive %:          {final_telemetry.get('recent_avg_inactive_pct', 0):.2f}%")
        print(f"Path Utilization:          {final_telemetry.get('recent_path_utilization', 0):.2%}")
        print("=" * 60)
        
        return experiment_results


def create_sample_datasets() -> Tuple[List[str], List[str]]:
    """Create sample training and test datasets."""
    from graph_tokenizer import create_sample_corpus
    
    # Use the sample corpus and split it
    full_corpus = create_sample_corpus()
    
    # Simple split: 80% train, 20% test
    split_idx = int(0.8 * len(full_corpus))
    train_texts = full_corpus[:split_idx]
    test_texts = full_corpus[split_idx:]
    
    # Add some variations
    train_texts.extend([
        "the quick brown fox jumps over the lazy dog repeatedly",
        "machine learning algorithms learn patterns from data",
        "artificial intelligence systems can perceive and act",
        "natural language processing bridges human and machine communication",
        "deep learning models require substantial computational resources"
    ])
    
    test_texts.extend([
        "foxes are quick animals that can jump high",
        "neural networks process information through interconnected nodes",
        "language models generate text based on learned patterns",
        "graph structures represent relationships between entities",
        "reinforcement learning agents learn from rewards and punishments"
    ])
    
    return train_texts, test_texts


if __name__ == "__main__":
    # Run the comparison experiment
    print("Initializing Parallel 3D-RNG Evaluation Harness...")
    
    # Create evaluator
    evaluator = ParallelRNGEvaluator(
        graph_dimensions=(12, 8, 8),  # Slightly larger for better performance
        hidden_size=16,
        input_face_size=(4, 4),       # 16 input channels
        output_face_size=(4, 4)       # 16 output channels
    )
    
    # Get sample data
    train_texts, test_texts = create_sample_datasets()
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Run experiment
    results = evaluator.run_comparison_experiment(
        train_texts=train_texts,
        test_texts=test_texts,
        epochs=3,  # Reduced for demo
        chunk_size=16
    )
    
    # Save checkpoint
    evaluator.save_checkpoint("parallel_3drng_checkpoint")
    
    print("\nExperiment completed. Checkpoint saved.")