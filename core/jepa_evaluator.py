"""
JEPA Evaluator for 3D-RNG World Engine
Lead Multi-Modal Data Scientist & Training Architect Implementation

This script implements Phase 3: Comparative JEPA Evaluator for the 3D-RNG World Engine,
tracking local prediction error convergence and memory footprint to demonstrate O(1) 
scaling properties compared to standard transformer-based approaches.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import time
import psutil
import os
from collections import defaultdict, deque


class JEPALocalPredictionTracker:
    """
    Tracks and logs the "Local Prediction Error" across the grid over time.
    Should converge downward as the model learns world physics.
    """
    
    def __init__(self, world_model):
        """
        Initialize JEPA Local Prediction Tracker.
        
        Args:
            world_model: The 3D-RNG world model (PredictiveCodingWorldCore)
        """
        self.world_model = world_model
        
        # Tracking variables
        self.prediction_error_history = deque(maxlen=10000)  # Long-term history
        self.node_wise_error_history = defaultdict(lambda: deque(maxlen=1000))  # Per-node tracking
        self.layer_wise_error_history = defaultdict(lambda: deque(maxlen=1000))  # Per-layer (x-plane) tracking
        
        # Convergence metrics
        self.convergence_window = 100  # Window for convergence detection
        self.error_threshold = 0.01   # Threshold for considering converged
        self.is_converged = False
        self.convergence_epoch = None
        
        # Statistics
        self.total_updates = 0
        self.start_time = time.time()
        
        print("JEPALocalPredictionTracker initialized")
    
    def update(self, epoch: int):
        """
        Update prediction error tracking for current epoch.
        
        Args:
            epoch: Current training epoch
        """
        # Get world statistics
        stats = self.world_model.get_world_statistics()
        avg_prediction_error = stats['average_prediction_error']
        
        # Record overall error
        self.prediction_error_history.append((epoch, avg_prediction_error))
        self.total_updates += 1
        
        # Record node-wise errors if available
        if hasattr(self.world_model, 'nodes'):
            for coord, node in self.world_model.nodes.items():
                if hasattr(node, 'prediction_errors') and node.prediction_errors:
                    # Calculate average error for this node
                    node_errors = list(node.prediction_errors.values())
                    if node_errors:
                        avg_node_error = np.mean([np.linalg.norm(err) for err in node_errors])
                        self.node_wise_error_history[coord].append((epoch, avg_node_error))
                        
                        # Also track by x-layer (for spatial analysis)
                        x_layer = coord[0]
                        self.layer_wise_error_history[x_layer].append((epoch, avg_node_error))
        
        # Check for convergence
        self._check_convergence(epoch)
    
    def _check_convergence(self, epoch: int):
        """Check if prediction error has converged."""
        if len(self.prediction_error_history) < self.convergence_window:
            return
            
        # Get recent errors
        recent_errors = [error for _, error in list(self.prediction_error_history)[-self.convergence_window:]]
        recent_errors = np.array(recent_errors)
        
        # Check if error is below threshold and stable
        mean_error = np.mean(recent_errors)
        std_error = np.std(recent_errors)
        
        if mean_error < self.error_threshold and std_error < self.error_threshold * 0.5:
            if not self.is_converged:
                self.is_converged = True
                self.convergence_epoch = epoch
                print(f"JEPA Convergence Detected at epoch {epoch}: "
                      f"Mean Error = {mean_error:.6f}, Std = {std_error:.6f}")
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get convergence information."""
        return {
            'is_converged': self.is_converged,
            'convergence_epoch': self.convergence_epoch,
            'current_error': self.prediction_error_history[-1][1] if self.prediction_error_history else None,
            'error_history_length': len(self.prediction_error_history)
        }
    
    def get_spatial_error_map(self) -> Dict[int, float]:
        """
        Get average prediction error by x-layer (spatial dimension).
        
        Returns:
            Dictionary mapping x-coordinate to average error in that layer
        """
        layer_errors = {}
        for x_layer, error_history in self.layer_wise_error_history.items():
            if error_history:
                recent_errors = [error for _, error in list(error_history)[-100:]]  # Last 100 updates
                if recent_errors:
                    layer_errors[x_layer] = np.mean(recent_errors)
        return layer_errors
    
    def get_temporal_error_profile(self) -> Tuple[List[int], List[float]]:
        """
        Get temporal error profile for plotting.
        
        Returns:
            Tuple of (epochs, errors) for plotting
        """
        if not self.prediction_error_history:
            return [], []
        
        epochs, errors = zip(*self.prediction_error_history)
        return list(epochs), list(errors)


class JEPAMemoryProfiler:
    """
    Tracks memory footprint over extremely long continuous sequence injections
    to prove O(1) memory scaling (no KV cache growth).
    """
    
    def __init__(self):
        """Initialize JEPA Memory Profiler."""
        self.process = psutil.Process(os.getpid())
        self.memory_history = deque(maxlen=10000)  # (epoch, memory_mb)
        self.sequence_length_history = deque(maxlen=10000)  # (epoch, sequence_length)
        self.baseline_memory = None
        
        print("JEPAMemoryProfiler initialized")
    
    def update(self, epoch: int, sequence_length: Optional[int] = None):
        """
        Update memory tracking for current epoch.
        
        Args:
            epoch: Current training epoch
            sequence_length: Length of input sequence processed (if applicable)
        """
        # Get current memory usage
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
        
        # Set baseline on first update
        if self.baseline_memory is None:
            self.baseline_memory = memory_mb
        
        # Record memory usage
        self.memory_history.append((epoch, memory_mb))
        
        # Record sequence length if provided
        if sequence_length is not None:
            self.sequence_length_history.append((epoch, sequence_length))
    
    def get_memory_growth_rate(self) -> float:
        """
        Calculate memory growth rate (slope of memory vs sequence length).
        
        Returns:
            Growth rate in MB per unit sequence length (should approach 0 for O(1))
        """
        if len(self.sequence_length_history) < 10 or len(self.memory_history) < 10:
            return 0.0
            
        # Get recent data points
        seq_lengths = [seq_len for _, seq_len in list(self.sequence_length_history)[-100:]]
        memory_values = [mem for _, mem in list(self.memory_history)[-100:]]
        
        if len(seq_lengths) != len(memory_values) or len(seq_lengths) < 2:
            return 0.0
            
        # Calculate linear regression slope
        seq_lengths = np.array(seq_lengths)
        memory_values = np.array(memory_values)
        
        # Remove baseline to see growth from start
        memory_growth = memory_values - np.mean(memory_values[:10])  # Subtract early average
        
        if np.std(seq_lengths) > 0:
            slope = np.cov(seq_lengths, memory_growth)[0, 1] / np.var(seq_lengths)
        else:
            slope = 0.0
            
        return slope
    
    def get_memory_efficiency_ratio(self) -> float:
        """
        Calculate memory efficiency ratio compared to linear growth expectation.
        
        Returns:
            Ratio of actual growth to expected linear growth (should be << 1 for O(1))
        """
        if len(self.sequence_length_history) < 10:
            return 1.0
            
        seq_lengths = [seq_len for _, seq_len in list(self.sequence_length_history)[-100:]]
        memory_values = [mem for _, mem in list(self.memory_history)[-100:]]
        
        if len(seq_lengths) < 2:
            return 1.0
            
        max_seq_len = max(seq_lengths)
        min_seq_len = min(seq_lengths)
        max_mem = max(memory_values)
        min_mem = min(memory_values)
        
        if max_seq_len > min_seq_len:
            # Expected linear growth: memory increase per unit sequence length
            expected_growth_per_unit = (max_mem - min_mem) / (max_seq_len - min_seq_len)
            actual_growth_rate = self.get_memory_growth_rate()
            
            if expected_growth_per_unit != 0:
                ratio = abs(actual_growth_rate / expected_growth_per_unit)
                return min(ratio, 10.0)  # Cap at 10 for reporting
        return 1.0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        if not self.memory_history:
            return {}
            
        current_memory = self.memory_history[-1][1]
        baseline_memory = self.baseline_memory or current_memory
        
        return {
            'current_memory_mb': current_memory,
            'baseline_memory_mb': baseline_memory,
            'memory_increase_mb': current_memory - baseline_memory,
            'memory_growth_rate_mb_per_unit': self.get_memory_growth_rate(),
            'memory_efficiency_ratio': self.get_memory_efficiency_ratio(),
            'history_length': len(self.memory_history)
        }


class JEPAEvaluator:
    """
    Main JEPA Evaluator combining local prediction tracking and memory profiling.
    Provides comprehensive evaluation of the 3D-RNG World Engine's learning properties.
    """
    
    def __init__(self, world_model):
        """
        Initialize JEPA Evaluator.
        
        Args:
            world_model: The 3D-RNG world model (PredictiveCodingWorldCore)
        """
        self.world_model = world_model
        self.prediction_tracker = JEPALocalPredictionTracker(world_model)
        self.memory_profiler = JEPAMemoryProfiler()
        
        # Evaluation metrics
        self.evaluation_history = []
        self.best_epoch = None
        self.best_error = float('inf')
        
        print("JEPAEvaluator initialized")
    
    def evaluate_epoch(self, epoch: int, sequence_length: Optional[int] = None):
        """
        Perform comprehensive evaluation for current epoch.
        
        Args:
            epoch: Current training epoch
            sequence_length: Length of input sequence processed (for memory profiling)
        """
        # Update prediction tracking
        self.prediction_tracker.update(epoch)
        
        # Update memory profiling
        self.memory_profiler.update(epoch, sequence_length)
        
        # Get current metrics
        prediction_metrics = self.prediction_tracker.get_convergence_info()
        memory_metrics = self.memory_profiler.get_memory_stats()
        spatial_error_map = self.prediction_tracker.get_spatial_error_map()
        temporal_profile = self.prediction_tracker.get_temporal_error_profile()
        
        # Compile evaluation results
        eval_result = {
            'epoch': epoch,
            'timestamp': time.time(),
            'prediction_error': prediction_metrics['current_error'],
            'is_converged': prediction_metrics['is_converged'],
            'convergence_epoch': prediction_metrics['convergence_epoch'],
            'memory_mb': memory_metrics.get('current_memory_mb', 0),
            'memory_increase_mb': memory_metrics.get('memory_increase_mb', 0),
            'memory_efficiency_ratio': memory_metrics.get('memory_efficiency_ratio', 1.0),
            'spatial_error_map': spatial_error_map,
            'temporal_profile': temporal_profile
        }
        
        # Track best performance
        current_error = prediction_metrics['current_error']
        if current_error is not None and current_error < self.best_error:
            self.best_error = current_error
            self.best_epoch = epoch
        
        # Store in history
        self.evaluation_history.append(eval_result)
        
        # Print periodic summary
        if epoch % 100 == 0:
            self._print_evaluation_summary(epoch)
    
    def _print_evaluation_summary(self, epoch: int):
        """Print evaluation summary for current epoch."""
        if not self.evaluation_history:
            return
            
        latest = self.evaluation_history[-1]
        print(f"\n=== JEPA Evaluation Summary (Epoch {epoch}) ===")
        print(f"Prediction Error: {latest['prediction_error']:.6f}")
        print(f"Converged: {latest['is_converged']}")
        if latest['is_converged']:
            print(f"Convergence Epoch: {latest['convergence_epoch']}")
        print(f"Memory Usage: {latest['memory_mb']:.1f} MB "
              f"(+{latest['memory_increase_mb']:.1f} MB from baseline)")
        print(f"Memory Efficiency Ratio: {latest['memory_efficiency_ratio']:.4f}")
        print(f"Best Error: {self.best_error:.6f} at Epoch {self.best_epoch}")
        print("=" * 50)
    
    def get_evaluation_report(self) -> Dict[str, Any]:
        """
        Get comprehensive evaluation report.
        
        Returns:
            Dictionary with complete evaluation results
        """
        if not self.evaluation_history:
            return {"error": "No evaluation data available"}
        
        latest = self.evaluation_history[-1]
        
        # Calculate trends
        if len(self.evaluation_history) >= 10:
            recent_errors = [e['prediction_error'] for e in self.evaluation_history[-10:] 
                           if e['prediction_error'] is not None]
            error_trend = "improving" if len(recent_errors) >= 2 and recent_errors[-1] < recent_errors[0] else "stable"
        else:
            error_trend = "insufficient_data"
        
        # Extract per-epoch history for visualization
        epoch_history = []
        for eval_entry in self.evaluation_history:
            epoch_history.append({
                'epoch': eval_entry.get('epoch', 0),
                'prediction_error': eval_entry.get('prediction_error', 0),
                'memory_mb': eval_entry.get('memory_mb', 0),
                'memory_increase_mb': eval_entry.get('memory_increase_mb', 0),
                'is_converged': eval_entry.get('is_converged', False),
                'timestamp': eval_entry.get('timestamp', 0)
            })
        
        report = {
            'overall_assessment': {
                'is_converged': latest['is_converged'],
                'convergence_epoch': latest['convergence_epoch'],
                'final_prediction_error': latest['prediction_error'],
                'best_prediction_error': self.best_error,
                'best_epoch': self.best_epoch,
                'error_trend': error_trend
            },
            'memory_efficiency': {
                'current_memory_mb': latest['memory_mb'],
                'memory_increase_mb': latest['memory_increase_mb'],
                'memory_growth_rate_mb_per_unit': latest.get('memory_growth_rate_mb_per_unit', 0),
                'memory_efficiency_ratio': latest['memory_efficiency_ratio'],
                'memory_scaling_quality': 'O(1)' if latest['memory_efficiency_ratio'] < 0.1 else 
                                        'sub-linear' if latest['memory_efficiency_ratio'] < 0.5 else
                                        'linear_or_worse'
            },
            'spatial_dynamics': {
                'spatial_error_map': latest['spatial_error_map'],
                'error_homogeneity': self._calculate_spatial_homogeneity(latest['spatial_error_map'])
            },
            'temporal_dynamics': {
                'temporal_profile_length': len(latest['temporal_profile'][0]) if latest['temporal_profile'][0] else 0,
                'recent_error_trend': error_trend
            },
            'evaluation_history_length': len(self.evaluation_history),
            'epoch_history': epoch_history  # Per-epoch data for visualization
        }
        
        return report
    
    def _calculate_spatial_homogeneity(self, spatial_error_map: Dict[int, float]) -> str:
        """Calculate how homogeneous the error is across spatial layers."""
        if not spatial_error_map or len(spatial_error_map) < 2:
            return "insufficient_data"
        
        errors = list(spatial_error_map.values())
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        if mean_error > 0:
            cv = std_error / mean_error  # Coefficient of variation
            if cv < 0.1:
                return "highly_homogeneous"
            elif cv < 0.3:
                return "moderately_homogeneous"
            else:
                return "heterogeneous"
        else:
            return "uniform_zero_error"
    
    def save_evaluation_report(self, filepath: str):
        """
        Save evaluation report to file.
        
        Args:
            filepath: Path to save the report
        """
        import json
        
        report = self.get_evaluation_report()
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        report_serializable = convert_numpy(report)
        
        with open(filepath, 'w') as f:
            json.dump(report_serializable, f, indent=2)
        
        print(f"Evaluation report saved to {filepath}")


def create_jepa_evaluator_example():
    """
    Create an example showing how to use the JEPA Evaluator.
    """
    print("Creating JEPA Evaluator example...")
    print("Note: This example requires a world_model instance.")
    print("In practice, this would be initialized from world_core.py or predictive_coding.py")
    
    # This is a structural example - actual usage requires initialized world_model
    print("\nJEPAEvaluator is ready for integration with:")
    print("- PredictiveCodingWorldCore (from world_core.py or predictive_coding.py)")
    print("\nEvaluation capabilities:")
    print("  - Local prediction error tracking (should converge downward)")
    print("  - Memory footprint tracking (should show O(1) scaling)")
    print("  - Spatial error mapping (across x-layers)")
    print("  - Temporal error profiling")
    print("  - Convergence detection")
    
    return None  # Return None since we can't instantiate without dependencies


if __name__ == "__main__":
    # Run the example
    create_jepa_evaluator_example()