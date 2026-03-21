"""
Main Execution Script for 3D-RNG World Engine
Lead Integration & Operations Engineer Implementation

This script wires together all 3D-RNG World Model modules, executes a Moving MNIST sanity check,
and implements strict NaN/Explosion safeguards as specified in Phase 10: System Ignition & Debugging.
"""

import numpy as np
import time
import sys
import traceback
from typing import Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
import os

# Import our 3D-RNG modules
try:
    from world_core import WorldCore3D
    from predictive_coding import PredictiveCodingWorldCore
    from spatial_tokenizer import SpatialTokenizer
    from world_curriculum import WorldCurriculumTrainer
    from jepa_evaluator import JEPAEvaluator
    print("Successfully imported all 3D-RNG modules")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are in the same directory")
    sys.exit(1)


class MovingMNISTSimulator:
    """
    Simulates a Moving MNIST dataset for sanity checking.
    Generates simple bouncing square patterns that mimic digit movement.
    """
    
    def __init__(self, frame_size: Tuple[int, int] = (64, 64), 
                 digit_size: int = 8, speed: Tuple[float, float] = (2.0, 1.5)):
        """
        Initialize Moving MNIST simulator.
        
        Args:
            frame_size: Size of video frames (height, width)
            digit_size: Size of the moving "digit" square
            speed: Movement speed in (y, x) directions
        """
        self.frame_size = frame_size
        self.digit_size = digit_size
        self.speed = np.array(speed, dtype=float)
        self.position = np.array([frame_size[0]//2, frame_size[1]//2], dtype=float)
        self.velocity = self.speed.copy()
        
    def next_frame(self) -> np.ndarray:
        """
        Generate next frame in the sequence.
        
        Returns:
            Frame of shape (height, width, 1) with values in [0, 1]
        """
        # Create blank frame
        frame = np.zeros(self.frame_size, dtype=np.float32)
        
        # Update position
        self.position += self.velocity
        
        # Bounce off edges
        if self.position[0] <= self.digit_size//2 or self.position[0] >= self.frame_size[0] - self.digit_size//2:
            self.velocity[0] = -self.velocity[0]
            self.position[0] = np.clip(self.position[0], self.digit_size//2, 
                                     self.frame_size[0] - self.digit_size//2)
            
        if self.position[1] <= self.digit_size//2 or self.position[1] >= self.frame_size[1] - self.digit_size//2:
            self.velocity[1] = -self.velocity[1]
            self.position[1] = np.clip(self.position[1], self.digit_size//2, 
                                     self.frame_size[1] - self.digit_size//2)
        
        # Draw square digit
        y_start = int(self.position[0] - self.digit_size//2)
        y_end = int(self.position[0] + self.digit_size//2)
        x_start = int(self.position[1] - self.digit_size//2)
        x_end = int(self.position[1] + self.digit_size//2)
        
        # Ensure bounds
        y_start = max(0, y_start)
        y_end = min(self.frame_size[0], y_end)
        x_start = max(0, x_start)
        x_end = min(self.frame_size[1], x_end)
        
        if y_end > y_start and x_end > x_start:
            frame[y_start:y_end, x_start:x_end] = 1.0
            
        # Add some noise for realism
        frame += np.random.randn(*self.frame_size) * 0.05
        frame = np.clip(frame, 0, 1)
        
        # Add channel dimension
        return frame[:, :, np.newaxis]
    
    def reset(self):
        """Reset simulator to initial state."""
        self.position = np.array([self.frame_size[0]//2, self.frame_size[1]//2], dtype=float)
        # Random initial velocity
        angle = np.random.uniform(0, 2*np.pi)
        speed_magnitude = np.linalg.norm(self.speed)
        self.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed_magnitude * 0.5


def check_for_nan_inf(tensor: np.ndarray, name: str, coordinates: Optional[Tuple[int, int, int]] = None) -> bool:
    """
    Check for NaN or Inf values in a tensor and report detailed information.
    
    Args:
        tensor: Tensor to check
        name: Name of the tensor for reporting
        coordinates: Optional coordinates for node-specific reporting
        
    Returns:
        True if NaN/Inf found, False otherwise
    """
    if tensor is None:
        return False
        
    has_nan = np.isnan(tensor).any()
    has_inf = np.isinf(tensor).any()
    
    if has_nan or has_inf:
        print(f"\n{'='*60}")
        print(f"NAN/INF DETECTED in {name}")
        if coordinates:
            print(f"At coordinates: {coordinates}")
        print(f"{'='*60}")
        print(f"Tensor shape: {tensor.shape}")
        print(f"Tensor dtype: {tensor.dtype}")
        print(f"NaN count: {np.isnan(tensor).sum()}")
        print(f"Inf count: {np.isinf(tensor).sum()}")
        
        # Show some statistics
        finite_vals = tensor[np.isfinite(tensor)]
        if len(finite_vals) > 0:
            print(f"Finite values - Min: {finite_vals.min():.6f}, Max: {finite_vals.max():.6f}, Mean: {finite_vals.mean():.6f}")
        
        # Show locations of NaN/Inf if tensor is not too large
        if tensor.size < 1000:
            nan_locations = np.where(np.isnan(tensor))
            inf_locations = np.where(np.isinf(tensor))
            if len(nan_locations[0]) > 0:
                print(f"NaN locations: {list(zip(*nan_locations))}")
            if len(inf_locations[0]) > 0:
                print(f"Inf locations: {list(zip(*inf_locations))}")
        else:
            # For large tensors, show first few problematic locations
            nan_mask = np.isnan(tensor)
            inf_mask = np.isinf(tensor)
            if nan_mask.any():
                coords = np.where(nan_mask)
                print(f"First 5 NaN locations: {list(zip(*[c[:5] for c in coords]))}")
            if inf_mask.any():
                coords = np.where(inf_mask)
                print(f"First 5 Inf locations: {list(zip(*[c[:5] for c in coords]))}")
        
        print(f"{'='*60}\n")
        return True
    
    return False


def validate_world_state(world_model, epoch: int) -> bool:
    """
    Validate the world model state for numerical stability.
    
    Args:
        world_model: The world model instance
        epoch: Current epoch number
        
    Returns:
        True if state is valid, False if NaN/Inf detected
    """
    try:
        # Check shared weights
        if hasattr(world_model, 'shared_weights'):
            if check_for_nan_inf(world_model.shared_weights, "shared_weights"):
                return False
        
        # Check node states
        if hasattr(world_model, 'nodes'):
            for coord, node in world_model.nodes.items():
                # Check hidden state
                if check_for_nan_inf(node.hidden_state, f"node_{coord}_hidden_state", coord):
                    # Dump debug information for this node
                    print(f"Debug info for exploding node {coord}:")
                    print(f"  Bias: {node.bias}")
                    if hasattr(node, 'predicted_neighbor_states'):
                        print(f"  Predicted neighbor states: {len(node.predicted_neighbor_states)} entries")
                    if hasattr(node, 'prediction_errors'):
                        print(f"  Prediction errors: {len(node.prediction_errors)} entries")
                    if hasattr(node, 'connection_weights'):
                        print(f"  Connection weights: {len(node.connection_weights)} entries")
                    return False
                
                # Check bias
                if check_for_nan_inf(node.bias, f"node_{coord}_bias", coord):
                    return False
                
                # Check predictive coding specific states
                if hasattr(node, 'predicted_neighbor_states'):
                    for neighbor_coord, pred_state in node.predicted_neighbor_states.items():
                        if check_for_nan_inf(pred_state, f"node_{coord}_pred_{neighbor_coord}", coord):
                            return False
                
                if hasattr(node, 'prediction_errors'):
                    for neighbor_coord, error in node.prediction_errors.items():
                        if check_for_nan_inf(error, f"node_{coord}_error_{neighbor_coord}", coord):
                            return False
                
                if hasattr(node, 'connection_weights'):
                    for neighbor_coord, weight in node.connection_weights.items():
                        if not np.isfinite(weight):
                            print(f"Non-finite connection weight: node {coord} -> {neighbor_coord}: {weight}")
                            return False
        
        return True
        
    except Exception as e:
        print(f"Error during world state validation: {e}")
        traceback.print_exc()
        return False


def create_synthetic_text_stream(num_frames: int) -> list:
    """
    Create a synthetic text stream corresponding to the Moving MNIST sequence.
    
    Args:
        num_frames: Number of frames in the sequence
        
    Returns:
        List of text descriptions
    """
    # Simple descriptions based on position quadrants
    descriptions = [
        "moving upper left",
        "moving upper right", 
        "moving lower left",
        "moving lower right",
        "bouncing vertically",
        "bouncing horizontally",
        "moving diagonally",
        "centered position"
    ]
    
    text_stream = []
    for i in range(num_frames):
        # Cycle through descriptions with some variation
        desc_idx = (i // 50) % len(descriptions)
        # Add frame number for uniqueness
        text_stream.append(f"{descriptions[desc_idx]} frame {i}")
    
    return text_stream


def main():
    """Main execution function for 3D-RNG World Engine sanity check."""
    print("="*70)
    print("3D-RNG WORLD ENGINE - SYSTEM IGNITION & DEBUGGING")
    print("Phase 10: Moving MNIST Sanity Check")
    print("="*70)
    
    # Configuration
    WORLD_DIMENSIONS = (10, 10, 10)  # Manageable size for testing
    HIDDEN_SIZE = 32
    VISION_FACE_SIZE = (8, 8)        # 8x8 grid on vision face (x=0)
    TEXT_FACE_SIZE = (4, 4)          # 4x4 grid on text face (x=1)
    ACTION_ZONE_SIZE = (2, 2)        # 2x2 grid on action zone (x=2)
    PATCH_SIZE = (8, 8)              # 8x8 patches for 64x64 frames
    EMBED_DIM = 32                   # Match hidden size
    NUM_EPOCHS = 500                 # As specified in requirements
    
    print(f"Configuration:")
    print(f"  World dimensions: {WORLD_DIMENSIONS}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Vision face: {VISION_FACE_SIZE}")
    print(f"  Text face: {TEXT_FACE_SIZE}")
    print(f"  Action zone: {ACTION_ZONE_SIZE}")
    print(f"  Patch size: {PATCH_SIZE}")
    print(f"  Embed dim: {EMBED_DIM}")
    print(f"  Training epochs: {NUM_EPOCHS}")
    print()
    
    try:
        # Initialize components
        print("Initializing 3D-RNG components...")
        
        # Option 1: Use the predictive coding world (local learning)
        world_model = PredictiveCodingWorldCore(
            dim_x=WORLD_DIMENSIONS[0],
            dim_y=WORLD_DIMENSIONS[1],
            dim_z=WORLD_DIMENSIONS[2],
            hidden_size=HIDDEN_SIZE,
            leak_rate=0.1,
            learning_rate=0.01,
            vision_face_size=VISION_FACE_SIZE,
            text_face_size=TEXT_FACE_SIZE,
            action_zone_size=ACTION_ZONE_SIZE
        )
        
        # Option 2: Use the basic world core (continuous latent)
        # world_model = WorldCore3D(
        #     dim_x=WORLD_DIMENSIONS[0],
        #     dim_y=WORLD_DIMENSIONS[1],
        #     dim_z=WORLD_DIMENSIONS[2],
        #     hidden_size=HIDDEN_SIZE,
        #     leak_rate=0.1,
        #     vision_face_size=VISION_FACE_SIZE,
        #     text_face_size=TEXT_FACE_SIZE,
        #     action_zone_size=ACTION_ZONE_SIZE
        # )
        
        print("[OK] World model initialized")
        
        # Initialize spatial tokenizer
        spatial_tokenizer = SpatialTokenizer(
            vision_face_size=VISION_FACE_SIZE,
            text_face_size=TEXT_FACE_SIZE,
            patch_size=PATCH_SIZE,
            embed_dim=EMBED_DIM,
            in_channels=1  # Grayscale for Moving MNIST simulation
        )
        print("[OK] Spatial tokenizer initialized")
        
        # Initialize world curriculum trainer
        curriculum_trainer = WorldCurriculumTrainer(
            world_model=world_model,
            spatial_tokenizer=spatial_tokenizer
        )
        print("[OK] World curriculum trainer initialized")
        
        # Initialize JEPA evaluator
        jepa_evaluator = JEPAEvaluator(world_model)
        print("[OK] JEPA evaluator initialized")
        
        # Initialize Moving MNIST simulator
        mnist_simulator = MovingMNISTSimulator(
            frame_size=(64, 64),
            digit_size=8,
            speed=(2.0, 1.5)
        )
        print("[OK] Moving MNIST simulator initialized")
        
        # Generate synthetic text stream
        text_stream = create_synthetic_text_stream(NUM_EPOCHS)
        print("[OK] Synthetic text stream generated")
        
        # Training metrics
        epoch_times = []
        memory_usage = []
        validation_passed = True
        nan_epoch = None
        
        print("\nStarting training loop...")
        print("-" * 70)
        
        # Main training loop
        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()
            
            # Generate next frame
            frame = mnist_simulator.next_frame()
            
            # Get corresponding text
            text = text_stream[epoch] if epoch < len(text_stream) else "ongoing sequence"
            
            # Prepare inputs using spatial tokenizer
            vision_mapping, text_mapping = spatial_tokenizer.tokenize_multi_modal(frame, text)
            
            # Convert mappings to arrays for world model
            vision_input = None
            text_input = None
            
            if vision_mapping:
                # Convert vision mapping to array format
                vision_height, vision_width = VISION_FACE_SIZE
                vision_array = np.zeros((vision_height * vision_width, EMBED_DIM))
                idx = 0
                for y in range(vision_height):
                    for z in range(vision_width):
                        coord = (0, y, z)
                        if coord in vision_mapping:
                            vision_array[idx] = vision_mapping[coord]
                        idx += 1
                vision_input = vision_array
            
            if text_mapping:
                # Convert text mapping to array format
                text_height, text_width = TEXT_FACE_SIZE
                text_array = np.zeros((text_height * text_width, EMBED_DIM))
                idx = 0
                for y in range(text_height):
                    for z in range(text_width):
                        coord = (1, y, z)
                        if coord in text_mapping:
                            text_array[idx] = text_mapping[coord]
                        idx += 1
                text_input = text_array
            
            # For Stage 1 (Observation), we don't inject actions initially
            action_input = None
            
            # Advance the world model
            try:
                outputs = world_model.tick_world(
                    vision_input=vision_input,
                    text_input=text_input,
                    action_input=action_input
                )
            except Exception as e:
                print(f"\nERROR during world tick at epoch {epoch}: {e}")
                traceback.print_exc()
                break
            
            # Validate world state for NaN/Inf
            if not validate_world_state(world_model, epoch):
                validation_passed = False
                nan_epoch = epoch
                print(f"\nVALIDATION FAILED at epoch {epoch}")
                break
            
            # Update JEPA evaluator
            # Estimate sequence length (could be actual frames processed)
            jepa_evaluator.evaluate_epoch(epoch, sequence_length=epoch+1)
            
            # Record timing
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            # Record memory usage
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_usage.append(memory_mb)
            
            # Progress reporting
            if epoch % 50 == 0 or epoch < 10:
                stats = world_model.get_world_statistics()
                print(f"Epoch {epoch:3d} | "
                      f"Time: {epoch_time*1000:5.1f}ms | "
                      f"Mem: {memory_mb:5.1f}MB | "
                      f"Act: {stats['activation_ratio']:.3f} | "
                      f"Err: {stats.get('average_prediction_error', 0):.4f}")
            
            # Early stopping if we're taking too long per epoch
            if epoch_time > 5.0:  # More than 5 seconds per epoch is too slow
                print(f"\nWarning: Epoch {epoch} took {epoch_time:.1f}s - consider reducing model size")
        
        print("-" * 70)
        
        # Final evaluation
        if validation_passed and nan_epoch is None:
            print("[OK] SUCCESS: Completed all epochs without NaN/Inf detection!")
            
            # Get final statistics
            final_stats = world_model.get_world_statistics()
            jepa_report = jepa_evaluator.get_evaluation_report()
            
            print(f"\nFinal Results:")
            print(f"  Epochs completed: {NUM_EPOCHS}")
            print(f"  Final activation ratio: {final_stats['activation_ratio']:.3f}")
            print(f"  Final prediction error: {jepa_report['overall_assessment']['final_prediction_error']:.6f}")
            print(f"  Convergence status: {'Converged' if jepa_report['overall_assessment']['is_converged'] else 'Not converged'}")
            if jepa_report['overall_assessment']['is_converged']:
                print(f"  Convergence epoch: {jepa_report['overall_assessment']['convergence_epoch']}")
            print(f"  Best prediction error: {jepa_report['overall_assessment']['best_prediction_error']:.6f} at epoch {jepa_report['overall_assessment']['best_epoch']}")
            print(f"  Memory usage: {jepa_report['memory_efficiency']['current_memory_mb']:.1f} MB")
            print(f"  Memory scaling quality: {jepa_report['memory_efficiency']['memory_scaling_quality']}")
            
            # Check memory footprint consistency (from tick 100 to 500)
            if len(memory_usage) >= 500:
                memory_100_500 = memory_usage[100:500]
                memory_std = np.std(memory_100_500)
                memory_mean = np.mean(memory_100_500)
                memory_cv = memory_std / memory_mean if memory_mean > 0 else 0
                
                print(f"\nMemory Consistency Check (epochs 100-500):")
                print(f"  Mean memory: {memory_mean:.1f} MB")
                print(f"  Std deviation: {memory_std:.1f} MB")
                 print(f"  Coefficient of variation: {memory_cv:.3f}")
                 
                 if memory_cv < 0.1:  # Less than 10% variation
                     print("  [OK] PASS: Memory footprint remains stable (O(1) scaling)")
                 else:
                     print("  [!] WARNING: Memory footprint shows significant variation")
            
            # Save JEPA evaluation report
            jepa_evaluator.save_evaluation_report("jepa_evaluation_report.json")
            print(f"  ✓ Saved JEPA evaluation report to jepa_evaluation_report.json")
            
            # Create and save prediction error plot
            if len(jepa_evaluator.prediction_tracker.prediction_error_history) > 0:
                epochs, errors = zip(*jepa_evaluator.prediction_tracker.prediction_error_history)
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, errors, 'b-', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Local Prediction Error')
                plt.title('3D-RNG Local Prediction Error Convergence (Moving MNIST)')
                plt.grid(True, alpha=0.3)
                plt.yscale('log')
                plt.savefig('local_prediction_error_curve.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved prediction error curve to local_prediction_error_curve.png")
            
        else:
            print(f"✗ FAILURE: NaN/Inf detected at epoch {nan_epoch}")
            print("The system has encountered numerical instability.")
            print("This provides valuable debugging information for fixing the architecture.")
            
            # Save debug information
            debug_info = {
                'failed_epoch': nan_epoch,
                'world_dimensions': WORLD_DIMENSIONS,
                'hidden_size': HIDDEN_SIZE,
                'epochs_completed': nan_epoch if nan_epoch is not None else NUM_EPOCHS
            }
            
            import json
            with open('debug_failure_info.json', 'w') as f:
                json.dump(debug_info, f, indent=2)
            print(f"  ✓ Saved debug information to debug_failure_info.json")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        validation_passed = False
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        traceback.print_exc()
        validation_passed = False
    
    print("\n" + "="*70)
    if validation_passed and nan_epoch is None:
        print("SYSTEM IGNITION SUCCESSFUL")
        print("The 3D-RNG World Engine has passed the Moving MNIST sanity check!")
    else:
        print("SYSTEM IGNITION FAILED")
        print("The 3D-RNG World Engine encountered issues requiring debugging.")
        print("Check the generated debug files for detailed failure analysis.")
    print("="*70)
    
    return validation_passed and nan_epoch is None


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)