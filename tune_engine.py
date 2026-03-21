"""
Hyperparameter Tuning Script for 3D-RNG World Engine
Lead ML Systems Tuner & Hyperparameter Optimizer

This script performs a grid search over learning_rate and leak_rate to optimize
the PredictiveCodingWorldCore for better prediction performance.
"""

import numpy as np
import time
import json
import os
from typing import Dict, Any, Tuple, List

# Import 3D-RNG modules
from core.predictive_coding import PredictiveCodingWorldCore


# Configuration
WORLD_DIMENSIONS = (10, 10, 10)  # Manageable size for testing
HIDDEN_SIZE = 32
VISION_FACE_SIZE = (4, 4)
TEXT_FACE_SIZE = (2, 2)
ACTION_ZONE_SIZE = (2, 2)

# Grid search parameters
LEARNING_RATES = [0.05, 0.1, 0.25, 0.5]
LEAK_RATES = [0.05, 0.2, 0.5]
NUM_EPOCHS = 50  # Shortened for tuning speed

# MoE parameters (disabled for tuning to focus on base neurochemistry)
USE_MOE = False
NUM_EXPERTS = 8
MOE_K = 2
NUM_BLOCKS = 8


def run_training_epoch(world_model: PredictiveCodingWorldCore,
                      vision_input: np.ndarray,
                      text_input: np.ndarray) -> float:
    """
    Run a single training epoch and return the prediction error.
    
    Returns:
        Average prediction error across all nodes
    """
    # Advance world with inputs
    outputs = world_model.tick_world(vision_input, text_input, None)
    
    # Calculate prediction error from world statistics
    stats = world_model.get_world_statistics()
    return stats.get('average_prediction_error', 0.0)


def evaluate_configuration(learning_rate: float, leak_rate: float) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate a single hyperparameter configuration.
    
    Returns:
        Tuple of (final_prediction_error, config_dict)
    """
    config = {
        'learning_rate': learning_rate,
        'leak_rate': leak_rate,
        'hidden_size': HIDDEN_SIZE,
        'dimensions': WORLD_DIMENSIONS,
        'epochs': NUM_EPOCHS
    }
    
    # Initialize world model with specific hyperparameters
    world_model = PredictiveCodingWorldCore(
        dim_x=WORLD_DIMENSIONS[0],
        dim_y=WORLD_DIMENSIONS[1],
        dim_z=WORLD_DIMENSIONS[2],
        hidden_size=HIDDEN_SIZE,
        leak_rate=leak_rate,
        learning_rate=learning_rate,
        vision_face_size=VISION_FACE_SIZE,
        text_face_size=TEXT_FACE_SIZE,
        action_zone_size=ACTION_ZONE_SIZE,
        use_moe=USE_MOE,
        num_experts=NUM_EXPERTS,
        moe_k=MOE_K,
        num_blocks=NUM_BLOCKS
    )
    
    # Run training epochs
    prediction_errors = []
    
    for epoch in range(NUM_EPOCHS):
        # Generate vision input (moving pattern)
        vision_phase = epoch * 0.1
        vision_input = np.zeros((len(world_model.vision_face_coords), HIDDEN_SIZE))
        for i in range(len(world_model.vision_face_coords)):
            vision_input[i] = np.sin(np.arange(HIDDEN_SIZE) * 0.1 + vision_phase + i) * 0.2
        
        # Generate text input (periodic pulses)
        text_input = np.zeros((len(world_model.text_face_coords), HIDDEN_SIZE))
        if epoch % 3 == 0:  # Pulse every 3 steps
            text_input[:, :] = 0.5
        
        # Run epoch
        pred_error = run_training_epoch(world_model, vision_input, text_input)
        prediction_errors.append(pred_error)
    
    # Return final prediction error (average of last 10 epochs)
    final_error = np.mean(prediction_errors[-10:]) if len(prediction_errors) >= 10 else np.mean(prediction_errors)
    
    return final_error, config


def main():
    """Run the hyperparameter grid search."""
    print("=" * 70)
    print("3D-RNG World Engine Hyperparameter Tuning")
    print("=" * 70)
    print(f"Grid Search Space:")
    print(f"  Learning Rates: {LEARNING_RATES}")
    print(f"  Leak Rates: {LEAK_RATES}")
    print(f"  Epochs per config: {NUM_EPOCHS}")
    print(f"  Total configurations: {len(LEARNING_RATES) * len(LEAK_RATES)}")
    print("=" * 70)
    print()
    
    # Store results
    results = []
    
    # Grid search
    total_configs = len(LEARNING_RATES) * len(LEAK_RATES)
    config_num = 0
    
    for lr in LEARNING_RATES:
        for leak in LEAK_RATES:
            config_num += 1
            print(f"[{config_num}/{total_configs}] Testing: learning_rate={lr}, leak_rate={leak}")
            
            start_time = time.time()
            try:
                pred_error, config = evaluate_configuration(lr, leak)
                elapsed = time.time() - start_time
                
                results.append({
                    'learning_rate': lr,
                    'leak_rate': leak,
                    'prediction_error': pred_error,
                    'elapsed_time': elapsed
                })
                
                print(f"  -> Prediction Error: {pred_error:.4f} (took {elapsed:.1f}s)")
                
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"  -> ERROR: {str(e)} (took {elapsed:.1f}s)")
                results.append({
                    'learning_rate': lr,
                    'leak_rate': leak,
                    'prediction_error': float('inf'),
                    'elapsed_time': elapsed,
                    'error': str(e)
                })
    
    # Sort by prediction error (lower is better)
    results.sort(key=lambda x: x['prediction_error'])
    
    # Print leaderboard
    print()
    print("=" * 70)
    print("LEADERBOARD (sorted by Prediction Error - lower is better)")
    print("=" * 70)
    print(f"{'Rank':<6}{'LR':<10}{'Leak Rate':<12}{'Pred Error':<15}{'Time (s)':<10}")
    print("-" * 70)
    
    for i, result in enumerate(results):
        rank = i + 1
        lr = result['learning_rate']
        leak = result['leak_rate']
        pred_err = result['prediction_error']
        elapsed = result['elapsed_time']
        print(f"{rank:<6}{lr:<10}{leak:<12}{pred_err:<15.4f}{elapsed:<10.1f}")
    
    print("=" * 70)
    
    # Get best configuration
    best = results[0]
    best_config = {
        'learning_rate': best['learning_rate'],
        'leak_rate': best['leak_rate'],
        'prediction_error': best['prediction_error'],
        'search_space': {
            'learning_rates': LEARNING_RATES,
            'leak_rates': LEAK_RATES,
            'epochs': NUM_EPOCHS
        }
    }
    
    print(f"\nBest Configuration:")
    print(f"  Learning Rate: {best['learning_rate']}")
    print(f"  Leak Rate: {best['leak_rate']}")
    print(f"  Prediction Error: {best['prediction_error']:.4f}")
    
    # Save best config
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/best_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\nBest config saved to artifacts/best_config.json")
    
    return best_config


if __name__ == "__main__":
    main()
