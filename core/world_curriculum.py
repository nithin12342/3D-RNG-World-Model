"""
World Curriculum Trainer for 3D-RNG World Engine
Lead Multi-Modal Data Scientist & Training Architect Implementation

This script implements Phase 2: World Curriculum Trainer for the 3D-RNG World Engine,
implementing a strict Unsupervised-to-Action-Conditioned curriculum learning approach
with Stage 1 (Observation) and Stage 2 (Action-Rollout) as specified.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Callable
import itertools
import random
from collections import deque


class WorldCurriculumTrainer:
    """
    Implements the curriculum learning pipeline for the 3D-RNG World Engine.
    Follows a strict Unsupervised-to-Action-Conditioned curriculum:
    Stage 1: Pure observation/prediction (self-supervised)
    Stage 2: Action-conditioned rollout with future prediction evaluation
    """
    
    def __init__(self,
                 world_model,  # PredictiveCodingWorldCore or WorldCore3D instance
                 spatial_tokenizer,  # SpatialTokenizer instance
                 curriculum_stages: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the World Curriculum Trainer.
        
        Args:
            world_model: The 3D-RNG world model (PredictiveCodingWorldCore)
            spatial_tokenizer: The spatial tokenizer for multi-modal input
            curriculum_stages: List of curriculum stage configurations
        """
        self.world_model = world_model
        self.spatial_tokenizer = spatial_tokenizer
        
        # Default curriculum stages if not provided
        if curriculum_stages is None:
            self.curriculum_stages = [
                {
                    'name': 'Stage_1_Observation',
                    'focus': 'unsupervised_world_modeling',
                    'objective': 'learn_spatial_temporal_dynamics_through_prediction',
                    'sensory_dropout': 0.0,  # No dropout - full sensory input
                    'action_injection': False,  # No action injection
                    'future_prediction_horizon': 0,  # Not evaluating future prediction yet
                    'rollout_length': 1,  # Single step observation
                    'exploration_rate': 0.3,  # High exploration for discovery
                    'duration_epochs': 1000
                },
                {
                    'name': 'Stage_2_Action_Rollout',
                    'focus': 'action_conditioned_prediction',
                    'objective': 'learn_action_consequences_and_planning',
                    'sensory_dropout': 0.5,  # Dropout future sensory inputs during rollout
                    'action_injection': True,  # Inject action vectors
                    'future_prediction_horizon': 5,  # Predict 5 steps ahead
                    'rollout_length': 10,  # Rollout 10 steps
                    'exploration_rate': 0.1,  # Lower exploitation for precision
                    'duration_epochs': 2000
                }
            ]
        else:
            self.curriculum_stages = curriculum_stages
            
        self.current_stage_idx = 0
        self.current_epoch = 0
        self.stage_epoch = 0
        
        # For Stage 2: Action-Rollout
        self.action_buffer = deque(maxlen=100)  # Store recent actions for RandOpt DSP
        self.prediction_accuracy_history = deque(maxlen=200)  # Track prediction accuracy
        self.best_action_sequences = []  # Store distilled action sequences
        
        print(f"WorldCurriculumTrainer initialized with {len(self.curriculum_stages)} stages")
        for i, stage in enumerate(self.curriculum_stages):
            print(f"  Stage {i+1}: {stage['name']} - {stage['objective']}")
    
    def get_current_stage(self) -> Dict[str, Any]:
        """Get the current curriculum stage configuration."""
        if self.current_stage_idx >= len(self.curriculum_stages):
            # Return the final stage if we've progressed beyond
            return self.curriculum_stages[-1]
        return self.curriculum_stages[self.current_stage_idx]
    
    def advance_stage(self):
        """Advance to the next curriculum stage."""
        if self.current_stage_idx < len(self.curriculum_stages) - 1:
            self.current_stage_idx += 1
            self.stage_epoch = 0
            print(f"Advancing to Stage {self.current_stage_idx + 1}: "
                  f"{self.curriculum_stages[self.current_stage_idx]['name']}")
        else:
            print("Already at final curriculum stage")
    
    def update_epoch(self):
        """Update epoch counters and check for stage advancement."""
        self.current_epoch += 1
        self.stage_epoch += 1
        
        current_stage = self.get_current_stage()
        if self.stage_epoch >= current_stage['duration_epochs']:
            self.advance_stage()
    
    def prepare_stage_1_input(self, 
                             video_frame: Optional[np.ndarray] = None,
                             text: Optional[str] = None) -> Tuple[Optional[np.ndarray], 
                                                                 Optional[np.ndarray], 
                                                                 Optional[np.ndarray]]:
        """
        Prepare inputs for Stage 1 (Observation): Pure unsupervised learning.
        
        Args:
            video_frame: Optional video frame
            text: Optional text
            
        Returns:
            Tuple of (vision_input, text_input, action_input) for world model
        """
        # Get current stage config
        stage = self.get_current_stage()
        
        # Tokenize multi-modal input
        vision_mapping, text_mapping = self.spatial_tokenizer.tokenize_multi_modal(
            video_frame, text
        )
        
        # Convert mappings to arrays for world model injection
        vision_input = self._mapping_to_array(vision_mapping, 
                                            self.spatial_tokenizer.vision_face_size)
        text_input = self._mapping_to_array(text_mapping,
                                          self.spatial_tokenizer.text_face_size)
        
        # Stage 1: No action injection (pure observation)
        action_input = None
        
        # Apply sensory dropout if specified (typically 0 for Stage 1)
        if stage['sensory_dropout'] > 0:
            vision_input = self._apply_dropout(vision_input, stage['sensory_dropout'])
            text_input = self._apply_dropout(text_input, stage['sensory_dropout'])
        
        return vision_input, text_input, action_input
    
    def prepare_stage_2_input(self,
                             video_frame: Optional[np.ndarray] = None,
                             text: Optional[str] = None,
                             action_vector: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], 
                                                                                  Optional[np.ndarray], 
                                                                                  Optional[np.ndarray]]:
        """
        Prepare inputs for Stage 2 (Action-Rollout): Action-conditioned prediction.
        
        Args:
            video_frame: Optional video frame
            text: Optional text
            action_vector: Action vector to inject into Action Zone
            
        Returns:
            Tuple of (vision_input, text_input, action_input) for world model
        """
        # Get current stage config
        stage = self.get_current_stage()
        
        # Tokenize multi-modal input
        vision_mapping, text_mapping = self.spatial_tokenizer.tokenize_multi_modal(
            video_frame, text
        )
        
        # Convert mappings to arrays
        vision_input = self._mapping_to_array(vision_mapping,
                                            self.spatial_tokenizer.vision_face_size)
        text_input = self._mapping_to_array(text_mapping,
                                          self.spatial_tokenizer.text_face_size)
        
        # Stage 2: Action injection enabled
        if stage['action_injection'] and action_vector is not None:
            action_input = action_vector.copy()
        else:
            # Generate exploratory action if none provided
            action_input = self._generate_exploratory_action(stage['exploration_rate'])
        
        # Apply sensory dropout for future steps during rollout
        # (This would be applied internally during the rollout process)
        
        return vision_input, text_input, action_input
    
    def _mapping_to_array(self, mapping: Dict[Tuple[int, int, int], np.ndarray],
                         face_size: Tuple[int, int]) -> np.ndarray:
        """
        Convert coordinate mapping to array format for world model injection.
        
        Args:
            mapping: Dictionary mapping (x, y, z) coordinates to vectors
            face_size: Size of the face grid (height, width)
            
        Returns:
            Array of shape (num_positions, embed_dim)
        """
        height, width = face_size
        num_positions = height * width
        
        # Determine embedding dimension from first non-empty mapping
        embed_dim = None
        for vector in mapping.values():
            if vector is not None and len(vector) > 0:
                embed_dim = len(vector)
                break
        
        if embed_dim is None:
            # Default embedding dimension
            embed_dim = getattr(self.spatial_tokenizer, 'embed_dim', 384)
        
        # Initialize output array
        output_array = np.zeros((num_positions, embed_dim))
        
        # Fill array from mapping
        idx = 0
        for y in range(height):
            for z in range(width):
                coord = (0, y, z) if face_size == self.spatial_tokenizer.vision_face_size else (1, y, z)
                if coord in mapping and mapping[coord] is not None:
                    output_array[idx] = mapping[coord]
                idx += 1
                
        return output_array
    
    def _apply_dropout(self, array: Optional[np.ndarray], dropout_rate: float) -> Optional[np.ndarray]:
        """
        Apply dropout to input array.
        
        Args:
            array: Input array
            dropout_rate: Dropout rate (0 to 1)
            
        Returns:
            Array with dropout applied
        """
        if array is None:
            return None
            
        mask = np.random.random(array.shape) > dropout_rate
        return array * mask
    
    def _generate_exploratory_action(self, exploration_rate: float) -> np.ndarray:
        """
        Generate an exploratory action vector.
        
        Args:
            exploration_rate: Rate of exploration (0 to 1)
            
        Returns:
            Action vector
        """
        # Action dimension should match world model's action zone size
        action_height, action_width = self.spatial_tokenizer.action_zone_size
        action_size = action_height * action_width
        embed_dim = getattr(self.spatial_tokenizer, 'embed_dim', 384)
        
        # Base action: small random vector
        action = np.random.randn(action_size, embed_dim) * 0.1
        
        # Add exploratory noise based on exploration rate
        if exploration_rate > 0:
            noise = np.random.randn(*action.shape) * exploration_rate * 0.5
            action += noise
            
        return action
    
    def train_stage_1_observation(self,
                                 video_stream: List[np.ndarray],
                                 text_stream: List[str],
                                 num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute Stage 1: Observation curriculum.
        Pure unsupervised learning through prediction error minimization.
        
        Args:
            video_stream: List of video frames
            text_stream: List of corresponding text descriptions
            num_epochs: Number of epochs to train (overrides stage duration)
            
        Returns:
            Training statistics
        """
        stage = self.get_current_stage()
        epochs = num_epochs if num_epochs is not None else stage['duration_epochs']
        
        print(f"\n=== Starting {stage['name']} ===")
        print(f"Objective: {stage['objective']}")
        print(f"Training for {epochs} epochs")
        
        # Reset world model for fresh start
        self.world_model.reset_world()
        
        # Training metrics
        total_prediction_error = 0.0
        epoch_errors = []
        
        # Create data iterator (cycle through streams if needed)
        data_length = min(len(video_stream), len(text_stream))
        if data_length == 0:
            raise ValueError("Video and text streams must have matching non-zero length")
        
        for epoch in range(epochs):
            # Get current data point (with cycling)
            idx = epoch % data_length
            video_frame = video_stream[idx]
            text = text_stream[idx]
            
            # Prepare Stage 1 input
            vision_input, text_input, action_input = self.prepare_stage_1_input(
                video_frame, text
            )
            
            # Advance world model one step
            outputs = self.world_model.tick_world(
                vision_input=vision_input,
                text_input=text_input,
                action_input=action_input
            )
            
            # Calculate prediction error (from world model statistics)
            stats = self.world_model.get_world_statistics()
            prediction_error = stats['average_prediction_error']
            total_prediction_error += prediction_error
            epoch_errors.append(prediction_error)
            
            # Log progress
            if epoch % 100 == 0:
                avg_error = np.mean(epoch_errors[-100:]) if len(epoch_errors) >= 100 else np.mean(epoch_errors)
                print(f"Epoch {epoch}: Avg Prediction Error = {avg_error:.4f}")
        
        # Final statistics
        final_stats = self.world_model.get_world_statistics()
        avg_prediction_error = total_prediction_error / epochs
        
        results = {
            'stage': stage['name'],
            'epochs_completed': epochs,
            'average_prediction_error': avg_prediction_error,
            'final_activation_ratio': final_stats['activation_ratio'],
            'final_refractory_ratio': final_stats['refractory_ratio'],
            'epoch_errors': epoch_errors
        }
        
        print(f"=== {stage['name']} Complete ===")
        print(f"Average Prediction Error: {avg_prediction_error:.4f}")
        print(f"Final Activation Ratio: {final_stats['activation_ratio']:.3f}")
        
        return results
    
    def train_stage_2_action_rollout(self,
                                   video_stream: List[np.ndarray],
                                   text_stream: List[str],
                                   num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute Stage 2: Action-Rollout curriculum.
        Learn action consequences through future prediction and RandOpt DSP.
        
        Args:
            video_stream: List of video frames
            text_stream: List of corresponding text descriptions
            num_epochs: Number of epochs to train (overrides stage duration)
            
        Returns:
            Training statistics
        """
        stage = self.get_current_stage()
        epochs = num_epochs if num_epochs is not None else stage['duration_epochs']
        
        print(f"\n=== Starting {stage['name']} ===")
        print(f"Objective: {stage['objective']}")
        print(f"Training for {epochs} epochs")
        print(f"Rollout length: {stage['rollout_length']}")
        print(f"Future prediction horizon: {stage['future_prediction_horizon']}")
        
        # Training metrics
        total_prediction_error = 0.0
        rollout_accuracies = []
        epoch_errors = []
        
        # Create data iterator
        data_length = min(len(video_stream), len(text_stream))
        if data_length == 0:
            raise ValueError("Video and text streams must have matching non-zero length")
        
        for epoch in range(epochs):
            # Get current data point
            idx = epoch % data_length
            video_frame = video_stream[idx]
            text = text_stream[idx]
            
            # Generate action for this step (could come from policy or exploration)
            action_vector = self._generate_exploratory_action(stage['exploration_rate'])
            self.action_buffer.append(action_vector.copy())
            
            # Execute action rollout
            rollout_result = self._execute_action_rollout(
                video_frame, text, action_vector, stage
            )
            
            # Accumulate metrics
            total_prediction_error += rollout_result['prediction_error']
            rollout_accuracies.append(rollout_result['accuracy'])
            epoch_errors.append(rollout_result['prediction_error'])
            
            # Log progress
            if epoch % 100 == 0:
                recent_accuracy = np.mean(rollout_accuracies[-100:]) if len(rollout_accuracies) >= 100 else np.mean(rollout_accuracies)
                recent_error = np.mean(epoch_errors[-100:]) if len(epoch_errors) >= 100 else np.mean(epoch_errors)
                print(f"Epoch {epoch}: Avg Accuracy = {recent_accuracy:.3f}, Avg Error = {recent_error:.4f}")
        
        # Apply RandOpt DSP to distill best action sequences
        self._apply_randopt_dsp()
        
        # Final statistics
        final_stats = self.world_model.get_world_statistics()
        avg_prediction_error = total_prediction_error / epochs
        avg_accuracy = np.mean(rollout_accuracies) if rollout_accuracies else 0.0
        
        results = {
            'stage': stage['name'],
            'epochs_completed': epochs,
            'average_prediction_error': avg_prediction_error,
            'average_accuracy': avg_accuracy,
            'final_activation_ratio': final_stats['activation_ratio'],
            'final_refractory_ratio': final_stats['refractory_ratio'],
            'rollout_accuracies': rollout_accuracies,
            'epoch_errors': epoch_errors,
            'best_action_sequences': self.best_action_sequences.copy()
        }
        
        print(f"=== {stage['name']} Complete ===")
        print(f"Average Prediction Error: {avg_prediction_error:.4f}")
        print(f"Average Accuracy: {avg_accuracy:.3f}")
        print(f"Distilled Action Sequences: {len(self.best_action_sequences)}")
        
        return results
    
    def _execute_action_rollout(self,
                               video_frame: np.ndarray,
                               text: str,
                               initial_action: np.ndarray,
                               stage: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single action rollout for Stage 2 training.
        
        Args:
            video_frame: Initial video frame
            text: Initial text
            initial_action: Initial action vector to inject
            stage: Current stage configuration
            
        Returns:
            Rollout results dictionary
        """
        rollout_length = stage['rollout_length']
        prediction_horizon = stage['future_prediction_horizon']
        sensory_dropout = stage['sensory_dropout']
        
        # Reset to initial state for consistent rollout
        initial_state = self._capture_world_state()
        
        # Inject initial action and sensory input
        vision_input, text_input, _ = self.prepare_stage_2_input(
            video_frame, text, initial_action
        )
        
        # Apply sensory input (no dropout for initial step)
        outputs = self.world_model.tick_world(
            vision_input=vision_input,
            text_input=text_input,
            action_input=initial_action
        )
        
        # Store initial prediction for comparison
        initial_prediction = outputs['prediction_target'].copy() if 'prediction_target' in outputs else None
        
        # Roll forward without sensory input (or with dropout)
        rollout_predictions = []
        rollout_actuals = []
        
        for step in range(1, rollout_length):
            # For future steps, we may zero out sensory input or apply dropout
            if step >= prediction_horizon:
                # Beyond prediction horizon: no sensory input (pure rollout)
                vision_input = None
                text_input = None
            else:
                # Within prediction horizon: apply sensory dropout
                # In practice, we'd use actual future frames, but for simulation we use dropout
                vision_input = np.zeros_like(vision_input) if vision_input is not None else None
                text_input = np.zeros_like(text_input) if text_input is not None else None
                
                # Apply dropout
                if vision_input is not None:
                    vision_input = self._apply_dropout(vision_input, sensory_dropout)
                if text_input is not None:
                    text_input = self._apply_dropout(text_input, sensory_dropout)
            
            # No action injection during rollout (let dynamics evolve)
            action_input = None
            
            # Advance world
            outputs = self.world_model.tick_world(
                vision_input=vision_input,
                text_input=text_input,
                action_input=action_input
            )
            
            # Store prediction and actual state
            if 'prediction_target' in outputs:
                rollout_predictions.append(outputs['prediction_target'].copy())
            if 'state' in outputs:
                rollout_actuals.append(outputs['state'].copy())
        
        # Calculate rollout accuracy
        accuracy = self._calculate_rollout_accuracy(
            rollout_predictions, rollout_actuals, initial_prediction
        )
        
        # Calculate prediction error (deviation from initial state prediction)
        prediction_error = self._calculate_prediction_error(
            rollout_predictions, rollout_actuals
        )
        
        # Restore initial state for next iteration
        self._restore_world_state(initial_state)
        
        return {
            'prediction_error': prediction_error,
            'accuracy': accuracy,
            'rollout_length': rollout_length,
            'predictions': rollout_predictions,
            'actuals': rollout_actuals
        }
    
    def _capture_world_state(self) -> Dict[str, Any]:
        """Capture the current state of the world model for restoration."""
        state = {
            'node_states': {},
            'shared_weights': self.world_model.shared_weights.copy() if hasattr(self.world_model, 'shared_weights') else None,
            'connection_weights': {}
        }
        
        # Save node states
        if hasattr(self.world_model, 'nodes'):
            for coord, node in self.world_model.nodes.items():
                state['node_states'][coord] = {
                    'hidden_state': node.hidden_state.copy(),
                    'refractory_counter': node.refractory_counter,
                    'bias': node.bias.copy() if hasattr(node, 'bias') else None
                }
                
                # Save predictive coding specific states if available
                if hasattr(node, 'predicted_neighbor_states'):
                    state['node_states'][coord]['predicted_neighbor_states'] = {
                        k: v.copy() for k, v in node.predicted_neighbor_states.items()
                    }
                if hasattr(node, 'prediction_errors'):
                    state['node_states'][coord]['prediction_errors'] = {
                        k: v.copy() for k, v in node.prediction_errors.items()
                    }
                if hasattr(node, 'connection_weights'):
                    state['node_states'][coord]['connection_weights'] = node.connection_weights.copy()
        
        return state
    
    def _restore_world_state(self, state: Dict[str, Any]):
        """Restore the world model to a previously captured state."""
        if state is None:
            return
            
        # Restore shared weights
        if state['shared_weights'] is not None and hasattr(self.world_model, 'shared_weights'):
            self.world_model.shared_weights[:] = state['shared_weights']
        
        # Restore node states
        if hasattr(self.world_model, 'nodes') and state['node_states']:
            for coord, node_state in state['node_states'].items():
                if coord in self.world_model.nodes:
                    node = self.world_model.nodes[coord]
                    node.hidden_state[:] = node_state['hidden_state']
                    node.refractory_counter = node_state['refractory_counter']
                    if node_state['bias'] is not None and hasattr(node, 'bias'):
                        node.bias[:] = node_state['bias']
                    
                    # Restore predictive coding states
                    if 'predicted_neighbor_states' in node_state and hasattr(node, 'predicted_neighbor_states'):
                        node.predicted_neighbor_states = {
                            k: v.copy() for k, v in node_state['predicted_neighbor_states'].items()
                        }
                    if 'prediction_errors' in node_state and hasattr(node, 'prediction_errors'):
                        node.prediction_errors = {
                            k: v.copy() for k, v in node_state['prediction_errors'].items()
                        }
                    if 'connection_weights' in node_state and hasattr(node, 'connection_weights'):
                        node.connection_weights = node_state['connection_weights'].copy()
    
    def _calculate_rollout_accuracy(self,
                                  predictions: List[np.ndarray],
                                  actuals: List[np.ndarray],
                                  initial_prediction: Optional[np.ndarray]) -> float:
        """
        Calculate accuracy of action rollout predictions.
        
        Args:
            predictions: List of predicted states
            actuals: List of actual states
            initial_prediction: Initial prediction for baseline
            
        Returns:
            Accuracy score (0 to 1)
        """
        if not predictions or not actuals:
            return 0.0
            
        # Ensure we have matching lengths
        min_len = min(len(predictions), len(actuals))
        if min_len == 0:
            return 0.0
            
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]
        
        # Calculate prediction accuracy as 1 - normalized error
        total_error = 0.0
        total_norm = 0.0
        
        for pred, actual in zip(predictions, actuals):
            error = np.linalg.norm(pred - actual)
            norm = np.linalg.norm(actual) + 1e-8  # Avoid division by zero
            total_error += error
            total_norm += norm
        
        if total_norm > 0:
            normalized_error = total_error / total_norm
            accuracy = max(0.0, 1.0 - normalized_error)
        else:
            accuracy = 0.0
            
        return accuracy
    
    def _calculate_prediction_error(self,
                                  predictions: List[np.ndarray],
                                  actuals: List[np.ndarray]) -> float:
        """
        Calculate average prediction error over rollout.
        
        Args:
            predictions: List of predicted states
            actuals: List of actual states
            
        Returns:
            Average prediction error
        """
        if not predictions or not actuals:
            return float('inf')
            
        min_len = min(len(predictions), len(actuals))
        if min_len == 0:
            return float('inf')
            
        errors = []
        for pred, actual in zip(predictions[:min_len], actuals[:min_len]):
            error = np.linalg.norm(pred - actual)
            errors.append(error)
            
        return np.mean(errors) if errors else float('inf')
    
    def _apply_randopt_dsp(self):
        """
        Apply RandOpt DSP (Random Optimization Digital Signal Processing) 
        to distill the most accurate future-predicting spatial paths.
        """
        print("Applying RandOpt DSP to distill best action sequences...")
        
        if len(self.action_buffer) < 10:
            print("Insufficient action history for RandOpt DSP")
            return
        
        # Convert action buffer to array for processing
        action_history = np.array(list(self.action_buffer))
        accuracy_history = np.array(list(self.prediction_accuracy_history))
        
        if len(accuracy_history) == 0:
            # Use exploration-based selection if no accuracy history
            # Select actions with moderate exploration (not too random, not too fixed)
            action_norms = np.linalg.norm(action_history, axis=(1, 2))
            # Prefer medium-norm actions (balanced exploration/exploitation)
            sorted_indices = np.argsort(np.abs(action_norms - np.median(action_norms)))
            best_indices = sorted_indices[:min(10, len(sorted_indices))]
        else:
            # Select top-performing action sequences based on accuracy
            # Use weighted combination of recency and accuracy
            recency_weights = np.exp(-np.arange(len(accuracy_history))[::-1] / 20.0)  # Exponential recency weighting
            recency_weights = recency_weights / np.sum(recency_weights)
            
            combined_scores = accuracy_history * recency_weights[-len(accuracy_history):]
            best_indices = np.argsort(combined_scores)[-10:]  # Top 10
        
        # Extract best action sequences
        self.best_action_sequences = []
        for idx in best_indices:
            if idx < len(action_history):
                sequence = action_history[idx:idx+min(5, len(action_history)-idx)]  # Up to 5-step sequences
                if len(sequence) > 0:
                    self.best_action_sequences.append(sequence.copy())
        
        print(f"Distilled {len(self.best_action_sequences)} best action sequences")
    
    def get_curriculum_progress(self) -> Dict[str, Any]:
        """
        Get current curriculum progress information.
        
        Returns:
            Dictionary with progress metrics
        """
        current_stage = self.get_current_stage()
        progress_in_stage = self.stage_epoch / max(current_stage['duration_epochs'], 1)
        
        return {
            'current_stage_idx': self.current_stage_idx,
            'current_stage_name': current_stage['name'],
            'current_epoch': self.current_epoch,
            'stage_epoch': self.stage_epoch,
            'progress_in_stage': min(progress_in_stage, 1.0),
            'total_stages': len(self.curriculum_stages),
            'is_final_stage': self.current_stage_idx >= len(self.curriculum_stages) - 1
        }


def create_world_curriculum_example():
    """
    Create an example showing how to use the World Curriculum Trainer.
    """
    print("Creating World Curriculum Trainer example...")
    print("Note: This example requires world_model and spatial_tokenizer instances.")
    print("In practice, these would be initialized from world_core.py and spatial_tokenizer.py")
    
    # This is a structural example - actual usage requires initialized components
    print("\nWorldCurriculumTrainer is ready for integration with:")
    print("- PredictiveCodingWorldCore (from world_core.py or predictive_coding.py)")
    print("- SpatialTokenizer (from spatial_tokenizer.py)")
    print("\nCurriculum stages:")
    print("  Stage 1: Unsupervised observation/prediction learning")
    print("  Stage 2: Action-conditioned rollout with RandOpt DSP")
    
    return None  # Return None since we can't instantiate without dependencies


if __name__ == "__main__":
    # Run the example
    create_world_curriculum_example()