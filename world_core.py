"""
Continuous Latent Core for 3D-RNG World Engine
Chief AI Scientist & World Model Architect Implementation

This script implements Phase 1: Continuous Latent Core for the 3D-RNG World Engine,
transforming the architecture from an episodic engine into a continuous multi-modal
world foundation model with persistent latent states and local predictive coding.
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional, Set
from collections import defaultdict
import itertools


class WorldNode:
    """
    Represents a single node in the 3D Recursive Neural Graph with continuous latent dynamics.
    Implements leaky integrator mechanism for persistent hidden states.
    """
    
    def __init__(self, coordinates: Tuple[int, int, int], hidden_size: int, 
                 leak_rate: float = 0.1):
        """
        Initialize a World Node with continuous latent dynamics.
        
        Args:
            coordinates: Tuple of (x, y, z) spatial position
            hidden_size: Dimensionality of the hidden state vector
            leak_rate: Rate of state decay (alpha in leaky integrator)
        """
        self.coordinates = coordinates
        self.hidden_size = hidden_size
        self.leak_rate = leak_rate  # Alpha in leaky integrator: h_t = (1-α)h_{t-1} + α*activation
        
        # Continuous latent state (persists across time steps)
        self.hidden_state = np.zeros(hidden_size)  # h_t
        self.bias = np.random.randn(hidden_size) * 0.1  # Node-specific bias
        self.refractory_counter = 0  # Tracks cooldown period
        
        # For predictive coding: store prediction of next state
        self.predicted_next_state = np.zeros(hidden_size)
        self.prediction_error = np.zeros(hidden_size)
        
    def update_state_continuous(self, incoming_state: np.ndarray, 
                               shared_weights: np.ndarray,
                               activation: str = 'tanh') -> np.ndarray:
        """
        Applies continuous update with leaky integrator and aggressive decay:
        h_t = ((1 - α) * h_{t-1} + α * activation(W_shared * incoming + b_v)) * decay_factor
        
        Args:
            incoming_state: Input state from connected nodes
            shared_weights: Global weight matrix shared across all nodes
            activation: Activation function ('tanh' or 'relu')
            
        Returns:
            Updated hidden state vector (h_t)
        """
        # --- ADVANCED LOGICAL ANALYZER ENFORCEMENT: STRICT kWTA & DECAY ---
        # Convert to torch tensors
        incoming_signals = torch.from_numpy(incoming_state).float()
        shared_weights_tensor = torch.from_numpy(shared_weights).float()
        hidden_state_tensor = torch.from_numpy(self.hidden_state).float()
        bias_tensor = torch.from_numpy(self.bias).float()
        
        # 1. Calculate signal magnitudes
        signal_mags = torch.norm(incoming_signals, dim=-1)
        
        # 2. Determine 15% sparsity threshold
        k_active = max(1, int(signal_mags.shape[0] * 0.15))
        top_k_vals, _ = torch.topk(signal_mags, k_active, dim=-1)
        threshold = top_k_vals[-1:]
        
        # 3. Generate binary mask (1.0 for winners, 0.0 for losers)
        kwta_mask = (signal_mags >= threshold).float().unsqueeze(-1)
        
        # 4. Eradicate weak signals BEFORE activation
        sparse_signals = incoming_signals * kwta_mask
        
        # 5. Damped Leaky Integrator Update (gamma = 0.80)
        decay_factor = 0.80
        self.hidden_state = (((1 - self.leak_rate) * hidden_state_tensor + self.leak_rate * torch.tanh(torch.matmul(shared_weights_tensor, sparse_signals) + bias_tensor)) * decay_factor).numpy()
        # --- END ENFORCEMENT ---
        
        return self.hidden_state
    
    def predict_next_state(self, shared_weights: np.ndarray,
                          activation: str = 'tanh') -> np.ndarray:
        """
        Predict the next state of this node based on current state and shared weights.
        Used for local predictive coding.
        
        Args:
            shared_weights: Global weight matrix
            activation: Activation function
            
        Returns:
            Predicted next state vector
        """
        weighted_input = np.dot(shared_weights, self.hidden_state)
        pre_activation = weighted_input + self.bias
        
        if activation == 'tanh':
            predicted = np.tanh(pre_activation)
        elif activation == 'relu':
            predicted = np.maximum(0, pre_activation)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
            
        self.predicted_next_state = predicted
        return predicted
    
    def update_prediction_error(self, actual_next_state: np.ndarray):
        """
        Update prediction error based on actual next state.
        
        Args:
            actual_next_state: Actual next state vector
        """
        self.prediction_error = actual_next_state - self.predicted_next_state
    
    def tick_refractory(self):
        """Decrements the refractory counter by 1 per time step."""
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
    
    def is_refractory(self) -> bool:
        """Check if node is currently in refractory state."""
        return self.refractory_counter > 0
    
    def set_refractory(self, steps: int):
        """Set refractory period for the node."""
        self.refractory_counter = steps


class WorldCore3D:
    """
    3D Recursive Neural Graph implementing Continuous Latent Dynamics
    with Multi-Modal Injection Zones for World Modeling.
    
    Key Features:
    - Continuous latent state with leaky integrator dynamics
    - Designated multi-modal injection zones (Vision_Face, Text_Face, Action_Zone)
    - Local predictive coding mechanism replacing global reinforcement
    - Persistent hidden states that evolve smoothly over time
    """
    
    def __init__(self, dim_x: int, dim_y: int, dim_z: int, hidden_size: int,
                 leak_rate: float = 0.1,
                 vision_face_size: Tuple[int, int] = (4, 4),
                 text_face_size: Tuple[int, int] = (2, 2),
                 action_zone_size: Tuple[int, int] = (2, 2)):
        """
        Initialize the 3D World Core with continuous dynamics and multi-modal zones.
        
        Args:
            dim_x, dim_y, dim_z: Dimensions of the 3D grid
            hidden_size: Dimensionality of hidden state vectors
            leak_rate: Rate of state decay in leaky integrator
            vision_face_size: Size of vision input face grid
            text_face_size: Size of text input face grid
            action_zone_size: Size of action injection zone
        """
        self.dimensions = (dim_x, dim_y, dim_z)
        self.hidden_size = hidden_size
        self.leak_rate = leak_rate
        
        # Define multi-modal injection zones
        self.vision_face_size = vision_face_size
        self.text_face_size = text_face_size
        self.action_zone_size = action_zone_size
        
        # Validate zone sizes fit within dimensions
        if vision_face_size[0] > dim_y or vision_face_size[1] > dim_z:
            raise ValueError("Vision face size exceeds Y-Z dimensions")
        if text_face_size[0] > dim_y or text_face_size[1] > dim_z:
            raise ValueError("Text face size exceeds Y-Z dimensions")
        if action_zone_size[0] > dim_y or action_zone_size[1] > dim_z:
            raise ValueError("Action zone size exceeds Y-Z dimensions")
        
        # Create 3D grid of world nodes
        self.nodes: Dict[Tuple[int, int, int], WorldNode] = {}
        for x in range(dim_x):
            for y in range(dim_y):
                for z in range(dim_z):
                    node = WorldNode((x, y, z), hidden_size, leak_rate)
                    self.nodes[(x, y, z)] = node
        
        # Initialize connection weights (for routing/prediction)
        self.connection_weights: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], float] = {}
        self._initialize_connections()
        
        # Shared weight matrix for all nodes (W_shared)
        limit = np.sqrt(6.0 / (hidden_size + hidden_size))
        self.shared_weights = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        
        # Define multi-modal injection zones
        # Vision face: x=0 plane (visual input)
        self.vision_face_coords = self._generate_face_coords(0, vision_face_size)
        # Text face: x=1 plane (linguistic/symbolic input)
        self.text_face_coords = self._generate_face_coords(1, text_face_size)
        # Action zone: x=2 plane (motor/behavioral output)
        self.action_zone_coords = self._generate_face_coords(2, action_zone_size)
        # State observation face: x=max-1 plane (internal state readout)
        self.state_face_coords = self._generate_face_coords(dim_x - 2, vision_face_size)
        # Prediction target face: x=max plane (what we're trying to predict)
        self.prediction_target_face_coords = self._generate_face_coords(dim_x - 1, vision_face_size)
        
        print(f"Initialized WorldCore3D:")
        print(f"  Dimensions: {self.dimensions}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Leak rate: {leak_rate}")
        print(f"  Vision face ({len(self.vision_face_coords)} nodes): x=0")
        print(f"  Text face ({len(self.text_face_coords)} nodes): x=1")
        print(f"  Action zone ({len(self.action_zone_coords)} nodes): x=2")
        print(f"  State face ({len(self.state_face_coords)} nodes): x={dim_x-2}")
        print(f"  Prediction target face ({len(self.prediction_target_face_coords)} nodes): x={dim_x-1}")
    
    def _generate_face_coords(self, fixed_x: int, face_size: Tuple[int, int]) -> List[Tuple[int, int, int]]:
        """Generate coordinate list for a face at fixed x dimension."""
        coords = []
        size_y, size_z = face_size
        for y in range(size_y):
            for z in range(size_z):
                coords.append((fixed_x, y, z))
        return coords
    
    def _initialize_connections(self):
        """Initialize connection weights for local prediction (6-connected grid)."""
        # Define 6 orthogonal directions: +/- x, +/- y, +/- z
        directions = [
            (1, 0, 0), (-1, 0, 0),  # x-axis
            (0, 1, 0), (0, -1, 0),  # y-axis
            (0, 0, 1), (0, 0, -1)   # z-axis
        ]
        
        for (x, y, z), node in self.nodes.items():
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                # Check if neighbor is within bounds
                if (0 <= nx < self.dimensions[0] and 
                    0 <= ny < self.dimensions[1] and 
                    0 <= nz < self.dimensions[2]):
                    neighbor_coord = (nx, ny, nz)
                    # Initialize connection weight for prediction (current -> neighbor)
                    self.connection_weights[((x, y, z), neighbor_coord)] = 1.0
    
    def inject_multi_modal_input(self, 
                                vision_input: Optional[np.ndarray] = None,
                                text_input: Optional[np.ndarray] = None,
                                action_input: Optional[np.ndarray] = None):
        """
        Inject multi-modal inputs into designated zones.
        
        Args:
            vision_input: Input for vision face [num_vision_nodes, hidden_size]
            text_input: Input for text face [num_text_nodes, hidden_size]
            action_input: Input for action zone [num_action_nodes, hidden_size]
        """
        # Inject vision input
        if vision_input is not None:
            self._inject_to_face(vision_input, self.vision_face_coords)
        
        # Inject text input
        if text_input is not None:
            self._inject_to_face(text_input, self.text_face_coords)
        
        # Inject action input
        if action_input is not None:
            self._inject_to_face(action_input, self.action_zone_coords)
    
    def _inject_to_face(self, input_data: np.ndarray, face_coords: List[Tuple[int, int, int]]):
        """
        Helper method to inject input data into a specific face.
        
        Args:
            input_data: Input tensor [num_input_nodes, hidden_size] or [hidden_size]
            face_coords: List of coordinates for the target face
        """
        num_face_nodes = len(face_coords)
        
        # Handle input formatting
        if input_data.ndim == 1:
            # Broadcast single input to all face nodes
            input_vectors = np.tile(input_data, (num_face_nodes, 1))
        elif input_data.ndim == 2:
            if input_data.shape[0] == num_face_nodes:
                input_vectors = input_data
            elif input_data.shape[0] == 1:
                # Broadcast single vector to all face nodes
                input_vectors = np.tile(input_data[0], (num_face_nodes, 1))
            else:
                raise ValueError(f"Input data shape {input_data.shape} incompatible with face size {num_face_nodes}")
        else:
            raise ValueError("Input data must be 1D or 2D array")
        
        # Inject into face nodes
        for i, (node_coord, input_vector) in enumerate(zip(face_coords, input_vectors)):
            if node_coord in self.nodes:
                # For injection, we directly set the hidden state (overriding leaky integrator temporarily)
                self.nodes[node_coord].hidden_state = input_vector.copy()
    
    def tick_world(self, 
                    vision_input: Optional[np.ndarray] = None,
                    text_input: Optional[np.ndarray] = None,
                    action_input: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Advance the world core by one time step.
        
        Args:
            vision_input: Optional vision input for this time step
            text_input: Optional text input for this time step
            action_input: Optional action input for this time step
            
        Returns:
            Dictionary containing outputs from various faces
        """
        # Step 1: Inject multi-modal inputs
        self.inject_multi_modal_input(vision_input, text_input, action_input)
        
        # Step 2: Generate predictions for all nodes (predictive coding phase)
        predictions = {}
        for coord, node in self.nodes.items():
            predictions[coord] = node.predict_next_state(self.shared_weights, activation='tanh')
        
        # Step 3: Update all nodes with continuous dynamics
        # First, collect all incoming signals for kWTA processing
        all_incoming_states = []
        all_coords = []
        
        for coord, node in self.nodes.items():
            # Gather incoming states from connected neighbors
            incoming_states = []
            weights = []
            
            for (src_coord, dst_coord), weight in self.connection_weights.items():
                if dst_coord == coord:  # Connection pointing TO this node
                    if src_coord in self.nodes:
                        incoming_states.append(self.nodes[src_coord].hidden_state)
                        weights.append(weight)
            
            # Compute weighted average of incoming states
            if incoming_states:
                # Weighted sum
                weighted_incoming = np.zeros(self.hidden_size)
                total_weight = 0.0
                for state, weight in zip(incoming_states, weights):
                    weighted_incoming += state * weight
                    total_weight += weight
                
                if total_weight > 0:
                    incoming_state = weighted_incoming / total_weight
                else:
                    incoming_state = np.zeros(self.hidden_size)
            else:
                incoming_state = np.zeros(self.hidden_size)
            
            all_incoming_states.append(incoming_state)
            all_coords.append(coord)
        
        # Apply Lateral Inhibition (kWTA) - Top K% survive using exact specification
        if len(all_incoming_states) > 0:
            # Convert to torch tensor for kWTA operation
            incoming_signals = np.array(all_incoming_states)  # Shape: [num_nodes, hidden_size]
            signal_mags = torch.norm(torch.from_numpy(incoming_signals).float(), dim=-1)  # Shape: [num_nodes]
            
            # Define k: k_active = max(1, int(signal_mags.shape[0] * 0.15))
            k_active = max(1, int(signal_mags.shape[0] * 0.15))
            
            # Get threshold: top_k_vals, _ = torch.topk(signal_mags, k_active, dim=-1)
            top_k_vals, _ = torch.topk(signal_mags, k_active, dim=-1)
            threshold = top_k_vals[:, -1:]  # Shape: [num_nodes, 1] - we need the last value for each
            
            # Create mask: kwta_mask = (signal_mags >= top_k_vals[:, -1:]).float().unsqueeze(-1)
            kwta_mask = (signal_mags >= threshold.squeeze(-1)).float().unsqueeze(-1)  # Shape: [num_nodes, 1]
            
            # Apply mask: sparse_signals = incoming_signals * kwta_mask
            sparse_signals = incoming_signals * kwta_mask.numpy()  # Convert back to numpy
            
            # Update node state using the sparse signals and an aggressive decay
            for i, (coord, node) in enumerate(self.nodes.items()):
                # Get the sparse signal for this node
                sparse_signal = sparse_signals[i]
                
                # Update state with continuous dynamics using sparse signals and aggressive decay
                # h_t = ((1 - leak_rate) * h_t_prev + leak_rate * torch.tanh(sparse_signals)) * 0.95
                # Since we're using numpy, we'll use np.tanh
                retained_state = (1.0 - self.leak_rate) * node.hidden_state
                innovation = self.leak_rate * np.tanh(sparse_signal)
                new_state = (retained_state + innovation) * 0.95  # Aggressive decay factor
                node.hidden_state = new_state
        

        
        # Step 4: Update prediction errors (compare prediction with actual)
        for coord, node in self.nodes.items():
            if coord in predictions:
                actual_state = node.hidden_state
                predicted_state = predictions[coord]
                node.update_prediction_error(actual_state - predicted_state)
        
        # Step 5: Apply refractory periods based on prediction error magnitude
        for coord, node in self.nodes.items():
            error_magnitude = np.linalg.norm(node.prediction_error)
            # High prediction error -> longer refractory period (exploration)
            # Low prediction error -> shorter refractory period (exploitation)
            if error_magnitude > 0.5:
                node.set_refractory(2)  # Explore more
            elif error_magnitude < 0.1:
                node.set_refractory(0)  # Exploit more
            else:
                node.set_refractory(1)  # Neutral
        
        # Step 6: Decrement refractory counters
        for node in self.nodes.values():
            node.tick_refractory()
        
        # Step 7: Collect outputs from various faces
        outputs = {}
        
        # Vision output (what the system "sees" internally)
        vision_states = []
        for coord in self.vision_face_coords:
            if coord in self.nodes:
                vision_states.append(self.nodes[coord].hidden_state.copy())
        outputs['vision'] = np.array(vision_states) if vision_states else np.zeros((len(self.vision_face_coords), self.hidden_size))
        
        # Text output (linguistic/symbolic processing)
        text_states = []
        for coord in self.text_face_coords:
            if coord in self.nodes:
                text_states.append(self.nodes[coord].hidden_state.copy())
        outputs['text'] = np.array(text_states) if text_states else np.zeros((len(self.text_face_coords), self.hidden_size))
        
        # Action output (motor/behavioral commands)
        action_states = []
        for coord in self.action_zone_coords:
            if coord in self.nodes:
                action_states.append(self.nodes[coord].hidden_state.copy())
        outputs['action'] = np.array(action_states) if action_states else np.zeros((len(self.action_zone_coords), self.hidden_size))
        
        # State output (internal state observation)
        state_states = []
        for coord in self.state_face_coords:
            if coord in self.nodes:
                state_states.append(self.nodes[coord].hidden_state.copy())
        outputs['state'] = np.array(state_states) if state_states else np.zeros((len(self.state_face_coords), self.hidden_size))
        
        # Prediction target output (what we're trying to predict)
        prediction_states = []
        for coord in self.prediction_target_face_coords:
            if coord in self.nodes:
                prediction_states.append(self.nodes[coord].hidden_state.copy())
        outputs['prediction_target'] = np.array(prediction_states) if prediction_states else np.zeros((len(self.prediction_target_face_coords), self.hidden_size))
        
        return outputs
    
    def get_world_statistics(self) -> Dict[str, float]:
        """
        Calculate statistics about the world state for monitoring.
        
        Returns:
            Dictionary with world statistics
        """
        total_nodes = len(self.nodes)
        active_count = 0
        total_magnitude = 0.0
        total_prediction_error = 0.0
        refractory_count = 0
        
        for node in self.nodes.values():
            magnitude = np.linalg.norm(node.hidden_state)
            error_magnitude = np.linalg.norm(node.prediction_error)
            
            total_magnitude += magnitude
            total_prediction_error += error_magnitude
            
            if magnitude > 0.1:  # Threshold for considering active
                active_count += 1
            if node.is_refractory():
                refractory_count += 1
        
        activation_ratio = active_count / total_nodes if total_nodes > 0 else 0.0
        avg_magnitude = total_magnitude / total_nodes if total_nodes > 0 else 0.0
        avg_prediction_error = total_prediction_error / total_nodes if total_nodes > 0 else 0.0
        refractory_ratio = refractory_count / total_nodes if total_nodes > 0 else 0.0
        
        return {
            'total_nodes': total_nodes,
            'active_nodes': active_count,
            'activation_ratio': activation_ratio,
            'average_magnitude': avg_magnitude,
            'average_prediction_error': avg_prediction_error,
            'refractory_count': refractory_count,
            'refractory_ratio': refractory_ratio
        }
    
    def reset_world(self):
        """Reset the world core to initial state."""
        # Reset node states
        for node in self.nodes.values():
            node.hidden_state = np.zeros(self.hidden_size)
            node.refractory_counter = 0
            node.bias = np.random.randn(self.hidden_size) * 0.1
            node.predicted_next_state = np.zeros(self.hidden_size)
            node.prediction_error = np.zeros(self.hidden_size)
        
        # Reset connection weights to initial value
        for edge in self.connection_weights:
            self.connection_weights[edge] = 1.0
        
        # Reset shared weights
        limit = np.sqrt(6.0 / (self.hidden_size + self.hidden_size))
        self.shared_weights = np.random.uniform(-limit, limit, (self.hidden_size, self.hidden_size))


def create_world_training_example():
    """
    Create an example showing how to use the World Core for training.
    """
    print("Creating WorldCore3D for Continuous Latent Dynamics...")
    
    # Create world core with multi-modal zones
    world = WorldCore3D(
        dim_x=10, dim_y=8, dim_z=8,
        hidden_size=32,
        leak_rate=0.1,
        vision_face_size=(4, 4),
        text_face_size=(2, 2),
        action_zone_size=(2, 2)
    )
    
    print(f"WorldCore3D created with {len(world.nodes)} nodes")
    print(f"Vision face: {len(world.vision_face_coords)} nodes")
    print(f"Text face: {len(world.text_face_coords)} nodes")
    print(f"Action zone: {len(world.action_zone_coords)} nodes")
    
    # Simulate a few time steps
    print("\nSimulating world dynamics...")
    for t in range(5):
        # Generate random multi-modal inputs
        vision_input = np.random.randn(len(world.vision_face_coords), world.hidden_size) * 0.1
        text_input = np.random.randn(len(world.text_face_coords), world.hidden_size) * 0.1
        action_input = np.random.randn(len(world.action_zone_coords), world.hidden_size) * 0.1
        
        # Advance world
        outputs = world.tick_world(vision_input, text_input, action_input)
        
        # Get statistics
        stats = world.get_world_statistics()
        
        print(f"Time step {t}:")
        print(f"  Activation ratio: {stats['activation_ratio']:.3f}")
        print(f"  Avg prediction error: {stats['average_prediction_error']:.3f}")
        print(f"  Refractory ratio: {stats['refractory_ratio']:.3f}")
    
    print("\nWorldCore3D ready for predictive coding implementation!")
    return world


if __name__ == "__main__":
    # Run the example
    create_world_training_example()