"""
Parallel 3D Recursive Neural Graph (3D-RNG) Implementation
Lead ML Systems Engineer & Data Scientist Implementation

This script implements Phase 2: Parallel Generation for the 3D-RNG architecture,
supporting parallel 'Spatial Chunk Decoding' where multiple tokens can be generated
simultaneously from an output face.

Key modifications from base implementation:
1. Defines Input_Face and Output_Face as coordinate grids instead of single nodes
2. Updated forward_probe to handle wave of inputs routing simultaneously
3. Enhanced traceback_reinforcement to handle multi-output reward vectors
4. Aggregates pheromone updates for overlapping paths to prevent weight explosions
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Set
from collections import defaultdict
import itertools


class Node:
    """
    Represents a single node in the 3D Recursive Neural Graph.
    Unchanged from base implementation.
    """
    
    def __init__(self, coordinates: Tuple[int, int, int], hidden_size: int):
        """
        Initialize a Node at the given spatial coordinates.
        
        Args:
            coordinates: Tuple of (x, y, z) spatial position
            hidden_size: Dimensionality of the hidden state vector
        """
        self.coordinates = coordinates
        self.hidden_state = np.zeros(hidden_size)  # h_t initialized to zero
        self.bias = np.random.randn(hidden_size) * 0.1  # Small random bias
        self.refractory_counter = 0  # Tracks cooldown period
        
    def update_state(self, incoming_state: np.ndarray, shared_weights: np.ndarray, 
                     activation: str = 'tanh') -> np.ndarray:
        """
        Applies the recursive formula: h_t = activation(W_shared * h_{t-1} + b_v)
        
        Args:
            incoming_state: Previous state vector from parent node (h_{t-1})
            shared_weights: Global weight matrix shared across all nodes
            activation: Activation function ('tanh' or 'relu')
            
        Returns:
            Updated hidden state vector (h_t)
        """
        # Matrix multiplication: W_shared * h_{t-1}
        weighted_input = np.dot(shared_weights, incoming_state)
        # Add node-specific bias: W_shared * h_{t-1} + b_v
        pre_activation = weighted_input + self.bias
        # Apply activation function
        if activation == 'tanh':
            self.hidden_state = np.tanh(pre_activation)
        elif activation == 'relu':
            self.hidden_state = np.maximum(0, pre_activation)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
            
        return self.hidden_state
    
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


class ParallelNeuralGraph3D:
    """
    3D Recursive Neural Graph implementing Spatially-Routed Reinforcement Learning
    with parallel spatial chunk decoding capabilities.
    
    Key Features:
    - Defines Input_Face and Output_Face as coordinate grids
    - Supports wave-based parallel input propagation
    - Generates tensor outputs from output face (multiple tokens simultaneously)
    - Enhanced traceback reinforcement for multi-output learning
    """
    
    def __init__(self, dim_x: int, dim_y: int, dim_z: int, hidden_size: int,
                 input_face_size: Tuple[int, int] = (4, 4),
                 output_face_size: Tuple[int, int] = (4, 4)):
        """
        Initialize the 3D grid with defined input and output faces.
        
        Args:
            dim_x, dim_y, dim_z: Dimensions of the 3D grid
            hidden_size: Dimensionality of hidden state vectors
            input_face_size: Size of input face grid (default 4x4 for 16 tokens)
            output_face_size: Size of output face grid (default 4x4 for 16 tokens)
        """
        self.dimensions = (dim_x, dim_y, dim_z)
        self.hidden_size = hidden_size
        self.input_face_size = input_face_size
        self.output_face_size = output_face_size
        
        # Validate face sizes fit within dimensions
        if input_face_size[0] > dim_y or input_face_size[1] > dim_z:
            raise ValueError("Input face size exceeds Y-Z dimensions")
        if output_face_size[0] > dim_y or output_face_size[1] > dim_z:
            raise ValueError("Output face size exceeds Y-Z dimensions")
        
        # Create 3D grid of nodes
        self.nodes: Dict[Tuple[int, int, int], Node] = {}
        for x in range(dim_x):
            for y in range(dim_y):
                for z in range(dim_z):
                    node = Node((x, y, z), hidden_size)
                    self.nodes[(x, y, z)] = node
        
        # Initialize pheromone weights (tau) for all valid edges to 1.0
        self.pheromones: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], float] = {}
        self._initialize_pheromones()
        
        # Shared weight matrix for all nodes (W_shared)
        limit = np.sqrt(6.0 / (hidden_size + hidden_size))
        self.shared_weights = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        
        # Evaporation rate (rho)
        self.evaporation_rate = 0.05
        
        # Define input and output faces
        # Input face: x=0 plane, y,z coordinates within face size
        self.input_face_coords = self._generate_face_coords(0, input_face_size)
        # Output face: x=max plane, y,z coordinates within face size  
        self.output_face_coords = self._generate_face_coords(dim_x - 1, output_face_size)
        
        print(f"Initialized ParallelNeuralGraph3D:")
        print(f"  Dimensions: {self.dimensions}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Input face ({len(self.input_face_coords)} nodes): x=0, y in [0,{input_face_size[0]-1}], z in [0,{input_face_size[1]-1}]")
        print(f"  Output face ({len(self.output_face_coords)} nodes): x={dim_x-1}, y in [0,{output_face_size[0]-1}], z in [0,{output_face_size[1]-1}]")
    
    def _generate_face_coords(self, fixed_x: int, face_size: Tuple[int, int]) -> List[Tuple[int, int, int]]:
        """Generate coordinate list for a face at fixed x dimension."""
        coords = []
        size_y, size_z = face_size
        for y in range(size_y):
            for z in range(size_z):
                coords.append((fixed_x, y, z))
        return coords
    
    def _initialize_pheromones(self):
        """Initialize pheromone weights for all orthogonal neighboring edges."""
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
                    # Initialize pheromone weight for edge (current -> neighbor)
                    self.pheromones[((x, y, z), neighbor_coord)] = 1.0
    
    def get_valid_neighbors(self, current_coord: Tuple[int, int, int], 
                           path_history: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Returns adjacent orthogonal nodes that are NOT currently in a refractory state
        and not in the recent path history (to prevent backtracking).
        """
        # Define 6 orthogonal directions
        directions = [
            (1, 0, 0), (-1, 0, 0),  # x-axis
            (0, 1, 0), (0, -1, 0),  # y-axis
            (0, 0, 1), (0, 0, -1)   # z-axis
        ]
        
        valid_neighbors = []
        x, y, z = current_coord
        
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            neighbor_coord = (nx, ny, nz)
            
            # Check bounds
            if not (0 <= nx < self.dimensions[0] and 
                    0 <= ny < self.dimensions[1] and 
                    0 <= nz < self.dimensions[2]):
                continue
                
            # Check if neighbor exists
            if neighbor_coord not in self.nodes:
                continue
                
            # Check refractory constraint: not in recent path history
            if neighbor_coord in path_history:
                continue
                
            # Check if node is refractory
            neighbor_node = self.nodes[neighbor_coord]
            if neighbor_node.is_refractory():
                continue
                
            valid_neighbors.append(neighbor_coord)
            
        return valid_neighbors
    
    def _select_next_node(self, current_coord: Tuple[int, int, int], 
                         valid_neighbors: List[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
        """
        Selects next neighbor based on pheromone probabilities:
        P(u -> v) = tau_{u,v} / sum(tau_{u,k}) for all valid neighbors k
        """
        if not valid_neighbors:
            return None
            
        # Calculate probabilities based on pheromone weights
        probabilities = []
        total_pheromone = 0.0
        
        for neighbor in valid_neighbors:
            # Get pheromone weight for edge (current -> neighbor)
            edge_key = (current_coord, neighbor)
            tau = self.pheromones.get(edge_key, 1.0)  # Default to 1.0 if not found
            probabilities.append(tau)
            total_pheromone += tau
        
        # Normalize to get probabilities
        if total_pheromone > 0:
            probabilities = [p / total_pheromone for p in probabilities]
        else:
            # Equal probability if all pheromones are zero
            probabilities = [1.0 / len(valid_neighbors)] * len(valid_neighbors)
        
        # Select neighbor based on probabilities
        selected_index = np.random.choice(len(valid_neighbors), p=probabilities)
        return valid_neighbors[selected_index]
    
    def forward_probe(self, input_data: np.ndarray, 
                     max_steps: int = 1000) -> Tuple[np.ndarray, List[List[Tuple[int, int, int]]]]:
        """
        Executes the Guided Non-Backtracking DFS for parallel inputs.
        Allows a "wave" of inputs to route simultaneously from input face.
        When signal hits the output face, it captures the output state.
        
        Args:
            input_data: Input tensor of shape [num_input_nodes, hidden_size] 
                       or [hidden_size] to broadcast to all input nodes
            max_steps: Maximum steps to prevent infinite loops (safety mechanism)
            
        Returns:
            Tuple of (output_tensor, list_of_paths)
            - output_tensor: Shape [num_output_nodes, hidden_size] 
            - list_of_paths: List of paths taken (one per output node that was reached)
        """
        # Handle input data formatting
        if input_data.ndim == 1:
            # Broadcast single input to all input nodes
            input_vectors = np.tile(input_data, (len(self.input_face_coords), 1))
        elif input_data.ndim == 2:
            if input_data.shape[0] == len(self.input_face_coords):
                input_vectors = input_data
            elif input_data.shape[0] == 1:
                # Broadcast single vector to all input nodes
                input_vectors = np.tile(input_data[0], (len(self.input_face_coords), 1))
            else:
                raise ValueError(f"Input data shape {input_data.shape} incompatible with input face size {len(self.input_face_coords)}")
        else:
            raise ValueError("Input data must be 1D or 2D array")
        
        # Initialize tracking for each input node
        num_inputs = len(self.input_face_coords)
        paths = [[] for _ in range(num_inputs)]  # One path per input
        current_coords = list(self.input_face_coords)  # Current position for each input
        active_paths = [True] * num_inputs  # Track which paths are still active
        steps_taken = [0] * num_inputs  # Steps taken per path
        
        # Set initial states for input nodes
        for i, (node_coord, input_vector) in enumerate(zip(self.input_face_coords, input_vectors)):
            node = self.nodes[node_coord]
            node.hidden_state = input_vector.copy()
            paths[i].append(node_coord)  # Start path with input node
        
        # Main propagation loop
        max_active_steps = 0
        while max(steps_taken) < max_steps and any(active_paths):
            max_active_steps += 1
            
            # Process each active path
            for i in range(num_inputs):
                if not active_paths[i]:
                    continue
                    
                current_coord = current_coords[i]
                current_node = self.nodes[current_coord]
                
                # Get valid neighbors (not refractory, not in recent path)
                # Exclude current node from history check to allow immediate revisit? 
                # Actually, we want to prevent backtracking, so exclude recent history
                recent_history = paths[i][-max(3, len(paths[i])//2):] if len(paths[i]) > 3 else paths[i][:-1]
                valid_neighbors = self.get_valid_neighbors(current_coord, recent_history)
                
                if not valid_neighbors:
                    # Dead end - deactivate this path
                    active_paths[i] = False
                    continue
                
                # Select next node based on pheromone probabilities
                next_coord = self._select_next_node(current_coord, valid_neighbors)
                
                if next_coord is None:
                    active_paths[i] = False
                    continue
                
                # Move to next node
                next_node = self.nodes[next_coord]
                
                # Update next node's state using current node's state as input
                incoming_state = current_node.hidden_state
                next_node.update_state(incoming_state, self.shared_weights, activation='tanh')
                
                # Update tracking
                paths[i].append(next_coord)
                current_coords[i] = next_coord
                steps_taken[i] += 1
                
                # Check if we've reached output face
                if next_coord in self.output_face_coords:
                    # Path has reached output - could continue or stop here
                    # For now, let it continue but mark as reached output
                    pass
                
                # Optional: Set refractory period on current node to prevent immediate backtracking
                current_node.set_refractory(1)  # Refractory for 1 step
        
        # Collect output states from output face nodes
        output_states = []
        output_paths = []  # Only include paths that reached output face
        
        for i in range(num_inputs):
            final_coord = current_coords[i]
            if final_coord in self.output_face_coords:
                # This path reached the output face
                output_states.append(self.nodes[final_coord].hidden_state.copy())
                output_paths.append(paths[i])
            # Else: path didn't reach output, we ignore it for output tensor
        
        # Convert to tensor
        if output_states:
            output_tensor = np.array(output_states)  # Shape: [num_reached_outputs, hidden_size]
        else:
            # No outputs reached - return zeros with expected shape
            output_tensor = np.zeros((len(self.output_face_coords), self.hidden_size))
        
        return output_tensor, output_paths
    
    def traceback_reinforcement(self, list_of_paths: List[List[Tuple[int, int, int]]], 
                               reward_vector: np.ndarray, 
                               evaporation_rate: Optional[float] = None):
        """
        Unwinds multiple paths. Evaporates all pheromones slightly, then heavily rewards 
        the specific edges used in the paths based on the outcome.
        
        Handles overlapping paths by aggregating rewards to prevent weight explosions.
        
        Implements: tau_{u,v} <- (1 - rho)*tau_{u,v} + sum_of_rewards_for_edge
        
        Args:
            list_of_paths: List of paths (each path is list of coordinates)
            reward_vector: Reward values for each path (shape: [num_paths])
            evaporation_rate: Evaporation rate rho (uses instance default if None)
        """
        if evaporation_rate is None:
            evaporation_rate = self.evaporation_rate
            
        # Apply global evaporation to all pheromones first
        # tau <- (1 - rho) * tau
        for edge in self.pheromones:
            self.pheromones[edge] *= (1.0 - evaporation_rate)
        
        # Aggregate rewards for each edge across all paths
        edge_rewards = defaultdict(float)
        
        # Process each path with its corresponding reward
        for path, reward in zip(list_of_paths, reward_vector):
            # Unwind the path and accumulate rewards for each edge
            for i in range(len(path) - 1):
                u_coord = path[i]
                v_coord = path[i + 1]
                edge_key = (u_coord, v_coord)
                
                # Accumulate reward for this edge
                edge_rewards[edge_key] += reward
                
                # Also reinforce reverse edge for biological plausibility
                reverse_edge_key = (v_coord, u_coord)
                if reverse_edge_key in self.pheromones:
                    edge_rewards[reverse_edge_key] += reward
        
        # Apply aggregated rewards to pheromones
        # tau <- (1 - rho)*tau + aggregated_reward
        for edge_key, total_reward in edge_rewards.items():
            if edge_key in self.pheromones:
                current_tau = self.pheromones[edge_key]
                new_tau = (1.0 - evaporation_rate) * current_tau + total_reward
                self.pheromones[edge_key] = new_tau
            # Also handle reverse edges that might not be in original pheromones
            elif edge_key[::-1] in self.pheromones:  # Check if reverse exists
                # Actually, we should only update existing edges to maintain graph structure
                pass
    
    def get_activation_statistics(self) -> Dict[str, float]:
        """
        Calculate statistics about node activation for monitoring sparsity.
        """
        total_nodes = len(self.nodes)
        active_count = 0
        total_magnitude = 0.0
        
        for node in self.nodes.values():
            magnitude = np.linalg.norm(node.hidden_state)
            total_magnitude += magnitude
            # --- EPSILON-SAFE METRIC OVERRIDE ---
            # Do not count microscopic floating-point noise as an active node
            epsilon_threshold = 1e-4
            if magnitude > epsilon_threshold:  # Epsilon threshold for considering active
                active_count += 1
                
        activation_ratio = active_count / total_nodes if total_nodes > 0 else 0.0
        avg_magnitude = total_magnitude / total_nodes if total_nodes > 0 else 0.0
        
        return {
            'total_nodes': total_nodes,
            'active_nodes': active_count,
            'activation_ratio': activation_ratio,
            'average_magnitude': avg_magnitude
        }
    
    def get_face_activation_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate activation statistics specifically for input and output faces.
        """
        def face_stats(face_coords):
            if not face_coords:
                return {'count': 0, 'active': 0, 'ratio': 0.0, 'avg_magnitude': 0.0}
            
            active = 0
            total_mag = 0.0
            for coord in face_coords:
                if coord in self.nodes:
                    magnitude = np.linalg.norm(self.nodes[coord].hidden_state)
                    total_mag += magnitude
                    # --- EPSILON-SAFE METRIC OVERRIDE ---
                    epsilon_threshold = 1e-4
                    if magnitude > epsilon_threshold:
                        active += 1
            
            return {
                'count': len(face_coords),
                'active': active,
                'ratio': active / len(face_coords) if face_coords else 0.0,
                'avg_magnitude': total_mag / len(face_coords) if face_coords else 0.0
            }
        
        return {
            'input_face': face_stats(self.input_face_coords),
            'output_face': face_stats(self.output_face_coords)
        }
    
    def reset_graph(self):
        """Reset the graph to initial state (useful for multiple trials)."""
        # Reset node states
        for node in self.nodes.values():
            node.hidden_state = np.zeros(self.hidden_size)
            node.refractory_counter = 0
            node.bias = np.random.randn(self.hidden_size) * 0.1  # Reset bias
            
        # Reset pheromones to initial value
        for edge in self.pheromones:
            self.pheromones[edge] = 1.0
            
        # Reset shared weights
        limit = np.sqrt(6.0 / (self.hidden_size + self.hidden_size))
        self.shared_weights = np.random.uniform(-limit, limit, (self.hidden_size, self.hidden_size))


def create_parallel_training_example():
    """
    Create an example showing how to use the parallel 3D-RNG for training.
    """
    print("Creating Parallel 3D-RNG for Spatial Chunk Decoding...")
    
    # Create graph with 4x4 input and output faces (16 chunks each)
    graph = ParallelNeuralGraph3D(
        dim_x=10, dim_y=8, dim_z=8, 
        hidden_size=16,
        input_face_size=(4, 4),
        output_face_size=(4, 4)
    )
    
    # Example: Create input data for 16 chunks
    # Each chunk gets its own input vector
    num_chunks = len(graph.input_face_coords)  # Should be 16
    input_data = np.random.randn(num_chunks, 16) * 0.5
    
    print(f"Input data shape: {input_data.shape}")
    print(f"Number of input channels: {len(graph.input_face_coords)}")
    print(f"Number of output channels: {len(graph.output_face_coords)}")
    
    # Forward pass
    print("\nExecuting parallel forward probe...")
    output_tensor, paths = graph.forward_probe(input_data, max_steps=50)
    
    print(f"Output tensor shape: {output_tensor.shape}")
    print(f"Number of successful paths: {len(paths)}")
    if paths:
        print(f"Average path length: {np.mean([len(p) for p in paths]):.1f}")
    
    # Example reward vector (one reward per path)
    # In practice, this would come from a loss function comparing output to target
    if len(paths) > 0:
        rewards = np.random.rand(len(paths)) * 2 - 1  # Random rewards between -1 and 1
        print(f"\nApplying traceback reinforcement with rewards: {rewards}")
        graph.traceback_reinforcement(paths, rewards)
        print("Reinforcement applied.")
    
    # Show statistics
    stats = graph.get_activation_statistics()
    face_stats = graph.get_face_activation_stats()
    
    print(f"\nActivation Statistics:")
    print(f"  Overall: {stats['active_nodes']}/{stats['total_nodes']} active ({stats['activation_ratio']:.2%})")
    print(f"  Input Face: {face_stats['input_face']['active']}/{face_stats['input_face']['count']} active ({face_stats['input_face']['ratio']:.2%})")
    print(f"  Output Face: {face_stats['output_face']['active']}/{face_stats['output_face']['count']} active ({face_stats['output_face']['ratio']:.2%})")
    
    return graph, input_data, output_tensor, paths


if __name__ == "__main__":
    # Run the example
    create_parallel_training_example()