"""
3D Recursive Neural Graph (3D-RNG) Implementation
Core Developer Agent Implementation

This script implements the foundational topological structures and mathematical 
routing logic for the 3D Recursive Neural Graph as defined in spec_def.md.

Architectural Constraints:
1. NO BACKPROPAGATION: Does not use torch.autograd, loss.backward(), or standard deep learning optimizers
2. NO SEQUENTIAL LAYERS: Uses a 3D coordinate grid topology, not flat layer arrays
3. SHARED STATE ENGINE: Every node applies the same shared weight matrix for state updates
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import heapq


class Node:
    """
    Represents a single node in the 3D Recursive Neural Graph.
    
    Each node maintains:
    - Spatial coordinates (x, y, z)
    - Hidden state vector (h_t) for recursive computation
    - Localized bias parameter (b_v)
    - Refractory counter to prevent back-tracking
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


class NeuralGraph3D:
    """
    3D Recursive Neural Graph implementing Spatially-Routed Reinforcement Learning.
    
    The graph consists of:
    - A 3D grid of Node objects
    - Pheromone weights (tau) on edges between neighboring nodes
    - Shared weight matrix for all node computations
    - Guided Non-Backtracking Depth-First Search for signal routing
    """
    
    def __init__(self, dim_x: int, dim_y: int, dim_z: int, hidden_size: int):
        """
        Initialize the 3D grid, create Node instances, and set all adjacent edge pheromones to 1.0.
        
        Args:
            dim_x, dim_y, dim_z: Dimensions of the 3D grid
            hidden_size: Dimensionality of hidden state vectors
        """
        self.dimensions = (dim_x, dim_y, dim_z)
        self.hidden_size = hidden_size
        
        # Create 3D grid of nodes
        self.nodes: Dict[Tuple[int, int, int], Node] = {}
        for x in range(dim_x):
            for y in range(dim_y):
                for z in range(dim_z):
                    node = Node((x, y, z), hidden_size)
                    self.nodes[(x, y, z)] = node
        
        # Initialize pheromone weights (tau) for all valid edges to 1.0
        # Using dictionary with key as (from_coord, to_coord) for sparse representation
        self.pheromones: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], float] = {}
        self._initialize_pheromones()
        
        # Shared weight matrix for all nodes (W_shared)
        # Using Xavier/Glorot initialization for stability
        limit = np.sqrt(6.0 / (hidden_size + hidden_size))
        self.shared_weights = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        
        # Evaporation rate (rho) - can be adjusted
        self.evaporation_rate = 0.05
        
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
        
        Implements the Refractory Constraint: Cannot route to the immediate past node,
        or any node accessed in the last N steps of this specific DFS trace.
        
        Args:
            current_coord: Current node coordinates
            path_history: List of coordinates visited in current DFS trace
            
        Returns:
            List of valid neighbor coordinates
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
            # Using last N steps where N could be path length or fixed value
            # For simplicity, we'll avoid immediate backtracking and recent loops
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
        
        Args:
            current_coord: Current node coordinates
            valid_neighbors: List of valid neighbor coordinates
            
        Returns:
            Selected neighbor coordinate or None if no valid neighbors
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
    
    def forward_probe(self, input_data: np.ndarray, start_coord: Tuple[int, int, int], 
                     max_steps: int = 1000) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """
        Executes the Guided Non-Backtracking DFS.
        Signal propagates from start_coord through the graph following pheromone trails
        with refractory constraints to prevent loops.
        
        Args:
            input_data: Input vector to inject at start node
            start_coord: Starting coordinates (typically on input face)
            max_steps: Maximum steps to prevent infinite loops (safety mechanism)
            
        Returns:
            Tuple of (final_output_state, path_coordinates_visited)
        """
        # Reset refractory states for all nodes (optional, could be done per simulation)
        # For now, we'll rely on path history and per-node refractory counters
        
        # Initialize path tracking
        path_stack = [start_coord]
        current_coord = start_coord
        steps = 0
        
        # Set input at start node
        start_node = self.nodes[start_coord]
        # For simplicity, we'll set the hidden state directly from input
        # In practice, might want to process input through some transformation
        if input_data.shape != (self.hidden_size,):
            # Resize or project input to hidden size if needed
            if input_data.size >= self.hidden_size:
                start_node.hidden_state = input_data.flat[:self.hidden_size]
            else:
                # Pad or repeat input
                padded = np.zeros(self.hidden_size)
                padded[:len(input_data.flat)] = input_data.flat
                start_node.hidden_state = padded
        else:
            start_node.hidden_state = input_data.copy()
        
        # Mark start node as visited in path (but not necessarily refractory yet)
        # We'll use path_stack for history checking
        
        while steps < max_steps:
            current_node = self.nodes[current_coord]
            
            # Get valid neighbors (not refractory, not in recent path)
            valid_neighbors = self.get_valid_neighbors(current_coord, path_stack[:-1])  # Exclude current from history check
            
            if not valid_neighbors:
                # Dead end - signal cannot proceed further
                break
                
            # Select next node based on pheromone probabilities
            next_coord = self._select_next_node(current_coord, valid_neighbors)
            
            if next_coord is None:
                break
                
            # Move to next node
            next_node = self.nodes[next_coord]
            
            # Update next node's state using current node's state as input
            incoming_state = current_node.hidden_state
            next_node.update_state(incoming_state, self.shared_weights, activation='tanh')
            
            # Add to path
            path_stack.append(next_coord)
            current_coord = next_coord
            steps += 1
            
            # Optional: Set refractory period on current node to prevent immediate backtracking
            # This implements part of the refractory constraint
            current_node.set_refractory(1)  # Refractory for 1 step
            
        # Return final state and path
        final_node = self.nodes[current_coord]
        return final_node.hidden_state.copy(), path_stack
    
    def traceback_reinforcement(self, path_stack: List[Tuple[int, int, int]], 
                               reward: float, evaporation_rate: Optional[float] = None):
        """
        Unwinds the DFS stack. Evaporates all pheromones slightly, then heavily rewards 
        the specific edges used in the path_stack based on the outcome.
        
        Implements: tau_{u,v} <- (1 - rho)*tau_{u,v} + R for each edge in path
        
        Args:
            path_stack: List of coordinates visited in the path (in order)
            reward: Reward value (+1 for correct, -0.5 for incorrect)
            evaporation_rate: Evaporation rate rho (uses instance default if None)
        """
        if evaporation_rate is None:
            evaporation_rate = self.evaporation_rate
            
        # Apply global evaporation to all pheromones first
        # tau <- (1 - rho) * tau
        for edge in self.pheromones:
            self.pheromones[edge] *= (1.0 - evaporation_rate)
        
        # Then reinforce edges in the path
        # For each consecutive pair in path: (u,v) in path
        for i in range(len(path_stack) - 1):
            u_coord = path_stack[i]
            v_coord = path_stack[i + 1]
            edge_key = (u_coord, v_coord)
            
            # Apply learning rule: tau <- (1 - rho)*tau + R
            current_tau = self.pheromones.get(edge_key, 1.0)
            new_tau = (1.0 - evaporation_rate) * current_tau + reward
            self.pheromones[edge_key] = new_tau
            
            # Also reinforce reverse edge? Typically in ant colony optimization, 
            # both directions might be reinforced, but for directed graph we might only do forward
            # For biological plausibility, let's reinforce both directions with same reward
            reverse_edge_key = (v_coord, u_coord)
            if reverse_edge_key in self.pheromones:
                current_tau_rev = self.pheromones[reverse_edge_key]
                new_tau_rev = (1.0 - evaporation_rate) * current_tau_rev + reward
                self.pheromones[reverse_edge_key] = new_tau_rev
    
    def get_activation_statistics(self) -> Dict[str, float]:
        """
        Calculate statistics about node activation for monitoring sparsity.
        
        Returns:
            Dictionary with activation statistics
        """
        total_nodes = len(self.nodes)
        # For simplicity, we'll consider a node "active" if its hidden state has significant magnitude
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
            
        # Reset shared weights (optional - might want to keep learned weights)
        limit = np.sqrt(6.0 / (self.hidden_size + self.hidden_size))
        self.shared_weights = np.random.uniform(-limit, limit, (self.hidden_size, self.hidden_size))


# Example usage and testing functions
def create_xor_scenario():
    """
    Create a simple scenario for testing XOR logic as mentioned in the harness.
    This would involve setting up input/output nodes and training procedure.
    """
    # This is a placeholder for the actual implementation
    # In a full implementation, we would:
    # 1. Create a graph of appropriate dimensions
    # 2. Define input nodes (left face) and output nodes (right face) 
    # 3. Implement training loop using the traceback_reinforcement method
    # 4. Test with XOR inputs: [0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0
    pass


def create_mnist_scenario():
    """
    Create scenario for MNIST testing as mentioned in the harness.
    """
    # Placeholder for MNIST implementation
    pass


if __name__ == "__main__":
    # Example of creating and testing a small graph
    print("Creating 3D Recursive Neural Graph...")
    graph = NeuralGraph3D(dim_x=10, dim_y=10, dim_z=10, hidden_size=16)
    
    print(f"Graph dimensions: {graph.dimensions}")
    print(f"Number of nodes: {len(graph.nodes)}")
    print(f"Number of pheromone edges: {len(graph.pheromones)}")
    
    # Test forward probe with random input
    input_vector = np.random.randn(16) * 0.5  # Small random input
    start_coord = (0, 5, 5)  # Left face (x=0)
    
    print("\nExecuting forward probe...")
    final_state, path = graph.forward_probe(input_vector, start_coord, max_steps=50)
    
    print(f"Path length: {len(path)}")
    print(f"Path coordinates: {path}")
    print(f"Final state shape: {final_state.shape}")
    print(f"Final state magnitude: {np.linalg.norm(final_state):.4f}")
    
    # Show activation statistics
    stats = graph.get_activation_statistics()
    print(f"\nActivation Statistics:")
    print(f"  Active nodes: {stats['active_nodes']}/{stats['total_nodes']} ({stats['activation_ratio']:.2%})")
    print(f"  Average magnitude: {stats['average_magnitude']:.4f}")
    
    # Test traceback reinforcement with a reward
    print("\nApplying traceback reinforcement...")
    reward = 1.0  # Correct prediction
    graph.traceback_reinforcement(path, reward)
    
    print("Reinforcement applied.")
    print(f"Sample pheromone values: {list(graph.pheromones.values())[:5]}")