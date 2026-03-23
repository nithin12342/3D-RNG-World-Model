"""
Local Predictive Coding for 3D-RNG World Engine
Chief AI Scientist & World Model Architect Implementation

This script implements Phase 2: Local Predictive Coding for the 3D-RNG World Engine,
replacing global Traceback Reinforcement with localized Self-Supervised Learning where
each node predicts its neighbors' states and updates connections based on prediction error.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Set, Any
from collections import defaultdict
import itertools

# Import Sparse MoE layer
try:
    from core.moe_layer import SparseMoE
    MOE_AVAILABLE = True
except ImportError:
    MOE_AVAILABLE = False
    print("Warning: MoE layer not available.")

# Import LocalOptimizerBridge for Hebbian gradient learning
try:
    from core.optimization import LocalOptimizerBridge
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("Warning: LocalOptimizerBridge not available.")

# Import Neurosymbolic modules for composition
try:
    from core.cognitive_controller import CognitiveController as CognitiveControllerBase
    from core.neurosymbolic_kg import NeurosymbolicKG
    from core.agentic_sandbox import AgenticSandbox
    NEUROSYMBOLIC_AVAILABLE = True
except ImportError as e:
    NEUROSYMBOLIC_AVAILABLE = False
    print(f"Warning: Neurosymbolic modules not available: {e}")


class CognitiveController:
    """
    Wrapper class for cognitive operations.
    Provides a clean interface for the engine to interact with cognitive modules.
    
    This follows the Composition over Inheritance principle - wrapping the available
    cognitive components behind a simple public interface.
    """
    
    def __init__(self, hidden_size: int = 768):
        """
        Initialize CognitiveController with wrapped cognitive components.
        
        Args:
            hidden_size: Dimensionality of hidden state vectors
        """
        self.hidden_size = hidden_size
        
        if NEUROSYMBOLIC_AVAILABLE:
            # Initialize the base cognitive controller
            self._controller = CognitiveControllerBase()
        else:
            self._controller = None
        
        # Internal state tracking
        self.current_plan: Optional[Dict[str, Any]] = None
        self.last_hidden_state: Optional[np.ndarray] = None
        
    def run(self, input_data: Any) -> Dict[str, Any]:
        """
        Process hidden state through cognitive controller.
        
        Args:
            input_data: Either a numpy array or a dictionary with:
                - 'prompt': decoded string summary from VectorDecoder
                - 'hidden_state': raw hidden state numpy array
                - 'text_action_count': count of text/action nodes
                - 'kg_count': count of KG nodes
            
        Returns:
            Dictionary containing cognitive processing results
        """
        # Handle both legacy numpy array input and new dictionary input
        if isinstance(input_data, dict):
            # New format: dictionary from VectorDecoder
            hidden_state = input_data.get('hidden_state')
            prompt = input_data.get('prompt', '')
            text_action_count = input_data.get('text_action_count', 0)
            kg_count = input_data.get('kg_count', 0)
        else:
            # Legacy format: raw numpy array
            hidden_state = input_data
            prompt = ''
            text_action_count = 0
            kg_count = 0
        
        self.last_hidden_state = hidden_state.copy()
        
        if self._controller is not None:
            # Use the base controller for processing
            # Convert hidden_state magnitude to a task description
            state_magnitude = float(np.linalg.norm(hidden_state))
            task = f"Process hidden state with magnitude {state_magnitude:.4f}"
            
            # Run cognitive processing
            result = {
                'hidden_state_shape': hidden_state.shape,
                'processed': True,
                'state_magnitude': state_magnitude,
                'controller_state': self._controller.get_state(),
                'decoded_prompt': prompt,  # Include the decoded prompt for reference
                'text_action_count': text_action_count,
                'kg_count': kg_count
            }
        else:
            # Fallback processing when modules aren't available
            result = {
                'hidden_state_shape': hidden_state.shape,
                'processed': True,
                'state_magnitude': float(np.linalg.norm(hidden_state)),
                'decoded_prompt': prompt,
                'text_action_count': text_action_count,
                'kg_count': kg_count
            }
        
        return result


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    From "Root Mean Square Layer Normalization" (Zhang et al., 2019).
    Applies normalization based on RMS of activations, without centering.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.
        
        Args:
            hidden_size: Dimension of input
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        # Learnable scale parameter
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            
        Returns:
            Normalized tensor
        """
        # Compute RMS (Root Mean Square)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize
        normalized = x / rms
        # Scale
        return normalized * self.weight


class BlockAttentionResiduals(nn.Module):
    """
    Block Attention Residuals implementation based on arXiv:2603.15031v1.
    
    Key features:
    - Partitions execution sequence into N blocks
    - Creates block representations by summing outputs within each block
    - Uses learned pseudo-query vectors for inter-block attention
    - Applies RMSNorm to keys to prevent magnitude dominance
    - Achieves O(Ld) memory efficiency vs O(L^2d) for standard attention
    
    Mathematical formulation:
    - Block representation: b_n = Σ_{i∈block_n} output_i
    - Attention: a_l = softmax(w_l · k_{0:n-1}) where w_l is learned query
    - Output: y_l = Σ_{n} a_l[n] · v_n + input_l (residual)
    """
    
    def __init__(self, hidden_size: int, num_blocks: int = 8, 
                 num_layers: int = 4, eps: float = 1e-6):
        """
        Initialize Block Attention Residuals.
        
        Args:
            hidden_size: Dimension of hidden states
            num_blocks: Number of blocks to partition sequence into
            num_layers: Number of layers with learned query vectors
            eps: Epsilon for RMSNorm stability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.eps = eps
        
        # RMSNorm for keys (prevents magnitude dominance)
        self.key_norm = RMSNorm(hidden_size, eps)
        
        # Learnable pseudo-query vectors w_l ∈ ℝ^d for each layer
        # Shape: (num_layers, hidden_size)
        self.query_vectors = nn.Parameter(
            torch.randn(num_layers, hidden_size) * 0.01
        )
        
        # Block projection (optional: project block representations)
        self.block_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Layer norm for output
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform(self.block_projection.weight)
        nn.init.xavier_uniform(self.output_projection.weight)
    
    def compute_block_representations(self, layer_outputs: torch.Tensor, 
                                       block_size: int) -> torch.Tensor:
        """
        Compute block representations by summing outputs within each block.
        
        Args:
            layer_outputs: Tensor of shape (batch, seq_len, hidden_size)
            block_size: Number of elements per block
            
        Returns:
            Block representations of shape (batch, num_blocks, hidden_size)
        """
        batch_size, seq_len, hidden_size = layer_outputs.shape
        
        # Pad sequence to be divisible by num_blocks
        padded_len = ((seq_len + block_size - 1) // block_size) * block_size
        if padded_len > seq_len:
            padding = padded_len - seq_len
            layer_outputs = F.pad(layer_outputs, (0, 0, 0, padding))
        
        # Reshape into blocks: (batch, num_blocks, block_size, hidden_size)
        num_blocks = padded_len // block_size
        reshaped = layer_outputs.view(batch_size, num_blocks, block_size, hidden_size)
        
        # Sum within each block to get block representation b_n
        block_repr = reshaped.sum(dim=2)  # (batch, num_blocks, hidden_size)
        
        return block_repr
    
    def forward(self, inputs: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """
        Forward pass with block attention residuals.
        
        Args:
            inputs: Input tensor of shape (batch, seq_len, hidden_size)
            layer_idx: Index of current layer (0 to num_layers-1)
            
        Returns:
            Output with attention-weighted block residuals added
        """
        batch_size, seq_len, hidden_size = inputs.shape
        
        # Compute block size based on sequence length and num_blocks
        block_size = max(1, seq_len // self.num_blocks)
        
        # Compute block representations
        block_repr = self.compute_block_representations(inputs, block_size)
        
        # Project block representations
        keys = self.block_projection(block_repr)  # (batch, num_blocks, hidden_size)
        values = block_repr  # Use raw block representations as values
        
        # Apply RMSNorm to keys (prevents magnitude dominance)
        keys = self.key_norm(keys)
        
        # Get learned query vector for this layer
        # Shape: (batch, 1, hidden_size)
        query = self.query_vectors[layer_idx].unsqueeze(0).unsqueeze(0)
        query = query.expand(batch_size, 1, -1)
        
        # Compute attention scores: a_l = softmax(w_l · k_{0:n-1})
        # Squeeze seq_len dimension for single query
        query = query.squeeze(1)  # (batch, hidden_size)
        keys = keys.squeeze(1)  # (batch, hidden_size) if num_blocks=1
        
        if keys.dim() == 3 and keys.shape[1] > 1:
            # Multiple blocks: compute dot products
            # query: (batch, 1, hidden_size), keys: (batch, num_blocks, hidden_size)
            query_expanded = query.unsqueeze(1)  # (batch, 1, hidden_size)
            attention_scores = torch.bmm(query_expanded, keys.transpose(1, 2)).squeeze(1)  # (batch, num_blocks)
        else:
            # Single block case
            attention_scores = (query * keys.squeeze(1)).sum(dim=-1)  # (batch,)
        
        # Softmax over blocks
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, num_blocks)
        
        # Compute attention-weighted sum of block values
        if values.shape[1] > 1:
            # Multiple blocks
            attention_weights_expanded = attention_weights.unsqueeze(-1)  # (batch, num_blocks, 1)
            attended = (values * attention_weights_expanded).sum(dim=1)  # (batch, hidden_size)
        else:
            # Single block
            attended = values.squeeze(1) * attention_weights  # (batch, hidden_size)
        
        # Expand to sequence length
        attended = attended.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, hidden_size)
        
        # Project output
        attended = self.output_projection(attended)
        
        # Residual connection with input
        output = self.layer_norm(inputs + attended)
        
        return output


class BlockAttentionPredictiveCodingNode:
    """
    Predictive Coding Node with Block Attention Residuals.
    
    Extends functionality with:
    - Learned pseudo-query vector w_l for inter-block attention
    - Block-wise attention over previous layer representations
    - All methods from PredictiveCodingNode
    """
    
    def __init__(self, coordinates: Tuple[int, int, int], hidden_size: int,
                 leak_rate: float = 0.1, learning_rate: float = 0.01,
                 num_blocks: int = 8):
        """
        Initialize Block Attention Predictive Coding Node.
        
        Args:
            coordinates: Tuple of (x, y, z) spatial position
            hidden_size: Dimensionality of the hidden state vector
            leak_rate: Rate of state decay in leaky integrator
            learning_rate: Learning rate for connection weight updates
            num_blocks: Number of blocks for attention partitioning
        """
        self.coordinates = coordinates
        self.hidden_size = hidden_size
        self.leak_rate = leak_rate
        self.learning_rate = learning_rate
        self.num_blocks = num_blocks
        
        # Continuous latent state
        self.hidden_state = np.zeros(hidden_size)
        self.bias = np.random.randn(hidden_size) * 0.1
        self.refractory_counter = 0
        
        # Predictive coding components
        self.predicted_neighbor_states = {}
        self.prediction_errors = {}
        self.connection_weights = {}
        
        # Initialize neighbor connections
        self.neighbors = []
        
        # Learned pseudo-query vector w_l ∈ ℝ^d
        self.query_vector = np.random.randn(hidden_size) * 0.01
        
        # Block attention history
        self.block_history: List[np.ndarray] = []
        self.max_block_history = num_blocks
    
    # === Methods from PredictiveCodingNode ===
    
    def update_state_continuous(self, incoming_state: np.ndarray, 
                               shared_weights: np.ndarray,
                               activation: str = 'tanh') -> np.ndarray:
        """Apply continuous update with leaky integrator."""
        retained_state = (1.0 - self.leak_rate) * self.hidden_state
        weighted_input = np.dot(shared_weights, incoming_state)
        pre_activation = weighted_input + self.bias
        
        if activation == 'tanh':
            activated_state = np.tanh(pre_activation)
        elif activation == 'relu':
            activated_state = np.maximum(0, pre_activation)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        innovation = self.leak_rate * activated_state
        self.hidden_state = retained_state + innovation
        return self.hidden_state
    
    def predict_neighbor_states(self, shared_weights: np.ndarray,
                               activation: str = 'tanh') -> Dict[Tuple[int, int, int], np.ndarray]:
        """Predict neighbor states."""
        predictions = {}
        for neighbor_coord in self.neighbors:
            weighted_input = np.dot(shared_weights, self.hidden_state)
            pre_activation = weighted_input + self.bias
            if activation == 'tanh':
                predicted_state = np.tanh(pre_activation)
            elif activation == 'relu':
                predicted_state = np.maximum(0, pre_activation)
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            predictions[neighbor_coord] = predicted_state
        self.predicted_neighbor_states = predictions
        return predictions
    
    def update_prediction_errors(self, actual_neighbor_states: Dict[Tuple[int, int, int], np.ndarray]):
        """Update prediction errors."""
        errors = {}
        for neighbor_coord in self.neighbors:
            if neighbor_coord in self.predicted_neighbor_states and neighbor_coord in actual_neighbor_states:
                error = actual_neighbor_states[neighbor_coord] - self.predicted_neighbor_states[neighbor_coord]
                errors[neighbor_coord] = error
        self.prediction_errors = errors
    
    def update_connection_weights(self):
        """Update connection weights based on prediction errors."""
        evaporation_rate = 0.05
        for neighbor_coord in self.neighbors:
            if neighbor_coord in self.connection_weights:
                self.connection_weights[neighbor_coord] *= (1.0 - evaporation_rate)
                self.connection_weights[neighbor_coord] = max(self.connection_weights[neighbor_coord], 0.01)
            
            if neighbor_coord in self.prediction_errors:
                error = self.prediction_errors[neighbor_coord]
                error_magnitude = np.linalg.norm(error)
                error_penalty = min(1.0, error_magnitude ** 2)
                learning_signal = self.learning_rate * (1.0 - 2.0 * error_penalty)
                pre_activity = self.hidden_state
                post_activity = self.predicted_neighbor_states.get(neighbor_coord, np.zeros(self.hidden_size))
                correlation = np.dot(pre_activity, post_activity) / (self.hidden_size)
                if neighbor_coord in self.connection_weights:
                    self.connection_weights[neighbor_coord] += learning_signal * correlation
                    self.connection_weights[neighbor_coord] = np.clip(self.connection_weights[neighbor_coord], 0.01, 5.0)
                else:
                    self.connection_weights[neighbor_coord] = 0.01 + learning_signal * correlation
    
    def tick_refractory(self):
        """Decrement refractory counter."""
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
    
    def is_refractory(self) -> bool:
        """Check if in refractory state."""
        return self.refractory_counter > 0
    
    def set_refractory(self, steps: int):
        """Set refractory period."""
        self.refractory_counter = steps
    
    def update_with_block_attention(self, incoming_state: np.ndarray,
                                    block_representations: List[np.ndarray],
                                    shared_weights: np.ndarray,
                                    activation: str = 'tanh') -> np.ndarray:
        """
        Update node state with block attention residuals.
        
        Computes:
        1. Standard leaky integrator update
        2. Block attention over previous block representations
        3. Combines both for final state
        
        Args:
            incoming_state: Input state from neighbors
            block_representations: List of previous block representations
            shared_weights: Global weight matrix
            activation: Activation function
            
        Returns:
            Updated hidden state
        """
        # Step 1: Standard leaky integrator update
        retained_state = (1.0 - self.leak_rate) * self.hidden_state
        weighted_input = np.dot(shared_weights, incoming_state)
        pre_activation = weighted_input + self.bias
        
        if activation == 'tanh':
            activated_state = np.tanh(pre_activation)
        elif activation == 'relu':
            activated_state = np.maximum(0, pre_activation)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        innovation = self.leak_rate * activated_state
        base_update = retained_state + innovation
        
        # Step 2: Block attention over previous blocks
        if block_representations and len(block_representations) > 0:
            # Compute attention weights: softmax(w_l · k_{0:n-1})
            # Keys are RMS-normalized block representations
            keys = []
            for block_repr in block_representations:
                # Apply RMS normalization to keys
                rms = np.sqrt(np.mean(block_repr ** 2) + 1e-6)
                normalized_key = block_repr / rms
                keys.append(normalized_key)
            
            # Compute query-key similarities
            query = self.query_vector
            scores = [np.dot(query, key) for key in keys]
            
            # Softmax over scores
            scores_exp = np.exp(scores - np.max(scores))  # numerical stability
            attention_weights = scores_exp / (np.sum(scores_exp) + 1e-6)
            
            # Weighted sum of block values
            attended_state = np.zeros(self.hidden_size)
            for w, block_repr in zip(attention_weights, block_representations):
                attended_state += w * block_repr
            
            # Combine base update with attended state
            final_state = base_update + 0.1 * attended_state  # Small residual weight
        else:
            final_state = base_update
        
        self.hidden_state = final_state
        
        return self.hidden_state
    
    def store_block_representation(self, block_output: np.ndarray):
        """
        Store current output as a block representation for future attention.
        
        Args:
            block_output: Output to store as block representation
        """
        self.block_history.append(block_output.copy())
        
        # Keep only recent blocks (limited history)
        if len(self.block_history) > self.max_block_history:
            self.block_history.pop(0)


class PredictiveCodingNode:
    """
    Extended World Node with predictive coding capabilities.
    Each node predicts its neighbors' states and learns from prediction errors.
    """
    
    def __init__(self, coordinates: Tuple[int, int, int], hidden_size: int, 
                 leak_rate: float = 0.1, learning_rate: float = 0.01):
        """
        Initialize a Predictive Coding Node.
        
        Args:
            coordinates: Tuple of (x, y, z) spatial position
            hidden_size: Dimensionality of the hidden state vector
            leak_rate: Rate of state decay in leaky integrator
            learning_rate: Learning rate for connection weight updates
        """
        self.coordinates = coordinates
        self.hidden_size = hidden_size
        self.leak_rate = leak_rate
        self.learning_rate = learning_rate  # For updating connection weights
        
        # Continuous latent state (persists across time steps)
        self.hidden_state = np.zeros(hidden_size)  # h_t
        self.bias = np.random.randn(hidden_size) * 0.1  # Node-specific bias
        self.refractory_counter = 0  # Tracks cooldown period
        
        # Predictive coding components
        self.predicted_neighbor_states: Dict[Tuple[int, int, int], np.ndarray] = {}  # Predictions for each neighbor
        self.prediction_errors: Dict[Tuple[int, int, int], np.ndarray] = {}  # Errors for each neighbor connection
        self.connection_weights: Dict[Tuple[int, int, int], float] = {}  # Weights to each neighbor
        
        # Initialize neighbor connections (will be set by WorldCore)
        self.neighbors: List[Tuple[int, int, int]] = []
        
    def update_state_continuous(self, incoming_state: np.ndarray, 
                               shared_weights: np.ndarray,
                               activation: str = 'tanh') -> np.ndarray:
        """
        Applies continuous update with leaky integrator:
        h_t = (1 - α) * h_{t-1} + α * activation(W_shared * incoming + b_v)
        
        Args:
            incoming_state: Input state from connected nodes
            shared_weights: Global weight matrix shared across all nodes
            activation: Activation function ('tanh' or 'relu')
            
        Returns:
            Updated hidden state vector (h_t)
        """
        # Leaky integrator component: retain (1-α) of previous state
        retained_state = (1.0 - self.leak_rate) * self.hidden_state
        
        # Innovation component: α * activation(W_shared * incoming + b_v)
        weighted_input = np.dot(shared_weights, incoming_state)
        pre_activation = weighted_input + self.bias
        
        if activation == 'tanh':
            activated_state = np.tanh(pre_activation)
        elif activation == 'relu':
            activated_state = np.maximum(0, pre_activation)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
            
        innovation = self.leak_rate * activated_state
        
        # Update state
        self.hidden_state = retained_state + innovation
        
        return self.hidden_state
    
    def predict_neighbor_states(self, shared_weights: np.ndarray,
                               activation: str = 'tanh') -> Dict[Tuple[int, int, int], np.ndarray]:
        """
        Predict the next state of each neighbor based on current state.
        This is the core of local predictive coding: each node predicts what its neighbors will experience.
        
        Args:
            shared_weights: Global weight matrix
            activation: Activation function
            
        Returns:
            Dictionary mapping neighbor coordinates to predicted state vectors
        """
        predictions = {}
        
        # Predict what each neighbor's state will be next
        for neighbor_coord in self.neighbors:
            # For neighbor prediction, we use the current node's state as input
            # (simulating what the neighbor would receive from this node)
            weighted_input = np.dot(shared_weights, self.hidden_state)
            pre_activation = weighted_input + self.bias  # Same bias as this node
            
            if activation == 'tanh':
                predicted_state = np.tanh(pre_activation)
            elif activation == 'relu':
                predicted_state = np.maximum(0, pre_activation)
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
                
            predictions[neighbor_coord] = predicted_state
            
        self.predicted_neighbor_states = predictions
        return predictions
    
    def update_prediction_errors(self, actual_neighbor_states: Dict[Tuple[int, int, int], np.ndarray]):
        """
        Update prediction errors based on actual neighbor states.
        
        Args:
            actual_neighbor_states: Dictionary mapping neighbor coordinates to actual state vectors
        """
        errors = {}
        
        for neighbor_coord in self.neighbors:
            if neighbor_coord in self.predicted_neighbor_states and neighbor_coord in actual_neighbor_states:
                # Prediction error = actual - predicted
                error = actual_neighbor_states[neighbor_coord] - self.predicted_neighbor_states[neighbor_coord]
                errors[neighbor_coord] = error
                
        self.prediction_errors = errors
    
    def update_connection_weights(self):
        """
        Update connection weights based on prediction errors.
        Lower error = stronger connection (Hebbian-like learning).
        Implements: w <- w * (1 - evaporation) + η * (1 - ||error||^2) * pre_post_correlation
        Increased evaporation rate and stronger penalty for high error.
        """
        # Increased evaporation rate for faster decay of unused paths
        evaporation_rate = 0.05  # Base evaporation rate (increased from original 0.01 implied)
        
        for neighbor_coord in self.neighbors:
            # Apply evaporation to all connections (decay unused paths)
            if neighbor_coord in self.connection_weights:
                self.connection_weights[neighbor_coord] *= (1.0 - evaporation_rate)
                # Ensure weight doesn't go below minimum
                self.connection_weights[neighbor_coord] = max(self.connection_weights[neighbor_coord], 0.01)
            
            if neighbor_coord in self.prediction_errors:
                error = self.prediction_errors[neighbor_coord]
                error_magnitude = np.linalg.norm(error)
                
                # Enhanced learning signal: stronger penalty for high error
                # Use squared error for more aggressive punishment of large errors
                # Small error -> positive weight update (strengthen connection)
                # Large error -> strong negative weight update (weaken connection significantly)
                error_penalty = min(1.0, error_magnitude ** 2)  # Squared error, capped at 1.0
                learning_signal = self.learning_rate * (1.0 - 2.0 * error_penalty)  # Increased penalty factor
                
                # Hebbian-like update: correlation between pre (this node) and post (neighbor)
                # In practice, we use the current hidden state as pre-synaptic activity
                pre_activity = self.hidden_state
                # For post-synaptic, we'd ideally use the neighbor's state, but we approximate
                # with the prediction since we don't have the neighbor's actual state during update
                post_activity = self.predicted_neighbor_states.get(neighbor_coord, np.zeros(self.hidden_size))
                
                # Outer product for weight update matrix (simplified to scalar for connection weight)
                correlation = np.dot(pre_activity, post_activity) / (self.hidden_size)  # Normalize
                
                # Update weight with evaporation already applied above
                if neighbor_coord in self.connection_weights:
                    self.connection_weights[neighbor_coord] += learning_signal * correlation
                    # Keep weights in reasonable bounds
                    self.connection_weights[neighbor_coord] = np.clip(
                        self.connection_weights[neighbor_coord], 0.01, 5.0
                    )
                else:
                    self.connection_weights[neighbor_coord] = 0.01 + learning_signal * correlation
    
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


class VectorDecoder:
    """
    Vector Decoder for bridging continuous tensor states to LLM string inputs.
    
    This follows the Interface Segregation Principle - providing a clean decode
    interface that translates raw hidden states into meaningful string summaries.
    """
    
    def __init__(self, hidden_size: int = 768):
        """
        Initialize the Vector Decoder.
        
        Args:
            hidden_size: Dimensionality of hidden state vectors
        """
        self.hidden_size = hidden_size
    
    def decode_to_string(self, state: np.ndarray) -> str:
        """
        Decode a continuous hidden state vector into a formatted string summary.
        
        Args:
            state: Hidden state vector [hidden_size]
            
        Returns:
            Formatted string summary with mathematical moments
        """
        # Calculate mathematical moments
        mean_val = float(np.mean(state))
        max_magnitude = float(np.max(np.abs(state)))
        std_val = float(np.std(state))
        
        # Format into structured prompt string
        summary = (
            f"State Vector Magnitude: {max_magnitude:.4f}. "
            f"Mean: {mean_val:.4f}. "
            f"StdDev: {std_val:.4f}. "
            f"Requesting graph evaluation..."
        )
        
        return summary
    
    def decode_batch(self, states: List[np.ndarray]) -> str:
        """
        Decode a batch of hidden states into a combined summary.
        
        Args:
            states: List of hidden state vectors
            
        Returns:
            Combined string summary
        """
        if not states:
            return "No state vectors to decode."
        
        # Stack and compute aggregate statistics
        state_stack = np.stack(states)
        mean_val = float(np.mean(state_stack))
        max_magnitude = float(np.max(np.abs(state_stack)))
        std_val = float(np.std(state_stack))
        
        summary = (
            f"Combined State Analysis: "
            f"Max Magnitude: {max_magnitude:.4f}. "
            f"Mean: {mean_val:.4f}. "
            f"StdDev: {std_val:.4f}. "
            f"Vector count: {len(states)}. "
            f"Requesting cognitive evaluation..."
        )
        
        return summary


class PredictiveCodingWorldCore:
    """
    3D World Core implementing Local Predictive Coding.
    Replaces global reinforcement with localized self-supervised prediction.
    Now with Sparse MoE and Block Attention Residuals for dynamic depth routing.
    """
    
    def __init__(self, dim_x: int, dim_y: int, dim_z: int, hidden_size: int,
                 leak_rate: float = 0.1,
                 learning_rate: float = 0.01,
                 vision_face_size: Tuple[int, int] = (4, 4),
                 text_face_size: Tuple[int, int] = (2, 2),
                 action_zone_size: Tuple[int, int] = (2, 2),
                 use_moe: bool = True,
                 num_experts: int = 8,
                 moe_k: int = 2,
                 num_blocks: int = 8,
                 cognitive_controller=None,
                 kg=None,
                 sandbox=None):
        """
        Initialize the Predictive Coding World Core.
        
        Args:
            dim_x, dim_y, dim_z: Dimensions of the 3D grid
            hidden_size: Dimensionality of hidden state vectors
            leak_rate: Rate of state decay in leaky integrator
            learning_rate: Learning rate for connection weight updates
            vision_face_size: Size of vision input face grid
            text_face_size: Size of text input face grid
            action_zone_size: Size of action injection zone
            use_moe: Enable Sparse Mixture of Experts
            num_experts: Number of expert networks
            moe_k: Top-k experts to route to
            num_blocks: Number of blocks for Block Attention Residuals
        """
        self.dimensions = (dim_x, dim_y, dim_z)
        self.hidden_size = hidden_size
        self.leak_rate = leak_rate
        self.learning_rate = learning_rate
        
        # Sparse MoE Configuration
        self.use_moe = use_moe and MOE_AVAILABLE
        self.num_experts = num_experts
        self.moe_k = moe_k
        self.num_blocks = num_blocks
        
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
        
        # Create 3D grid of Block Attention Predictive Coding nodes
        self.nodes: Dict[Tuple[int, int, int], BlockAttentionPredictiveCodingNode] = {}
        for x in range(dim_x):
            for y in range(dim_y):
                for z in range(dim_z):
                    node = BlockAttentionPredictiveCodingNode(
                        (x, y, z), hidden_size, leak_rate, learning_rate, num_blocks
                    )
                    self.nodes[(x, y, z)] = node
        
        # Establish neighbor connections (6-connected grid)
        self._establish_neighbor_connections()
        
        # Shared weight matrix for all nodes (W_shared)
        limit = np.sqrt(6.0 / (hidden_size + hidden_size))
        self.shared_weights = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        
        # Initialize MoE layer
        if self.use_moe:
            self.moe_layer = SparseMoE(
                hidden_size=hidden_size,
                num_experts=num_experts,
                k=moe_k,
                intermediate_size=hidden_size * 4,
                dropout=0.1,
                noise_std=1.0,
                capacity_factor=1.25,
                eval_capacity_factor=2.0
            )
            # Set to train mode to enable gradient computation
            self.moe_layer.train()
            print(f"  PredictiveCodingWorldCore MoE: Enabled ({num_experts} experts, k={moe_k})")
        else:
            self.moe_layer = None
            print(f"  PredictiveCodingWorldCore MoE: Disabled")
        
        # Initialize LocalOptimizerBridge for Hebbian gradient learning
        self.optimizer_bridge: Optional[LocalOptimizerBridge] = None
        self._moe_output_pt: Optional[torch.Tensor] = None  # Store PyTorch output for gradient computation
        if OPTIMIZATION_AVAILABLE and self.use_moe and self.moe_layer is not None:
            self.optimizer_bridge = LocalOptimizerBridge(
                moe_parameters=list(self.moe_layer.parameters()),
                learning_rate=1e-3,
                embed_dim=hidden_size
            )
            print(f"  LocalOptimizerBridge: Initialized for MoE gradients with D-MMD")
        
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
        
        # --- COMPOSITION: Wire in neurosymbolic modules (Dependency Injection) ---
        # Using Composition over Inheritance - encapsulate cognitive modules as member objects
        # Dependencies are now injected via constructor (DIP compliance)
        if cognitive_controller is not None:
            self.cognitive_controller = cognitive_controller
            print(f"  Cognitive Controller: Injected (DIP)")
        elif NEUROSYMBOLIC_AVAILABLE:
            self.cognitive_controller = CognitiveController(hidden_size=hidden_size)
            print(f"  Cognitive Controller: Initialized (fallback)")
        else:
            self.cognitive_controller = None
            print(f"  Cognitive Controller: Not available")
        
        if kg is not None:
            self.kg = kg
            print(f"  Neurosymbolic KG: Injected (DIP)")
        elif NEUROSYMBOLIC_AVAILABLE:
            self.kg = NeurosymbolicKG()
            print(f"  Neurosymbolic KG: Initialized (fallback)")
        else:
            self.kg = None
            print(f"  Neurosymbolic KG: Not available")
        
        if sandbox is not None:
            self.sandbox = sandbox
            print(f"  Agentic Sandbox: Injected (DIP)")
        elif NEUROSYMBOLIC_AVAILABLE:
            self.sandbox = AgenticSandbox()
            print(f"  Agentic Sandbox: Initialized (fallback)")
        else:
            self.sandbox = None
            print(f"  Agentic Sandbox: Not available")
        
        # Initialize Vector Decoder for bridging tensor states to LLM string inputs
        self.vector_decoder = VectorDecoder(hidden_size=hidden_size)
        print(f"  Vector Decoder: Initialized")
        
        # === D-MMD Teacher-Student State ===
        # Tick counter for periodic D-MMD distillation (every 5th tick)
        self.tick_count: int = 0
        self.d_mmd_distillation_interval: int = 5
        
        # Define KG face (X=4) and Agentic face (X=5) coordinates
        # These planes will store knowledge graph and agentic tool state
        self.kg_face_coords = self._generate_face_coords(4, vision_face_size)
        self.agentic_face_coords = self._generate_face_coords(5, action_zone_size)
        
        print(f"Initialized PredictiveCodingWorldCore:")
        print(f"  Dimensions: {self.dimensions}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Leak rate: {leak_rate}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Vision face ({len(self.vision_face_coords)} nodes): x=0")
        print(f"  Text face ({len(self.text_face_coords)} nodes): x=1")
        print(f"  Action zone ({len(self.action_zone_coords)} nodes): x=2")
        print(f"  KG face ({len(self.kg_face_coords)} nodes): x=4")
        print(f"  Agentic face ({len(self.agentic_face_coords)} nodes): x=5")
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
    
    def _establish_neighbor_connections(self):
        """Establish neighbor connections for each node (6-connected grid)."""
        # Define 6 orthogonal directions: +/- x, +/- y, +/- z
        directions = [
            (1, 0, 0), (-1, 0, 0),  # x-axis
            (0, 1, 0), (0, -1, 0),  # y-axis
            (0, 0, 1), (0, 0, -1)   # z-axis
        ]
        
        for (x, y, z), node in self.nodes.items():
            node.neighbors = []  # Reset neighbors
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                # Check if neighbor is within bounds
                if (0 <= nx < self.dimensions[0] and 
                    0 <= ny < self.dimensions[1] and 
                    0 <= nz < self.dimensions[2]):
                    neighbor_coord = (nx, ny, nz)
                    node.neighbors.append(neighbor_coord)
                    # Initialize connection weight
                    if neighbor_coord not in node.connection_weights:
                        node.connection_weights[neighbor_coord] = 1.0
    
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
        Advance the world core by one time step using local predictive coding.
        
        Args:
            vision_input: Optional vision input for this time step
            text_input: Optional text input for this time step
            action_input: Optional action input for this time step
            
        Returns:
            Dictionary containing outputs from various faces
        """
        # Step 1: Inject multi-modal inputs
        self.inject_multi_modal_input(vision_input, text_input, action_input)
        
        # Step 2: Generate predictions for all neighbors (predictive coding phase)
        all_predictions = {}  # Maps (source_coord, target_coord) -> predicted_state
        for coord, node in self.nodes.items():
            # Each node predicts what its neighbors will experience
            neighbor_predictions = node.predict_neighbor_states(self.shared_weights, activation='tanh')
            for neighbor_coord, prediction in neighbor_predictions.items():
                all_predictions[(coord, neighbor_coord)] = prediction
        
        # Step 3: Update all nodes with continuous dynamics
        # Each node receives input from its connected neighbors
        for coord, node in self.nodes.items():
            # Gather incoming states from connected neighbors
            incoming_states = []
            weights = []
            
            for neighbor_coord in node.neighbors:
                if neighbor_coord in self.nodes:
                    # Get the actual state of the neighbor
                    neighbor_state = self.nodes[neighbor_coord].hidden_state
                    # Get the connection weight from this node to the neighbor
                    weight = node.connection_weights.get(neighbor_coord, 1.0)
                    
                    incoming_states.append(neighbor_state)
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
            
            # Update node state with continuous dynamics
            node.update_state_continuous(incoming_state, self.shared_weights, activation='tanh')
        
        # --- SPARSE MoE DYNAMIC ROUTING ---
        # NOTE: Removed torch.no_grad() to enable gradient computation for Hebbian learning
        if hasattr(self, 'moe_layer') and self.moe_layer is not None:
            # Gather states and convert to PyTorch
            node_coords = list(self.nodes.keys())
            states_np = np.array([self.nodes[c].hidden_state for c in node_coords])
            states_pt = torch.from_numpy(states_np).float().unsqueeze(1)  # Add seq dim
            
            # Route through MoE experts - UNFROZEN for gradient computation
            moe_out_pt = self.moe_layer(states_pt)
            
            # Preserve the computational graph for the Hebbian backward pass
            # (DO NOT detach here - we need the graph to flow to moe_layer parameters)
            self._moe_output_pt = moe_out_pt
            
            # Detach ONLY for the continuous NumPy physics engine
            moe_out_np = moe_out_pt.detach().squeeze(1).numpy()
            
            # Update states and store block representations
            for i, coord in enumerate(node_coords):
                self.nodes[coord].hidden_state = moe_out_np[i]
                # Store representation for block attention history
                self.nodes[coord].store_block_representation(moe_out_np[i])
        # --- END MoE ROUTING ---
        
        # === D-MMD TEACHER-STUDENT DISTILLATION ===
        # Every 5th tick, perform distillation from SC-MCTS teacher to MoE student
        self.tick_count += 1
        if self.tick_count % self.d_mmd_distillation_interval == 0:
            if self.optimizer_bridge is not None and self._moe_output_pt is not None:
                try:
                    # === Generate Teacher Target via SC-MCTS ===
                    # Use cognitive controller with KG to generate high-quality target
                    teacher_target_np = None
                    if hasattr(self, 'kg') and self.kg is not None and hasattr(self.kg, 'mcts'):
                        # Generate a reasoning query based on current hidden states
                        # Get average hidden state as query representation
                        avg_hidden = np.mean([n.hidden_state for n in self.nodes.values()], axis=0)
                        query_magnitude = float(np.linalg.norm(avg_hidden))
                        
                        # Create a query for SC-MCTS
                        query = f"reasoning_depth_{int(query_magnitude * 100)}"
                        
                        # Run SC-MCTS search to get reasoning path
                        context = {"hidden_states": avg_hidden}
                        paths, _ = self.kg.mcts.search(start_node_id=0, query=query, context=context)
                        
                        # Extract target logits from SC-MCTS reasoning path
                        if paths and len(paths) > 0:
                            # Use the best path's final state as teacher target
                            best_path = paths[0]
                            teacher_target_np = np.array(best_path.get('value', avg_hidden))
                    
                    # If no KG/MCTS available, fall back to using current MoE output as target
                    # (essentially self-distillation for initialization)
                    if teacher_target_np is None:
                        teacher_target_np = self._moe_output_pt.detach().cpu().numpy()
                    
                    # Convert teacher target to PyTorch tensor with matching shape
                    teacher_target_pt = torch.from_numpy(teacher_target_np).float()
                    
                    # Ensure shape matches student output
                    if teacher_target_pt.dim() == 2:
                        teacher_target_pt = teacher_target_pt.unsqueeze(1)  # Add seq dim
                    
                    # Apply D-MMD gradients (Student-Teacher entropy cancellation)
                    d_mmd_loss = self.optimizer_bridge.apply_d_mmd_gradients(
                        self._moe_output_pt,
                        teacher_target_pt
                    )
                    print(f"  [D-MMD] Tick {self.tick_count}: distillation loss = {d_mmd_loss:.6f}")
                    
                except Exception as e:
                    print(f"  [D-MMD] Warning: distillation failed: {e}")
        # === END D-MMD DISTILLATION ===
        
        # --- ACTIVE ENGINE: HARD-ZERO SPATIAL OVERRIDE (NUMPY) ---
        # 1. Extract all hidden states and their magnitudes
        all_states = np.array([node.hidden_state for node in self.nodes.values()])
        state_mags = np.linalg.norm(all_states, axis=-1)
        
        # 2. Find the global 15% threshold across the flattened volume
        k_active = max(1, int(len(state_mags) * 0.15))
        global_thresh = np.partition(state_mags, -k_active)[-k_active]
        
        # 3. Enforce strict sparsity and aggressive decay
        for node, mag in zip(self.nodes.values(), state_mags):
            if mag < global_thresh:
                node.hidden_state = np.zeros_like(node.hidden_state)
            else:
                node.hidden_state *= 0.80
        # --- END OVERRIDE ---
        


        # Step 4: Gather actual neighbor states for error calculation
        actual_neighbor_states_dict: Dict[Tuple[int, int, int], Dict[Tuple[int, int, int], np.ndarray]] = {}
        for coord, node in self.nodes.items():
            actual_neighbor_states = {}
            for neighbor_coord in node.neighbors:
                if neighbor_coord in self.nodes:
                    actual_neighbor_states[neighbor_coord] = self.nodes[neighbor_coord].hidden_state.copy()
            actual_neighbor_states_dict[coord] = actual_neighbor_states
        
        # Step 5: Update prediction errors and connection weights (local learning)
        for coord, node in self.nodes.items():
            if coord in actual_neighbor_states_dict:
                # Update prediction errors
                node.update_prediction_errors(actual_neighbor_states_dict[coord])
                # Update connection weights based on errors
                node.update_connection_weights()
        
        # --- LOCAL HEBBIAN GRADIENT BRIDGE ---
        # Apply localized gradients using the stored MoE output and local prediction errors
        # This enables O(1) memory scaling by immediately calling backward() and step()
        if self.optimizer_bridge is not None and self._moe_output_pt is not None:
            # Calculate local target states from prediction errors for each node
            # The target is: current_state - prediction_error (corrective signal)
            node_coords = list(self.nodes.keys())
            
            # Create target tensor from prediction errors
            target_states = []
            execution_penalty = 0.0
            
            # --- TDD GRADIENT BRIDGE: Execute code and compute penalty ---
            # Try to execute code via sandbox if available
            sandbox_success = None
            if hasattr(self, 'sandbox') and self.sandbox is not None:
                # Generate code from hidden states (simplified: use first node's state)
                # In production, this would decode the hidden state to actual Python code
                try:
                    # Sample code for execution testing
                    sample_code = '''
def test_function(x):
    return x * 2

result = test_function(21)
print(result)
'''
                    execution_result = self.sandbox.execute_python(sample_code)
                    
                    # Check if execution was successful
                    sandbox_success = not execution_result.startswith("Error:")
                    
                    if not sandbox_success:
                        # Heavy penalty for execution failure
                        execution_penalty = 1.0
                        if hasattr(self, '_debug_logging') and self._debug_logging:
                            print(f"  [TDD] Execution failed: {execution_result[:50]}...")
                    else:
                        execution_penalty = 0.0
                        if hasattr(self, '_debug_logging') and self._debug_logging:
                            print(f"  [TDD] Execution successful!")
                except Exception as e:
                    sandbox_success = False
                    execution_penalty = 0.5  # Moderate penalty for sandbox errors
                    if hasattr(self, '_debug_logging') and self._debug_logging:
                        print(f"  [TDD] Sandbox error: {e}")
            # --- END TDD GRADIENT BRIDGE ---
            
            for coord in node_coords:
                node = self.nodes[coord]
                # Get aggregate prediction error for this node
                if coord in node.prediction_errors and len(node.prediction_errors) > 0:
                    # Use the mean prediction error as the correction signal
                    errors = list(node.prediction_errors.values())
                    mean_error = np.mean(errors, axis=0)
                    # Target = current state - error (to reduce error)
                    target = node.hidden_state - mean_error
                else:
                    # No prediction error available, use current state as target
                    target = node.hidden_state
                
                # Apply TDD penalty: if execution failed, heavily penalize this pathway
                if execution_penalty > 0:
                    # Push hidden state toward zero (suppress active experts)
                    target = target * (1.0 - execution_penalty)
                
                target_states.append(target)
            
            # Convert target to PyTorch tensor
            target_np = np.array(target_states)
            target_pt = torch.from_numpy(target_np).float().unsqueeze(1)
            
            # Apply local gradients using the bridge
            # This computes MSE between MoE output and target, then backward() + step()
            try:
                local_loss = self.optimizer_bridge.apply_moe_gradients(
                    self._moe_output_pt,
                    target_pt
                )
                # Clear stored output after gradient application
                self._moe_output_pt = None
                if hasattr(self, '_debug_logging') and self._debug_logging:
                    print(f"  [Hebbian] Local gradient applied, loss: {local_loss:.6f}")
            except Exception as e:
                # If gradient application fails, clear stored output and continue
                self._moe_output_pt = None
                if hasattr(self, '_debug_logging') and self._debug_logging:
                    print(f"  [Hebbian] Gradient application warning: {e}")
        # --- END HEBBIAN GRADIENT BRIDGE ---
        
        # Step 6: Apply refractory periods based on aggregate prediction error
        for coord, node in self.nodes.items():
            # Calculate aggregate prediction error magnitude
            total_error = 0.0
            error_count = 0
            if coord in node.prediction_errors:
                for error in node.prediction_errors.values():
                    total_error += np.linalg.norm(error)
                    error_count += 1
            
            avg_error = total_error / max(error_count, 1)
            
            # High prediction error -> longer refractory period (exploration)
            # Low prediction error -> shorter refractory period (exploitation)
            if avg_error > 0.5:
                node.set_refractory(2)  # Explore more
            elif avg_error < 0.1:
                node.set_refractory(0)  # Exploit more
            else:
                node.set_refractory(1)  # Neutral
        
        # Step 7: Decrement refractory counters
        for node in self.nodes.values():
            node.tick_refractory()
        
        # Step 8: Collect outputs from various faces
        outputs = {}
        
        # --- SRP: COGNITIVE PROCESSING (with Vector Decoding Bridge) ---
        # Extract average hidden state from X=2 (text/action plane) and X=4 (KG plane)
        # and pass to cognitive controller for processing
        # This keeps the physics loop clean; cognitive logic is handled by the controller
        # 
        # CRITICAL: Raw tensors are NOT passed to LLM - they must be decoded first!
        # This follows Interface Segregation - VectorDecoder translates continuous states
        # to string summaries before LLM consumption.
        if self.cognitive_controller is not None:
            # Extract states from X=2 plane (text/action)
            text_action_states = []
            for coord in self.text_face_coords:
                if coord in self.nodes:
                    text_action_states.append(self.nodes[coord].hidden_state)
            
            # Extract states from X=4 plane (KG face)
            kg_states = []
            for coord in self.kg_face_coords:
                if coord in self.nodes:
                    kg_states.append(self.nodes[coord].hidden_state)
            
            # Combine states for cognitive processing
            combined_states = []
            if text_action_states:
                combined_states.extend(text_action_states)
            if kg_states:
                combined_states.extend(kg_states)
            
            if combined_states:
                avg_hidden_state = np.mean(combined_states, axis=0)
                
                # VECTOR DECODING BRIDGE: Convert raw tensor to string summary
                # This is the critical bridging layer - NO raw vectors hitting the LLM!
                decoded_prompt = self.vector_decoder.decode_batch(combined_states)
                
                # Create a string-based input for the cognitive controller
                # Pass the decoded prompt string instead of raw tensor
                cognitive_input = {
                    'prompt': decoded_prompt,
                    'hidden_state': avg_hidden_state,  # Keep for internal processing
                    'text_action_count': len(text_action_states),
                    'kg_count': len(kg_states)
                }
                
                # Pass to cognitive controller - clean interface, no direct state manipulation
                cognitive_result = self.cognitive_controller.run(cognitive_input)
                # Store result for later use (e.g., tool execution in sandbox)
                outputs['cognitive_result'] = cognitive_result
                outputs['decoded_prompt'] = decoded_prompt  # Store for debugging
        # --- END COGNITIVE PROCESSING ---
        
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
        total_connection_weight = 0.0
        connection_count = 0
        
        for node in self.nodes.values():
            magnitude = np.linalg.norm(node.hidden_state)
            error_magnitude = 0.0
            error_count = 0
            
            # Calculate average prediction error
            if node.prediction_errors:
                for error in node.prediction_errors.values():
                    error_magnitude += np.linalg.norm(error)
                    error_count += 1
            
            avg_node_error = error_magnitude / max(error_count, 1)
            total_prediction_error += avg_node_error
            
            # Connection statistics
            for weight in node.connection_weights.values():
                total_connection_weight += weight
                connection_count += 1
            
            # --- EPSILON-SAFE METRIC OVERRIDE ---
            # Do not count microscopic floating-point noise as an active node
            epsilon_threshold = 1e-4
            if magnitude > epsilon_threshold:  # Epsilon threshold for considering active
                active_count += 1
            if node.is_refractory():
                refractory_count += 1
        
        activation_ratio = active_count / total_nodes if total_nodes > 0 else 0.0
        avg_magnitude = total_magnitude / total_nodes if total_nodes > 0 else 0.0
        avg_prediction_error = total_prediction_error / total_nodes if total_nodes > 0 else 0.0
        refractory_ratio = refractory_count / total_nodes if total_nodes > 0 else 0.0
        avg_connection_weight = total_connection_weight / max(connection_count, 1)
        
        return {
            'total_nodes': total_nodes,
            'active_nodes': active_count,
            'activation_ratio': activation_ratio,
            'average_magnitude': avg_magnitude,
            'average_prediction_error': avg_prediction_error,
            'refractory_count': refractory_count,
            'refractory_ratio': refractory_ratio,
            'average_connection_weight': avg_connection_weight
        }
    
    def reset_world(self):
        """Reset the world core to initial state."""
        # Reset node states
        for node in self.nodes.values():
            node.hidden_state = np.zeros(self.hidden_size)
            node.refractory_counter = 0
            node.bias = np.random.randn(self.hidden_size) * 0.1
            
            # Reset predictive coding components
            node.predicted_neighbor_states.clear()
            node.prediction_errors.clear()
            # Keep connection weights but reset to initial values
            for neighbor_coord in node.connection_weights:
                node.connection_weights[neighbor_coord] = 1.0
            node.neighbors = []  # Will be re-established
            
        # Re-establish neighbor connections
        self._establish_neighbor_connections()
        
        # Reset shared weights
        limit = np.sqrt(6.0 / (self.hidden_size + self.hidden_size))
        self.shared_weights = np.random.uniform(-limit, limit, (self.hidden_size, self.hidden_size))


def create_predictive_coding_example():
    """
    Create an example showing how to use the Predictive Coding World Core.
    """
    print("Creating PredictiveCodingWorldCore for Local Predictive Coding...")
    
    # Create predictive coding world core
    world = PredictiveCodingWorldCore(
        dim_x=10, dim_y=8, dim_z=8,
        hidden_size=32,
        leak_rate=0.1,
        learning_rate=0.01,
        vision_face_size=(4, 4),
        text_face_size=(2, 2),
        action_zone_size=(2, 2)
    )
    
    print(f"PredictiveCodingWorldCore created with {len(world.nodes)} nodes")
    print(f"Each node has {len(list(world.nodes.values())[0].neighbors)} neighbors")
    
    # Simulate a few time steps to demonstrate learning
    print("\nSimulating predictive coding dynamics...")
    for t in range(10):
        # Generate inputs with some temporal structure
        # Vision input: moving pattern
        vision_phase = t * 0.1
        vision_input = np.zeros((len(world.vision_face_coords), world.hidden_size))
        for i in range(len(world.vision_face_coords)):
            vision_input[i] = np.sin(np.arange(world.hidden_size) * 0.1 + vision_phase + i) * 0.2
        
        # Text input: periodic pulses
        text_input = np.zeros((len(world.text_face_coords), world.hidden_size))
        if t % 3 == 0:  # Pulse every 3 steps
            text_input[:, :] = 0.5
        
        # Action input: random exploration
        action_input = np.random.randn(len(world.action_zone_coords), world.hidden_size) * 0.1
        
        # Advance world
        outputs = world.tick_world(vision_input, text_input, action_input)
        
        # Get statistics
        stats = world.get_world_statistics()
        
        if t % 2 == 0:  # Print every other step to reduce output
            print(f"Time step {t}:")
            print(f"  Activation ratio: {stats['activation_ratio']:.3f}")
            print(f"  Avg prediction error: {stats['average_prediction_error']:.3f}")
            print(f"  Avg connection weight: {stats['average_connection_weight']:.3f}")
            print(f"  Refractory ratio: {stats['refractory_ratio']:.3f}")
    
    print("\nPredictiveCodingWorldCore ready for JEPA decoder implementation!")
    return world


if __name__ == "__main__":
    # Run the example
    create_predictive_coding_example()