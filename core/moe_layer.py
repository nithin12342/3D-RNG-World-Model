"""
Sparse Mixture of Experts (MoE) Layer for 3D-RNG World Engine
Chief AI Architect & Advanced Routing Specialist Implementation

This module implements Sparse Mixture of Experts with noisy top-k gating for
conditional computation in the 3D-RNG architecture. Enables dynamic routing of
signals to a subset of expert networks, achieving state-of-the-art scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class NoisyTopKGating(nn.Module):
    """
    Noisy Top-K Gating mechanism for sparse MoE.
    
    Implements the gating mechanism from "Outrageously Large Neural Networks"
    (Shazeer et al., 2017) with added noise for load balancing.
    
    Key features:
    - Adds noise to logits before softmax for load balancing
    - Selects top-k experts with sparse activation
    - Learns expert bias terms for better routing
    """
    
    def __init__(self, hidden_size: int, num_experts: int, k: int = 2, 
                 noise_std: float = 1.0, gate_bias: float = 0.0):
        """
        Initialize Noisy Top-K Gating.
        
        Args:
            hidden_size: Input dimension (d_model)
            num_experts: Total number of expert networks
            k: Number of experts to route to (top-k)
            noise_std: Standard deviation of noise for load balancing
            gate_bias: Initial bias for expert routing logits
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.k = k
        self.noise_std = noise_std
        self.noise_epsilon = 1e-6
        
        # Gate linear layer: projects hidden states to expert logits
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Expert bias (learnable per-expert routing preference)
        self.register_buffer('expert_bias', torch.ones(num_experts) * gate_bias)
        
        # Learnable noise weight for controlling noise magnitude
        self.noise_weight = nn.Parameter(torch.zeros(num_experts))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize gate weights with Xavier initialization."""
        nn.init.xavier_uniform_(self.gate.weight)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gating weights for expert routing.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size) or 
               (batch_size, hidden_size) for single token batch
            
        Returns:
            Tuple of (gated_outputs, expert_weights, expert_indices)
            - gated_outputs: Weighted sum of expert outputs
            - expert_weights: Softmax weights for selected experts
            - expert_indices: Indices of selected experts
        """
        # Compute raw logits from gate
        logits = self.gate(x)  # (batch, seq_len, num_experts) or (batch, num_experts)
        
        # Add expert bias for routing preference
        logits = logits + self.expert_bias
        
        # Add noise for load balancing (only during training)
        if self.training and self.noise_std > 0:
            # Learnable noise magnitude per expert
            noise_magnitude = self.noise_std * torch.sigmoid(self.noise_weight)
            noise = torch.randn_like(logits) * noise_magnitude
            logits = logits + noise
        
        # Compute softmax weights
        weights = F.softmax(logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(weights, self.k, dim=-1)
        
        # Normalize top-k weights (they don't sum to 1 anymore after selection)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + self.noise_epsilon)
        
        # Create sparse gating output (zeros for non-selected experts)
        # This creates the sparse computation pattern
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(-1, top_k_indices, top_k_weights)
        
        return sparse_weights, top_k_weights, top_k_indices


class ExpertNetwork(nn.Module):
    """
    Individual Expert Network for MoE layer.
    
    Each expert is a feed-forward network with two layers and GELU activation.
    Based on the MLP block from "Attention is All You Need" (Vaswani et al., 2017).
    """
    
    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None,
                 dropout: float = 0.1):
        """
        Initialize Expert Network.
        
        Args:
            hidden_size: Input and output dimension
            intermediate_size: Hidden layer dimension (default: 4 * hidden_size)
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or hidden_size * 4
        
        # First linear projection (up-projection)
        self.up_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        
        # Second linear projection (down-projection)
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size, bias=False)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize expert weights."""
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert network.
        
        Args:
            x: Input tensor
            
        Returns:
            Expert output
        """
        # Up-projection with GELU activation
        hidden = self.up_proj(x)
        hidden = F.gelu(hidden)
        
        # Down-projection with dropout
        output = self.dropout(self.down_proj(hidden))
        
        return output


class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts Layer.
    
    Routes incoming signals to a subset of expert networks using noisy top-k gating.
    The output is a weighted combination of the selected experts' outputs.
    
    This implementation follows the MoE formulation:
    y = Σ_{i=1}^{n} G(x)_i * E_i(x)
    
    where G(x) is the gating function and E_i are the expert networks.
    
    Key features:
    - Noisy top-k gating for load balancing
    - Learnable expert bias for routing preference
    - Dropout for regularization
    - Optional capacity factor for expert utilization control
    """
    
    def __init__(self, hidden_size: int, num_experts: int = 8, k: int = 2,
                 intermediate_size: Optional[int] = None, dropout: float = 0.1,
                 noise_std: float = 1.0, capacity_factor: float = 1.25,
                 eval_capacity_factor: float = 2.0):
        """
        Initialize Sparse MoE Layer.
        
        Args:
            hidden_size: Input and output dimension (d_model)
            num_experts: Total number of expert networks
            k: Number of experts to route to (top-k)
            intermediate_size: Intermediate dimension for experts (default: 4 * hidden_size)
            dropout: Dropout rate for expert outputs
            noise_std: Standard deviation of noise for load balancing
            capacity_factor: Multiplier for expert capacity during training
            eval_capacity_factor: Multiplier for expert capacity during evaluation
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        
        # Ensure k <= num_experts
        assert k <= num_experts, f"k ({k}) must be <= num_experts ({num_experts})"
        
        # Noisy top-k gating mechanism
        self.gating = NoisyTopKGating(
            hidden_size=hidden_size,
            num_experts=num_experts,
            k=k,
            noise_std=noise_std
        )
        
        # Create pool of expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dropout=dropout
            )
            for _ in range(num_experts)
        ])
        
        # Output layer normalization for stability
        self.output_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        # Residual connection for gradient flow
        self.residual_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Sparse MoE.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size) or
               (batch_size, hidden_size)
               
        Returns:
            MoE output of same shape as input
        """
        original_shape = x.shape
        is_batched = len(original_shape) == 3
        
        if not is_batched:
            # Add sequence dimension if single token
            x = x.unsqueeze(1)
        
        batch_size, seq_len, hidden_size = x.shape
        
        # Get gating weights
        # Shape: (batch, seq_len, num_experts)
        gating_weights, top_k_weights, top_k_indices = self.gating(x)
        
        # Determine expert capacity
        capacity = int(seq_len * self.capacity_factor if self.training 
                      else seq_len * self.eval_capacity_factor)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Track expert utilization for monitoring
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Create mask for tokens going to this expert
            expert_mask = gating_weights[..., expert_idx] > 0
            
            if expert_mask.any():
                # Count tokens for this expert (clamped to capacity)
                num_tokens = expert_mask.sum().item()
                expert_usage[expert_idx] = num_tokens
                
                # Clamp to capacity (drop tokens if overloaded)
                if num_tokens > capacity:
                    # Select first 'capacity' tokens (deterministic)
                    expert_mask = expert_mask.clone()
                    cumsum = expert_mask.cumsum(dim=1)
                    expert_mask = expert_mask & (cumsum <= capacity)
                
                # Get input tokens for this expert
                expert_input = x[expert_mask]
                
                if expert_input.numel() > 0:
                    # Process through expert network
                    expert_output = self.experts[expert_idx](expert_input)
                    
                    # Get gating weights for these tokens
                    expert_gating = gating_weights[expert_mask, expert_idx].unsqueeze(-1)
                    
                    # Weight expert output by gating weight
                    weighted_output = expert_output * expert_gating
                    
                    # Accumulate to output (scatter_add for efficiency)
                    output[expert_mask] += weighted_output
        
        # Apply output normalization
        output = self.output_norm(output)
        
        # Add residual connection (identity mapping)
        # This helps with gradient flow and maintains base representation
        output = output + self.residual_dropout(x)
        
        # Reshape to original shape if needed
        if not is_batched:
            output = output.squeeze(1)
        
        return output
    
    def get_expert_utilization(self) -> torch.Tensor:
        """
        Get current expert utilization statistics.
        
        Returns:
            Tensor of shape (num_experts,) with utilization counts
        """
        # This would need to be tracked during forward pass
        # Return placeholder - in practice would be accumulated during training
        return torch.zeros(self.num_experts)


class BlockSparseMoE(nn.Module):
    """
    Block-level Sparse MoE for hierarchical routing.
    
    Partitions the hidden state into blocks and applies MoE routing within each block.
    This provides finer-grained control over computation allocation.
    """
    
    def __init__(self, hidden_size: int, num_experts: int = 4, k: int = 1,
                 num_blocks: int = 4, dropout: float = 0.1):
        """
        Initialize Block Sparse MoE.
        
        Args:
            hidden_size: Total hidden size
            num_experts: Number of experts per block
            k: Number of experts to route to per block
            num_blocks: Number of blocks to partition hidden state into
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        
        assert hidden_size % num_blocks == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_blocks ({num_blocks})"
        
        # Create MoE for each block
        self.block_moes = nn.ModuleList([
            SparseMoE(
                hidden_size=self.block_size,
                num_experts=num_experts,
                k=k,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through block-level MoE.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            MoE output of same shape
        """
        # Split into blocks
        blocks = torch.split(x, self.block_size, dim=-1)
        
        # Process each block through its MoE
        block_outputs = [self.block_moes[i](blocks[i]) for i in range(self.num_blocks)]
        
        # Concatenate back together
        output = torch.cat(block_outputs, dim=-1)
        
        return output


class ConditionalExpertRouting(nn.Module):
    """
    Conditional Expert Routing with learned routing policies.
    
    Implements a learnable router that can adaptively select experts based on
    input context, enabling dynamic depth routing (Mixture of Depths).
    """
    
    def __init__(self, hidden_size: int, num_experts: int, num_routing_policies: int = 4):
        """
        Initialize Conditional Expert Routing.
        
        Args:
            hidden_size: Input dimension
            num_experts: Total number of experts
            num_routing_policies: Number of learned routing policies
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_routing_policies = num_routing_policies
        
        # Policy encoder: learns different routing strategies
        self.policy_encoder = nn.Linear(hidden_size, num_routing_policies)
        
        # Expert selector: uses policy to select experts
        self.expert_selector = nn.Linear(num_routing_policies, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            ExpertNetwork(hidden_size) for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Forward pass with conditional routing.
        
        Args:
            x: Input tensor
            temperature: Routing temperature (higher = more random)
            
        Returns:
            Expert output
        """
        # Encode routing policy
        policy_logits = self.policy_encoder(x)
        
        # Apply temperature for stochastic routing
        if temperature != 1.0:
            policy_logits = policy_logits / temperature
        
        # Softmax over policies
        policy_weights = F.softmax(policy_logits, dim=-1)
        
        # Select experts based on policy
        expert_weights = self.expert_selector(policy_weights)
        expert_weights = F.softmax(expert_weights, dim=-1)
        
        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        
        # Weight by expert selection
        output = torch.sum(expert_outputs * expert_weights.unsqueeze(-2), dim=-1)
        
        return output


# Utility functions for MoE integration

def create_moe_layer(config: dict) -> SparseMoE:
    """
    Factory function to create MoE layer from configuration.
    
    Args:
        config: Dictionary with MoE configuration
        
    Returns:
        Initialized SparseMoE layer
    """
    return SparseMoE(
        hidden_size=config.get('hidden_size', 256),
        num_experts=config.get('num_experts', 8),
        k=config.get('k', 2),
        intermediate_size=config.get('intermediate_size', None),
        dropout=config.get('dropout', 0.1),
        noise_std=config.get('noise_std', 1.0),
        capacity_factor=config.get('capacity_factor', 1.25),
        eval_capacity_factor=config.get('eval_capacity_factor', 2.0)
    )


def count_moe_parameters(moe_layer: SparseMoE) -> dict:
    """
    Count parameters in MoE layer.
    
    Args:
        moe_layer: SparseMoE layer
        
    Returns:
        Dictionary with parameter counts
    """
    gate_params = sum(p.numel() for p in moe_layer.gating.parameters())
    expert_params = sum(p.numel() for p in moe_layer.experts.parameters())
    norm_params = sum(p.numel() for p in moe_layer.output_norm.parameters())
    
    return {
        'gate_parameters': gate_params,
        'expert_parameters': expert_params,
        'normalization_parameters': norm_params,
        'total_parameters': gate_params + expert_params + norm_params,
        'num_experts': moe_layer.num_experts
    }
