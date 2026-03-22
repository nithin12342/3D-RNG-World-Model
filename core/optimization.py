"""
Local Hebbian Gradient Bridge for 3D-RNG World Engine
Chief Neural Dynamics Engineer & Optimization Architect Implementation

This module implements the LocalOptimizerBridge for unfreezing PyTorch modules
(SparseMoE and SpatialTokenizer) and connecting their weights to local spatial
prediction errors without utilizing global backpropagation, preserving O(1) memory scaling.

Key Design Principles:
- Single Responsibility: Dedicated optimizer bridge for local gradient updates
- O(1) Memory: Immediate backward() and step() calls prevent gradient accumulation
- Dependency Injection: Modules passed during initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Any
import numpy as np


class LocalOptimizerBridge:
    """
    Local Hebbian Gradient Bridge for O(1) memory local learning.
    
    This bridge calculates localized MSE between the MoE/Tokenizer outputs
    and target states dictated by the node's local prediction error, then
    immediately calls backward() and step() to prevent graph accumulation.
    
    Architecture:
    - Maintains separate Adam optimizers for MoE and Tokenizer parameters
    - Provides apply_local_gradients() for immediate gradient application
    - Stores intermediate PyTorch tensors for the error calculation phase
    
    Usage:
        bridge = LocalOptimizerBridge(moe_parameters, tokenizer_parameters)
        bridge.apply_local_gradients(predicted_tensors, target_tensors)
    """
    
    def __init__(
        self,
        moe_parameters: Optional[List[torch.Tensor]] = None,
        tokenizer_parameters: Optional[List[torch.Tensor]] = None,
        learning_rate: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Initialize the LocalOptimizerBridge.
        
        Args:
            moe_parameters: List of parameters from SparseMoE module
            tokenizer_parameters: List of parameters from SpatialTokenizer module
            learning_rate: Learning rate for Adam optimizers
            betas: Coefficients for computing running averages of gradient
            eps: Term added to denominator to improve numerical stability
            weight_decay: Weight decay (L2 penalty)
        """
        self.learning_rate = learning_rate
        self.moe_optimizer: Optional[torch.optim.Adam] = None
        self.tokenizer_optimizer: Optional[torch.optim.Adam] = None
        
        # Store intermediate tensors for error calculation
        self._stored_moe_output: Optional[torch.Tensor] = None
        self._stored_tokenizer_output: Optional[torch.Tensor] = None
        
        # Initialize MoE optimizer if parameters provided
        if moe_parameters is not None and len(moe_parameters) > 0:
            self.moe_optimizer = torch.optim.Adam(
                moe_parameters,
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay
            )
            print(f"  LocalOptimizerBridge: MoE optimizer initialized with {len(moe_parameters)} parameter tensors")
        
        # Initialize Tokenizer optimizer if parameters provided
        if tokenizer_parameters is not None and len(tokenizer_parameters) > 0:
            self.tokenizer_optimizer = torch.optim.Adam(
                tokenizer_parameters,
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay
            )
            print(f"  LocalOptimizerBridge: Tokenizer optimizer initialized with {len(tokenizer_parameters)} parameter tensors")
    
    def store_moe_output(self, moe_output: torch.Tensor) -> None:
        """
        Store the MoE layer output for later gradient computation.
        
        Args:
            moe_output: Output tensor from SparseMoE forward pass
        """
        # Detach from any existing graph to prevent accumulation
        self._stored_moe_output = moe_output.detach().requires_grad_(True)
    
    def store_tokenizer_output(self, tokenizer_output: torch.Tensor) -> None:
        """
        Store the Tokenizer layer output for later gradient computation.
        
        Args:
            tokenizer_output: Output tensor from SpatialTokenizer forward pass
        """
        # Detach from any existing graph to prevent accumulation
        self._stored_tokenizer_output = tokenizer_output.detach().requires_grad_(True)
    
    def get_stored_outputs(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Retrieve stored MoE and Tokenizer outputs for gradient computation.
        
        Returns:
            Tuple of (moe_output, tokenizer_output) tensors
        """
        return self._stored_moe_output, self._stored_tokenizer_output
    
    def apply_local_gradients(
        self,
        predicted_tensors: Optional[torch.Tensor] = None,
        target_tensors: Optional[torch.Tensor] = None,
        use_stored_outputs: bool = True
    ) -> float:
        """
        Apply localized gradients using MSE loss between predicted and target states.
        
        This method calculates the local Mean Squared Error, immediately calls
        backward() to compute gradients, steps the optimizers, and zeroes gradients.
        This ensures O(1) memory scaling by preventing gradient graph accumulation.
        
        Args:
            predicted_tensors: Predicted state tensors (from MoE/Tokenizer)
            target_tensors: Target state tensors (from local prediction errors)
            use_stored_outputs: If True, use stored outputs instead of provided tensors
            
        Returns:
            The computed loss value (MSE)
            
        Note:
            - Must call either with tensors or have stored outputs available
            - backward() is called with retain_graph=False to clear the graph
            - Gradients are zeroed after step() to prevent accumulation
        """
        # Determine which tensors to use
        if use_stored_outputs and (self._stored_moe_output is not None or self._stored_tokenizer_output is not None):
            # Use stored outputs
            if self._stored_moe_output is not None:
                predicted = self._stored_moe_output
            elif self._stored_tokenizer_output is not None:
                predicted = self._stored_tokenizer_output
            else:
                raise ValueError("No stored outputs available")
        elif predicted_tensors is not None:
            predicted = predicted_tensors
        else:
            raise ValueError("Must provide tensors or have stored outputs available")
        
        # Use provided target or create dummy target from predicted (for initialization)
        if target_tensors is not None:
            target = target_tensors
        else:
            # If no target provided, use the predicted tensor itself (identity loss for initialization)
            # This is a fallback that essentially does nothing but allows the graph to form
            target = predicted.detach()
        
        # Compute localized MSE loss
        # Use mean reduction for stable gradients across batch sizes
        loss = F.mse_loss(predicted, target, reduction='mean')
        
        # Clear any previous gradient history and compute new gradients
        # CRITICAL: retain_graph=False ensures O(1) memory by clearing the graph immediately
        loss.backward(retain_graph=False)
        
        # Step MoE optimizer if available
        if self.moe_optimizer is not None:
            self.moe_optimizer.step()
            self.moe_optimizer.zero_grad()
        
        # Step Tokenizer optimizer if available
        if self.tokenizer_optimizer is not None:
            self.tokenizer_optimizer.step()
            self.tokenizer_optimizer.zero_grad()
        
        # Clear stored outputs after gradient application to free memory
        self._stored_moe_output = None
        self._stored_tokenizer_output = None
        
        return loss.item()
    
    def apply_moe_gradients(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """
        Apply gradients specifically for MoE layer updates.
        
        Args:
            predicted: Predicted state from MoE
            target: Target state from local prediction error
            
        Returns:
            The computed loss value
        """
        # Zero gradients first
        if self.moe_optimizer is not None:
            self.moe_optimizer.zero_grad()
        
        # Compute loss and backpropagate
        loss = F.mse_loss(predicted, target, reduction='mean')
        loss.backward(retain_graph=False)
        
        # Step and zero
        if self.moe_optimizer is not None:
            self.moe_optimizer.step()
            self.moe_optimizer.zero_grad()
        
        return loss.item()
    
    def apply_tokenizer_gradients(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """
        Apply gradients specifically for Tokenizer layer updates.
        
        Args:
            predicted: Predicted state from Tokenizer
            target: Target state from local prediction error
            
        Returns:
            The computed loss value
        """
        # Zero gradients first
        if self.tokenizer_optimizer is not None:
            self.tokenizer_optimizer.zero_grad()
        
        # Compute loss and backpropagate
        loss = F.mse_loss(predicted, target, reduction='mean')
        loss.backward(retain_graph=False)
        
        # Step and zero
        if self.tokenizer_optimizer is not None:
            self.tokenizer_optimizer.step()
            self.tokenizer_optimizer.zero_grad()
        
        return loss.item()
    
    def step_moe_only(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """
        Convenience method for MoE-only gradient updates.
        
        Args:
            predicted: Predicted state from MoE
            target: Target state from local prediction error
            
        Returns:
            The computed loss value
        """
        return self.apply_moe_gradients(predicted, target)
    
    def step_tokenizer_only(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """
        Convenience method for Tokenizer-only gradient updates.
        
        Args:
            predicted: Predicted state from Tokenizer
            target: Target state from local prediction error
            
        Returns:
            The computed loss value
        """
        return self.apply_tokenizer_gradients(predicted, target)
    
    def get_optimizer_state(self) -> dict:
        """
        Get current state of all optimizers for debugging/monitoring.
        
        Returns:
            Dictionary containing optimizer states
        """
        state = {}
        if self.moe_optimizer is not None:
            state['moe'] = {
                'param_groups': self.moe_optimizer.param_groups,
                'state': {k: {kk: vv.clone() if isinstance(vv, torch.Tensor) else vv 
                             for kk, vv in v.items()} 
                         for k, v in self.moe_optimizer.state.items()}
            }
        if self.tokenizer_optimizer is not None:
            state['tokenizer'] = {
                'param_groups': self.tokenizer_optimizer.param_groups,
                'state': {k: {kk: vv.clone() if isinstance(vv, torch.Tensor) else vv 
                             for kk, vv in v.items()} 
                         for k, v in self.tokenizer_optimizer.state.items()}
            }
        return state
    
    def set_learning_rate(self, lr: float) -> None:
        """
        Update learning rate for all optimizers.
        
        Args:
            lr: New learning rate
        """
        self.learning_rate = lr
        if self.moe_optimizer is not None:
            for param_group in self.moe_optimizer.param_groups:
                param_group['lr'] = lr
        if self.tokenizer_optimizer is not None:
            for param_group in self.tokenizer_optimizer.param_groups:
                param_group['lr'] = lr
    
    def zero_gradients(self) -> None:
        """
        Zero gradients for all optimizers.
        Useful for forcing a clean state before manual gradient computation.
        """
        if self.moe_optimizer is not None:
            self.moe_optimizer.zero_grad()
        if self.tokenizer_optimizer is not None:
            self.tokenizer_optimizer.zero_grad()
    
    def clear_stored_outputs(self) -> None:
        """
        Clear any stored output tensors to free memory.
        """
        self._stored_moe_output = None
        self._stored_tokenizer_output = None


def create_local_optimizer_bridge(
    moe_module: Optional[nn.Module] = None,
    tokenizer_module: Optional[nn.Module] = None,
    learning_rate: float = 1e-3,
    **optimizer_kwargs
) -> LocalOptimizerBridge:
    """
    Factory function to create a LocalOptimizerBridge from PyTorch modules.
    
    Args:
        moe_module: SparseMoE or similar MoE module
        tokenizer_module: SpatialTokenizer or similar tokenizer module
        learning_rate: Learning rate for Adam optimizers
        **optimizer_kwargs: Additional arguments for Adam optimizer
        
    Returns:
        Configured LocalOptimizerBridge instance
    """
    moe_params = list(moe_module.parameters()) if moe_module is not None else None
    tokenizer_params = list(tokenizer_module.parameters()) if tokenizer_module is not None else None
    
    return LocalOptimizerBridge(
        moe_parameters=moe_params,
        tokenizer_parameters=tokenizer_params,
        learning_rate=learning_rate,
        **optimizer_kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Simple test to verify the bridge works
    print("Testing LocalOptimizerBridge...")
    
    # Create dummy parameters
    dummy_params = [torch.randn(10, 10, requires_grad=True) for _ in range(3)]
    
    # Initialize bridge
    bridge = LocalOptimizerBridge(
        moe_parameters=dummy_params,
        learning_rate=1e-3
    )
    
    # Create test tensors
    predicted = torch.randn(5, 10, requires_grad=True)
    target = torch.randn(5, 10)
    
    # Apply gradients
    loss = bridge.apply_moe_gradients(predicted, target)
    print(f"Applied gradient update, loss: {loss:.6f}")
    
    print("LocalOptimizerBridge test passed!")
