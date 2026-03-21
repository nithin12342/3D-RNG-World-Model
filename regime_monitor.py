"""
Regime Monitor for 3D-RNG Financial Applications
Lead Quantitative ML Engineer Implementation

This script implements regime-adaptive evaporation rate control for the 3D-RNG architecture,
dynamically adjusting the evaporation rate based on recent reward performance to detect
and adapt to changing market regimes.
"""

import numpy as np
from typing import Tuple, List, Optional, Deque, Dict
from collections import deque
import time


class RegimeMonitor:
    """
    Monitors trading performance to detect regime changes and adapt evaporation rate.
    
    Features:
    - Tracks rolling average of Traceback Reward
    - Detects regime shifts when performance deteriorates consistently
    - Dynamically increases evaporation rate to "flush" old pheromone pathways
    - Gradually returns evaporation rate to baseline when performance improves
    """
    
    def __init__(self,
                 baseline_evaporation: float = 0.05,
                 regime_threshold: float = -0.01,
                 regime_consecutive_epochs: int = 5,
                 evaporation_multiplier: float = 5.0,
                 recovery_rate: float = 0.1,
                 max_evaporation: float = 0.5):
        """
        Initialize the Regime Monitor.
        
        Args:
            baseline_evaporation: Normal evaporation rate (rho)
            regime_threshold: Average reward below this indicates potential regime shift
            regime_consecutive_epochs: Number of consecutive bad epochs to trigger regime shift
            evaporation_multiplier: How much to increase evaporation during regime shift
            recovery_rate: Rate at which evaporation returns to baseline after regime shift
            max_evaporation: Maximum allowed evaporation rate
        """
        self.baseline_evaporation = baseline_evaporation
        self.regime_threshold = regime_threshold
        self.regime_consecutive_epochs = regime_consecutive_epochs
        self.evaporation_multiplier = evaporation_multiplier
        self.recovery_rate = recovery_rate
        self.max_evaporation = max_evaporation
        
        # Current state
        self.current_evaporation = baseline_evaporation
        self.regime_shift_detected = False
        self.epochs_in_regime_shift = 0
        
        # History tracking
        self.reward_history: Deque[float] = deque(maxlen=100)  # Keep last 100 rewards
        self.regime_history: List[bool] = []  # Track when regime shifts were detected
        self.epoch_counter = 0
        
        # Performance tracking
        self.consecutive_bad_epochs = 0
        self.best_reward_seen = float('-inf')
        self.worst_reward_seen = float('inf')
    
    def update(self, reward: float) -> float:
        """
        Update the regime monitor with a new reward and return the current evaporation rate.
        
        Args:
            reward: The Traceback Reward from the most recent epoch
            
        Returns:
            Current evaporation rate to use for the next epoch
        """
        self.epoch_counter += 1
        self.reward_history.append(reward)
        
        # Update best/worst rewards seen
        self.best_reward_seen = max(self.best_reward_seen, reward)
        self.worst_reward_seen = min(self.worst_reward_seen, reward)
        
        # Check if this is a bad epoch (below regime threshold)
        is_bad_epoch = reward < self.regime_threshold
        
        if is_bad_epoch:
            self.consecutive_bad_epochs += 1
        else:
            # Reset consecutive bad epochs counter
            self.consecutive_bad_epochs = 0
            # If we were in a regime shift, start recovery
            if self.regime_shift_detected:
                self._start_recovery()
        
        # Check for regime shift trigger
        if not self.regime_shift_detected and self.consecutive_bad_epochs >= self.regime_consecutive_epochs:
            self._trigger_regime_shift()
        
        # If in regime shift, apply recovery logic
        if self.regime_shift_detected:
            self._apply_recovery()
        
        return self.current_evaporation
    
    def _trigger_regime_shift(self):
        """Trigger a regime shift response."""
        self.regime_shift_detected = True
        self.epochs_in_regime_shift = 0
        self.regime_history.append(True)
        
        # Increase evaporation rate to flush old pathways
        new_evaporation = min(
            self.baseline_evaporation * self.evaporation_multiplier,
            self.max_evaporation
        )
        self.current_evaporation = new_evaporation
        
        print(f"[RegimeMonitor] Regime shift detected at epoch {self.epoch_counter}")
        print(f"  Consecutive bad epochs: {self.consecutive_bad_epochs}")
        print(f"  Recent avg reward: {self.get_recent_average_reward():.4f}")
        print(f"  Increasing evaporation from {self.baseline_evaporation:.4f} to {self.current_evaporation:.4f}")
    
    def _start_recovery(self):
        """Start the recovery process after regime shift."""
        self.regime_shift_detected = False
        self.regime_history.append(False)
        print(f"[RegimeMonitor] Regime shift recovery started at epoch {self.epoch_counter}")
        print(f"  Performance improved: {self.reward_history[-1]:.4f} >= {self.regime_threshold:.4f}")
    
    def _apply_recovery(self):
        """Apply gradual recovery of evaporation rate toward baseline."""
        self.epochs_in_regime_shift += 1
        
        # Gradually move evaporation rate back toward baseline
        if self.current_evaporation > self.baseline_evaporation:
            # Decrease evaporation toward baseline
            delta = (self.current_evaporation - self.baseline_evaporation) * self.recovery_rate
            self.current_evaporation = max(self.baseline_evaporation, self.current_evaporation - delta)
        elif self.current_evaporation < self.baseline_evaporation:
            # Increase evaporation toward baseline (shouldn't happen in normal case)
            delta = (self.baseline_evaporation - self.current_evaporation) * self.recovery_rate
            self.current_evaporation = min(self.baseline_evaporation, self.current_evaporation + delta)
        
        # Check if we've recovered sufficiently
        if abs(self.current_evaporation - self.baseline_evaporation) < 0.001:
            self.current_evaporation = self.baseline_evaporation
            if self.epochs_in_regime_shift > 10:  # Only print if we were in shift for a while
                print(f"[RegimeMonitor] Evaporation rate recovered to baseline at epoch {self.epoch_counter}")
    
    def get_current_evaporation(self) -> float:
        """Get the current evaporation rate."""
        return self.current_evaporation
    
    def get_recent_average_reward(self, window: int = 10) -> float:
        """
        Get the average reward over the last N epochs.
        
        Args:
            window: Number of recent epochs to average
            
        Returns:
            Average reward over the window
        """
        if len(self.reward_history) == 0:
            return 0.0
        
        recent_rewards = list(self.reward_history)[-min(window, len(self.reward_history)):]
        return np.mean(recent_rewards)
    
    def get_regime_statistics(self) -> Dict:
        """
        Get statistics about regime detection and performance.
        
        Returns:
            Dictionary with regime monitoring statistics
        """
        return {
            'epoch_counter': self.epoch_counter,
            'current_evaporation': self.current_evaporation,
            'baseline_evaporation': self.baseline_evaporation,
            'regime_shift_detected': self.regime_shift_detected,
            'consecutive_bad_epochs': self.consecutive_bad_epochs,
            'regime_shift_count': sum(self.regime_history),
            'recent_avg_reward': self.get_recent_average_reward(10),
            'best_reward_seen': self.best_reward_seen,
            'worst_reward_seen': self.worst_reward_seen,
            'reward_history_length': len(self.reward_history)
        }
    
    def is_in_regime_shift(self) -> bool:
        """Check if currently in a detected regime shift."""
        return self.regime_shift_detected
    
    def reset(self):
        """Reset the regime monitor to initial state."""
        self.__init__(
            baseline_evaporation=self.baseline_evaporation,
            regime_threshold=self.regime_threshold,
            regime_consecutive_epochs=self.regime_consecutive_epochs,
            evaporation_multiplier=self.evaporation_multiplier,
            recovery_rate=self.recovery_rate,
            max_evaporation=self.max_evaporation
        )


class AdaptiveEvaporationController:
    """
    Controls the evaporation rate in the 3D-RNG based on regime detection.
    This would be integrated into the ParallelNeuralGraph3D class.
    """
    
    def __init__(self, 
                 baseline_evaporation: float = 0.05,
                 regime_threshold: float = -0.01,
                 regime_consecutive_epochs: int = 5,
                 evaporation_multiplier: float = 5.0,
                 recovery_rate: float = 0.1,
                 max_evaporation: float = 0.5):
        """
        Initialize the Adaptive Evaporation Controller.
        
        Args:
            Same as RegimeMonitor parameters
        """
        self.regime_monitor = RegimeMonitor(
            baseline_evaporation=baseline_evaporation,
            regime_threshold=regime_threshold,
            regime_consecutive_epochs=regime_consecutive_epochs,
            evaporation_multiplier=evaporation_multiplier,
            recovery_rate=recovery_rate,
            max_evaporation=max_evaporation
        )
        
        # For integration with NeuralGraph3D
        self.last_evaporation_update = 0
    
    def update_evaporation(self, reward: float) -> float:
        """
        Update evaporation rate based on reward and return the new rate.
        
        Args:
            reward: The Traceback Reward from the most recent epoch
            
        Returns:
            Current evaporation rate
        """
        return self.regime_monitor.update(reward)
    
    def get_evaporation_rate(self) -> float:
        """Get the current evaporation rate."""
        return self.regime_monitor.get_current_evaporation()
    
    def get_regime_info(self) -> Dict:
        """Get regime monitoring information."""
        return self.regime_monitor.get_regime_statistics()


def create_regime_monitor_example():
    """Create an example showing how to use the regime monitor."""
    print("Creating Regime Monitor example...")
    
    # Create regime monitor
    monitor = RegimeMonitor(
        baseline_evaporation=0.05,
        regime_threshold=-0.01,
        regime_consecutive_epochs=3,  # Lower for demo
        evaporation_multiplier=3.0,   # Lower for demo
        recovery_rate=0.2,
        max_evaporation=0.3
    )
    
    # Simulate a sequence of rewards showing a regime shift
    print("\nSimulating reward sequence with regime shift...")
    
    # Good performance period
    good_rewards = [0.02, 0.01, 0.03, 0.02, 0.01]
    print("Good performance period:")
    for i, reward in enumerate(good_rewards):
        evaporation = monitor.update(reward)
        print(f"  Epoch {i+1}: Reward={reward:.3f}, Evaporation={evaporation:.3f}")
    
    # Poor performance period (regime shift)
    poor_rewards = [-0.02, -0.03, -0.01, -0.04, -0.02]
    print("\nPoor performance period (regime shift):")
    for i, reward in enumerate(poor_rewards):
        evaporation = monitor.update(reward)
        status = "REGIME SHIFT!" if monitor.is_in_regime_shift() else ""
        print(f"  Epoch {len(good_rewards)+i+1}: Reward={reward:.3f}, Evaporation={evaporation:.3f} {status}")
    
    # Recovery period
    recovery_rewards = [0.01, 0.02, 0.01, 0.03, 0.02]
    print("\nRecovery period:")
    for i, reward in enumerate(recovery_rewards):
        evaporation = monitor.update(reward)
        status = "RECOVERING" if monitor.is_in_regime_shift() else "RECOVERED"
        print(f"  Epoch {len(good_rewards)+len(poor_rewards)+i+1}: Reward={reward:.3f}, Evaporation={evaporation:.3f} [{status}]")
    
    # Print final statistics
    stats = monitor.get_regime_statistics()
    print(f"\nFinal Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return monitor


if __name__ == "__main__":
    # Run the example
    create_regime_monitor_example()