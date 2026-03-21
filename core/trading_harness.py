"""
Trading Harness for 3D-RNG Financial Applications
Lead Quantitative ML Engineer Implementation

This script implements the trading-specific components for the 3D-RNG architecture,
including output interpretation, Sharpe-based reward calculation, and position sizing.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import pandas as pd


class TradingOutputInterpreter:
    """
    Interprets the 3D-RNG output face activations as trading actions.
    
    The Output_Face generates a 3-element action vector per position:
    [Position (1 for Long, -1 for Short), Conviction (0 to 1), Stop_Loss_Threshold]
    """
    
    def __init__(self, 
                 output_face_size: Tuple[int, int] = (4, 4),
                 position_threshold: float = 0.1,
                 conviction_threshold: float = 0.3):
        """
        Initialize the Trading Output Interpreter.
        
        Args:
            output_face_size: Dimensions of the output face (height, width)
            position_threshold: Minimum activation to consider a position signal
            conviction_threshold: Minimum conviction to execute a trade
        """
        self.output_face_size = output_face_size
        self.position_threshold = position_threshold
        self.conviction_threshold = conviction_threshold
        
        # Calculate total number of output positions
        self.num_positions = output_face_size[0] * output_face_size[1]
    
    def interpret_output(self, 
                        output_tensor: np.ndarray) -> List[Dict]:
        """
        Interpret the output tensor as trading actions.
        
        Args:
            output_tensor: Shape [num_output_nodes, 3] where last dim is [position, conviction, stop_loss]
            
        Returns:
            List of trading action dictionaries
        """
        if output_tensor.ndim != 2 or output_tensor.shape[1] != 3:
            raise ValueError("Output tensor must have shape [num_nodes, 3]")
        
        actions = []
        
        for i in range(output_tensor.shape[0]):
            position_raw = output_tensor[i, 0]   # Raw position signal (-1 to 1)
            conviction_raw = output_tensor[i, 1] # Raw conviction signal (0 to 1)
            stop_loss_raw = output_tensor[i, 2]  # Raw stop loss signal (0 to 1)
            
            # Convert raw signals to actionable values
            position = np.tanh(position_raw)  # Ensures [-1, 1] range
            conviction = np.clip(conviction_raw, 0, 1)  # Ensures [0, 1] range
            stop_loss = np.clip(stop_loss_raw, 0, 0.5)  # Max 50% stop loss
            
            # Only create action if conviction is above threshold
            if conviction >= self.conviction_threshold:
                action = {
                    'position': np.sign(position) if abs(position) > self.position_threshold else 0,
                    'position_strength': abs(position),  # How strong the signal is
                    'conviction': conviction,
                    'stop_loss': stop_loss,
                    'output_index': i
                }
                actions.append(action)
        
        return actions
    
    def aggregate_actions(self, 
                         actions: List[Dict]) -> Dict:
        """
        Aggregate multiple position signals into a single portfolio action.
        
        Args:
            actions: List of individual action dictionaries
            
        Returns:
            Aggregated action dictionary
        """
        if not actions:
            return {
                'position': 0,
                'conviction': 0,
                'stop_loss': 0,
                'reason': 'No signals above threshold'
            }
        
        # Weight actions by conviction and position strength
        total_weight = 0
        weighted_position = 0
        avg_conviction = 0
        max_stop_loss = 0
        
        for action in actions:
            weight = action['conviction'] * action['position_strength']
            total_weight += weight
            weighted_position += action['position'] * weight
            avg_conviction += action['conviction'] * weight
            max_stop_loss = max(max_stop_loss, action['stop_loss'])
        
        if total_weight > 0:
            final_position = weighted_position / total_weight
            final_conviction = avg_conviction / total_weight
        else:
            final_position = 0
            final_conviction = 0
        
        # Apply conviction threshold to final decision
        if final_conviction < self.conviction_threshold:
            final_position = 0
        
        return {
            'position': np.clip(final_position, -1, 1),
            'conviction': final_conviction,
            'stop_loss': max_stop_loss,
            'num_signals': len(actions),
            'reason': f'Aggregated {len(actions)} signals'
        }


def calculate_sharpe_reward(prediction: np.ndarray, 
                           actual_future_return: np.ndarray,
                           volatility_lookback: int = 20,
                           risk_free_rate: float = 0.0) -> float:
    """
    Calculate a Sharpe ratio-based reward for trading predictions.
    
    This reward function encourages profitable, low-volatility decisions
    and heavily penalizes predictions that would lead to stop loss hits.
    
    Args:
        prediction: Model's predicted action/position (scalar or array)
        actual_future_return: Actual future returns over the holding period
        volatility_lookback: Period used to estimate volatility
        risk_free_rate: Risk-free rate for Sharpe calculation
        
    Returns:
        Scalar reward value (higher is better)
    """
    # Handle scalar inputs
    if np.isscalar(prediction):
        prediction = np.array([prediction])
    if np.isscalar(actual_future_return):
        actual_future_return = np.array([actual_future_return])
    
    # Ensure same length
    min_len = min(len(prediction), len(actual_future_return))
    if min_len == 0:
        return 0.0
    
    prediction = prediction[:min_len]
    actual_future_return = actual_future_return[:min_len]
    
    # Calculate strategy returns: prediction * actual_return
    # (Assuming prediction is position: -1 to 1)
    strategy_returns = prediction * actual_future_return
    
    # Calculate metrics
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns)
    
    # Avoid division by zero
    if std_return == 0:
        sharpe_ratio = 0 if mean_return == 0 else np.inf * np.sign(mean_return)
    else:
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
    
    # Apply penalties for large losses (stop loss hits)
    # Heavily penalize negative returns beyond a threshold
    loss_penalty = 0
    for ret in strategy_returns:
        if ret < -0.02:  # More than 2% loss
            loss_penalty += abs(ret) * 2  # Double penalty for losses
    
    # Combine Sharpe ratio with loss penalty
    # Normalize Sharpe to reasonable range and subtract penalty
    sharpe_component = np.tanh(sharpe_ratio)  # Maps to [-1, 1]
    penalty_component = min(loss_penalty, 2.0)  # Cap penalty
    
    reward = sharpe_component - penalty_component
    
    # Ensure reward is in reasonable range
    reward = np.clip(reward, -2.0, 2.0)
    
    return float(reward)


def calculate_sortino_reward(prediction: np.ndarray,
                            actual_future_return: np.ndarray,
                            target_return: float = 0.0,
                            risk_free_rate: float = 0.0) -> float:
    """
    Calculate a Sortino ratio-based reward (focuses on downside risk).
    
    Args:
        prediction: Model's predicted action/position
        actual_future_return: Actual future returns
        target_return: Minimum acceptable return
        risk_free_rate: Risk-free rate
        
    Returns:
        Scalar reward value
    """
    # Handle scalar inputs
    if np.isscalar(prediction):
        prediction = np.array([prediction])
    if np.isscalar(actual_future_return):
        actual_future_return = np.array([actual_future_return])
    
    # Ensure same length
    min_len = min(len(prediction), len(actual_future_return))
    if min_len == 0:
        return 0.0
    
    prediction = prediction[:min_len]
    actual_future_return = actual_future_return[:min_len]
    
    # Calculate strategy returns
    strategy_returns = prediction * actual_future_return
    
    # Calculate downside deviation
    excess_returns = strategy_returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        downside_deviation = 0
    else:
        downside_deviation = np.sqrt(np.mean(downside_returns**2))
    
    # Calculate Sortino ratio
    mean_excess = np.mean(excess_returns)
    if downside_deviation == 0:
        sortino_ratio = 0 if mean_excess == 0 else np.inf * np.sign(mean_excess)
    else:
        sortino_ratio = mean_excess / downside_deviation
    
    # Normalize and return
    reward = np.tanh(sortino_ratio)
    return float(reward)


def calculate_max_drawdown_reward(prediction: np.ndarray,
                                 actual_future_return: np.ndarray,
                                 max_allowed_drawdown: float = 0.1) -> float:
    """
    Calculate reward based on maximum drawdown constraint.
    
    Args:
        prediction: Model's predicted action/position
        actual_future_return: Actual future returns
        max_allowed_drawdown: Maximum allowed drawdown (e.g., 0.1 for 10%)
        
    Returns:
        Scalar reward value
    """
    # Handle scalar inputs
    if np.isscalar(prediction):
        prediction = np.array([prediction])
    if np.isscalar(actual_future_return):
        actual_future_return = np.array([actual_future_return])
    
    # Ensure same length
    min_len = min(len(prediction), len(actual_future_return))
    if min_len == 0:
        return 0.0
    
    prediction = prediction[:min_len]
    actual_future_return = actual_future_return[:min_len]
    
    # Calculate strategy returns
    strategy_returns = prediction * actual_future_return
    
    # Calculate cumulative returns
    cum_returns = np.cumprod(1 + strategy_returns) - 1
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cum_returns)
    
    # Calculate drawdown
    drawdown = (running_max - cum_returns) / (1 + running_max)
    drawdown = np.where(np.isnan(drawdown), 0, drawdown)  # Handle division by zero
    
    # Maximum drawdown
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Reward based on drawdown constraint
    if max_dd <= max_allowed_drawdown:
        # Within constraint - reward based on returns
        total_return = np.prod(1 + strategy_returns) - 1
        reward = np.tanh(total_return * 10)  # Scale for tanh
    else:
        # Exceeded constraint - heavy penalty
        reward = -np.tanh((max_dd - max_allowed_drawdown) * 20)
    
    return float(reward)


class PositionSizer:
    """
    Implements various position sizing strategies for risk management.
    """
    
    @staticmethod
    def fixed_fractional(equity: float, 
                        risk_per_trade: float,
                        stop_loss_pct: float) -> float:
        """
        Calculate position size based on fixed fractional risk.
        
        Args:
            equity: Current account equity
            risk_per_trade: Fraction of equity to risk per trade (e.g., 0.02 for 2%)
            stop_loss_pct: Stop loss as fraction of position value (e.g., 0.05 for 5%)
            
        Returns:
            Position size as fraction of equity
        """
        if stop_loss_pct <= 0:
            return 0
        return risk_per_trade / stop_loss_pct
    
    @staticmethod
    def kelly_fraction(win_rate: float,
                      avg_win: float,
                      avg_loss: float) -> float:
        """
        Calculate Kelly fraction for optimal position sizing.
        
        Args:
            win_rate: Probability of winning trade (0 to 1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
            
        Returns:
            Kelly fraction (can be negative -> don't trade)
        """
        if avg_loss <= 0:
            return 0
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        return max(0, kelly)  # Never bet negative Kelly
    
    @staticmethod
    def volatility_scaled(volatility: float,
                         target_volatility: float = 0.01) -> float:
        """
        Scale position inversely to volatility.
        
        Args:
            volatility: Current market volatility (e.g., std of returns)
            target_volatility: Desired portfolio volatility
            
        Returns:
            Volatility scaling factor
        """
        if volatility <= 0:
            return 1.0
        return target_volatility / volatility


def create_trading_example():
    """Create an example showing how to use the trading harness components."""
    print("Creating Trading Harness example...")
    
    # Create output interpreter
    interpreter = TradingOutputInterpreter(
        output_face_size=(4, 4),
        position_threshold=0.1,
        conviction_threshold=0.3
    )
    
    # Simulate output from 3D-RNG (16 positions, 3 values each)
    output_tensor = np.random.randn(16, 3) * 0.5
    
    # Interpret output
    actions = interpreter.interpret_output(output_tensor)
    print(f"Generated {len(actions)} individual actions")
    
    # Aggregate actions
    aggregated = interpreter.aggregate_actions(actions)
    print(f"Aggregated action: {aggregated}")
    
    # Example reward calculation
    prediction = np.array([0.5])  # Long position
    actual_return = np.array([0.02])  # 2% gain
    
    sharpe_reward = calculate_sharpe_reward(prediction, actual_return)
    sortino_reward = calculate_sortino_reward(prediction, actual_return)
    dd_reward = calculate_max_drawdown_reward(prediction, actual_return)
    
    print(f"Sharpe reward: {sharpe_reward:.3f}")
    print(f"Sortino reward: {sortino_reward:.3f}")
    print(f"Drawdown reward: {dd_reward:.3f}")
    
    # Example position sizing
    equity = 100000  # $100k account
    risk_per_trade = 0.02  # 2% risk per trade
    stop_loss_pct = 0.05   # 5% stop loss
    
    position_size = PositionSizer.fixed_fractional(equity, risk_per_trade, stop_loss_pct)
    print(f"Position size: {position_size:.2%} of equity (${equity * position_size:.2f})")
    
    return interpreter, actions, aggregated


if __name__ == "__main__":
    # Run the example
    create_trading_example()