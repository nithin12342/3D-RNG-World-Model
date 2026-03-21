"""
Regime Shift Backtest Runner for 3D-RNG Financial Applications
Lead Quant Validator & Backtesting Engineer Implementation

This script executes a comparative backtest between the 3D-RNG with Adaptive Evaporation
and a standard baseline neural network (MLP/LSTM) during the 2020 COVID-19 market crash.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
import yaml
import json
import os
from datetime import datetime, timedelta

# Import our custom components
from quant_dataloader import FinancialDataLoader
from trading_harness import TradingOutputInterpreter, calculate_sharpe_reward
from regime_monitor import RegimeMonitor
from parallel_graph_impl import ParallelNeuralGraph3D


class BaselineFinancialModel(nn.Module):
    """
    Standard baseline model (MLP or LSTM) for comparison with 3D-RNG.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 model_type: str = 'mlp'):
        """
        Initialize the baseline model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of layers (for LSTM) or hidden layers (for MLP)
            model_type: 'mlp' or 'lstm'
        """
        super(BaselineFinancialModel, self).__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if model_type == 'lstm':
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 3)  # [position, conviction, stop_loss]
        elif model_type == 'mlp':
            layers = []
            prev_size = input_size
            for i in range(num_layers):
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                prev_size = hidden_size
            layers.append(nn.Linear(hidden_size, 3))
            self.network = nn.Sequential(*layers)
        else:
            raise ValueError("model_type must be 'mlp' or 'lstm'")
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size]
               or [batch_size, input_size] for MLP
            
        Returns:
            Output tensor of shape [batch_size, 3]
        """
        if self.model_type == 'lstm':
            lstm_out, _ = self.lstm(x)
            # Use last time step output
            last_out = lstm_out[:, -1, :]
            output = self.fc(last_out)
        else:  # MLP
            output = self.network(x)
        
        return output


class FinancialBacktester:
    """
    Backtesting framework for comparing 3D-RNG with baseline models during regime shifts.
    """
    
    def __init__(self,
                 symbols: List[str] = ['SPY', 'QQQ', 'TLT', 'GLD', 'VIX'],
                 start_date: str = '2018-01-01',
                 end_date: str = '2020-12-31',
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001):
        """
        Initialize the backtester.
        
        Args:
            symbols: List of ticker symbols to include
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Starting portfolio value
            transaction_cost: Transaction cost as fraction of trade value
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Initialize components
        self.data_loader = None
        self.threed_rng = None
        self.baseline_model = None
        self.trading_interpreter = TradingOutputInterpreter()
        self.regime_monitor = RegimeMonitor(
            baseline_evaporation=0.05,
            regime_threshold=-0.01,
            regime_consecutive_epochs=3,
            evaporation_multiplier=5.0,
            recovery_rate=0.1,
            max_evaporation=0.3
        )
        
        # Results tracking
        self.threed_rng_equity_curve = []
        self.baseline_equity_curve = []
        self.threed_rng_trades = []
        self.baseline_trades = []
        self.regime_events = []
        self.dates = []
        
    def load_and_prepare_data(self, data_dir: str = './'):
        """
        Load historical data and prepare it for backtesting.
        
        Args:
            data_dir: Directory containing the data files
        """
        print(f"Loading historical data for {self.symbols} from {self.start_date} to {self.end_date}")
        
        # Try to load existing data files, otherwise create synthetic data for demo
        data_files_exist = all(
            os.path.exists(os.path.join(data_dir, f"{symbol}.csv")) 
            for symbol in self.symbols
        )
        
        if data_files_exist:
            # Load actual data
            data_dict = {}
            for symbol in self.symbols:
                df = pd.read_csv(os.path.join(data_dir, f"{symbol}.csv"), index_col=0, parse_dates=True)
                # Filter date range
                df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
                data_dict[symbol] = df['Close']  # Use closing prices
            
            # Combine into DataFrame
            self.raw_data = pd.DataFrame(data_dict)
            print(f"Loaded actual data: {self.raw_data.shape}")
        else:
            # Create synthetic data that mimics market regimes
            print("Creating synthetic market data with regime shifts...")
            self.raw_data = self._create_synthetic_market_data()
        
        # Initialize data loader
        self.data_loader = FinancialDataLoader(window_size=20)  # 20-day lookback
        
        # Manually set the data (bypassing file loading for speed)
        self.data_loader.raw_data = self.raw_data
        self.data_loader.compute_log_returns()
        self.data_loader.scale_features()
        
        print(f"Prepared data: {self.data_loader.scaled_data.shape}")
        print(f"Date range: {self.data_loader.scaled_data.index[0]} to {self.data_loader.scaled_data.index[-1]}")
    
    def _create_synthetic_market_data(self) -> pd.DataFrame:
        """
        Create synthetic market data that includes:
        - 2018: Low volatility period
        - 2019: Bull market 
        - Q1 2020: COVID-19 crash (regime shift)
        - Rest of 2020: Recovery
        """
        # Create date range
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        # Remove weekends for simplicity
        dates = dates[dates.weekday < 5]
        
        n_days = len(dates)
        data = {}
        
        # Define regime parameters
        regimes = [
            # 2018: Low volatility
            {'start': 0, 'end': int(0.3 * n_days), 'drift': 0.0002, 'vol': 0.008},
            # 2019: Bull market
            {'start': int(0.3 * n_days), 'end': int(0.7 * n_days), 'drift': 0.0008, 'vol': 0.012},
            # Q1 2020: COVID crash
            {'start': int(0.7 * n_days), 'end': int(0.8 * n_days), 'drift': -0.003, 'vol': 0.04},
            # Recovery
            {'start': int(0.8 * n_days), 'end': n_days, 'drift': 0.0005, 'vol': 0.015}
        ]
        
        for i, symbol in enumerate(self.symbols):
            prices = [100.0]  # Starting price
            
            for regime in regimes:
                start_idx = regime['start']
                end_idx = regime['end']
                drift = regime['drift']
                vol = regime['vol']
                
                # Generate returns for this regime
                n_regime_days = end_idx - start_idx
                if n_regime_days > 0:
                    returns = np.random.normal(drift, vol, n_regime_days)
                    for ret in returns:
                        prices.append(prices[-1] * (1 + ret))
            
            # Trim to correct length
            prices = prices[1:n_days+1]
            data[symbol] = prices
        
        return pd.DataFrame(data, index=dates)
    
    def initialize_models(self, input_face_size: Tuple[int, int] = (4, 4)):
        """
        Initialize both the 3D-RNG and baseline models.
        
        Args:
            input_face_size: Dimensions of the input/output faces
        """
        # Calculate input dimensions
        n_assets = len(self.symbols)
        input_vector_size = input_face_size[0] * input_face_size[1] * n_assets
        
        print(f"Initializing models with {n_assets} assets...")
        print(f"Input face size: {input_face_size}")
        print(f"Input vector size: {input_vector_size}")
        
        # Initialize 3D-RNG
        self.threed_rng = ParallelNeuralGraph3D(
            dim_x=10, dim_y=8, dim_z=8,  # 3D dimensions
            hidden_size=32,
            input_face_size=input_face_size,
            output_face_size=input_face_size
        )
        
        # Initialize baseline model (using MLP for simplicity)
        # Use the same dimensionality as 3D-RNG for fair comparison
        self.baseline_model = BaselineFinancialModel(
            input_size=32,  # Match 3D-RNG hidden_size
            hidden_size=64,
            num_layers=2,
            model_type='mlp'
        )
        
        # Initialize optimizer for baseline model
        self.baseline_optimizer = torch.optim.Adam(self.baseline_model.parameters(), lr=0.001)
        self.baseline_criterion = nn.MSELoss()
        
        print("Models initialized successfully")
    
    def run_backtest(self) -> Dict:
        """
        Execute the comparative backtest.
        
        Returns:
            Dictionary containing backtest results and statistics
        """
        print("\nStarting comparative backtest...")
        print("=" * 50)
        
        # Reset results
        self.threed_rng_equity_curve = [self.initial_capital]
        self.baseline_equity_curve = [self.initial_capital]
        self.threed_rng_trades = []
        self.baseline_trades = []
        self.regime_events = []
        self.dates = list(self.data_loader.scaled_data.index)
        
        # Position tracking
        threed_rng_position = 0.0
        baseline_position = 0.0
        threed_rng_cash = self.initial_capital
        baseline_cash = self.initial_capital
        
        # Get number of time steps
        n_steps = len(self.data_loader.scaled_data) - self.data_loader.window_size
        
        print(f"Running backtest for {n_steps} days...")
        
        for step in range(n_steps):
            current_date = self.dates[step + self.data_loader.window_size]
            
            # Get window of data for this step
            window_data = self.data_loader.scaled_data.iloc[step:step + self.data_loader.window_size]
            
            # Create spatial tensor
            input_face_size = (4, 4)
            spatial_tensor = self.data_loader.create_spatial_tensor(window_data, input_face_size)
            
            # Flatten for model input
            flat_input = spatial_tensor.flatten()
            # Ensure input size matches hidden_size (32) by using PCA-like reduction or simple averaging
            # For now, we'll use a simple approach: take the first hidden_size elements or repeat to fill
            HIDDEN_SIZE = 32  # Must match the hidden_size used in ParallelNeuralGraph3D initialization
            if len(flat_input) >= HIDDEN_SIZE:
                # Take first hidden_size elements
                flat_input = flat_input[:HIDDEN_SIZE]
            else:
                # Repeat the input to fill hidden_size
                repeats = int(np.ceil(HIDDEN_SIZE / len(flat_input)))
                flat_input = np.tile(flat_input, repeats)[:HIDDEN_SIZE]
            input_tensor = torch.FloatTensor(flat_input).unsqueeze(0)  # Add batch dimension
            
            # === 3D-RNG Forward Pass ===
            # Convert to numpy for 3D-RNG
            np_input = flat_input.reshape(1, -1)  # 3D-RNG expects 2D [batch, features]
            
            # 3D-RNG forward probe
            threed_rng_output, paths = self.threed_rng.forward_probe(
                np_input[0],  # First (and only) batch element
                max_steps=30
            )
            
            # Interpret 3D-RNG output as trading action
            # The 3D-RNG output is [num_reached_outputs, hidden_size] where hidden_size=32
            # We need to convert this to [num_output_nodes, 3] for the trading interpreter
            # Strategy: Use first 3 dimensions of hidden state for each output node
            # For output nodes that weren't reached, use zeros
            
            # Initialize output array for all output face nodes
            output_nodes = len(self.threed_rng.output_face_coords)
            threed_rng_output_for_trading = np.zeros((output_nodes, 3))
            
            # Fill in values for nodes that were actually reached
            if threed_rng_output.shape[0] > 0 and threed_rng_output.shape[1] >= 3:
                # Use first 3 features as [position, conviction, stop_loss] proxies
                # Limit to the minimum of reached nodes and output nodes
                nodes_to_fill = min(threed_rng_output.shape[0], output_nodes)
                threed_rng_output_for_trading[:nodes_to_fill, :] = threed_rng_output[:nodes_to_fill, :3]
            # If no outputs reached or insufficient features, leave as zeros (no action)
            
            threed_rng_actions = self.trading_interpreter.interpret_output(threed_rng_output_for_trading)
            threed_rng_aggregated = self.trading_interpreter.aggregate_actions(threed_rng_actions)
            
            # === Baseline Model Forward Pass ===
            self.baseline_model.eval()
            with torch.no_grad():
                baseline_output = self.baseline_model(input_tensor)
                baseline_output_np = baseline_output.numpy().flatten()
                
                # Reshape baseline output similar to 3D-RNG
                if len(baseline_output_np) >= 3:
                    # Take first 3 elements as [position, conviction, stop_loss]
                    baseline_action_vector = baseline_output_np[:3]
                else:
                    # Pad if necessary
                    baseline_action_vector = np.pad(baseline_output_np, (0, 3 - len(baseline_output_np)), 'constant')
                
                # Interpret baseline output
                baseline_actions = self.trading_interpreter.interpret_output(
                    baseline_action_vector.reshape(1, 3)
                )
                baseline_aggregated = self.trading_interpreter.aggregate_actions(baseline_actions)
            
            # === Calculate Returns and Update Positions ===
            # Get actual returns for this day (next day's returns)
            if step + self.data_loader.window_size < len(self.data_loader.scaled_data):
                returns_data = self.data_loader.scaled_data.iloc[step + self.data_loader.window_size]
                # Use equally weighted portfolio return for simplicity
                daily_return = returns_data.mean()
            else:
                daily_return = 0.0
            
            # Update 3D-RNG position and equity
            if abs(threed_rng_aggregated['position']) > 0.01:  # Only trade if significant signal
                # Calculate trade size based on conviction and available capital
                trade_size = threed_rng_aggregated['conviction'] * 0.1  # Max 10% of capital per trade
                trade_value = threed_rng_cash * trade_size
                
                # Apply transaction cost
                cost = trade_value * self.transaction_cost
                threed_rng_cash -= cost
                
                # Update position
                position_change = trade_value * threed_rng_aggregated['position']
                threed_rng_position += position_change
                threed_rng_cash -= position_change  # Cash decreases when buying increases
                
                # Record trade
                self.threed_rng_trades.append({
                    'date': current_date,
                    'action': threed_rng_aggregated['position'],
                    'conviction': threed_rng_aggregated['conviction'],
                    'size': trade_value,
                    'cost': cost
                })
            
            # Update baseline position and equity
            if abs(baseline_aggregated['position']) > 0.01:
                trade_size = baseline_aggregated['conviction'] * 0.1
                trade_value = baseline_cash * trade_size
                
                cost = trade_value * self.transaction_cost
                baseline_cash -= cost
                
                position_change = trade_value * baseline_aggregated['position']
                baseline_position += position_change
                baseline_cash -= position_change
                
                self.baseline_trades.append({
                    'date': current_date,
                    'action': baseline_aggregated['position'],
                    'conviction': baseline_aggregated['conviction'],
                    'size': trade_value,
                    'cost': cost
                })
            
            # Calculate portfolio values
            # Assume we hold positions and they change with market returns
            threed_rng_portfolio_value = threed_rng_cash + (threed_rng_position * (1 + daily_return))
            baseline_portfolio_value = baseline_cash + (baseline_position * (1 + daily_return))
            
            # Actually, this is simplified - in reality we'd need to track individual positions
            # For simplicity, we'll assume the position represents our market exposure
            threed_rng_portfolio_value = threed_rng_cash * (1 + threed_rng_aggregated['position'] * daily_return * threed_rng_aggregated['conviction'])
            baseline_portfolio_value = baseline_cash * (1 + baseline_aggregated['position'] * daily_return * baseline_aggregated['conviction'])
            
            # Ensure we don't go negative
            threed_rng_portfolio_value = max(threed_rng_portfolio_value, 0.01 * self.initial_capital)
            baseline_portfolio_value = max(baseline_portfolio_value, 0.01 * self.initial_capital)
            
            self.threed_rng_equity_curve.append(threed_rng_portfolio_value)
            self.baseline_equity_curve.append(baseline_portfolio_value)
            
            # === Calculate Rewards and Update Models ===
            # 3D-RNG reward (for regime monitoring and learning)
            # Simplified: use the actual return as reward signal
            threed_rng_reward = daily_return * threed_rng_aggregated['position'] * threed_rng_aggregated['conviction']
            
            # Update regime monitor with 3D-RNG performance
            current_evaporation = self.regime_monitor.update(threed_rng_reward)
            
            # Apply evaporation to 3D-RNG (simulate learning)
            if len(paths) > 0 and len(threed_rng_actions) > 0:
                # Simple reward signal for demonstration
                reward_signal = np.array([threed_rng_reward]) if not np.isscalar(threed_rng_reward) else np.array([threed_rng_reward])
                self.threed_rng.traceback_reinforcement(paths, reward_signal)
            
            # Train baseline model (simplified - in reality would be more complex)
            # For demo, we'll just do a simple training step every 10 days
            if step % 10 == 0 and step > 0:
                self.baseline_model.train()
                # Create a simple target based on next day's returns
                if step + self.data_loader.window_size < len(self.data_loader.scaled_data):
                    target_returns = self.data_loader.scaled_data.iloc[step + self.data_loader.window_size]
                    # Convert asset returns to a single market return signal
                    market_return = target_returns.mean()  # Average return across assets
                    
                    # Convert to target format [position, conviction, stop_loss]
                    # Position: sign of return (-1, 0, or 1)
                    # Conviction: magnitude of return (clipped to 0-1)
                    # Stop loss: fixed value or based on volatility
                    position_target = np.sign(market_return)
                    conviction_target = min(abs(market_return) * 10, 1.0)  # Scale for visibility
                    stop_loss_target = 0.05  # Fixed 5% stop loss
                    
                    target_tensor = torch.FloatTensor([position_target, conviction_target, stop_loss_target]).unsqueeze(0)
                    
                    self.baseline_optimizer.zero_grad()
                    output = self.baseline_model(input_tensor)
                    loss = self.baseline_criterion(output, target_tensor)
                    loss.backward()
                    self.baseline_optimizer.step()
            
            # === Regime Event Logging ===
            if self.regime_monitor.is_in_regime_shift() and len(self.regime_events) == 0:
                # Record the start of regime shift
                self.regime_events.append({
                    'date': current_date,
                    'type': 'regime_shift_start',
                    'evaporation_rate': current_evaporation,
                    'reason': 'Regime shift detected by monitor'
                })
                print(f"[{current_date.strftime('%Y-%m-%d')}] REGIME SHIFT DETECTED - Evaporation increased to {current_evaporation:.4f}")
            
            elif not self.regime_monitor.is_in_regime_shift() and len(self.regime_events) > 0 and self.regime_events[-1]['type'] == 'regime_shift_start':
                # Record the end of regime shift
                self.regime_events.append({
                    'date': current_date,
                    'type': 'regime_shift_end',
                    'evaporation_rate': current_evaporation,
                    'reason': 'Performance recovered, returning to baseline'
                })
                print(f"[{current_date.strftime('%Y-%m-%d')}] REGIME SHIFT ENDED - Evaporation returned to {current_evaporation:.4f}")
            
            # Progress reporting
            if step % 50 == 0:
                print(f"Step {step}/{n_steps} - Date: {current_date.strftime('%Y-%m-%d')}")
                print(f"  3D-RNG Equity: ${self.threed_rng_equity_curve[-1]:.2f}")
                print(f"  Baseline Equity: ${self.baseline_equity_curve[-1]:.2f}")
                print(f"  3D-RNG Evaporation: {self.regime_monitor.get_current_evaporation():.4f}")
        
        print("\nBacktest completed!")
        
        # Calculate final statistics
        results = self._calculate_results()
        return results
    
    def _calculate_results(self) -> Dict:
        """Calculate performance statistics for both strategies."""
        # Convert equity curves to numpy arrays
        threed_rng_equity = np.array(self.threed_rng_equity_curve)
        baseline_equity = np.array(self.baseline_equity_curve)
        
        # Calculate returns
        threed_rng_returns = np.diff(threed_rng_equity) / threed_rng_equity[:-1]
        baseline_returns = np.diff(baseline_equity) / baseline_equity[:-1]
        
        # Remove any infinite or NaN values
        threed_rng_returns = threed_rng_returns[np.isfinite(threed_rng_returns)]
        baseline_returns = baseline_returns[np.isfinite(baseline_returns)]
        
        # Calculate statistics
        def calculate_stats(returns, equity_curve):
            if len(returns) == 0:
                return {
                    'total_return': 0.0,
                    'annualized_return': 0.0,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'final_equity': equity_curve[-1] if len(equity_curve) > 0 else 0
                }
            
            total_return = (equity_curve[-1] / equity_curve[0]) - 1
            # Annualize assuming 252 trading days per year
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            # Max drawdown
            rolling_max = np.maximum.accumulate(equity_curve)
            drawdown = (rolling_max - equity_curve) / rolling_max
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'final_equity': equity_curve[-1]
            }
        
        threed_rng_stats = calculate_stats(threed_rng_returns, threed_rng_equity)
        baseline_stats = calculate_stats(baseline_returns, baseline_equity)
        
        # Calculate regime statistics
        regime_stats = self.regime_monitor.get_regime_statistics()
        
        results = {
            'threed_rng': threed_rng_stats,
            'baseline': baseline_stats,
            'regime_events': self.regime_events,
            'threed_rng_trades': len(self.threed_rng_trades),
            'baseline_trades': len(self.baseline_trades),
            'dates': self.dates,
            'threed_rng_equity_curve': self.threed_rng_equity_curve,
            'baseline_equity_curve': self.baseline_equity_curve,
            'regime_stats': regime_stats
        }
        
        return results
    
    def save_results(self, filename: str = 'backtest_results.json'):
        """Save backtest results to file."""
        # This would be called after _calculate_results
        pass


def create_sample_data_files(data_dir: str = './'):
    """
    Create sample data files for the backtest if they don't exist.
    This is just for demonstration - in practice you'd use real data.
    """
    symbols = ['SPY', 'QQQ', 'TLT', 'GLD', 'VIX']
    
    for symbol in symbols:
        filename = os.path.join(data_dir, f"{symbol}.csv")
        if not os.path.exists(filename):
            print(f"Creating sample data file: {filename}")
            # This would create realistic OHLCV data
            # For brevity, we'll skip actual creation in this script
            pass


def main():
    """Main function to run the regime shift backtest."""
    print("=" * 70)
    print("3D-RNG REGIME SHIFT BACKTEST")
    print("Lead Quant Validator & Backtesting Engineer")
    print("=" * 70)
    
    # Initialize backtester
    backtester = FinancialBacktester(
        symbols=['SPY', 'QQQ', 'TLT', 'GLD', 'VIX'],
        start_date='2018-01-01',
        end_date='2020-12-31',
        initial_capital=100000.0,
        transaction_cost=0.001
    )
    
    # Load and prepare data
    backtester.load_and_prepare_data()
    
    # Initialize models
    backtester.initialize_models(input_face_size=(4, 4))
    
    # Run backtest
    results = backtester.run_backtest()
    
    # Print summary
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"3D-RNG Final Equity: ${results['threed_rng']['final_equity']:,.2f}")
    print(f"Baseline Final Equity: ${results['baseline']['final_equity']:,.2f}")
    print(f"3D-RNG Total Return: {results['threed_rng']['total_return']:.2%}")
    print(f"Baseline Total Return: {results['baseline']['total_return']:.2%}")
    print(f"3D-RNG Sharpe Ratio: {results['threed_rng']['sharpe_ratio']:.3f}")
    print(f"Baseline Sharpe Ratio: {results['baseline']['sharpe_ratio']:.3f}")
    print(f"3D-RNG Max Drawdown: {results['threed_rng']['max_drawdown']:.2%}")
    print(f"Baseline Max Drawdown: {results['baseline']['max_drawdown']:.2%}")
    
    print(f"\nRegime Events Detected: {len(results['regime_events'])}")
    for event in results['regime_events']:
        print(f"  {event['date'].strftime('%Y-%m-%d')}: {event['type']} (evap={event['evaporation_rate']:.4f})")
    
    print(f"\nTrading Activity:")
    print(f"  3D-RNG Trades: {results['threed_rng_trades']}")
    print(f"  Baseline Trades: {results['baseline_trades']}")
    
    # Calculate outperformance
    excess_return = results['threed_rng']['total_return'] - results['baseline']['total_return']
    print(f"\n3D-RNG Excess Return: {excess_return:.2%}")
    
    if excess_return > 0:
        print("PASS: 3D-RNG outperformed baseline!")
    else:
        print("FAIL: Baseline outperformed 3D-RNG")
    
    # Save results for tearsheet generation
    import json
    # Convert dates to strings for JSON serialization
    results_for_json = results.copy()
    results_for_json['dates'] = [d.strftime('%Y-%m-%d') for d in results['dates']]
    # Convert regime events dates
    for event in results_for_json['regime_events']:
        event['date'] = event['date'].strftime('%Y-%m-%d')
    
    with open('backtest_results.json', 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    print(f"\nResults saved to backtest_results.json")
    print("Run generate_tearsheet.py to create the analytics tear sheet.")
    
    return results


if __name__ == "__main__":
    # Run the backtest
    main()