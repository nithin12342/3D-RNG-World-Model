"""
Financial Integration Test for 3D-RNG Components
Tests the integration of quant_dataloader.py, trading_harness.py, and regime_monitor.py
with the existing 3D-RNG implementation.
"""

import numpy as np
import sys
import os

def test_quant_dataloader():
    """Test the FinancialDataLoader component."""
    print("Testing FinancialDataLoader...")
    
    try:
        from quant_dataloader import FinancialDataLoader, create_sample_financial_config, save_sample_config
        
        # Create and save sample config
        save_sample_config("test_financial_config.yaml")
        print("  Saved sample configuration")
        
        # Initialize data loader
        loader = FinancialDataLoader(config_path="test_financial_config.yaml", window_size=5)
        
        # Show asset mapping
        print(f"  Asset mapping: {loader.asset_coords}")
        print(f"  Number of assets: {len(loader.asset_coords)}")
        
        # Test with synthetic data (since we don't have real financial data)
        # Create synthetic price data for testing
        import pandas as pd
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        data = {}
        for asset in loader.asset_coords.keys():
            # Generate random walk prices
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            data[asset] = prices
        
        synthetic_df = pd.DataFrame(data, index=dates)
        synthetic_df.to_csv("test_financial_data.csv")
        print("  Created synthetic financial data")
        
        # Load the synthetic data
        loader.load_data("test_financial_data.csv")
        print(f"  Loaded data shape: {loader.raw_data.shape}")
        
        # Compute log returns
        loader.compute_log_returns()
        print(f"  Log returns shape: {loader.log_returns.shape}")
        
        # Scale features
        loader.scale_features()
        print(f"  Scaled data shape: {loader.scaled_data.shape}")
        print(f"  Feature range: [{loader.scaled_data.min().min():.3f}, {loader.scaled_data.max().max():.3f}]")
        
        # Test spatial tensor creation
        input_face_size = (4, 4)  # 4x4 grid
        if len(loader.scaled_data) > 0:
            recent_data = loader.scaled_data.iloc[-1:]  # Most recent row
            spatial_tensor = loader.create_spatial_tensor(recent_data, input_face_size)
            print(f"  Spatial tensor shape: {spatial_tensor.shape}")
            
            # Check that tensor has expected dimensions
            assert spatial_tensor.shape == (input_face_size[0], input_face_size[1], len(loader.asset_coords))
        
        # Test batch generation (if we have enough data)
        if len(loader.scaled_data) > loader.window_size:
            batch_count = 0
            for batch_x, batch_y in loader.generate_batches(batch_size=4, input_face_size=input_face_size):
                print(f"  Batch {batch_count}: X shape={batch_x.shape}, y shape={batch_y.shape}")
                batch_count += 1
                if batch_count >= 2:  # Just test a couple batches
                    break
        
        print("  * FinancialDataLoader test passed")
        return True
        
    except Exception as e:
        print(f"  ! FinancialDataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trading_harness():
    """Test the TradingHarness component."""
    print("\nTesting TradingHarness...")
    
    try:
        from core.trading_harness import TradingOutputInterpreter, calculate_sharpe_reward, calculate_sortino_reward, calculate_max_drawdown_reward, PositionSizer
        
        # Test TradingOutputInterpreter
        interpreter = TradingOutputInterpreter(
            output_face_size=(4, 4),
            position_threshold=0.1,
            conviction_threshold=0.3
        )
        
        # Simulate output from 3D-RNG (16 positions, 3 values each)
        output_tensor = np.random.randn(16, 3) * 0.5
        
        # Interpret output
        actions = interpreter.interpret_output(output_tensor)
        print(f"  Generated {len(actions)} individual actions")
        
        # Aggregate actions
        aggregated = interpreter.aggregate_actions(actions)
        print(f"  Aggregated action: position={aggregated['position']:.3f}, conviction={aggregated['conviction']:.3f}")
        
        # Test reward functions
        prediction = np.array([0.5])  # Long position
        actual_return = np.array([0.02])  # 2% gain
        
        sharpe_reward = calculate_sharpe_reward(prediction, actual_return)
        sortino_reward = calculate_sortino_reward(prediction, actual_return)
        dd_reward = calculate_max_drawdown_reward(prediction, actual_return)
        
        print(f"  Sharpe reward: {sharpe_reward:.3f}")
        print(f"  Sortino reward: {sortino_reward:.3f}")
        print(f"  Drawdown reward: {dd_reward:.3f}")
        
        # Test position sizing
        equity = 100000  # $100k account
        risk_per_trade = 0.02  # 2% risk per trade
        stop_loss_pct = 0.05   # 5% stop loss
        
        position_size = PositionSizer.fixed_fractional(equity, risk_per_trade, stop_loss_pct)
        print(f"  Position size: {position_size:.2%} of equity (${equity * position_size:.2f})")
        
        # Test edge cases
        # Empty actions
        empty_actions = interpreter.interpret_output(np.zeros((16, 3)))  # All zeros -> no conviction
        empty_aggregated = interpreter.aggregate_actions(empty_actions)
        assert empty_aggregated['position'] == 0
        
        print("  * TradingHarness test passed")
        return True
        
    except Exception as e:
        print(f"  ! TradingHarness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_regime_monitor():
    """Test the RegimeMonitor component."""
    print("\nTesting RegimeMonitor...")
    
    try:
        from regime_monitor import RegimeMonitor, AdaptiveEvaporationController
        
        # Create regime monitor
        monitor = RegimeMonitor(
            baseline_evaporation=0.05,
            regime_threshold=-0.01,
            regime_consecutive_epochs=3,  # Lower for demo
            evaporation_multiplier=3.0,   # Lower for demo
            recovery_rate=0.2,
            max_evaporation=0.3
        )
        
        # Test normal operation
        evaporation = monitor.update(0.02)  # Good reward
        assert evaporation == 0.05  # Should stay at baseline
        print(f"  Good reward (0.02): evaporation={evaporation:.3f}")
        
        # Test poor performance triggering regime shift
        poor_rewards = [-0.02, -0.03, -0.02]  # 3 consecutive bad epochs
        for i, reward in enumerate(poor_rewards):
            evaporation = monitor.update(reward)
            print(f"  Poor reward {i+1} ({reward:.3f}): evaporation={evaporation:.3f}, regime_shift={monitor.is_in_regime_shift()}")
        
        # Should be in regime shift now
        assert monitor.is_in_regime_shift() == True
        assert evaporation > 0.05  # Should be increased
        
        # Test recovery
        good_rewards = [0.02, 0.01, 0.03]
        for i, reward in enumerate(good_rewards):
            evaporation = monitor.update(reward)
            print(f"  Good reward {i+1} ({reward:.3f}): evaporation={evaporation:.3f}, regime_shift={monitor.is_in_regime_shift()}")
        
        # Should eventually recover
        stats = monitor.get_regime_statistics()
        print(f"  Final stats: {stats}")
        
        # Test AdaptiveEvaporationController
        controller = AdaptiveEvaporationController(
            baseline_evaporation=0.05,
            regime_threshold=-0.01,
            regime_consecutive_epochs=3,
            evaporation_multiplier=3.0,
            recovery_rate=0.2,
            max_evaporation=0.3
        )
        
        evap_rate = controller.update_evaporation(-0.02)
        print(f"  Controller test: evaporation rate={evap_rate:.3f}")
        
        print("  * RegimeMonitor test passed")
        return True
        
    except Exception as e:
        print(f"  ! RegimeMonitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration of all financial components."""
    print("\nTesting Financial Component Integration...")
    
    try:
        from core.quant_dataloader import FinancialDataLoader
        from core.trading_harness import TradingOutputInterpreter, calculate_sharpe_reward
        from core.regime_monitor import RegimeMonitor
        
        # Create components
        loader = FinancialDataLoader(window_size=5)
        interpreter = TradingOutputInterpreter()
        monitor = RegimeMonitor(regime_consecutive_epochs=2)  # Lower for demo
        
        # Simulate a simple workflow
        print("  Simulating financial trading workflow...")
        
        # 1. Generate synthetic market data
        import pandas as pd
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        n_assets = 4
        data = {}
        for i in range(n_assets):
            returns = np.random.normal(0.0005, 0.015, len(dates))  # Slightly positive drift
            prices = 100 * np.exp(np.cumsum(returns))
            data[f'ASSET_{i}'] = prices
        
        df = pd.DataFrame(data, index=dates)
        df.to_csv("integration_test_data.csv")
        
        # 2. Load and process data
        loader.load_data("integration_test_data.csv")
        loader.compute_log_returns()
        loader.scale_features()
        
        # 3. Simulate trading loop
        print("  Simulating trading epochs...")
        for epoch in range(5):
            # Get recent data for spatial tensor
            if len(loader.scaled_data) > loader.window_size:
                window_data = loader.scaled_data.iloc[-loader.window_size:]
                
                # Create spatial tensor (simplified - using first few assets)
                input_face_size = (2, 2)  # Small face for testing
                spatial_tensor = loader.create_spatial_tensor(window_data, input_face_size)
                
                # Flatten for 3D-RNG input (simulation)
                # In reality, this would go into the 3D-RNG forward_probe
                flat_input = spatial_tensor.flatten()[:16]  # Take first 16 elements
                if len(flat_input) < 16:
                    flat_input = np.pad(flat_input, (0, 16 - len(flat_input)), 'constant')
                
                # Simulate 3D-RNG output (normally from forward_probe)
                # Shape: [num_output_nodes, 3] for [position, conviction, stop_loss]
                simulated_output = np.random.randn(4, 3) * 0.3  # 2x2 face = 4 nodes
                
                # Interpret as trading actions
                actions = interpreter.interpret_output(simulated_output)
                aggregated = interpreter.aggregate_actions(actions)
                
                # Simulate future return (for reward calculation)
                # In reality, this would be the actual future return
                future_return = np.random.normal(0.001, 0.005)  # Small daily return
                
                # Calculate reward using Sharpe ratio
                # Prediction is the aggregated position
                prediction_val = np.array([aggregated['position'] * aggregated['conviction']])
                actual_return_val = np.array([future_return])
                
                reward = calculate_sharpe_reward(prediction_val, actual_return_val)
                
                # Update regime monitor
                evaporation_rate = monitor.update(reward)
                
                print(f"    Epoch {epoch+1}: Pos={aggregated['position']:.2f}, Conv={aggregated['conviction']:.2f}, "
                      f"Return={future_return:.4f}, Reward={reward:.3f}, Evap={evaporation_rate:.3f}")
        
        # Get final statistics
        final_stats = monitor.get_regime_statistics()
        print(f"  Final regime stats: {final_stats['epoch_counter']} epochs processed")
        
        print("  * Financial Component Integration test passed")
        return True
        
    except Exception as e:
        print(f"  ! Financial Component Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all financial integration tests."""
    print("=" * 70)
    print("3D-RNG FINANCIAL COMPONENT INTEGRATION TEST")
    print("=" * 70)
    
    tests = [
        test_quant_dataloader,
        test_trading_harness,
        test_regime_monitor,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 70)
    print(f"FINANCIAL TEST RESULTS: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All financial integration tests passed!")
        return 0
    else:
        print("❌ Some financial tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())