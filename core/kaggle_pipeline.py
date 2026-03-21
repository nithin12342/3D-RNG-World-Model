"""
Kaggle Data Downloader & Preprocessor for 3D-RNG
Lead Quant Data Engineer & ML Systems Architect Implementation

This script implements a data pipeline for ingesting and preprocessing 
financial data from Kaggle datasets for the 3D-RNG architecture.
"""

import numpy as np
import pandas as pd
import os
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class KaggleDataPipeline:
    """
    Downloads and preprocesses financial data for 3D-RNG.
    
    Features:
    - Downloads OHLCV data for S&P 500 stocks + VIX
    - Calculates log returns, RSI, and historical volatility
    - Normalizes features to [-1, 1] range using tanh
    - Provides cleaned data for spatial mapping
    """
    
    def __init__(self, 
                 symbols: Optional[List[str]] = None,
                 start_date: str = '2018-01-01',
                 end_date: str = '2020-12-31',
                 data_dir: str = './kaggle_data'):
        """
        Initialize the Kaggle Data Pipeline.
        
        Args:
            symbols: List of ticker symbols (if None, uses default S&P 500 basket)
            start_date: Start date for data collection
            end_date: End date for data collection
            data_dir: Directory to store downloaded data
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        
        # Default basket: 20 diverse S&P 500 stocks + VIX
        if symbols is None:
            self.symbols = [
                # Technology
                'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'ADBE',
                # Healthcare
                'JNJ', 'PFE', 'UNH',
                # Financial
                'JPM', 'BAC', 'GS',
                # Consumer
                'AMZN', 'WMT', 'PG',
                # Industrial
                'BA', 'CAT',
                # Energy
                'XOM',
                # Utilities
                'NEE',
                # VIX (Volatility Index)
                '^VIX'
            ]
        else:
            self.symbols = symbols
            
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Storage for processed data
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.processed_data: Optional[pd.DataFrame] = None
        self.feature_data: Optional[pd.DataFrame] = None
        
    def download_data(self, force_redownload: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV data for all symbols.
        
        Args:
            force_redownload: Whether to force redownload existing files
            
        Returns:
            Dictionary mapping symbols to their OHLCV DataFrames
        """
        print(f"Downloading data for {len(self.symbols)} symbols from {self.start_date} to {self.end_date}")
        
        for symbol in self.symbols:
            file_path = os.path.join(self.data_dir, f"{symbol.replace('^', '')}.csv")
            
            # Check if file exists and we don't need to redownload
            if os.path.exists(file_path) and not force_redownload:
                print(f"Loading {symbol} from cache...")
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                self.raw_data[symbol] = df
                continue
                
            try:
                print(f"Downloading {symbol}...")
                # Download data using yfinance
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.start_date, end=self.end_date)
                
                if df.empty:
                    print(f"Warning: No data found for {symbol}")
                    continue
                    
                # Keep only OHLCV columns
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                # Save to cache
                df.to_csv(file_path)
                self.raw_data[symbol] = df
                print(f"Downloaded {symbol}: {df.shape[0]} rows")
                
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
                # Create empty DataFrame as placeholder
                self.raw_data[symbol] = pd.DataFrame()
                
        print(f"Download complete. Successfully loaded {len([s for s in self.raw_data.keys() if not self.raw_data[s].empty])} symbols")
        return self.raw_data
    
    def calculate_log_returns(self) -> pd.DataFrame:
        """
        Calculate daily log returns for all symbols.
        
        Returns:
            DataFrame with log returns for each symbol
        """
        if not self.raw_data:
            raise ValueError("No raw data available. Call download_data() first.")
            
        log_returns_dict = {}
        
        for symbol, df in self.raw_data.items():
            if df.empty:
                continue
                
            # Calculate log returns: ln(P_t / P_{t-1})
            log_returns = np.log(df['Close'] / df['Close'].shift(1))
            log_returns_dict[symbol] = log_returns
            
        # Combine into DataFrame
        log_returns_df = pd.DataFrame(log_returns_dict)
        # Drop first row (NaN due to shift)
        log_returns_df = log_returns_df.iloc[1:].copy()
        
        self.log_returns = log_returns_df
        return log_returns_df
    
    def calculate_rsi(self, window: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI) for all symbols.
        
        Args:
            window: RSI calculation window (default 14 days)
            
        Returns:
            DataFrame with RSI values for each symbol
        """
        if not self.raw_data:
            raise ValueError("No raw data available. Call download_data() first.")
            
        rsi_dict = {}
        
        for symbol, df in self.raw_data.items():
            if df.empty:
                continue
                
            # Calculate price changes
            delta = df['Close'].diff()
            
            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calculate average gain and loss over window
            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Handle division by zero (when avg_loss is 0)
            rsi = rsi.fillna(100)  # When no losses, RSI = 100
            
            rsi_dict[symbol] = rsi
            
        # Combine into DataFrame
        rsi_df = pd.DataFrame(rsi_dict)
        self.rsi = rsi_df
        return rsi_df
    
    def calculate_historical_volatility(self, window: int = 20) -> pd.DataFrame:
        """
        Calculate historical volatility for all symbols.
        
        Args:
            window: Volatility calculation window (default 20 days)
            
        Returns:
            DataFrame with historical volatility for each symbol
        """
        if not hasattr(self, 'log_returns'):
            self.calculate_log_returns()
            
        # Calculate rolling standard deviation of log returns
        vol_dict = {}
        
        for symbol in self.log_returns.columns:
            if symbol not in self.raw_data or self.raw_data[symbol].empty:
                continue
                
            # Historical volatility: std of log returns * sqrt(252) for annualization
            vol = self.log_returns[symbol].rolling(window=window, min_periods=1).std() * np.sqrt(252)
            vol_dict[symbol] = vol
            
        # Combine into DataFrame
        vol_df = pd.DataFrame(vol_dict)
        self.historical_vol = vol_df
        return vol_df
    
    def normalize_features(self, 
                          log_returns_weight: float = 0.5,
                          rsi_weight: float = 0.3,
                          vol_weight: float = 0.2) -> pd.DataFrame:
        """
        Normalize and combine features into [-1, 1] range.
        
        Args:
            log_returns_weight: Weight for log returns in final feature
            rsi_weight: Weight for RSI in final feature
            vol_weight: Weight for volatility in final feature
            
        Returns:
            DataFrame with normalized features for each symbol
        """
        # Ensure we have all required data
        if not hasattr(self, 'log_returns'):
            self.calculate_log_returns()
        if not hasattr(self, 'rsi'):
            self.calculate_rsi()
        if not hasattr(self, 'historical_vol'):
            self.calculate_historical_volatility()
            
        # Normalize each component to [-1, 1] range
        features_dict = {}
        
        for symbol in self.symbols:
            if symbol not in self.log_returns.columns:
                continue
                
            # Get component data
            lr = self.log_returns[symbol].fillna(0)  # Fill NaN with 0
            rsi_val = self.rsi[symbol].fillna(50)    # Fill NaN with neutral RSI
            vol_val = self.historical_vol[symbol].fillna(self.historical_vol[symbol].mean())  # Fill NaN with mean
            
            # Normalize log returns: tanh naturally bounds to [-1, 1]
            lr_norm = np.tanh(lr * 100)  # Scale for sensitivity
            
            # Normalize RSI: map [0, 100] to [-1, 1]
            rsi_norm = (rsi_val - 50) / 50
            
            # Normalize volatility: use tanh to bound extreme values
            vol_norm = np.tanh(vol_val)  # Already scaled reasonably
            
            # Weighted combination
            combined = (log_returns_weight * lr_norm + 
                       rsi_weight * rsi_norm + 
                       vol_weight * vol_norm)
            
            # Final tanh to ensure [-1, 1] bounds
            features_dict[symbol] = np.tanh(combined)
            
        # Combine into DataFrame
        features_df = pd.DataFrame(features_dict)
        self.feature_data = features_df
        return features_df
    
    def get_latest_features(self, date: str) -> Optional[np.ndarray]:
        """
        Get the feature vector for all symbols on a specific date.
        
        Args:
            date: Date string in 'YYYY-MM-DD' format
            
        Returns:
            Numpy array of features for all symbols, or None if date not found
        """
        if self.feature_data is None:
            raise ValueError("Features not calculated. Call normalize_features() first.")
            
        try:
            # Get row for the specified date
            features_row = self.feature_data.loc[date]
            return features_row.values.astype(np.float32)
        except KeyError:
            # Date not found in index
            return None
        except Exception as e:
            print(f"Error getting features for {date}: {e}")
            return None
    
    def save_processed_data(self, filename: str = 'processed_features.csv'):
        """
        Save processed feature data to CSV.
        
        Args:
            filename: Name of the file to save
        """
        if self.feature_data is None:
            raise ValueError("No processed data to save. Call normalize_features() first.")
            
        file_path = os.path.join(self.data_dir, filename)
        self.feature_data.to_csv(file_path)
        print(f"Saved processed features to {file_path}")
        
    def get_data_info(self) -> Dict:
        """
        Get information about the downloaded and processed data.
        
        Returns:
            Dictionary with data information
        """
        info = {
            'symbols_requested': len(self.symbols),
            'symbols_downloaded': len([s for s in self.raw_data.keys() if not self.raw_data[s].empty]),
            'date_range': f"{self.start_date} to {self.end_date}",
            'data_points': 0,
            'features_calculated': False
        }
        
        if self.feature_data is not None:
            info['data_points'] = len(self.feature_data)
            info['features_calculated'] = True
            info['feature_symbols'] = list(self.feature_data.columns)
            
        return info


def create_sample_kaggle_config() -> Dict:
    """Create a sample configuration for the Kaggle pipeline."""
    config = {
        'symbols': [
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'ADBE',
            'JNJ', 'PFE', 'UNH',
            'JPM', 'BAC', 'GS',
            'AMZN', 'WMT', 'PG',
            'BA', 'CAT',
            'XOM',
            'NEE',
            '^VIX'
        ],
        'start_date': '2018-01-01',
        'end_date': '2020-12-31',
        'data_dir': './kaggle_data'
    }
    return config


def save_kaggle_config(config: Dict, filepath: str):
    """Save Kaggle pipeline configuration to file."""
    import json
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    # Example usage
    print("Creating Kaggle Data Pipeline...")
    
    # Initialize pipeline
    pipeline = KaggleDataPipeline()
    
    # Download data
    pipeline.download_data()
    
    # Calculate features
    pipeline.calculate_log_returns()
    pipeline.calculate_rsi()
    pipeline.calculate_historical_volatility()
    features = pipeline.normalize_features()
    
    # Show info
    info = pipeline.get_data_info()
    print(f"\nData Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Show sample features
    if len(features) > 0:
        print(f"\nSample features (first 5 rows):")
        print(features.head())
        print(f"\nFeature shape: {features.shape}")
    
    # Save processed data
    pipeline.save_processed_data()
    
    print("\nKaggle Data Pipeline ready for use!")