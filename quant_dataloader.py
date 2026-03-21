"""
Quantitative Data Loader for 3D-RNG Financial Applications
Lead Quantitative ML Engineer Implementation

This script implements a data loader for multivariate financial time-series data,
specifically designed for the 3D-RNG architecture with spatial feature mapping.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Generator, Union
from sklearn.preprocessing import RobustScaler
import yaml
import json


class FinancialDataLoader:
    """
    Loads and preprocesses multivariate financial time-series data for 3D-RNG.
    
    Features:
    - Ingests OHLCV data for multiple assets + macro indicators
    - Converts raw prices to log-returns
    - Applies robust scaling to bound data between -1.0 and 1.0
    - Implements spatial feature mapping to Input_Face coordinates
    - Yields rolling-window spatial tensors
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 window_size: int = 10,
                 feature_bounds: Tuple[float, float] = (-1.0, 1.0)):
        """
        Initialize the Financial Data Loader.
        
        Args:
            config_path: Path to configuration file defining asset mappings
            window_size: Size of rolling window for temporal context
            feature_bounds: Min/max bounds for scaled features
        """
        self.window_size = window_size
        self.feature_bounds = feature_bounds
        self.scaler = RobustScaler()
        
        # Asset to coordinate mapping (will be loaded from config or set manually)
        self.asset_coords: Dict[str, Tuple[int, int]] = {}
        self.coord_to_asset: Dict[Tuple[int, int], str] = {}
        
        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[np.ndarray] = None
        self.log_returns: Optional[np.ndarray] = None
        self.scaled_data: Optional[np.ndarray] = None
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load asset-to-coordinate mapping from configuration file."""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config = json.load(f)
            else:
                raise ValueError("Config file must be YAML or JSON")
        
        # Extract asset mappings
        if 'asset_mapping' in config:
            for asset, coord in config['asset_mapping'].items():
                self.asset_coords[asset] = tuple(coord)
                self.coord_to_asset[tuple(coord)] = asset
        else:
            # Default mapping for common financial assets
            self._set_default_mapping()
    
    def _set_default_mapping(self):
        """Set default asset-to-coordinate mapping for common financial instruments."""
        # Example mapping for 10 assets + 2 macro indicators on a 4x3 grid
        # (Adjust based on your actual Input_Face dimensions)
        default_mapping = {
            # Tech stocks (clustered together)
            'AAPL': (0, 0),   # Apple
            'MSFT': (0, 1),   # Microsoft
            'GOOGL': (0, 2),  # Google
            'NVDA': (0, 3),   # NVIDIA
            
            # Finance stocks
            'JPM': (1, 0),    # JPMorgan Chase
            'BAC': (1, 1),    # Bank of America
            'GS': (1, 2),     # Goldman Sachs
            
            # Healthcare
            'JNJ': (2, 0),    # Johnson & Johnson
            'PFE': (2, 1),    # Pfizer
            
            # Energy
            'XOM': (2, 2),    # Exxon Mobil
            'CVX': (2, 3),    # Chevron
            
            # Macro indicators
            'VIX': (3, 0),    # Volatility Index
            'DXY': (3, 1),    # US Dollar Index
        }
        
        for asset, coord in default_mapping.items():
            self.asset_coords[asset] = coord
            self.coord_to_asset[coord] = asset
    
    def load_data(self, data_path: str, 
                  price_columns: Optional[List[str]] = None,
                  date_column: str = 'Date'):
        """
        Load financial time-series data from file.
        
        Args:
            data_path: Path to CSV or Parquet file containing OHLCV data
            price_columns: List of column names to use for price data
            date_column: Name of the date column for indexing
        """
        # Load data based on file extension
        if data_path.endswith('.csv'):
            self.raw_data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            self.raw_data = pd.read_parquet(data_path)
        else:
            raise ValueError("Data file must be CSV or Parquet")
        
        # Set date as index if specified
        if date_column in self.raw_data.columns:
            self.raw_data[date_column] = pd.to_datetime(self.raw_data[date_column])
            self.raw_data.set_index(date_column, inplace=True)
        
        # If price_columns not specified, use all numeric columns
        if price_columns is None:
            price_columns = self.raw_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Validate that all price_columns exist
        missing_cols = [col for col in price_columns if col not in self.raw_data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
        # Store only the price data
        # Ensure we always get a DataFrame, even if selecting single column
        selected_data = self.raw_data[price_columns]
        if isinstance(selected_data, pd.Series):
            # Convert single column Series to DataFrame
            self.raw_data = selected_data.to_frame()
        else:
            self.raw_data = selected_data
        
        # Ensure column order matches asset mapping
        self._align_assets_with_mapping()
    
    def _align_assets_with_mapping(self):
        """Ensure data columns match the asset mapping."""
        # If we have asset mapping, reorder columns to match
        if self.asset_coords:
            # Get assets that are in both data and mapping
            data_assets = set(self.raw_data.columns)
            mapped_assets = set(self.asset_coords.keys())
            common_assets = data_assets.intersection(mapped_assets)
            
            if len(common_assets) == 0:
                raise ValueError("No overlapping assets between data and mapping")
            
            # Reorder data columns to match mapping order (by coordinate)
            ordered_assets = sorted(common_assets, 
                                  key=lambda asset: self.asset_coords[asset])
            self.raw_data = self.raw_data[ordered_assets]
    
    def compute_log_returns(self):
        """Convert raw prices to log-returns."""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Calculate log returns: ln(P_t / P_{t-1})
        self.log_returns = np.log(self.raw_data / self.raw_data.shift(1))
        # Drop the first row which is NaN due to shift
        self.log_returns = self.log_returns.iloc[1:].copy()
    
    def scale_features(self):
        """Apply robust scaling to bound features between -1 and 1."""
        if self.log_returns is None:
            raise ValueError("Log returns not computed. Call compute_log_returns() first.")
        
        # Fit scaler on the entire dataset (in production, you might want to fit only on training data)
        scaled_data = self.scaler.fit_transform(self.log_returns)
        
        # RobustScaler doesn't guarantee bounds, so we need to clip
        # First, transform to [0, 1] range using quantile approximation
        # Then scale to desired bounds
        self.scaled_data = np.clip(scaled_data, self.feature_bounds[0], self.feature_bounds[1])
        
        # Convert back to DataFrame for consistency
        self.scaled_data = pd.DataFrame(
            self.scaled_data, 
            index=self.log_returns.index, 
            columns=self.log_returns.columns
        )
    
    def create_spatial_tensor(self, 
                            data: pd.DataFrame, 
                            input_face_size: Tuple[int, int]) -> np.ndarray:
        """
        Convert temporal data to spatial tensor for Input_Face.
        
        Args:
            data: DataFrame with assets as columns and time as index
            input_face_size: Dimensions of the input face (height, width)
            
        Returns:
            Spatial tensor of shape [input_face_height, input_face_width, features]
        """
        height, width = input_face_size
        spatial_tensor = np.zeros((height, width, len(data.columns)))
        
        # Map each asset to its spatial coordinates
        for idx, asset in enumerate(data.columns):
            if asset in self.asset_coords:
                y, x = self.asset_coords[asset]
                # Ensure coordinates are within bounds
                if 0 <= y < height and 0 <= x < width:
                    spatial_tensor[y, x, idx] = data.iloc[-1][asset]  # Most recent value
                else:
                    # Handle out-of-bounds coordinates (shouldn't happen with proper mapping)
                    print(f"Warning: Asset {asset} at {self.asset_coords[asset]} is out of bounds for face {input_face_size}")
            else:
                # Asset not in mapping - could distribute evenly or skip
                print(f"Warning: Asset {asset} not found in coordinate mapping")
        
        return spatial_tensor
    
    def generate_batches(self, 
                        batch_size: int = 32,
                        input_face_size: Tuple[int, int] = (4, 4)) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate batches of spatial tensors for training.
        
        Yields:
            Tuple of (input_tensor, target_tensor) where:
            - input_tensor: Shape [batch_size, input_face_height, input_face_width, num_assets]
            - target_tensor: Shape [batch_size, num_assets] (next time step returns)
        """
        if self.scaled_data is None:
            raise ValueError("Data not scaled. Call scale_features() first.")
        
        # Calculate number of possible windows
        num_samples = len(self.scaled_data) - self.window_size
        if num_samples <= 0:
            raise ValueError(f"Not enough data for window size {self.window_size}. Need at least {self.window_size + 1} samples.")
        
        # Generate indices for batches
        indices = np.arange(num_samples)
        np.random.shuffle(indices)  # Shuffle for training
        
        for start_idx in range(0, len(indices), batch_size):
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            batch_inputs = []
            batch_targets = []
            
            for idx in batch_indices:
                # Get window of data
                window_data = self.scaled_data.iloc[idx:idx + self.window_size]
                # Get target (next time step)
                target_data = self.scaled_data.iloc[idx + self.window_size]
                
                # Convert window to spatial tensor
                spatial_tensor = self.create_spatial_tensor(window_data, input_face_size)
                batch_inputs.append(spatial_tensor)
                
                # Target is just the returns at next time step
                batch_targets.append(target_data.values)
            
            yield np.array(batch_inputs), np.array(batch_targets)
    
    def get_feature_dimensions(self) -> Tuple[int, int, int]:
        """
        Get the dimensions of the feature tensor.
        
        Returns:
            Tuple of (input_height, input_width, num_features)
        """
        if self.scaled_data is None:
            raise ValueError("No processed data available")
        
        # This would need to be configured based on actual input face size
        # For now, return a placeholder that should be configured externally
        return (4, 4, len(self.scaled_data.columns))  # Default 4x4 face
    
    def get_asset_list(self) -> List[str]:
        """Get list of assets in the data loader."""
        if self.scaled_data is None:
            return []
        return list(self.scaled_data.columns)
    
    def get_data_info(self) -> Dict:
        """Get information about the loaded data."""
        info = {
            'num_assets': len(self.scaled_data.columns) if self.scaled_data is not None else 0,
            'num_time_steps': len(self.scaled_data) if self.scaled_data is not None else 0,
            'window_size': self.window_size,
            'feature_bounds': self.feature_bounds,
            'asset_mapping': self.asset_coords.copy()
        }
        return info


def create_sample_financial_config() -> Dict:
    """Create a sample configuration file for financial assets."""
    config = {
        'asset_mapping': {
            'AAPL': [0, 0],
            'MSFT': [0, 1],
            'GOOGL': [0, 2],
            'NVDA': [0, 3],
            'JPM': [1, 0],
            'BAC': [1, 1],
            'GS': [1, 2],
            'JNJ': [2, 0],
            'PFE': [2, 1],
            'XOM': [2, 2],
            'CVX': [2, 3],
            'VIX': [3, 0],
            'DXY': [3, 1]
        },
        'window_size': 10,
        'feature_bounds': [-1.0, 1.0]
    }
    return config


def save_sample_config(filepath: str):
    """Save a sample configuration file."""
    config = create_sample_financial_config()
    if filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError("Filepath must end with .yaml, .yml, or .json")


if __name__ == "__main__":
    # Example usage
    print("Creating sample financial data loader...")
    
    # Create and save sample config
    save_sample_config("financial_config.yaml")
    print("Saved sample configuration to financial_config.yaml")
    
    # Initialize data loader
    loader = FinancialDataLoader(config_path="financial_config.yaml", window_size=10)
    
    # Show asset mapping
    print(f"Asset mapping: {loader.asset_coords}")
    
    # In a real scenario, you would load actual data here:
    # loader.load_data("financial_data.csv")
    # loader.compute_log_returns()
    # loader.scale_features()
    # 
    # # Generate batches
    # for batch_x, batch_y in loader.generate_batches(batch_size=32):
    #     print(f"Batch shape: X={batch_x.shape}, y={batch_y.shape}")
    #     break  # Just show first batch for demo
    
    print("FinancialDataLoader ready for use.")