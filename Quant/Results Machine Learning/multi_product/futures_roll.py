"""
Futures Roll Implementation
Based on Advances in Financial Machine Learning, López de Prado

This module handles the rolling of futures contracts to create continuous time series.

Key concepts:
- Roll dates: When to switch from front contract to next contract
- Roll methods: How to adjust prices (no adjustment, back adjustment, ratio adjustment)  
- Contract selection: Which contracts to include in the continuous series

Author: Sistema de Datos - Quant Analysis
Date: July 2025
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings

try:
    from ..data_structures.standard_data_structures import get_volume_bars
except ImportError:
    try:
        # Fallback for direct execution
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data_structures.standard_data_structures import get_volume_bars
    except ImportError:
        # Final fallback - create dummy function
        def get_volume_bars(file_path_or_df, threshold=1000, verbose=False):
            """Fallback function if volume bars not available"""
            if isinstance(file_path_or_df, str):
                if os.path.exists(file_path_or_df):
                    df = pd.read_csv(file_path_or_df)
                else:
                    raise FileNotFoundError(f"File not found: {file_path_or_df}")
            else:
                df = file_path_or_df.copy()
            
            # Ensure required columns exist
            required_cols = ['date_time', 'open', 'close', 'high', 'low', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'date_time' and 'timestamp' in df.columns:
                        df['date_time'] = pd.to_datetime(df['timestamp'])
                    elif col in ['open', 'close', 'high', 'low'] and 'price' in df.columns:
                        df[col] = df['price']
                    elif col == 'volume' and col not in df.columns:
                        df[col] = 1.0  # Default volume
                        
            return df


class FuturesRoll:
    """
    Handles futures contract rolling for continuous time series creation
    
    This class implements the methodology described in López de Prado's book
    for creating continuous futures time series by properly handling contract rolls.
    
    Example usage:
        # Basic usage
        roller = FuturesRoll(roll_method='back_adjust')
        tickers_df, expirations = roller.prepare_futures_dataset('data/')
        continuous_series = roller.create_continuous_series(tickers_df, expirations)
        
        # Advanced usage with custom parameters
        roller = FuturesRoll(
            roll_method='ratio_adjust',
            roll_days_before_expiry=3,
            min_volume_threshold=500
        )
    """
    
    def __init__(self, 
                 roll_method: str = 'back_adjust', 
                 roll_days_before_expiry: int = 5,
                 min_volume_threshold: float = 100,
                 volume_bar_threshold: int = 1000):
        """
        Initialize futures roll handler
        
        :param roll_method: ('back_adjust', 'ratio_adjust', 'no_adjust') Price adjustment method
        :param roll_days_before_expiry: (int) Days before expiry to perform roll
        :param min_volume_threshold: (float) Minimum volume threshold for contract inclusion
        :param volume_bar_threshold: (int) Volume threshold for creating volume bars
        """
        valid_methods = ['back_adjust', 'ratio_adjust', 'no_adjust']
        if roll_method not in valid_methods:
            raise ValueError(f"roll_method must be one of {valid_methods}")
            
        self.roll_method = roll_method
        self.roll_days_before_expiry = roll_days_before_expiry
        self.min_volume_threshold = min_volume_threshold
        self.volume_bar_threshold = volume_bar_threshold
        
    def prepare_futures_dataset(self, 
                               data_path: str = 'data/', 
                               contract_pattern: str = 'VIX',
                               threshold: Optional[int] = None,
                               verbose: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """
        Step 1: Prepare the dataset
        Compress each separate futures contract into volume bars
        
        This method implements the first step shown in López de Prado's methodology:
        1. Read all futures contract files matching the pattern
        2. Convert each to volume bars for consistent sampling
        3. Create expiration dictionary based on last traded date
        4. Combine all contracts into a single DataFrame
        
        :param data_path: (str) Path to futures data files
        :param contract_pattern: (str) Pattern to match contract files (e.g., 'VIX', 'ES', 'CL')
        :param threshold: (int) Volume threshold for bars (overrides instance setting)
        :param verbose: (bool) Verbose output
        :return: (tuple) Combined dataframe and expirations dictionary
        """
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
            
        threshold = threshold or self.volume_bar_threshold
        
        # Step 1a: Compress each separate futures contract into volume bars
        bar_data_frames_dict = {}
        processed_files = 0
        
        for f in sorted(os.listdir(data_path)):
            if contract_pattern in f and (f.endswith('.csv') or f.endswith('.xlsx')):
                file_path = os.path.join(data_path, f)
                
                try:
                    if verbose:
                        print(f"Processing {f}...")
                        
                    # Read and convert to volume bars
                    bars_df = get_volume_bars(
                        file_path_or_df=file_path, 
                        threshold=threshold, 
                        verbose=verbose
                    )
                    
                    # Validate data quality
                    if self._validate_contract_data(bars_df, f, verbose):
                        bar_data_frames_dict[f] = bars_df
                        processed_files += 1
                        
                        if verbose:
                            print(f"✓ {f}: {len(bars_df)} bars created")
                    else:
                        if verbose:
                            print(f"✗ {f}: Failed validation, skipping")
                            
                except Exception as e:
                    warnings.warn(f"Error processing {f}: {str(e)}")
                    if verbose:
                        print(f"✗ {f}: Error - {str(e)}")
                    continue
        
        if processed_files == 0:
            raise ValueError(f"No valid contract files found with pattern '{contract_pattern}' in {data_path}")
            
        # Step 1b: Create expirations dict based on latest traded day
        expirations_dict = {}
        for asset, df in bar_data_frames_dict.items():
            if not df.empty and 'date_time' in df.columns:
                # Ensure date_time is datetime
                if not pd.api.types.is_datetime64_any_dtype(df['date_time']):
                    df['date_time'] = pd.to_datetime(df['date_time'])
                expirations_dict[asset] = df['date_time'].iloc[-1]
        
        # Step 1c: Sort in ascending order by expiration dates
        expirations_dict = dict(sorted(expirations_dict.items(), 
                                     key=lambda item: item[1]))
        
        if verbose:
            print(f"\nExpiration schedule:")
            for contract, exp_date in expirations_dict.items():
                print(f"  {contract}: {exp_date.strftime('%Y-%m-%d')}")
        
        # Step 1d: Create data frame for futures roll
        tickers_df = pd.DataFrame()
        
        for asset, df in bar_data_frames_dict.items():
            if not df.empty:
                df_copy = df.copy()
                
                # Ensure date_time is datetime and set as index
                if not pd.api.types.is_datetime64_any_dtype(df_copy['date_time']):
                    df_copy['date_time'] = pd.to_datetime(df_copy['date_time'])
                    
                df_copy = df_copy.set_index('date_time')
                df_copy['ticker'] = asset
                
                # Select required columns
                required_cols = ['open', 'close', 'ticker']
                optional_cols = ['high', 'low', 'volume']
                
                cols_to_include = required_cols.copy()
                for col in optional_cols:
                    if col in df_copy.columns:
                        cols_to_include.append(col)
                
                tickers_df = pd.concat([tickers_df, df_copy[cols_to_include]])
        
        # Sort by date_time index
        tickers_df.sort_index(inplace=True)
        
        if verbose:
            print(f"\n✓ Dataset prepared: {len(tickers_df)} total bars across {processed_files} contracts")
            print(f"Date range: {tickers_df.index.min()} to {tickers_df.index.max()}")
        
        return tickers_df, expirations_dict
    
    def create_continuous_series(self, 
                                tickers_df: pd.DataFrame, 
                                expirations_dict: Dict,
                                verbose: bool = False) -> pd.DataFrame:
        """
        Create continuous futures series by rolling contracts
        
        This implements the core rolling logic:
        1. Determine active contract for each date
        2. Handle roll transitions based on expiry rules
        3. Apply price adjustments to maintain series continuity
        
        :param tickers_df: (pd.DataFrame) Combined futures data with ticker column
        :param expirations_dict: (dict) Contract expiration dates
        :param verbose: (bool) Verbose output
        :return: (pd.DataFrame) Continuous futures series with adjustments
        """
        
        if tickers_df.empty:
            raise ValueError("Input DataFrame is empty")
            
        if not expirations_dict:
            raise ValueError("Expirations dictionary is empty")
        
        # Sort contracts by expiration date
        sorted_contracts = list(expirations_dict.keys())
        
        if verbose:
            print(f"Creating continuous series using {len(sorted_contracts)} contracts")
            print(f"Roll method: {self.roll_method}")
            print(f"Roll days before expiry: {self.roll_days_before_expiry}")
        
        continuous_series = []
        current_contract_idx = 0
        roll_dates = []
        
        # Get unique dates sorted
        all_dates = sorted(tickers_df.index.unique())
        
        if verbose:
            print(f"Processing {len(all_dates)} unique dates...")
        
        for i, date in enumerate(all_dates):
            # Check if we need to roll to next contract
            if self._should_roll(date, sorted_contracts, current_contract_idx, expirations_dict):
                if current_contract_idx < len(sorted_contracts) - 1:
                    old_contract = sorted_contracts[current_contract_idx]
                    current_contract_idx += 1
                    new_contract = sorted_contracts[current_contract_idx]
                    
                    roll_dates.append({
                        'date': date,
                        'from_contract': old_contract,
                        'to_contract': new_contract
                    })
                    
                    if verbose:
                        print(f"  Roll on {date.strftime('%Y-%m-%d')}: {old_contract} → {new_contract}")
            
            # Get data for current active contract
            current_contract = sorted_contracts[current_contract_idx]
            contract_data = tickers_df[
                (tickers_df.index == date) & 
                (tickers_df['ticker'] == current_contract)
            ]
            
            if not contract_data.empty:
                row = contract_data.iloc[0].copy()
                row['active_contract'] = current_contract
                row['contract_number'] = current_contract_idx
                row['is_roll_date'] = any(rd['date'] == date for rd in roll_dates)
                continuous_series.append(row)
        
        if not continuous_series:
            raise ValueError("No continuous series data could be created")
        
        # Create DataFrame from series
        continuous_df = pd.DataFrame(continuous_series)
        continuous_df.index = [row.name for row in continuous_series]
        continuous_df.index.name = 'date_time'
        
        # Apply price adjustments based on roll method
        if self.roll_method == 'back_adjust':
            continuous_df = self._back_adjust_prices(continuous_df, verbose)
        elif self.roll_method == 'ratio_adjust':
            continuous_df = self._ratio_adjust_prices(continuous_df, verbose)
        # no_adjust: no changes needed
        
        # Add roll information
        continuous_df['roll_dates'] = continuous_df.index.isin([rd['date'] for rd in roll_dates])
        
        if verbose:
            print(f"✓ Continuous series created: {len(continuous_df)} bars")
            print(f"  {len(roll_dates)} roll events occurred")
            if self.roll_method != 'no_adjust':
                print(f"  Price adjustments applied using {self.roll_method} method")
        
        return continuous_df
    
    def _validate_contract_data(self, df: pd.DataFrame, filename: str, verbose: bool = False) -> bool:
        """
        Validate contract data quality
        
        :param df: Contract DataFrame
        :param filename: Contract filename for reporting
        :param verbose: Verbose output
        :return: True if data passes validation
        """
        if df.empty:
            if verbose:
                print(f"  Warning: {filename} has no data")
            return False
        
        required_cols = ['date_time', 'open', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            if verbose:
                print(f"  Warning: {filename} missing columns: {missing_cols}")
            return False
        
        # Check for sufficient volume if volume column exists
        if 'volume' in df.columns:
            total_volume = df['volume'].sum()
            if total_volume < self.min_volume_threshold:
                if verbose:
                    print(f"  Warning: {filename} has insufficient volume: {total_volume}")
                return False
        
        return True
    
    def _should_roll(self, 
                    current_date: pd.Timestamp, 
                    sorted_contracts: List[str], 
                    current_contract_idx: int,
                    expirations_dict: Dict) -> bool:
        """
        Determine if we should roll to the next contract
        
        Rolling logic based on López de Prado's recommendations:
        - Roll before expiry to avoid liquidity issues
        - Ensure next contract is available
        
        :param current_date: Current date being processed
        :param sorted_contracts: List of contracts sorted by expiration
        :param current_contract_idx: Index of current active contract
        :param expirations_dict: Contract expiration dates
        :return: (bool) True if should roll to next contract
        """
        # Can't roll if we're on the last contract
        if current_contract_idx >= len(sorted_contracts) - 1:
            return False
            
        current_contract = sorted_contracts[current_contract_idx]
        expiry_date = pd.to_datetime(expirations_dict[current_contract])
        
        # Calculate days to expiry
        days_to_expiry = (expiry_date - current_date).days
        
        # Roll if within the specified number of days before expiry
        return days_to_expiry <= self.roll_days_before_expiry
    
    def _back_adjust_prices(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Apply back adjustment to prices when rolling
        
        Back adjustment maintains price differences but shifts historical prices.
        This method preserves absolute price changes while creating a continuous series.
        
        Formula: Adjusted_Price = Original_Price + Cumulative_Adjustment
        where Cumulative_Adjustment accumulates the price gaps at each roll
        
        :param df: (pd.DataFrame) Continuous series before adjustment
        :param verbose: (bool) Verbose output
        :return: (pd.DataFrame) Back-adjusted continuous series
        """
        df = df.copy()
        
        # Find roll points where contract changes
        roll_mask = df['contract_number'].diff() != 0
        roll_points = df[roll_mask].index[1:]  # Skip first point
        
        if len(roll_points) == 0:
            if verbose:
                print("  No roll points found - no adjustments needed")
            return df
        
        cumulative_adjustment = 0
        adjustments_applied = []
        
        price_cols = ['open', 'close']
        if 'high' in df.columns:
            price_cols.append('high')
        if 'low' in df.columns:
            price_cols.append('low')
        
        for roll_point in roll_points:
            # Get the price difference at roll point
            roll_idx = df.index.get_loc(roll_point)
            prev_idx = roll_idx - 1
            
            if prev_idx >= 0:
                prev_close = df.iloc[prev_idx]['close']
                new_open = df.iloc[roll_idx]['close']  # Use close price for consistency
                price_gap = prev_close - new_open
                cumulative_adjustment += price_gap
                
                # Apply adjustment to all data from this roll point forward
                future_mask = df.index >= roll_point
                for col in price_cols:
                    if col in df.columns:
                        df.loc[future_mask, col] += cumulative_adjustment
                
                adjustments_applied.append({
                    'date': roll_point,
                    'price_gap': price_gap,
                    'cumulative_adjustment': cumulative_adjustment
                })
                
                if verbose:
                    print(f"    Back adjustment at {roll_point.strftime('%Y-%m-%d')}: "
                          f"gap={price_gap:.4f}, cumulative={cumulative_adjustment:.4f}")
        
        if verbose and adjustments_applied:
            total_adjustment = adjustments_applied[-1]['cumulative_adjustment']
            print(f"  ✓ Back adjustment complete: total adjustment = {total_adjustment:.4f}")
        
        return df
    
    def _ratio_adjust_prices(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Apply ratio adjustment to prices when rolling
        
        Ratio adjustment maintains percentage returns by scaling prices.
        This method preserves relative price movements while creating continuity.
        
        Formula: Adjusted_Price = Original_Price * Cumulative_Ratio
        where Cumulative_Ratio accumulates the price ratios at each roll
        
        :param df: (pd.DataFrame) Continuous series before adjustment
        :param verbose: (bool) Verbose output  
        :return: (pd.DataFrame) Ratio-adjusted continuous series
        """
        df = df.copy()
        
        # Find roll points where contract changes
        roll_mask = df['contract_number'].diff() != 0
        roll_points = df[roll_mask].index[1:]  # Skip first point
        
        if len(roll_points) == 0:
            if verbose:
                print("  No roll points found - no adjustments needed")
            return df
        
        cumulative_ratio = 1.0
        adjustments_applied = []
        
        price_cols = ['open', 'close']
        if 'high' in df.columns:
            price_cols.append('high')
        if 'low' in df.columns:
            price_cols.append('low')
        
        for roll_point in roll_points:
            # Get the price ratio at roll point
            roll_idx = df.index.get_loc(roll_point)
            prev_idx = roll_idx - 1
            
            if prev_idx >= 0:
                prev_close = df.iloc[prev_idx]['close']
                new_open = df.iloc[roll_idx]['close']  # Use close price for consistency
                
                if new_open != 0:  # Avoid division by zero
                    price_ratio = prev_close / new_open
                    cumulative_ratio *= price_ratio
                    
                    # Apply ratio adjustment to all data from this roll point forward
                    future_mask = df.index >= roll_point
                    for col in price_cols:
                        if col in df.columns:
                            df.loc[future_mask, col] *= price_ratio
                    
                    adjustments_applied.append({
                        'date': roll_point,
                        'price_ratio': price_ratio,
                        'cumulative_ratio': cumulative_ratio
                    })
                    
                    if verbose:
                        print(f"    Ratio adjustment at {roll_point.strftime('%Y-%m-%d')}: "
                              f"ratio={price_ratio:.6f}, cumulative={cumulative_ratio:.6f}")
        
        if verbose and adjustments_applied:
            total_ratio = adjustments_applied[-1]['cumulative_ratio']
            print(f"  ✓ Ratio adjustment complete: total ratio = {total_ratio:.6f}")
        
        return df
    
    def get_roll_schedule(self, expirations_dict: Dict) -> pd.DataFrame:
        """
        Generate roll schedule showing when contracts will be rolled
        
        This creates a forward-looking schedule of all planned roll dates
        based on the expiration dates and roll parameters.
        
        :param expirations_dict: (dict) Contract expiration dates
        :return: (pd.DataFrame) Roll schedule with dates and contract transitions
        """
        schedule = []
        
        contracts = list(expirations_dict.keys())
        
        for i in range(len(contracts) - 1):
            current_contract = contracts[i]
            next_contract = contracts[i + 1]
            
            expiry_date = pd.to_datetime(expirations_dict[current_contract])
            roll_date = expiry_date - timedelta(days=self.roll_days_before_expiry)
            
            schedule.append({
                'roll_date': roll_date,
                'from_contract': current_contract,
                'to_contract': next_contract,
                'expiry_date': expiry_date,
                'days_before_expiry': self.roll_days_before_expiry
            })
        
        schedule_df = pd.DataFrame(schedule)
        if not schedule_df.empty:
            schedule_df = schedule_df.sort_values('roll_date')
        
        return schedule_df
    
    def analyze_roll_costs(self, 
                          tickers_df: pd.DataFrame, 
                          expirations_dict: Dict) -> pd.DataFrame:
        """
        Analyze the costs associated with rolling contracts
        
        This method calculates the spread and liquidity costs at roll dates
        to help evaluate the efficiency of the rolling strategy.
        
        :param tickers_df: Combined futures data
        :param expirations_dict: Contract expiration dates  
        :return: DataFrame with roll cost analysis
        """
        roll_schedule = self.get_roll_schedule(expirations_dict)
        
        if roll_schedule.empty:
            return pd.DataFrame()
        
        roll_costs = []
        
        for _, roll in roll_schedule.iterrows():
            roll_date = roll['roll_date']
            from_contract = roll['from_contract']
            to_contract = roll['to_contract']
            
            # Get prices on roll date
            from_data = tickers_df[
                (tickers_df.index == roll_date) & 
                (tickers_df['ticker'] == from_contract)
            ]
            
            to_data = tickers_df[
                (tickers_df.index == roll_date) & 
                (tickers_df['ticker'] == to_contract)
            ]
            
            if not from_data.empty and not to_data.empty:
                from_price = from_data['close'].iloc[0]
                to_price = to_data['close'].iloc[0]
                
                # Calculate roll cost metrics
                price_spread = abs(from_price - to_price)
                price_spread_pct = (price_spread / from_price) * 100 if from_price != 0 else 0
                
                roll_costs.append({
                    'roll_date': roll_date,
                    'from_contract': from_contract,
                    'to_contract': to_contract,
                    'from_price': from_price,
                    'to_price': to_price,
                    'price_spread': price_spread,
                    'price_spread_pct': price_spread_pct
                })
        
        return pd.DataFrame(roll_costs)


# Convenience functions for easy usage
def prepare_vix_futures_dataset(data_path: str = 'data/', **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to prepare VIX futures dataset
    
    :param data_path: Path to VIX futures data files
    :param kwargs: Additional parameters for FuturesRoll
    :return: Combined dataframe and expirations dictionary
    """
    roller = FuturesRoll(**kwargs)
    return roller.prepare_futures_dataset(data_path, contract_pattern='VIX')


def create_continuous_vix_series(data_path: str = 'data/', 
                                roll_method: str = 'back_adjust',
                                verbose: bool = True,
                                **kwargs) -> pd.DataFrame:
    """
    Create continuous VIX futures series with single function call
    
    :param data_path: Path to VIX futures data files
    :param roll_method: Price adjustment method ('back_adjust', 'ratio_adjust', 'no_adjust')
    :param verbose: Verbose output
    :param kwargs: Additional parameters for FuturesRoll
    :return: Continuous VIX futures series
    """
    roller = FuturesRoll(roll_method=roll_method, **kwargs)
    tickers_df, expirations_dict = roller.prepare_futures_dataset(
        data_path, contract_pattern='VIX', verbose=verbose
    )
    return roller.create_continuous_series(tickers_df, expirations_dict, verbose=verbose)


def create_continuous_futures_series(data_path: str = 'data/',
                                    contract_pattern: str = 'ES',
                                    roll_method: str = 'back_adjust', 
                                    **kwargs) -> pd.DataFrame:
    """
    Generic function to create continuous futures series for any contract
    
    :param data_path: Path to futures data files
    :param contract_pattern: Pattern to match contract files (e.g., 'ES', 'CL', 'GC')
    :param roll_method: Price adjustment method
    :param kwargs: Additional parameters for FuturesRoll
    :return: Continuous futures series
    """
    roller = FuturesRoll(roll_method=roll_method, **kwargs)
    tickers_df, expirations_dict = roller.prepare_futures_dataset(
        data_path, contract_pattern=contract_pattern
    )
    return roller.create_continuous_series(tickers_df, expirations_dict)


# Main execution
if __name__ == "__main__":
    print("Futures Roll Module loaded successfully!")
    print("Use help(FuturesRoll) for detailed documentation.")
