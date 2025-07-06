"""
ETF Trick Implementation
Based on Advances in Financial Machine Learning, López de Prado

This module contains class for ETF trick generation and futures roll function, described in Marcos Lopez de Prado's
book 'Advances in Financial Machine Learning'. ETF trick class can generate ETF trick series either from .csv files
or from in memory pandas DataFrames.

The ETF trick is used to create synthetic price series from portfolio allocations, allowing for:
- Portfolio performance tracking
- Risk attribution analysis  
- Strategy backtesting with transaction costs
- Multi-asset portfolio construction

Key concepts:
- K: Portfolio value at time t
- w(t): Asset allocations (number of contracts)
- o(t): Open prices
- p(t): Close prices  
- d(t): Costs (rebalancing, carry, dividends)
- φ(t): Rate multipliers (dollar value per point)

Author: Sistema de Datos - Quant Analysis
Date: July 2025
"""

import warnings
import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any
import os


class ETFTrick:
    """
    Contains logic of vectorised ETF trick implementation.
    
    The ETF trick synthesizes a price series from portfolio allocations using the formula:
    K(t) = K(t-1) + Σ[w(t-1) * φ(t-1) * (o(t) - p(t-1)) + w(t) * φ(t) * (p(t) - o(t)) - d(t)]
    
    Where:
    - K(t): Portfolio value at time t
    - w(t): Asset allocations (number of contracts) at time t
    - o(t): Open prices at time t  
    - p(t): Close prices at time t
    - d(t): Transaction costs at time t
    - φ(t): Rate multipliers at time t
    
    Can be used for both memory data frames (pd.DataFrame) and csv files.
    All data frames/files should have the same date index and asset columns.
    
    Example usage:
        # In-memory usage
        etf = ETFTrick(open_df, close_df, alloc_df, costs_df, rates_df)
        etf_series = etf.get_etf_series()
        
        # CSV-based usage for large datasets
        etf = ETFTrick('open.csv', 'close.csv', 'alloc.csv', 'costs.csv', 'rates.csv')
        etf_series = etf.get_etf_series(batch_size=50000)
    """

    def __init__(self, 
                 open_df: Union[pd.DataFrame, str], 
                 close_df: Union[pd.DataFrame, str], 
                 alloc_df: Union[pd.DataFrame, str], 
                 costs_df: Union[pd.DataFrame, str], 
                 rates_df: Optional[Union[pd.DataFrame, str]] = None, 
                 index_col: int = 0):
        """
        Constructor

        Creates class object, for csv based files reads the first data chunk.

        :param open_df: (pd.DataFrame or str): open prices data frame or path to csv file,
         corresponds to o(t) from the book
        :param close_df: (pd.DataFrame or str): close prices data frame or path to csv file, corresponds to p(t)
        :param alloc_df: (pd.DataFrame or str): asset allocations data frame or path to csv file (in # of contracts),
         corresponds to w(t)
        :param costs_df: (pd.DataFrame or str): rebalance, carry and dividend costs of holding/rebalancing the
         position, corresponds to d(t)
        :param rates_df: (pd.DataFrame or str): dollar value of one point move of contract includes exchange rate,
         futures contracts multiplies). Corresponds to phi(t)
         For example, 1$ in VIX index, equals 1000$ in VIX futures contract value.
         If None then trivial (all values equal 1.0) is generated
        :param index_col: (int): positional index of index column. Used for to determine index column in csv files
        """
        
        # Store input parameters
        self.index_col = index_col
        self.is_csv_based = isinstance(open_df, str)
        
        # Initialize data containers
        self.data_dict = {}
        self.file_iterators = {}
        self.cache = {}
        self.current_k = 1.0  # Starting portfolio value
        
        if self.is_csv_based:
            # Store file paths
            self.file_paths = {
                'open': open_df,
                'close': close_df, 
                'alloc': alloc_df,
                'costs': costs_df,
                'rates': rates_df
            }
            
            # Validate files exist
            for name, path in self.file_paths.items():
                if path is not None and not os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")
            
            # Initialize file iterators
            self._init_csv_iterators()
        else:
            # Store DataFrames directly
            self.data_dict = {
                'open': open_df.copy(),
                'close': close_df.copy(),
                'alloc': alloc_df.copy(), 
                'costs': costs_df.copy(),
                'rates': rates_df.copy() if rates_df is not None else self._generate_trivial_rates(open_df)
            }
            
            # Validate data consistency
            self._validate_data_consistency()
            
        # Perform index alignment check
        self._index_check()
    
    def _init_csv_iterators(self):
        """Initialize CSV file iterators for batch processing"""
        for name, path in self.file_paths.items():
            if path is not None:
                self.file_iterators[name] = pd.read_csv(path, chunksize=None, index_col=self.index_col)
    
    def _generate_trivial_rates(self, reference_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trivial rates DataFrame (all values = 1.0) when rates_df is None
        
        :param reference_df: Reference DataFrame to copy structure from
        :return: DataFrame with all values equal to 1.0
        """
        rates_df = reference_df.copy()
        rates_df[:] = 1.0
        return rates_df
    
    def _validate_data_consistency(self):
        """Validate that all DataFrames have consistent structure"""
        if not self.is_csv_based:
            # Check that all DataFrames have same shape
            shapes = {name: df.shape for name, df in self.data_dict.items() if df is not None}
            first_shape = next(iter(shapes.values()))
            
            for name, shape in shapes.items():
                if shape != first_shape:
                    raise ValueError(f"DataFrame {name} has shape {shape}, expected {first_shape}")
            
            # Check that all DataFrames have same columns
            columns = {name: list(df.columns) for name, df in self.data_dict.items() if df is not None}
            first_columns = next(iter(columns.values()))
            
            for name, cols in columns.items():
                if cols != first_columns:
                    raise ValueError(f"DataFrame {name} has different columns than expected")

    def _append_previous_rows(self, cache: Dict[str, pd.DataFrame]):
        """
        Uses latest two rows from cache to append into current data. Used for csv based ETF trick, when the next
        batch is loaded and we need to recalculate K value which corresponds to previous batch.

        :param cache: (dict): dictionary which pd.DataFrames with latest 2 rows of open, close, alloc, costs, rates
        :return: (pd.DataFrame): data frame with close price differences (updates self.data_dict)
        """
        if not cache:
            return
            
        for name, cached_data in cache.items():
            if name in self.data_dict and cached_data is not None:
                # Prepend cache to current data
                self.data_dict[name] = pd.concat([cached_data, self.data_dict[name]])
                
        # Recalculate price differences with updated data
        self._calculate_price_differences()

    def generate_trick_components(self, cache: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculates all etf trick operations which can be vectorised. Outputs multilevel pandas data frame.

        Generated components:
        'w': alloc_df
        'h_t': h_t/K value from ETF trick algorithm from the book. Which K to use is based on previous values and
            cannot be vectorised.
        'close_open': close_df - open_df
        'price_diff': close price differences
        'costs': costs_df
        'rate': rates_df

        :param cache: (dict of pd.DataFrames): dictionary which contains latest 2 rows of open, close, rates, alloc,
            costs, rates data
        :return: (pd.DataFrame): pandas data frame with columns in a format: component_1/asset_name_1,
            component_1/asset_name_2, ..., component_6/asset_name_n
        """
        
        # Append previous rows if cache exists
        if cache:
            self._append_previous_rows(cache)
        
        # Calculate vectorized components
        components = {}
        
        # Component 1: Allocations (w)
        components['w'] = self.data_dict['alloc']
        
        # Component 2: h_t (placeholder, will be calculated iteratively)
        components['h_t'] = pd.DataFrame(
            index=self.data_dict['alloc'].index,
            columns=self.data_dict['alloc'].columns
        )
        
        # Component 3: Close - Open price differences
        components['close_open'] = self.data_dict['close'] - self.data_dict['open']
        
        # Component 4: Price differences (p(t) - p(t-1))
        components['price_diff'] = self.data_dict['close'].diff()
        
        # Component 5: Costs
        components['costs'] = self.data_dict['costs']
        
        # Component 6: Rates
        components['rate'] = self.data_dict['rates']
        
        # Create multi-level DataFrame
        result_data = {}
        for comp_name, comp_df in components.items():
            for col in comp_df.columns:
                result_data[f"{comp_name}/{col}"] = comp_df[col]
        
        return pd.DataFrame(result_data, index=components['w'].index)

    def _update_cache(self) -> Dict[str, pd.DataFrame]:
        """
        Updates cache (two previous rows) when new data batch is read into the memory. Cache is used to
        recalculate ETF trick value which corresponds to previous batch last row. That is why we need 2 previous rows
        for close price difference calculation

        :return: (dict): dictionary with open, close, alloc, costs and rates last 2 rows
        """
        cache = {}
        
        for name, df in self.data_dict.items():
            if df is not None and len(df) >= 2:
                # Store last 2 rows
                cache[name] = df.tail(2).copy()
            elif df is not None and len(df) == 1:
                # Store only available row
                cache[name] = df.tail(1).copy()
            else:
                cache[name] = None
                
        return cache

    def _chunk_loop(self, data_df: pd.DataFrame) -> pd.Series:
        """
        Single ETF trick iteration for currently stored(with needed components) data set in memory (data_df).
        For in-memory data set would yield complete ETF trick series, for csv based
        would generate ETF trick series for current batch.

        :param data_df: The data set on which to apply the ETF trick.
        :return: (pd.Series): pandas Series with ETF trick values
        """
        
        etf_values = []
        current_k = self.current_k
        
        # Get component DataFrames
        asset_cols = [col for col in self.data_dict['alloc'].columns]
        
        for i, (idx, row) in enumerate(data_df.iterrows()):
            if i == 0:
                # First row: initialize with current K value
                etf_values.append(current_k)
                continue
            
            # Calculate ETF trick for current row
            prev_idx = data_df.index[i-1]
            
            # Extract components for current and previous periods
            w_prev = self.data_dict['alloc'].loc[prev_idx]  # w(t-1)
            w_curr = self.data_dict['alloc'].loc[idx]       # w(t)
            
            o_curr = self.data_dict['open'].loc[idx]        # o(t)
            p_prev = self.data_dict['close'].loc[prev_idx]  # p(t-1)
            p_curr = self.data_dict['close'].loc[idx]       # p(t)
            
            phi_prev = self.data_dict['rates'].loc[prev_idx] # φ(t-1)
            phi_curr = self.data_dict['rates'].loc[idx]      # φ(t)
            
            d_curr = self.data_dict['costs'].loc[idx]        # d(t)
            
            # Calculate ETF trick components
            # Term 1: w(t-1) * φ(t-1) * (o(t) - p(t-1))
            term1 = (w_prev * phi_prev * (o_curr - p_prev)).sum()
            
            # Term 2: w(t) * φ(t) * (p(t) - o(t))  
            term2 = (w_curr * phi_curr * (p_curr - o_curr)).sum()
            
            # Term 3: -d(t) (costs)
            term3 = -d_curr.sum()
            
            # Calculate new K value
            current_k = current_k + term1 + term2 + term3
            etf_values.append(current_k)
        
        # Update current K for next batch
        self.current_k = current_k
        
        return pd.Series(etf_values, index=data_df.index, name='etf_trick')

    def _index_check(self):
        """
        Internal check for all price, rates and allocations data frames have the same index
        """
        if not self.is_csv_based:
            indices = {name: df.index for name, df in self.data_dict.items() if df is not None}
            
            if len(indices) > 1:
                first_index = next(iter(indices.values()))
                for name, index in indices.items():
                    if not index.equals(first_index):
                        warnings.warn(f"Index mismatch in DataFrame {name}")

    def _get_batch_from_csv(self, batch_size: int):
        """
        Reads the next batch of data sets from csv files and puts them in class variable data_dict

        :param batch_size: number of rows to read
        """
        try:
            self.data_dict = {}
            
            for name, path in self.file_paths.items():
                if path is not None:
                    # Read batch from CSV
                    df = pd.read_csv(path, 
                                   index_col=self.index_col, 
                                   nrows=batch_size,
                                   skiprows=getattr(self, f'_rows_read_{name}', 0))
                    
                    self.data_dict[name] = df
                    
                    # Track rows read
                    setattr(self, f'_rows_read_{name}', 
                           getattr(self, f'_rows_read_{name}', 0) + len(df))
                else:
                    # Generate trivial rates if not provided
                    if name == 'rates' and 'open' in self.data_dict:
                        self.data_dict[name] = self._generate_trivial_rates(self.data_dict['open'])
                    else:
                        self.data_dict[name] = None
                        
        except Exception as e:
            raise RuntimeError(f"Error reading batch from CSV files: {str(e)}")

    def _calculate_price_differences(self):
        """Calculate price differences for current data"""
        if 'close' in self.data_dict and self.data_dict['close'] is not None:
            # Will be calculated in generate_trick_components
            pass

    def _rewind_etf_trick(self, alloc_df: pd.DataFrame, etf_series: pd.Series):
        """
        ETF trick uses next open price information, when we process csv file in batches the last row in batch will have
        next open price value as nan, that is why when new batch comes, we need to rewind ETF trick values one step
        back, recalculate ETF trick value for the last row from previous batch using open price from latest batch
        received. This function rewinds values needed for ETF trick calculation recalculate

        :param alloc_df: (pd.DataFrame): data frame with allocations vectors
        :param etf_series (pd.Series): current computed ETF trick series
        """
        
        if len(etf_series) < 2:
            return etf_series
            
        # Get the last valid ETF value before potential NaN
        last_valid_idx = etf_series.dropna().index[-1]
        
        # Set current K to the last valid value for continuation
        self.current_k = etf_series.loc[last_valid_idx]
        
        return etf_series

    def _csv_file_etf_series(self, batch_size: int) -> pd.Series:
        """
        Csv based ETF trick series generation

        :param: batch_size: (int): Size of the batch that you would like to make use of
        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0
        """
        
        etf_series_parts = []
        cache = {}
        batch_count = 0
        
        # Initialize row counters
        for name in self.file_paths.keys():
            setattr(self, f'_rows_read_{name}', 0)
        
        while True:
            try:
                # Read next batch
                self._get_batch_from_csv(batch_size)
                
                # Check if we have data
                if any(df is None or df.empty for df in self.data_dict.values() if df is not None):
                    break
                
                # Generate components
                components_df = self.generate_trick_components(cache if batch_count > 0 else None)
                
                # Calculate ETF series for this batch
                batch_etf_series = self._chunk_loop(components_df)
                
                # Handle overlapping data from cache
                if batch_count > 0 and len(etf_series_parts) > 0:
                    # Remove overlap (first row of current batch was last row of previous batch)
                    batch_etf_series = batch_etf_series.iloc[1:]
                
                etf_series_parts.append(batch_etf_series)
                
                # Update cache for next iteration
                cache = self._update_cache()
                batch_count += 1
                
            except Exception as e:
                if "no more data" in str(e).lower() or len(self.data_dict.get('open', pd.DataFrame())) == 0:
                    break
                else:
                    raise e
        
        # Combine all parts
        if etf_series_parts:
            return pd.concat(etf_series_parts)
        else:
            return pd.Series(dtype=float, name='etf_trick')

    def _in_memory_etf_series(self) -> pd.Series:
        """
        In-memory based ETF trick series generation.

        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0
        """
        
        # Generate all components
        components_df = self.generate_trick_components()
        
        # Calculate ETF series
        return self._chunk_loop(components_df)

    def get_etf_series(self, batch_size: int = int(1e5)) -> pd.Series:
        """
        External method which defines which etf trick method to use.

        :param: batch_size: Size of the batch that you would like to make use of
        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0
        """
        
        if self.is_csv_based:
            return self._csv_file_etf_series(batch_size)
        else:
            return self._in_memory_etf_series()

    def reset(self):
        """
        Re-inits class object. This methods can be used to reset file iterators for multiple get_etf_trick() calls.
        """
        
        # Reset current K value
        self.current_k = 1.0
        
        # Reset cache
        self.cache = {}
        
        if self.is_csv_based:
            # Reset row counters
            for name in self.file_paths.keys():
                if hasattr(self, f'_rows_read_{name}'):
                    setattr(self, f'_rows_read_{name}', 0)
                    
            # Re-initialize iterators
            self._init_csv_iterators()
        
    def get_portfolio_statistics(self, etf_series: pd.Series) -> Dict[str, float]:
        """
        Calculate portfolio statistics from ETF series
        
        :param etf_series: ETF trick series
        :return: Dictionary with portfolio statistics
        """
        
        if etf_series.empty:
            return {}
        
        # Calculate returns
        returns = etf_series.pct_change().dropna()
        
        stats = {
            'total_return': (etf_series.iloc[-1] / etf_series.iloc[0]) - 1,
            'annualized_return': returns.mean() * 252,  # Assuming daily data
            'annualized_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(etf_series),
            'final_value': etf_series.iloc[-1],
            'num_observations': len(etf_series)
        }
        
        return stats
    
    def _calculate_max_drawdown(self, series: pd.Series) -> float:
        """Calculate maximum drawdown of the series"""
        running_max = series.expanding().max()
        drawdown = (series - running_max) / running_max
        return drawdown.min()


# Note: Futures rolling functionality has been moved to futures_roll.py
# Use FuturesRoll.create_continuous_series() instead


# Convenience functions
def create_etf_trick_from_csv(open_path: str, 
                             close_path: str, 
                             alloc_path: str, 
                             costs_path: str,
                             rates_path: Optional[str] = None,
                             batch_size: int = int(1e5)) -> pd.Series:
    """
    Convenience function to create ETF trick series from CSV files
    
    :param open_path: Path to open prices CSV
    :param close_path: Path to close prices CSV  
    :param alloc_path: Path to allocations CSV
    :param costs_path: Path to costs CSV
    :param rates_path: Path to rates CSV (optional)
    :param batch_size: Batch size for processing
    :return: ETF trick series
    """
    
    etf = ETFTrick(open_path, close_path, alloc_path, costs_path, rates_path)
    return etf.get_etf_series(batch_size)


def create_etf_trick_from_dataframes(open_df: pd.DataFrame,
                                   close_df: pd.DataFrame,
                                   alloc_df: pd.DataFrame, 
                                   costs_df: pd.DataFrame,
                                   rates_df: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Convenience function to create ETF trick series from DataFrames
    
    :param open_df: Open prices DataFrame
    :param close_df: Close prices DataFrame
    :param alloc_df: Allocations DataFrame
    :param costs_df: Costs DataFrame  
    :param rates_df: Rates DataFrame (optional)
    :return: ETF trick series
    """
    
    etf = ETFTrick(open_df, close_df, alloc_df, costs_df, rates_df)
    return etf.get_etf_series()


# Example usage and testing
if __name__ == "__main__":
    print("ETF Trick Module loaded successfully!")
    print("Use help(ETFTrick) for detailed documentation.")
