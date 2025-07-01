"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

Time bars generation logic
"""

# Imports
from typing import Union, Iterable, Optional
import numpy as np
import pandas as pd

try:
    from .base_bars import BaseBars
    from ..util.volume_classifier import get_tick_rule_buy_volume
except ImportError:
    try:
        from base_bars import BaseBars
    except ImportError:
        import sys
        import os
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        from base_bars import BaseBars
    
    # Fallback if util not available
    def get_tick_rule_buy_volume(close, volume):
        return volume * 0.5


class TimeBars(BaseBars):
    """
    Contains all of the logic to construct the time bars. This class shouldn't be used directly.
    Use get_time_bars instead
    """

    def __init__(self, resolution: str, num_units: int, batch_size: int = 20000000):
        """
        Constructor

        :param resolution: (str) Type of bar resolution: ['D', 'H', 'MIN', 'S']
        :param num_units: (int) Number of days, minutes, etc.
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        """
        super().__init__(metric='time_bars', batch_size=batch_size)
        
        self.resolution = resolution.upper()
        self.num_units = num_units
        
        # Time grouping parameters
        self.current_bar_start = None
        self.freq_string = self._get_freq_string()
        
        # Tick rule memory
        self.last_tick_direction = 1

    def _get_freq_string(self) -> str:
        """
        Convert resolution and num_units to pandas frequency string
        """
        freq_map = {
            'D': 'D',      # Daily
            'H': 'H',      # Hourly  
            'MIN': 'min',  # Minutes
            'S': 'S'       # Seconds
        }
        
        if self.resolution not in freq_map:
            raise ValueError(f"Resolution must be one of {list(freq_map.keys())}")
            
        return f"{self.num_units}{freq_map[self.resolution]}"

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for time bars
        """
        super()._reset_cache()
        self.current_bar_start = None

    def _extract_bars(self, data: pd.DataFrame) -> list:
        """
        For loop which compiles time bars.
        We did investigate the use of trying to solve this in a vectorised manner but found that a For loop worked well.

        :param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (list) Extracted bars
        """
        
        # Standardize column names
        if len(data.columns) >= 3:
            data.columns = ['date_time', 'price', 'volume'] + list(data.columns[3:])
        else:
            raise ValueError("Data must have at least 3 columns: date_time, price, volume")
            
        # Convert date_time to datetime if it's not already
        data['date_time'] = pd.to_datetime(data['date_time'])
        
        # Sort by timestamp
        data = data.sort_values('date_time').reset_index(drop=True)
        
        list_bars = []
        
        for _, row in data.iterrows():
            date_time = row['date_time']
            price = float(row['price'])
            volume = float(row['volume']) if not pd.isna(row['volume']) else 1.0
            
            # Determine the bar period this tick belongs to
            bar_start = self._get_bar_period(date_time)
            
            # Check if we need to finish the current bar
            if (self.current_bar_start is not None and 
                bar_start != self.current_bar_start and 
                self.cum_statistics['cum_ticks'] > 0):
                
                # Create the completed bar
                bar = self._create_bar(
                    date_time=self.current_bar_start + pd.Timedelta(self.freq_string) - pd.Timedelta(microseconds=1),
                    price=self.close_price,
                    high_price=self.high_price,
                    low_price=self.low_price,
                    open_price=self.open_price
                )
                
                list_bars.append(bar)
                self._reset_cache()
            
            # Start new bar if needed
            if self.current_bar_start is None:
                self.current_bar_start = bar_start
                self.open_price = price
                
            # Update bar data
            self._update_high_low(price)
            self.close_price = price
            
            # Apply tick rule for buy/sell classification
            if self.prev_price is not None:
                tick_direction = self._apply_tick_rule(price)
            else:
                tick_direction = 1
                
            self.prev_price = price
            
            # Update cumulative statistics
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_volume'] += volume
            dollar_value = price * volume
            self.cum_statistics['cum_dollar_value'] += dollar_value
            
            if tick_direction == 1:
                self.cum_statistics['cum_buy_volume'] += volume
                
        return list_bars
        
    def _get_bar_period(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        """
        Get the start of the bar period for a given timestamp
        
        :param timestamp: (pd.Timestamp) Current timestamp
        :return: (pd.Timestamp) Start of bar period
        """
        if self.resolution == 'D':
            return timestamp.normalize()
        elif self.resolution == 'H':
            return timestamp.floor(f'{self.num_units}H')
        elif self.resolution == 'MIN':
            return timestamp.floor(f'{self.num_units}min')
        elif self.resolution == 'S':
            return timestamp.floor(f'{self.num_units}S')
        else:
            raise ValueError(f"Unsupported resolution: {self.resolution}")
            
    def _apply_tick_rule(self, price: float) -> int:
        """
        Applies the tick rule as defined on page 29-30 of Advances in Financial Machine Learning.
        
        The tick rule classifies trades as buyer-initiated (1) or seller-initiated (-1).
        When price doesn't change, we use the last known tick direction (memory).
        
        :param price: (float) Current price
        :return: (int) 1 if uptick, -1 if downtick, maintains direction if no change
        """
        if self.prev_price is None:
            # Initialize with positive tick for first trade
            self.last_tick_direction = 1
            return 1
        elif price > self.prev_price:
            # Uptick: buyer-initiated
            self.last_tick_direction = 1
            return 1
        elif price < self.prev_price:
            # Downtick: seller-initiated
            self.last_tick_direction = -1
            return -1
        else:
            # No price change: use last known direction (López de Prado, page 29)
            return getattr(self, 'last_tick_direction', 1)


def get_time_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                  resolution: str = 'MIN', num_units: int = 1, 
                  batch_size: int = 20000000, verbose: bool = True, 
                  to_csv: bool = False, output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
    """
    Creates time bars: aggregates transactions into time-based bars.
    
    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing
                            raw tick data in the format[date_time, price, volume]
    :param resolution: (str) Type of bar resolution: ['D', 'H', 'MIN', 'S'] for Daily, Hourly, Minutes, Seconds
    :param num_units: (int) Number of resolution units (e.g., 5 for 5-minute bars)
    :param batch_size: (int) Number of rows to read in from the csv, per batch
    :param verbose: (bool) Flag whether to print message on each processed batch or not
    :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
    :param output_path: (str) Path to results file, if to_csv = True
    :return: (pd.DataFrame or None) Time bars
    """
    
    bars = TimeBars(resolution=resolution, num_units=num_units, batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
