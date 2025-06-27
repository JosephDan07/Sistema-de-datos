"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of time, tick, volume, and dollar bars.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018,
pg 25) to build the more interesting features for predicting financial time series data.

These financial data structures have better statistical properties when compared to those based on fixed time interval
sampling. A great paper to read more about this is titled: The Volume Clock: Insights into the high frequency paradigm,
Lopez de Prado, et al.

Many of the projects going forward will require Dollar and Volume bars.
"""

# Imports
from typing import Union, Iterable, Optional

import numpy as np
import pandas as pd

from .base_bars import BaseBars


class StandardBars(BaseBars):
    """
    Contains all of the logic to construct the standard bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_bars which will create an instance of this
    class and then construct the standard bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, metric: str, threshold: int = 50000, batch_size: int = 20000000):
        """
        Constructor

        :param metric: (str) Type of bar to create. Example: "tick_bars", "volume_bars", "dollar_bars"
        :param threshold: (int) Threshold at which to sample
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        """
        super().__init__(metric, batch_size)
        
        # Threshold for sampling
        self.threshold = threshold
        
        # Current cumulative value being tracked
        self.cum_value = 0

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for standard bars
        """
        super()._reset_cache()
        self.cum_value = 0

    def _extract_bars(self, data: pd.DataFrame) -> list:
        """
        For loop which compiles the various bars: tick, volume, or dollar.
        We did investigate the use of trying to solve this in a vectorised manner but found that a For loop worked well.

        :param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (list) Extracted bars
        """
        
        # Standardize column names
        if len(data.columns) >= 3:
            data.columns = ['date_time', 'price', 'volume'] + list(data.columns[3:])
        else:
            raise ValueError("Data must have at least 3 columns: date_time, price, volume")
            
        list_bars = []
        
        for row in data.values:
            # Extract row data
            date_time = row[0]
            price = float(row[1])
            volume = float(row[2]) if len(row) > 2 else 1.0
            
            # Update high and low
            self._update_high_low(price)
            
            # Set open price for new bar
            if self.open_price is None:
                self.open_price = price
                
            # Set previous price for tick classification
            if self.prev_price is not None:
                tick_direction = self._apply_tick_rule(price)
            else:
                tick_direction = 1
                
            self.prev_price = price
            self.close_price = price
            
            # Update cumulative statistics
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_volume'] += volume
            dollar_value = price * volume
            self.cum_statistics['cum_dollar_value'] += dollar_value
            
            if tick_direction == 1:
                self.cum_statistics['cum_buy_volume'] += volume
                
            # Update the specific cumulative value based on bar type
            if self.metric == 'tick_bars':
                self.cum_value += 1
            elif self.metric == 'volume_bars':
                self.cum_value += volume
            elif self.metric == 'dollar_bars':
                self.cum_value += dollar_value
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
                
            # Check if we should create a bar
            if self.cum_value >= self.threshold:
                # Create bar
                bar = self._create_bar(
                    date_time=date_time,
                    price=self.close_price,
                    high_price=self.high_price,
                    low_price=self.low_price,
                    open_price=self.open_price
                )
                
                list_bars.append(bar)
                
                # Reset for next bar
                self._reset_cache()
                
        return list_bars
        
    def _apply_tick_rule(self, price: float) -> int:
        """
        Applies the tick rule as defined on page 29.
        
        :param price: (float) Current price
        :return: (int) 1 if uptick, -1 if downtick, 0 if no change
        """
        if price > self.prev_price:
            return 1
        elif price < self.prev_price:
            return -1
        else:
            return 0


def get_tick_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                  threshold: int = 1000, batch_size: int = 20000000, 
                  verbose: bool = True, to_csv: bool = False, 
                  output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
    """
    Creates tick bars: aggregates transactions into bars by number of transactions.
    
    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing
                            raw tick data in the format[date_time, price, volume]
    :param threshold: (int) Number of ticks to aggregate into one bar
    :param batch_size: (int) Number of rows to read in from the csv, per batch
    :param verbose: (bool) Flag whether to print message on each processed batch or not
    :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
    :param output_path: (str) Path to results file, if to_csv = True
    :return: (pd.DataFrame or None) Tick bars
    """
    
    bars = StandardBars(metric='tick_bars', threshold=threshold, batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)


def get_volume_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                    threshold: int = 1000, batch_size: int = 20000000, 
                    verbose: bool = True, to_csv: bool = False, 
                    output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
    """
    Creates volume bars: aggregates transactions into bars by cumulative volume.
    
    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing
                            raw tick data in the format[date_time, price, volume]
    :param threshold: (int) Volume to aggregate into one bar
    :param batch_size: (int) Number of rows to read in from the csv, per batch
    :param verbose: (bool) Flag whether to print message on each processed batch or not
    :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
    :param output_path: (str) Path to results file, if to_csv = True
    :return: (pd.DataFrame or None) Volume bars
    """
    
    bars = StandardBars(metric='volume_bars', threshold=threshold, batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)


def get_dollar_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                    threshold: int = 100000, batch_size: int = 20000000, 
                    verbose: bool = True, to_csv: bool = False, 
                    output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
    """
    Creates dollar bars: aggregates transactions into bars by cumulative dollar volume.
    
    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing
                            raw tick data in the format[date_time, price, volume]
    :param threshold: (int) Dollar volume to aggregate into one bar
    :param batch_size: (int) Number of rows to read in from the csv, per batch
    :param verbose: (bool) Flag whether to print message on each processed batch or not
    :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
    :param output_path: (str) Path to results file, if to_csv = True
    :return: (pd.DataFrame or None) Dollar bars
    """
    
    bars = StandardBars(metric='dollar_bars', threshold=threshold, batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
