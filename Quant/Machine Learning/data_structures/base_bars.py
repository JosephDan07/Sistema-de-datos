"""
A base class for the various bar types. Includes the logic shared between classes, to minimise the amount of
duplicated code.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, Generator, Iterable, Optional, List
import warnings

import numpy as np
import pandas as pd


def _crop_data_frame_in_batches(df: pd.DataFrame, chunksize: int) -> list:
    """
    Splits df into chunks of chunksize

    :param df: (pd.DataFrame) Dataframe to split
    :param chunksize: (int) Number of rows in chunk
    :return: (list) Chunks (pd.DataFrames)
    """
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    
    chunks = []
    for start in range(0, len(df), chunksize):
        end = min(start + chunksize, len(df))
        chunks.append(df.iloc[start:end].copy())
    
    return chunks


def ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exponentially weighted moving average
    
    :param values: (np.ndarray) Values to calculate EWMA for
    :param alpha: (float) Smoothing factor between 0 and 1
    :return: (np.ndarray) EWMA values
    """
    if alpha <= 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    result = np.empty_like(values, dtype=np.float64)
    result[0] = values[0]
    
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
    
    return result


class BaseBars(ABC):
    """
    Abstract base class which contains the structure which is shared between the various standard and information
    driven bars. There are some methods contained in here that would only be applicable to information bars but
    they are included here so as to avoid a complicated nested class structure.
    """

    def __init__(self, metric: str, batch_size: int = 20000000):
        """
        Constructor

        :param metric: (str) Type of bar to create. Example: dollar_bars, tick_imbalance, etc.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        """
        # Base properties
        self.metric = metric
        self.batch_size = int(batch_size)
        
        # Bar construction
        self.open_price = None
        self.prev_price = None
        self.high_price = None
        self.low_price = None
        self.close_price = None
        
        # Caches for bar construction
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}
        
        # Information-driven bar caches
        self.imbalance_array = []
        self.expected_imbalance_window = []
        self.imbalance_tick_statistics = {'num_ticks_bar': []}
        
        # Run bar caches  
        self.runs_array = []
        self.expected_runs_window = []
        self.run_tick_statistics = {'num_ticks_bar': []}

    def _reset_cache(self):
        """
        Reset the cache elements
        """
        self.open_price = None
        self.prev_price = None  
        self.high_price = None
        self.low_price = None
        self.close_price = None
        
        # Reset statistics
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}

    def batch_run(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame], verbose: bool = True, 
                  to_csv: bool = False, output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
        """
        Reads csv file(s) or pd.DataFrame in batches and then constructs the financial data structure in the form of a DataFrame.
        The csv file or DataFrame must have only 3 columns: date_time, price, & volume.

        :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing
                                raw tick data  in the format[date_time, price, volume]
        :param verbose: (bool) Flag whether to print message on each processed batch or not
        :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
        :param output_path: (str) Path to results file, if to_csv = True

        :return: (pd.DataFrame or None) Financial data structure
        """
        
        if to_csv and output_path is None:
            raise ValueError("output_path must be specified when to_csv=True")
            
        list_bars = []
        
        # Process data in batches
        for batch in self._batch_iterator(file_path_or_df):
            if verbose:
                print(f'Processing batch with {len(batch)} rows...')
                
            # Extract bars from current batch
            list_bars_batch = self._extract_bars(batch)
            list_bars.extend(list_bars_batch)
            
            if verbose and list_bars_batch:
                print(f'Generated {len(list_bars_batch)} bars from batch')
        
        if not list_bars:
            warnings.warn("No bars were generated. Check your data format and thresholds.", UserWarning)
            return pd.DataFrame()
            
        # Convert to DataFrame
        bars_df = pd.DataFrame(list_bars)
        
        if to_csv:
            bars_df.to_csv(output_path, index=False)
            if verbose:
                print(f'Bars saved to {output_path}')
            return None
        else:
            if verbose:
                print(f'Generated {len(bars_df)} total bars')
            return bars_df

    def _batch_iterator(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame]) -> Generator[pd.DataFrame, None, None]:
        """
        :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame
                                containing raw tick data in the format[date_time, price, volume]
        """
        
        if isinstance(file_path_or_df, pd.DataFrame):
            # Process DataFrame in chunks
            chunks = _crop_data_frame_in_batches(file_path_or_df, self.batch_size)
            for chunk in chunks:
                yield chunk
                
        elif isinstance(file_path_or_df, str):
            # Single file processing
            for chunk in pd.read_csv(file_path_or_df, chunksize=self.batch_size):
                yield chunk
                
        elif isinstance(file_path_or_df, Iterable):
            # Multiple files processing
            for file_path in file_path_or_df:
                for chunk in pd.read_csv(file_path, chunksize=self.batch_size):
                    yield chunk
        else:
            raise ValueError("file_path_or_df must be a DataFrame, string path, or iterable of string paths")

    def _read_first_row(self, file_path: str):
        """
        :param file_path: (str) Path to the csv file containing raw tick data in the format[date_time, price, volume]
        """
        first_row = pd.read_csv(file_path, nrows=1)
        return first_row

    def run(self, data: Union[list, tuple, pd.DataFrame]) -> list:
        """
        Reads a List, Tuple, or Dataframe and then constructs the financial data structure in the form of a list.
        The List, Tuple, or DataFrame must have only 3 attrs: date_time, price, & volume.

        :param data: (list, tuple, or pd.DataFrame) Dict or ndarray containing raw tick data in the format[date_time, price, volume]

        :return: (list) Financial data structure
        """
        
        if isinstance(data, pd.DataFrame):
            df_data = data
        else:
            # Convert list/tuple to DataFrame
            if isinstance(data, (list, tuple)) and len(data) > 0:
                if len(data[0]) != 3:
                    raise ValueError("Each data point must have exactly 3 elements: [date_time, price, volume]")
                df_data = pd.DataFrame(data, columns=['date_time', 'price', 'volume'])
            else:
                raise ValueError("Data must be a non-empty list, tuple, or DataFrame")
        
        return self._extract_bars(df_data)

    @abstractmethod  
    def _extract_bars(self, data: pd.DataFrame) -> list:
        """
        This method is required by all the bar types and is used to create the desired bars.

        :param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (list) Bars built using the current batch.
        """
        pass
        
    def _update_high_low(self, price: float):
        """
        Update the high and low prices for current bar
        
        :param price: (float) Current price
        """
        if self.high_price is None or price > self.high_price:
            self.high_price = price
        if self.low_price is None or price < self.low_price:
            self.low_price = price
            
    def _create_bar(self, date_time, price: float, high_price: float, low_price: float, 
                   open_price: float) -> dict:
        """
        Create a bar dictionary with OHLC data and statistics
        
        :param date_time: Bar timestamp
        :param price: (float) Close price
        :param high_price: (float) High price
        :param low_price: (float) Low price  
        :param open_price: (float) Open price
        :return: (dict) Bar data
        """
        
        # Calculate VWAP if we have volume data
        vwap = (self.cum_statistics['cum_dollar_value'] / 
                self.cum_statistics['cum_volume']) if self.cum_statistics['cum_volume'] > 0 else price
                
        return {
            'date_time': date_time,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': price,
            'volume': self.cum_statistics['cum_volume'],
            'cum_buy_volume': self.cum_statistics['cum_buy_volume'],
            'vwap': vwap,
            'num_ticks': self.cum_statistics['cum_ticks']
        }

    @abstractmethod
    def _reset_cache(self):
        """
        This method is required by all the bar types. It describes how cache should be reset
        when new bar is sampled.
        """

    @staticmethod
    def _assert_csv(test_batch: pd.DataFrame):
        """
        Tests that the csv file read has the format: date_time, price, and volume.
        If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.

        :param test_batch: (pd.DataFrame) The first row of the dataset.
        """
        assert test_batch.shape[1] == 3, 'Must have only 3 columns in csv: date_time, price, & volume.'
        assert isinstance(test_batch.iloc[0, 1], float), 'price column in csv not float.'
        assert not isinstance(test_batch.iloc[0, 2], str), 'volume column in csv not int or float.'

        try:
            pd.to_datetime(test_batch.iloc[0, 0])
        except ValueError:
            raise ValueError('csv file, column 0, not a date time format:',
                             test_batch.iloc[0, 0])

    def _update_high_low(self, price: float) -> Union[float, float]:
        """
        Update the high and low prices using the current price.

        :param price: (float) Current price
        :return: (tuple) Updated high and low prices
        """

        pass

    def _create_bars(self, date_time: str, price: float, high_price: float, low_price: float, list_bars: list) -> None:
        """
        Given the inputs, construct a bar which has the following fields: date_time, open, high, low, close, volume,
        cum_buy_volume, cum_ticks, cum_dollar_value.
        These bars are appended to list_bars, which is later used to construct the final bars DataFrame.

        :param date_time: (str) Timestamp of the bar
        :param price: (float) The current price
        :param high_price: (float) Highest price in the period
        :param low_price: (float) Lowest price in the period
        :param list_bars: (list) List to which we append the bars
        """

        pass

    def _apply_tick_rule(self, price: float) -> int:
        """
        Applies the tick rule as defined on page 29 of Advances in Financial Machine Learning.

        :param price: (float) Price at time t
        :return: (int) The signed tick
        """

        pass

    def _get_imbalance(self, price: float, signed_tick: int, volume: float) -> float:
        """
        Advances in Financial Machine Learning, page 29.

        Get the imbalance at a point in time, denoted as Theta_t

        :param price: (float) Price at t
        :param signed_tick: (int) signed tick, using the tick rule
        :param volume: (float) Volume traded at t
        :return: (float) Imbalance at time t
        """

        pass


class BaseImbalanceBars(BaseBars):
    """
    Base class for Imbalance Bars (EMA and Const) which implements imbalance bars calculation logic
    """

    def __init__(self, metric: str, batch_size: int,
                 expected_imbalance_window: int, exp_num_ticks_init: int,
                 analyse_thresholds: bool):
        """
        Constructor

        :param metric: (str) Type of imbalance bar to create. Example: dollar_imbalance.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        :param expected_imbalance_window: (int) Window used to estimate expected imbalance from previous trades
        :param exp_num_ticks_init: (int) Initial estimate for expected number of ticks in bar.
                                         For Const Imbalance Bars expected number of ticks equals expected number of ticks init
        :param analyse_thresholds: (bool) Flag to return thresholds values (theta, exp_num_ticks, exp_imbalance) in a
                                          form of Pandas DataFrame
        """

        pass

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for imbalance bars
        """

        pass

    def _extract_bars(self, data: Tuple[dict, pd.DataFrame]) -> list:
        """
        For loop which compiles the various imbalance bars: dollar, volume, or tick.

        :param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (list) Bars built using the current batch.
        """

        pass

    def _get_expected_imbalance(self, window: int):
        """
        Calculate the expected imbalance: 2P[b_t=1]-1, using a EWMA, pg 29
        :param window: (int) EWMA window for calculation
        :return: expected_imbalance: (np.ndarray) 2P[b_t=1]-1, approximated using a EWMA
        """

        pass

    @abstractmethod
    def _get_exp_num_ticks(self):
        """
        Abstract method which updates expected number of ticks when new run bar is formed
        """


# pylint: disable=too-many-instance-attributes
class BaseRunBars(BaseBars):
    """
    Base class for Run Bars (EMA and Const) which implements run bars calculation logic
    """

    def __init__(self, metric: str, batch_size: int, num_prev_bars: int,
                 expected_imbalance_window: int,
                 exp_num_ticks_init: int, analyse_thresholds: bool):
        """
        Constructor

        :param metric: (str) Type of imbalance bar to create. Example: dollar_imbalance.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        :param expected_imbalance_window: (int) Window used to estimate expected imbalance from previous trades
        :param exp_num_ticks_init: (int) Initial estimate for expected number of ticks in bar.
                                         For Const Imbalance Bars expected number of ticks equals expected number of ticks init
        :param analyse_thresholds: (bool) Flag to return thresholds values (thetas, exp_num_ticks, exp_runs) in Pandas DataFrame
        """

        pass

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for imbalance bars
        """

        pass

    def _extract_bars(self, data: Tuple[list, np.ndarray]) -> list:
        """
        For loop which compiles the various run bars: dollar, volume, or tick.

        :param data: (list or np.ndarray) Contains 3 columns - date_time, price, and volume.
        :return: (list) of bars built using the current batch.
        """


        pass

    def _get_expected_imbalance(self, array: list, window: int, warm_up: bool = False):
        """
        Advances in Financial Machine Learning, page 29.

        Calculates the expected imbalance: 2P[b_t=1]-1, using a EWMA.

        :param array: (list) of imbalances
        :param window: (int) EWMA window for calculation
        :parawm warm_up: (bool) flag of whether warm up period passed
        :return: expected_imbalance: (np.ndarray) 2P[b_t=1]-1, approximated using a EWMA
        """

        pass

    @abstractmethod
    def _get_exp_num_ticks(self):
        """
        Abstract method which updates expected number of ticks when new imbalance bar is formed
        """
