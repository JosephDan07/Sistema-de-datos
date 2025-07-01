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
    Exponentially weighted moving average following López de Prado specifications
    
    :param values: (np.ndarray) Values to calculate EWMA for
    :param alpha: (float) Smoothing factor between 0 and 1
    :return: (np.ndarray) EWMA values
    """
    if alpha <= 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    # Handle empty arrays
    if len(values) == 0:
        return np.array([], dtype=np.float64)
    
    # Direct implementation following López de Prado formula
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

    def __init__(self, metric: str, batch_size: int = 20000000, **kwargs):
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
        
        # Tick rule memory (López de Prado, page 29)
        self.last_tick_direction = 1
        
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
        
        # Warm-up period management (López de Prado recommendation)
        self.warm_up_period = 100  # Number of bars for warm-up
        self.is_warm_up_complete = False

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
                  to_csv: bool = False, output_path: Optional[str] = None, **kwargs) -> Union[pd.DataFrame, None]:
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
        total_bars_generated = 0
        for batch in self._batch_iterator(file_path_or_df):
            if verbose:
                print(f'Processing batch with {len(batch)} rows...')
            
            # Validate and clean data quality (re-enabled)
            batch = self._validate_data_quality(batch)
            if len(batch) == 0:
                if verbose:
                    print("Batch empty after data cleaning, skipping...")
                continue
                
            # Extract bars from current batch
            list_bars_batch = self._extract_bars(batch)
            list_bars.extend(list_bars_batch)
            
            # Check warm-up completion (re-enabled)
            total_bars_generated += len(list_bars_batch)
            self._check_warm_up_completion(total_bars_generated)
            
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
        Create a bar dictionary with OHLC data and microstructural statistics
        Following López de Prado's recommendations for comprehensive bar information
        
        :param date_time: Bar timestamp
        :param price: (float) Close price
        :param high_price: (float) High price
        :param low_price: (float) Low price  
        :param open_price: (float) Open price
        :return: (dict) Bar data with microstructural features
        """
        
        # Calculate VWAP if we have volume data
        vwap = (self.cum_statistics['cum_dollar_value'] / 
                self.cum_statistics['cum_volume']) if self.cum_statistics['cum_volume'] > 0 else price
        
        # Calculate buy volume percentage (López de Prado, page 30)
        buy_volume_pct = (self.cum_statistics['cum_buy_volume'] / 
                         self.cum_statistics['cum_volume']) if self.cum_statistics['cum_volume'] > 0 else 0.5
        
        # Calculate realized volatility (high-low estimator)
        realized_vol = np.log(high_price / low_price) if low_price > 0 and high_price > 0 else 0
        
        # Price change information
        price_change = price - open_price
        price_change_pct = (price_change / open_price) if open_price > 0 else 0
        
        return {
            'date_time': date_time,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': price,
            'volume': self.cum_statistics['cum_volume'],
            'cum_buy_volume': self.cum_statistics['cum_buy_volume'],
            'buy_volume_pct': buy_volume_pct,
            'vwap': vwap,
            'num_ticks': self.cum_statistics['cum_ticks'],
            'dollar_volume': self.cum_statistics['cum_dollar_value'],
            'realized_volatility': realized_vol,
            'price_change': price_change,
            'price_change_pct': price_change_pct
        }

    @abstractmethod
    def _reset_cache(self):
        """
        This method is required by all the bar types. It describes how cache should be reset
        when new bar is sampled.
        """
        pass

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

    def _apply_tick_rule(self, price: float) -> int:
        """
        Applies the tick rule as defined on page 29-30 of Advances in Financial Machine Learning.
        
        The tick rule classifies trades as buyer-initiated (1) or seller-initiated (-1).
        When price doesn't change, we use the last known tick direction (memory).

        :param price: (float) Price at time t
        :return: (int) The signed tick: 1 (buyer-initiated), -1 (seller-initiated)
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

    def _get_imbalance(self, price: float, signed_tick: int, volume: float) -> float:
        """
        Advances in Financial Machine Learning, page 29.

        Get the imbalance at a point in time, denoted as Theta_t

        :param price: (float) Price at t
        :param signed_tick: (int) signed tick, using the tick rule
        :param volume: (float) Volume traded at t
        :return: (float) Imbalance at time t
        """
        return signed_tick * volume

    def _validate_data_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean data quality following López de Prado's recommendations
        
        :param data: (pd.DataFrame) Raw tick data
        :return: (pd.DataFrame) Cleaned data
        """
        initial_count = len(data)
        
        # Remove rows with missing critical data
        data = data.dropna(subset=['price', 'volume'])
        
        # Remove zero or negative prices
        data = data[data['price'] > 0]
        
        # Remove zero volume trades (optional, depends on data source)
        # data = data[data['volume'] > 0]  # Uncomment if needed
        
        # Remove extreme outliers (> 10 standard deviations)
        price_mean = data['price'].mean()
        price_std = data['price'].std()
        if price_std > 0:
            price_outlier_mask = (np.abs(data['price'] - price_mean) < 10 * price_std)
            data = data[price_outlier_mask]
        
        cleaned_count = len(data)
        if initial_count > cleaned_count:
            warnings.warn(f"Data cleaning removed {initial_count - cleaned_count} rows "
                         f"({(initial_count - cleaned_count)/initial_count*100:.2f}%)")
        
        return data.reset_index(drop=True)
    
    def _check_warm_up_completion(self, num_bars_generated: int):
        """
        Check if warm-up period is complete
        
        :param num_bars_generated: (int) Number of bars generated so far
        """
        if not self.is_warm_up_complete and num_bars_generated >= self.warm_up_period:
            self.is_warm_up_complete = True
            if hasattr(self, 'verbose') and self.verbose:
                warnings.warn(f"Warm-up period complete after {self.warm_up_period} bars. "
                             "Statistical properties should now be stable.")


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
        super().__init__(metric, batch_size)
        self.expected_imbalance_window = expected_imbalance_window
        self.exp_num_ticks_init = exp_num_ticks_init
        self.analyse_thresholds = analyse_thresholds

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for imbalance bars
        """
        super()._reset_cache()

    def _extract_bars(self, data: pd.DataFrame) -> list:
        """
        For loop which compiles the various imbalance bars: dollar, volume, or tick.

        :param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (list) Bars built using the current batch.
        """
        # This will be implemented in concrete subclasses
        raise NotImplementedError("Subclasses must implement _extract_bars")

    def _get_expected_imbalance(self, window: int):
        """
        Calculate the expected imbalance: 2P[b_t=1]-1, using a EWMA, pg 29
        :param window: (int) EWMA window for calculation
        :return: expected_imbalance: (np.ndarray) 2P[b_t=1]-1, approximated using a EWMA
        """
        if len(self.expected_imbalance_window) < window:
            return np.nan
        return np.mean(self.expected_imbalance_window[-window:])

    @abstractmethod
    def _get_exp_num_ticks(self):
        """
        Abstract method which updates expected number of ticks when new run bar is formed
        """
        pass


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
        super().__init__(metric, batch_size)
        self.num_prev_bars = num_prev_bars
        self.expected_imbalance_window = expected_imbalance_window
        self.exp_num_ticks_init = exp_num_ticks_init
        self.analyse_thresholds = analyse_thresholds

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for imbalance bars
        """
        super()._reset_cache()

    def _extract_bars(self, data: pd.DataFrame) -> list:
        """
        For loop which compiles the various run bars: dollar, volume, or tick.

        :param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (list) of bars built using the current batch.
        """
        # This will be implemented in concrete subclasses
        raise NotImplementedError("Subclasses must implement _extract_bars")

    def _get_expected_imbalance(self, array: list, window: int, warm_up: bool = False):
        """
        Advances in Financial Machine Learning, page 29.

        Calculates the expected imbalance: 2P[b_t=1]-1, using a EWMA.

        :param array: (list) of imbalances
        :param window: (int) EWMA window for calculation
        :param warm_up: (bool) flag of whether warm up period passed
        :return: expected_imbalance: (np.ndarray) 2P[b_t=1]-1, approximated using a EWMA
        """
        if len(array) < window:
            return np.nan
        return np.mean(array[-window:])

    @abstractmethod
    def _get_exp_num_ticks(self):
        """
        Abstract method which updates expected number of ticks when new imbalance bar is formed
        """
        pass
