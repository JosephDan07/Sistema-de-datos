"""
A base class for the various bar types. Includes the logic shared between classes, to minimise the amount of
duplicated code.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, Generator, Iterable, Optional, List
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        :raises ValueError: If parameters are invalid
        """
        # Validate inputs
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("metric must be a non-empty string")
        
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        
        if batch_size > 50000000:  # 50M rows safety limit
            logger.warning(f"Very large batch_size ({batch_size}). This may cause memory issues.")
        
        # Base properties
        self.metric = metric.strip()
        self.batch_size = batch_size
        self.creation_time = datetime.now()
        
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
        self.warm_up_period = kwargs.get('warm_up_period', 100)
        self.is_warm_up_complete = False
        
        # Statistics tracking
        self.total_bars_generated = 0
        self.total_ticks_processed = 0
        self.processing_errors = 0
        
        logger.info(f"Initialized {self.metric} bars with batch_size={self.batch_size}")
        logger.debug(f"Warm-up period set to {self.warm_up_period} bars")

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
        :raises ValueError: If parameters are invalid
        :raises FileNotFoundError: If input file doesn't exist
        :raises Exception: For other processing errors
        """
        
        start_time = datetime.now()
        logger.info(f"Starting batch_run for {self.metric} bars")
        
        # Validate parameters
        if to_csv and output_path is None:
            raise ValueError("output_path must be specified when to_csv=True")
        
        if to_csv and not isinstance(output_path, str):
            raise ValueError("output_path must be a string")
            
        list_bars = []
        batch_count = 0
        
        try:
            # Process data in batches
            for batch in self._batch_iterator(file_path_or_df):
                batch_count += 1
                batch_start_time = datetime.now()
                
                if verbose:
                    logger.info(f'Processing batch {batch_count} with {len(batch)} rows...')
                
                # Validate first batch format
                if batch_count == 1:
                    try:
                        self._assert_csv(batch)
                        logger.debug("Data format validation passed")
                    except Exception as e:
                        logger.error(f"Data format validation failed: {e}")
                        raise
                
                # Validate and clean data quality
                try:
                    batch = self._validate_data_quality(batch)
                    if len(batch) == 0:
                        logger.warning(f"Batch {batch_count} empty after data cleaning, skipping...")
                        continue
                except Exception as e:
                    logger.error(f"Data quality validation failed for batch {batch_count}: {e}")
                    self.processing_errors += 1
                    continue
                    
                # Extract bars from current batch
                try:
                    list_bars_batch = self._extract_bars(batch)
                    list_bars.extend(list_bars_batch)
                    
                    # Update statistics
                    self.total_bars_generated += len(list_bars_batch)
                    self.total_ticks_processed += len(batch)
                    
                    # Check warm-up completion
                    self._check_warm_up_completion(self.total_bars_generated)
                    
                    batch_duration = datetime.now() - batch_start_time
                    if verbose and list_bars_batch:
                        logger.info(f'Generated {len(list_bars_batch)} bars from batch {batch_count} in {batch_duration.total_seconds():.2f}s')
                        
                except Exception as e:
                    logger.error(f"Error extracting bars from batch {batch_count}: {e}")
                    self.processing_errors += 1
                    continue
        
        except Exception as e:
            logger.error(f"Critical error in batch processing: {e}")
            raise
        
        # Final validation and output
        if not list_bars:
            logger.warning("No bars were generated. Check your data format and thresholds.")
            return pd.DataFrame()
            
        # Convert to DataFrame
        try:
            bars_df = pd.DataFrame(list_bars)
            
            # Add metadata
            bars_df.attrs['metric'] = self.metric
            bars_df.attrs['creation_time'] = self.creation_time
            bars_df.attrs['total_batches'] = batch_count
            bars_df.attrs['processing_errors'] = self.processing_errors
            bars_df.attrs['total_ticks_processed'] = self.total_ticks_processed
            
        except Exception as e:
            logger.error(f"Error creating DataFrame from bars: {e}")
            raise
        
        # Output handling
        total_duration = datetime.now() - start_time
        
        if to_csv:
            try:
                bars_df.to_csv(output_path, index=False)
                logger.info(f'Bars saved to {output_path}')
            except Exception as e:
                logger.error(f"Error saving bars to CSV: {e}")
                raise
            
            logger.info(f'Processing complete: {len(bars_df)} bars generated in {total_duration.total_seconds():.2f}s')
            return None
        else:
            logger.info(f'Processing complete: {len(bars_df)} bars generated in {total_duration.total_seconds():.2f}s')
            if self.processing_errors > 0:
                logger.warning(f"Processing completed with {self.processing_errors} errors")
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
        Enhanced with additional metadata and validation
        
        :param date_time: Bar timestamp
        :param price: (float) Close price
        :param high_price: (float) High price
        :param low_price: (float) Low price  
        :param open_price: (float) Open price
        :return: (dict) Bar data with microstructural features
        :raises ValueError: If price data is invalid
        """
        
        # Validate inputs
        if not all(isinstance(p, (int, float, np.integer, np.floating)) for p in [price, high_price, low_price, open_price]):
            raise ValueError("All price values must be numeric")
            
        if not all(p > 0 for p in [price, high_price, low_price, open_price]):
            raise ValueError("All price values must be positive")
            
        if low_price > high_price:
            logger.warning(f"Low price ({low_price}) > High price ({high_price}). Swapping values.")
            low_price, high_price = high_price, low_price
            
        if not (low_price <= open_price <= high_price):
            logger.warning(f"Open price ({open_price}) outside high-low range [{low_price}, {high_price}]")
            
        if not (low_price <= price <= high_price):
            logger.warning(f"Close price ({price}) outside high-low range [{low_price}, {high_price}]")
        
        # Calculate VWAP if we have volume data
        vwap = (self.cum_statistics['cum_dollar_value'] / 
                self.cum_statistics['cum_volume']) if self.cum_statistics['cum_volume'] > 0 else price
        
        # Calculate buy volume percentage (López de Prado, page 30)
        buy_volume_pct = (self.cum_statistics['cum_buy_volume'] / 
                         self.cum_statistics['cum_volume']) if self.cum_statistics['cum_volume'] > 0 else 0.5
        
        # Calculate realized volatility (high-low estimator)
        try:
            realized_vol = np.log(high_price / low_price) if low_price > 0 and high_price > 0 else 0
        except (ValueError, ZeroDivisionError):
            logger.warning("Could not calculate realized volatility, setting to 0")
            realized_vol = 0
        
        # Price change information
        price_change = price - open_price
        price_change_pct = (price_change / open_price) if open_price > 0 else 0
        
        # Additional microstructural features
        tick_size = (high_price - low_price) / self.cum_statistics['cum_ticks'] if self.cum_statistics['cum_ticks'] > 0 else 0
        
        # Dollar volume per tick
        dollar_per_tick = (self.cum_statistics['cum_dollar_value'] / 
                          self.cum_statistics['cum_ticks']) if self.cum_statistics['cum_ticks'] > 0 else 0
        
        # Volume per tick
        volume_per_tick = (self.cum_statistics['cum_volume'] / 
                          self.cum_statistics['cum_ticks']) if self.cum_statistics['cum_ticks'] > 0 else 0
        
        bar_data = {
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
            'price_change_pct': price_change_pct,
            'tick_size': tick_size,
            'dollar_per_tick': dollar_per_tick,
            'volume_per_tick': volume_per_tick,
            'warm_up_complete': self.is_warm_up_complete,
            'bar_method': self.metric
        }
        
        # Log bar creation (debug level)
        logger.debug(f"Created bar: {bar_data['date_time']}, Close: {price:.4f}, "
                    f"Volume: {self.cum_statistics['cum_volume']}, Ticks: {self.cum_statistics['cum_ticks']}")
        
        return bar_data

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
        Enhanced validation with better error messages and type checking.

        :param test_batch: (pd.DataFrame) The first row of the dataset.
        :raises ValueError: If data format is incorrect
        """
        if test_batch.empty:
            raise ValueError("Input DataFrame is empty")
        
        if test_batch.shape[1] != 3:
            raise ValueError(f'Must have exactly 3 columns in csv: date_time, price, & volume. '
                           f'Got {test_batch.shape[1]} columns: {list(test_batch.columns)}')
        
        # Expected column names (flexible matching)
        expected_columns = ['date_time', 'price', 'volume']
        actual_columns = list(test_batch.columns)
        
        # Check if column names match expected pattern
        if not all(any(expected in str(col).lower() for expected in ['date', 'time']) for col in [actual_columns[0]]):
            raise ValueError(f'First column should contain date/time information. Got: {actual_columns[0]}')
        
        if not any(keyword in str(actual_columns[1]).lower() for keyword in ['price', 'close', 'value']):
            raise ValueError(f'Second column should contain price information. Got: {actual_columns[1]}')
            
        if not any(keyword in str(actual_columns[2]).lower() for keyword in ['volume', 'vol', 'size']):
            raise ValueError(f'Third column should contain volume information. Got: {actual_columns[2]}')
        
        # Type validation with better error messages
        try:
            price_value = test_batch.iloc[0, 1]
            if pd.isna(price_value):
                raise ValueError('First price value is NaN')
            if not isinstance(price_value, (int, float, np.integer, np.floating)):
                raise ValueError(f'Price column must be numeric. Got type: {type(price_value)}')
            if price_value <= 0:
                raise ValueError(f'Price must be positive. Got: {price_value}')
        except (IndexError, TypeError) as e:
            raise ValueError(f'Error validating price column: {e}')
        
        try:
            volume_value = test_batch.iloc[0, 2]
            if pd.isna(volume_value):
                raise ValueError('First volume value is NaN')
            if not isinstance(volume_value, (int, float, np.integer, np.floating)):
                raise ValueError(f'Volume column must be numeric. Got type: {type(volume_value)}')
            if volume_value < 0:
                raise ValueError(f'Volume cannot be negative. Got: {volume_value}')
        except (IndexError, TypeError) as e:
            raise ValueError(f'Error validating volume column: {e}')

        # DateTime validation with better error handling
        try:
            datetime_value = test_batch.iloc[0, 0]
            if pd.isna(datetime_value):
                raise ValueError('First datetime value is NaN')
            parsed_datetime = pd.to_datetime(datetime_value)
            if pd.isna(parsed_datetime):
                raise ValueError('Datetime could not be parsed')
        except (ValueError, TypeError) as e:
            raise ValueError(f'DateTime validation failed for value "{test_batch.iloc[0, 0]}": {e}')
        except Exception as e:
            raise ValueError(f'Unexpected error in datetime validation: {e}')

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
        Enhanced with comprehensive logging and statistics
        
        :param data: (pd.DataFrame) Raw tick data
        :return: (pd.DataFrame) Cleaned data
        :raises ValueError: If data becomes empty after cleaning
        """
        if data.empty:
            logger.warning("Input data is empty")
            return data
            
        initial_count = len(data)
        logger.debug(f"Starting data validation with {initial_count} rows")
        
        # Track cleaning statistics
        cleaning_stats = {
            'initial_count': initial_count,
            'missing_data_removed': 0,
            'zero_price_removed': 0,
            'negative_price_removed': 0,
            'zero_volume_removed': 0,
            'negative_volume_removed': 0,
            'outliers_removed': 0
        }
        
        # Remove rows with missing critical data
        before_na = len(data)
        data = data.dropna(subset=['price', 'volume'])
        cleaning_stats['missing_data_removed'] = before_na - len(data)
        
        if cleaning_stats['missing_data_removed'] > 0:
            logger.debug(f"Removed {cleaning_stats['missing_data_removed']} rows with missing price/volume data")
        
        # Remove zero prices
        before_zero_price = len(data)
        data = data[data['price'] > 0]
        cleaning_stats['zero_price_removed'] = before_zero_price - len(data)
        
        if cleaning_stats['zero_price_removed'] > 0:
            logger.debug(f"Removed {cleaning_stats['zero_price_removed']} rows with zero prices")
        
        # Remove negative prices (additional check)
        before_neg_price = len(data)  
        data = data[data['price'] > 0]  # This also catches negative prices
        cleaning_stats['negative_price_removed'] = before_neg_price - len(data)
        
        # Handle volume validation (be more flexible with zero volume)
        if 'volume' in data.columns:
            before_neg_volume = len(data)
            data = data[data['volume'] >= 0]  # Allow zero volume but not negative
            cleaning_stats['negative_volume_removed'] = before_neg_volume - len(data)
            
            if cleaning_stats['negative_volume_removed'] > 0:
                logger.debug(f"Removed {cleaning_stats['negative_volume_removed']} rows with negative volume")
        
        # Remove extreme outliers (> 10 standard deviations) - More robust approach
        if len(data) > 10:  # Only apply if we have enough data
            try:
                # Use robust statistics (median and MAD instead of mean and std)
                price_median = data['price'].median()
                price_mad = data['price'].mad()  # Median Absolute Deviation
                
                if price_mad > 0:
                    # Use 10 MAD as threshold (more robust than 10 std)
                    threshold = 10 * price_mad
                    before_outliers = len(data)
                    price_outlier_mask = (np.abs(data['price'] - price_median) < threshold)
                    data = data[price_outlier_mask]
                    cleaning_stats['outliers_removed'] = before_outliers - len(data)
                    
                    if cleaning_stats['outliers_removed'] > 0:
                        logger.debug(f"Removed {cleaning_stats['outliers_removed']} price outliers using MAD method")
            except Exception as e:
                logger.warning(f"Could not apply outlier detection: {e}")
        
        cleaned_count = len(data)
        total_removed = initial_count - cleaned_count
        
        if total_removed > 0:
            removal_pct = (total_removed / initial_count) * 100
            logger.info(f"Data cleaning removed {total_removed} rows ({removal_pct:.2f}%)")
            
            # Log detailed statistics
            logger.debug(f"Cleaning breakdown: {cleaning_stats}")
            
            # Warning for excessive data loss
            if removal_pct > 10:
                logger.warning(f"High data loss during cleaning: {removal_pct:.2f}%. "
                             "Consider reviewing data quality and cleaning parameters.")
        
        # Final validation
        if cleaned_count == 0:
            logger.error("All data removed during cleaning process")
            raise ValueError("No valid data remaining after quality validation")
        
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

    def get_processing_statistics(self) -> dict:
        """
        Get comprehensive processing statistics for auditing and monitoring
        
        :return: (dict) Processing statistics and metadata
        """
        stats = {
            'metric': self.metric,
            'creation_time': self.creation_time,
            'batch_size': self.batch_size,
            'total_bars_generated': self.total_bars_generated,
            'total_ticks_processed': self.total_ticks_processed,
            'processing_errors': self.processing_errors,
            'warm_up_period': self.warm_up_period,
            'is_warm_up_complete': self.is_warm_up_complete,
            'last_tick_direction': self.last_tick_direction,
            'current_cum_statistics': self.cum_statistics.copy()
        }
        
        # Calculate derived statistics
        if self.total_ticks_processed > 0:
            stats['ticks_per_bar'] = self.total_ticks_processed / max(self.total_bars_generated, 1)
            stats['error_rate'] = self.processing_errors / self.total_ticks_processed
        else:
            stats['ticks_per_bar'] = 0
            stats['error_rate'] = 0
            
        logger.info(f"Processing statistics: {stats}")
        return stats
    
    def reset_statistics(self):
        """
        Reset processing statistics (useful for batch processing)
        """
        logger.info("Resetting processing statistics")
        self.total_bars_generated = 0
        self.total_ticks_processed = 0
        self.processing_errors = 0
        self.is_warm_up_complete = False
        self._reset_cache()
        
    def export_configuration(self) -> dict:
        """
        Export current configuration for reproducibility
        
        :return: (dict) Configuration parameters
        """
        config = {
            'metric': self.metric,
            'batch_size': self.batch_size,
            'warm_up_period': self.warm_up_period,
            'creation_time': self.creation_time.isoformat(),
            'class_name': self.__class__.__name__
        }
        
        return config

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
