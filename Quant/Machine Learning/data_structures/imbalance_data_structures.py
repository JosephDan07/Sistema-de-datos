"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

Imbalance bars generation logic: Tick, Volume, and Dollar Imbalance Bars
"""

from typing import Union, Iterable, Optional
import numpy as np
import pandas as pd

try:
    from .base_bars import BaseBars, ewma
except ImportError:
    from base_bars import BaseBars, ewma


class ImbalanceBars(BaseBars):
    """
    Contains all of the logic to construct imbalance bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_tick_imbalance_bars which will create an instance of this
    class and then construct the imbalance bars.
    """

    def __init__(self, metric: str, num_prev_bars: int = 3, expected_imbalance_window: int = 100, 
                 exp_num_ticks_init: int = 20000, exp_num_ticks_constraints: list = None,
                 batch_size: int = 20000000):
        """
        Constructor

        :param metric: (str) Type of imbalance bar to create. Example: "tick_imbalance", "volume_imbalance", "dollar_imbalance"
        :param num_prev_bars: (int) Number of previous bars used for expected imbalance window
        :param expected_imbalance_window: (int) Rolling window used for expected imbalance and number of ticks estimation
        :param exp_num_ticks_init: (int) Initial expected number of ticks per bar
        :param exp_num_ticks_constraints: (list) Constraints on expected number of ticks [min, max]
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        """
        super().__init__(metric, batch_size)
        
        # Imbalance bar specific parameters
        self.metric = metric
        self.num_prev_bars = num_prev_bars
        self.expected_imbalance_window = expected_imbalance_window
        self.exp_num_ticks_init = exp_num_ticks_init
        self.exp_num_ticks_constraints = exp_num_ticks_constraints or [10, np.inf]
        
        # Imbalance tracking
        self.imbalance_tick_statistics = {'num_ticks_bar': []}
        self.expected_imbalance_window = []
        self.imbalance_array = []
        self.warm_up_flag = False
        
        # Expected values
        self.expected_imbalance = np.nan
        self.expected_num_ticks = exp_num_ticks_init

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for imbalance bars
        """
        super()._reset_cache()
        self.imbalance_array = []

    def _extract_bars(self, data: pd.DataFrame) -> list:
        """
        For loop which compiles imbalance bars: tick, volume, or dollar.

        :param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (list) Extracted bars
        """
        
        # Standardize column names
        if len(data.columns) >= 3:
            data.columns = ['date_time', 'price', 'volume'] + list(data.columns[3:])
        else:
            raise ValueError("Data must have at least 3 columns: date_time, price, volume")
            
        list_bars = []
        
        for _, row in data.iterrows():
            # Extract row data
            date_time = row['date_time']
            price = float(row['price'])
            volume = float(row['volume']) if not pd.isna(row['volume']) else 1.0
            
            # Update high and low
            self._update_high_low(price)
            
            # Set open price for new bar
            if self.open_price is None:
                self.open_price = price
                
            # Apply tick rule
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
                
            # Calculate imbalance value based on metric type
            if self.metric == 'tick_imbalance':
                imbalance_value = tick_direction
            elif self.metric == 'volume_imbalance':
                imbalance_value = tick_direction * volume
            elif self.metric == 'dollar_imbalance':
                imbalance_value = tick_direction * dollar_value
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
                
            # Update imbalance array
            self.imbalance_array.append(imbalance_value)
            
            # Check if we should create a bar
            if self._should_create_bar():
                # Create bar
                bar = self._create_bar(
                    date_time=date_time,
                    price=self.close_price,
                    high_price=self.high_price,
                    low_price=self.low_price,
                    open_price=self.open_price
                )
                
                list_bars.append(bar)
                
                # Update imbalance statistics
                self._update_imbalance_statistics()
                
                # Reset for next bar
                self._reset_cache()
                
        return list_bars
        
    def _should_create_bar(self) -> bool:
        """
        Check if a bar should be created based on imbalance conditions
        
        :return: (bool) True if bar should be created
        """
        # Always create first bar after expected number of ticks
        if len(self.imbalance_tick_statistics['num_ticks_bar']) == 0:
            return self.cum_statistics['cum_ticks'] >= self.expected_num_ticks
            
        # Calculate current imbalance
        current_imbalance = abs(sum(self.imbalance_array))
        
        # Create bar if imbalance exceeds expected threshold or reached expected number of ticks
        return (current_imbalance >= abs(self.expected_imbalance) or 
                self.cum_statistics['cum_ticks'] >= self.expected_num_ticks)
                
    def _update_imbalance_statistics(self):
        """
        Update the expected imbalance and number of ticks for future bars
        """
        # Store number of ticks in this bar
        self.imbalance_tick_statistics['num_ticks_bar'].append(self.cum_statistics['cum_ticks'])
        
        # Store imbalance for this bar
        bar_imbalance = sum(self.imbalance_array)
        self.expected_imbalance_window.append(bar_imbalance)
        
        # Keep only recent window
        if len(self.expected_imbalance_window) > self.num_prev_bars:
            self.expected_imbalance_window = self.expected_imbalance_window[-self.num_prev_bars:]
            
        if len(self.imbalance_tick_statistics['num_ticks_bar']) > self.num_prev_bars:
            self.imbalance_tick_statistics['num_ticks_bar'] = \
                self.imbalance_tick_statistics['num_ticks_bar'][-self.num_prev_bars:]
                
        # Update expected imbalance (using mean of absolute values)
        if len(self.expected_imbalance_window) >= self.num_prev_bars:
            self.expected_imbalance = np.mean(np.abs(self.expected_imbalance_window[-self.num_prev_bars:]))
            
        # Update expected number of ticks
        if len(self.imbalance_tick_statistics['num_ticks_bar']) >= self.num_prev_bars:
            self.expected_num_ticks = int(np.mean(self.imbalance_tick_statistics['num_ticks_bar'][-self.num_prev_bars:]))
            
            # Apply constraints
            self.expected_num_ticks = max(self.exp_num_ticks_constraints[0], 
                                        min(self.expected_num_ticks, self.exp_num_ticks_constraints[1]))
                                        
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


class EMAImbalanceBars(ImbalanceBars):
    """
    EMA Imbalance Bars implementation
    """
    
    def __init__(self, metric: str, alpha: float = 0.3, **kwargs):
        """
        Constructor for EMA Imbalance Bars
        
        :param metric: (str) Type of imbalance bar
        :param alpha: (float) EMA smoothing factor
        """
        super().__init__(metric, **kwargs)
        self.alpha = alpha
        self.ema_expected_imbalance = None
        self.ema_expected_num_ticks = None
        
    def _update_imbalance_statistics(self):
        """
        Update expected values using EMA
        """
        super()._update_imbalance_statistics()
        
        # Calculate EMA for expected imbalance
        current_imbalance = abs(sum(self.imbalance_array))
        if self.ema_expected_imbalance is None:
            self.ema_expected_imbalance = current_imbalance
        else:
            self.ema_expected_imbalance = (self.alpha * current_imbalance + 
                                         (1 - self.alpha) * self.ema_expected_imbalance)
            
        # Calculate EMA for expected number of ticks
        current_num_ticks = self.cum_statistics['cum_ticks']
        if self.ema_expected_num_ticks is None:
            self.ema_expected_num_ticks = current_num_ticks
        else:
            self.ema_expected_num_ticks = (self.alpha * current_num_ticks + 
                                         (1 - self.alpha) * self.ema_expected_num_ticks)
            
        # Update expected values
        self.expected_imbalance = self.ema_expected_imbalance
        self.expected_num_ticks = max(self.exp_num_ticks_constraints[0], 
                                    min(int(self.ema_expected_num_ticks), self.exp_num_ticks_constraints[1]))


# Public API functions
def get_tick_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                           num_prev_bars: int = 3, expected_imbalance_window: int = 100,
                           exp_num_ticks_init: int = 20000, exp_num_ticks_constraints: list = None,
                           batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, 
                           output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
    """
    Creates tick imbalance bars
    """
    bars = ImbalanceBars(metric='tick_imbalance', num_prev_bars=num_prev_bars,
                        expected_imbalance_window=expected_imbalance_window,
                        exp_num_ticks_init=exp_num_ticks_init,
                        exp_num_ticks_constraints=exp_num_ticks_constraints,
                        batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)


def get_volume_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                             num_prev_bars: int = 3, expected_imbalance_window: int = 100,
                             exp_num_ticks_init: int = 20000, exp_num_ticks_constraints: list = None,
                             batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, 
                             output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
    """
    Creates volume imbalance bars
    """
    bars = ImbalanceBars(metric='volume_imbalance', num_prev_bars=num_prev_bars,
                        expected_imbalance_window=expected_imbalance_window,
                        exp_num_ticks_init=exp_num_ticks_init,
                        exp_num_ticks_constraints=exp_num_ticks_constraints,
                        batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)


def get_dollar_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                             num_prev_bars: int = 3, expected_imbalance_window: int = 100,
                             exp_num_ticks_init: int = 20000, exp_num_ticks_constraints: list = None,
                             batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, 
                             output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
    """
    Creates dollar imbalance bars
    """
    bars = ImbalanceBars(metric='dollar_imbalance', num_prev_bars=num_prev_bars,
                        expected_imbalance_window=expected_imbalance_window,
                        exp_num_ticks_init=exp_num_ticks_init,
                        exp_num_ticks_constraints=exp_num_ticks_constraints,
                        batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)


# EMA variants
def get_ema_tick_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                               alpha: float = 0.3, **kwargs) -> Union[pd.DataFrame, None]:
    """Creates EMA tick imbalance bars"""
    bars = EMAImbalanceBars(metric='tick_imbalance', alpha=alpha, **kwargs)
    return bars.batch_run(file_path_or_df, **kwargs)


def get_ema_volume_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                                 alpha: float = 0.3, **kwargs) -> Union[pd.DataFrame, None]:
    """Creates EMA volume imbalance bars"""
    bars = EMAImbalanceBars(metric='volume_imbalance', alpha=alpha, **kwargs)
    return bars.batch_run(file_path_or_df, **kwargs)


def get_ema_dollar_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                                 alpha: float = 0.3, **kwargs) -> Union[pd.DataFrame, None]:
    """Creates EMA dollar imbalance bars"""
    bars = EMAImbalanceBars(metric='dollar_imbalance', alpha=alpha, **kwargs)
    return bars.batch_run(file_path_or_df, **kwargs)


# Const variants (using fixed window)
def get_const_tick_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                                 **kwargs) -> Union[pd.DataFrame, None]:
    """Creates const tick imbalance bars"""
    return get_tick_imbalance_bars(file_path_or_df, **kwargs)


def get_const_volume_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                                   **kwargs) -> Union[pd.DataFrame, None]:
    """Creates const volume imbalance bars"""
    return get_volume_imbalance_bars(file_path_or_df, **kwargs)


def get_const_dollar_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                                   **kwargs) -> Union[pd.DataFrame, None]:
    """Creates const dollar imbalance bars"""
    return get_dollar_imbalance_bars(file_path_or_df, **kwargs)
