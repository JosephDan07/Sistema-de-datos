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
    from ..util.volume_classifier import get_tick_rule_buy_volume
except ImportError:
    try:
        from base_bars import BaseBars, ewma
    except ImportError:
        import sys
        import os
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        from base_bars import BaseBars, ewma
    
    # Fallback if util not available
    def get_tick_rule_buy_volume(close, volume):
        return volume * 0.5  # Simple fallback


class ImbalanceBars(BaseBars):
    """
    Contains all of the logic to construct imbalance bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_tick_imbalance_bars which will create an instance of this
    class and then construct the imbalance bars.
    
    Implementation follows López de Prado's formulas on pages 29-31:
    - θT = E[T] * |E[θ]| where E[θ] = 2P[bt = 1] - 1
    - Expected imbalance calculated using EWMA for adaptive estimation
    """

    def __init__(self, metric: str, num_prev_bars: int = 3, expected_imbalance_window: int = 100, 
                 exp_num_ticks_init: int = 20000, exp_num_ticks_constraints: list = None,
                 batch_size: int = 20000000, alpha: float = None, **kwargs):
        """
        Constructor

        :param metric: (str) Type of imbalance bar to create. Example: "tick_imbalance", "volume_imbalance", "dollar_imbalance"
        :param num_prev_bars: (int) Number of previous bars used for expected imbalance window
        :param expected_imbalance_window: (int) Rolling window used for expected imbalance and number of ticks estimation
        :param exp_num_ticks_init: (int) Initial expected number of ticks per bar
        :param exp_num_ticks_constraints: (list) Constraints on expected number of ticks [min, max]
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        :param alpha: (float) EWMA smoothing factor (if None, uses simple average)
        """
        super().__init__(metric, batch_size, **kwargs)
        
        # Imbalance bar specific parameters
        self.metric = metric
        self.num_prev_bars = num_prev_bars
        self.expected_imbalance_window = expected_imbalance_window
        self.exp_num_ticks_init = exp_num_ticks_init
        self.exp_num_ticks_constraints = exp_num_ticks_constraints or [10, np.inf]
        self.alpha = alpha  # EWMA smoothing factor
        
        # Imbalance tracking (López de Prado, page 29)
        self.imbalance_tick_statistics = {'num_ticks_bar': []}
        self.expected_imbalance_history = []  # For E[θ] calculation
        self.imbalance_array = []  # Current bar imbalances
        self.warm_up_flag = False
        
        # Expected values initialization
        self.expected_imbalance = 0.0  # E[θ] = 2P[bt = 1] - 1
        self.expected_num_ticks = exp_num_ticks_init  # E[T]
        
        # For P[bt = 1] calculation
        self.positive_ticks_history = []  # Track positive tick ratios

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
        Following López de Prado's formula: θT = E[T] * |E[θ]|
        where E[θ] = 2P[bt = 1] - 1
        
        :return: (bool) True if bar should be created
        """
        # Always create first bar after expected number of ticks (warm-up)
        if len(self.imbalance_tick_statistics['num_ticks_bar']) == 0:
            return self.cum_statistics['cum_ticks'] >= self.expected_num_ticks
            
        # Calculate current imbalance |θt|
        current_imbalance = abs(sum(self.imbalance_array))
        
        # Calculate threshold: θT = E[T] * |E[θ]|
        imbalance_threshold = self.expected_num_ticks * abs(self.expected_imbalance)
        
        # Create bar if imbalance exceeds threshold or reached maximum number of ticks
        max_ticks_threshold = self.expected_num_ticks * 3  # Safety valve
        return (current_imbalance >= imbalance_threshold or 
                self.cum_statistics['cum_ticks'] >= max_ticks_threshold)
                
    def _update_imbalance_statistics(self):
        """
        Update the expected imbalance and number of ticks for future bars
        Following López de Prado's formulas on pages 29-31:
        - E[θ] = 2P[bt = 1] - 1
        - P[bt = 1] estimated using EWMA or rolling average
        """
        # Store number of ticks in this bar
        current_num_ticks = self.cum_statistics['cum_ticks']
        self.imbalance_tick_statistics['num_ticks_bar'].append(current_num_ticks)
        
        # Calculate P[bt = 1] for this bar (proportion of positive ticks)
        positive_imbalances = sum(1 for x in self.imbalance_array if x > 0)
        total_imbalances = len(self.imbalance_array)
        prob_positive = positive_imbalances / total_imbalances if total_imbalances > 0 else 0.5
        
        # Store probability for future E[θ] calculation
        self.positive_ticks_history.append(prob_positive)
        
        # Keep only recent history
        max_history = max(self.num_prev_bars, 50)  # At least 50 for stability
        if len(self.positive_ticks_history) > max_history:
            self.positive_ticks_history = self.positive_ticks_history[-max_history:]
            
        if len(self.imbalance_tick_statistics['num_ticks_bar']) > max_history:
            self.imbalance_tick_statistics['num_ticks_bar'] = \
                self.imbalance_tick_statistics['num_ticks_bar'][-max_history:]
                
        # Update E[θ] = 2P[bt = 1] - 1
        if len(self.positive_ticks_history) >= self.num_prev_bars:
            if self.alpha is not None:
                # Use EWMA for adaptive estimation
                if not hasattr(self, '_ewma_prob_positive'):
                    self._ewma_prob_positive = np.mean(self.positive_ticks_history[-self.num_prev_bars:])
                else:
                    self._ewma_prob_positive = (self.alpha * prob_positive + 
                                              (1 - self.alpha) * self._ewma_prob_positive)
                avg_prob_positive = self._ewma_prob_positive
            else:
                # Use simple average
                avg_prob_positive = np.mean(self.positive_ticks_history[-self.num_prev_bars:])
            
            # Calculate E[θ] = 2P[bt = 1] - 1 (López de Prado, page 29)
            self.expected_imbalance = 2 * avg_prob_positive - 1
            
        # Update E[T] (expected number of ticks)
        if len(self.imbalance_tick_statistics['num_ticks_bar']) >= self.num_prev_bars:
            if self.alpha is not None:
                # Use EWMA for E[T]
                if not hasattr(self, '_ewma_num_ticks'):
                    self._ewma_num_ticks = np.mean(self.imbalance_tick_statistics['num_ticks_bar'][-self.num_prev_bars:])
                else:
                    self._ewma_num_ticks = (self.alpha * current_num_ticks + 
                                          (1 - self.alpha) * self._ewma_num_ticks)
                avg_num_ticks = self._ewma_num_ticks
            else:
                # Use simple average
                avg_num_ticks = np.mean(self.imbalance_tick_statistics['num_ticks_bar'][-self.num_prev_bars:])
            
            self.expected_num_ticks = int(avg_num_ticks)
            
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
    Uses Exponentially Weighted Moving Average for adaptive threshold estimation
    """
    
    def __init__(self, metric: str, alpha: float = 0.3, **kwargs):
        """
        Constructor for EMA Imbalance Bars
        
        :param metric: (str) Type of imbalance bar
        :param alpha: (float) EMA smoothing factor (0 < alpha <= 1)
        """
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
            
        # Pass alpha to parent class
        super().__init__(metric, alpha=alpha, **kwargs)
        self.alpha = alpha


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
