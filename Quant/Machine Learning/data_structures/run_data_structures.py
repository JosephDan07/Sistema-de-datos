"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

Run bars generation logic: Tick, Volume, and Dollar Run Bars
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


class RunBars(BaseBars):
    """
    Contains all of the logic to construct run bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_tick_run_bars which will create an instance of this
    class and then construct the run bars.
    
    Implementation follows López de Prado's formula on page 31:
    - A run is defined as a sequence of successive ticks with the same classification (buy or sell)
    - θT = max{∑bt,i+, ∑bt,i-} where bt,i+ and bt,i- are positive and negative runs
    """

    def __init__(self, metric: str, num_prev_bars: int = 3, expected_runs_window: int = 100, 
                 exp_num_ticks_init: int = 20000, exp_num_ticks_constraints: list = None,
                 batch_size: int = 20000000, alpha: float = None, **kwargs):
        """
        Constructor

        :param metric: (str) Type of run bar to create. Example: "tick_run", "volume_run", "dollar_run"
        :param num_prev_bars: (int) Number of previous bars used for expected runs window
        :param expected_runs_window: (int) Rolling window used for expected runs and number of ticks estimation
        :param exp_num_ticks_init: (int) Initial expected number of ticks per bar
        :param exp_num_ticks_constraints: (list) Constraints on expected number of ticks [min, max]
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        :param alpha: (float) EWMA smoothing factor (if None, uses simple average)
        """
        super().__init__(metric, batch_size, **kwargs)
        
        # Run bar specific parameters
        self.metric = metric
        self.num_prev_bars = num_prev_bars
        self.expected_runs_window = expected_runs_window
        self.exp_num_ticks_init = exp_num_ticks_init
        self.exp_num_ticks_constraints = exp_num_ticks_constraints or [10, np.inf]
        self.alpha = alpha  # EWMA smoothing factor
        
        # Run tracking (López de Prado, page 31)
        self.run_tick_statistics = {'num_ticks_bar': []}
        self.expected_runs_history = []  # For E[runs] calculation
        self.runs_array = []
        self.warm_up_flag = False
        
        # Current run tracking - tracks sequences of same-sign ticks
        self.current_run_length = 0  # Length of current run
        self.current_run_direction = 0  # Direction of current run (1 or -1)
        self.current_run_value = 0  # Cumulative value of current run
        self.pos_runs = 0  # Cumulative positive runs value
        self.neg_runs = 0  # Cumulative negative runs value
        self.completed_runs = []  # Store completed runs for this bar
        
        # Expected values
        self.expected_runs = 0.0  # E[max(pos_runs, neg_runs)]
        self.expected_num_ticks = exp_num_ticks_init

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for run bars
        """
        super()._reset_cache()
        self.runs_array = []
        self.current_run_value = 0
        self.current_run_direction = 0
        self.pos_runs = 0
        self.neg_runs = 0

    def _extract_bars(self, data: pd.DataFrame) -> list:
        """
        For loop which compiles run bars: tick, volume, or dollar.

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
                
            # Calculate run value based on metric type
            if self.metric == 'tick_run':
                run_value = tick_direction
            elif self.metric == 'volume_run':
                run_value = tick_direction * volume
            elif self.metric == 'dollar_run':
                run_value = tick_direction * dollar_value
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
                
            # Update runs
            self._update_runs(run_value)
            
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
                
                # Update run statistics
                self._update_run_statistics()
                
                # Reset for next bar
                self._reset_cache()
                
        return list_bars
        
    def _update_runs(self, run_value: float):
        """
        Update runs tracking based on current value
        Following López de Prado's definition: a run is a sequence of consecutive ticks 
        with the same classification (buy/sell)
        
        :param run_value: (float) Current run value (signed)
        """
        tick_direction = 1 if run_value > 0 else -1 if run_value < 0 else self.last_tick_direction
        
        if self.current_run_direction == 0:
            # Starting first run
            self.current_run_direction = tick_direction
            self.current_run_value = abs(run_value)
            self.current_run_length = 1
            
        elif self.current_run_direction == tick_direction:
            # Continuing current run
            self.current_run_value += abs(run_value)
            self.current_run_length += 1
            
        else:
            # Run direction changed - complete previous run
            if self.current_run_direction == 1:
                self.pos_runs += self.current_run_value
            else:
                self.neg_runs += self.current_run_value
                
            # Store completed run info
            self.completed_runs.append({
                'direction': self.current_run_direction,
                'value': self.current_run_value,
                'length': self.current_run_length
            })
            
            # Start new run
            self.current_run_direction = tick_direction
            self.current_run_value = abs(run_value)
            self.current_run_length = 1
        
    def _should_create_bar(self) -> bool:
        """
        Check if a bar should be created based on run conditions
        Following López de Prado's formula: θT = max{∑bt,i+, ∑bt,i-}
        
        :return: (bool) True if bar should be created
        """
        # Always create first bar after expected number of ticks (warm-up)
        if len(self.run_tick_statistics['num_ticks_bar']) == 0:
            return self.cum_statistics['cum_ticks'] >= self.expected_num_ticks
            
        # Calculate total positive and negative runs including current run
        total_pos_runs = self.pos_runs
        total_neg_runs = self.neg_runs
        
        # Add current incomplete run to appropriate total
        if self.current_run_direction == 1:
            total_pos_runs += self.current_run_value
        elif self.current_run_direction == -1:
            total_neg_runs += self.current_run_value
            
        # Calculate θT = max{∑bt,i+, ∑bt,i-} (López de Prado, page 31)
        current_theta = max(total_pos_runs, total_neg_runs)
        
        # Create bar if runs exceed expected threshold or reached maximum number of ticks
        max_ticks_threshold = self.expected_num_ticks * 3  # Safety valve
        return (current_theta >= self.expected_runs or 
                self.cum_statistics['cum_ticks'] >= max_ticks_threshold)
                
    def _update_run_statistics(self):
        """
        Update the expected runs and number of ticks for future bars
        Following López de Prado's methodology for adaptive threshold estimation
        """
        # Store number of ticks in this bar
        current_num_ticks = self.cum_statistics['cum_ticks']
        self.run_tick_statistics['num_ticks_bar'].append(current_num_ticks)
        
        # Calculate final θT = max{∑bt,i+, ∑bt,i-} for this bar
        total_pos_runs = self.pos_runs
        total_neg_runs = self.neg_runs
        
        # Include final incomplete run
        if self.current_run_direction == 1:
            total_pos_runs += self.current_run_value
        elif self.current_run_direction == -1:
            total_neg_runs += self.current_run_value
            
        bar_theta = max(total_pos_runs, total_neg_runs)
        self.expected_runs_history.append(bar_theta)
        
        # Keep only recent history
        max_history = max(self.num_prev_bars, 50)  # At least 50 for stability
        if len(self.expected_runs_history) > max_history:
            self.expected_runs_history = self.expected_runs_history[-max_history:]
            
        if len(self.run_tick_statistics['num_ticks_bar']) > max_history:
            self.run_tick_statistics['num_ticks_bar'] = \
                self.run_tick_statistics['num_ticks_bar'][-max_history:]
                
        # Update E[θ] (expected runs)
        if len(self.expected_runs_history) >= self.num_prev_bars:
            if self.alpha is not None:
                # Use EWMA for adaptive estimation
                if not hasattr(self, '_ewma_expected_runs'):
                    self._ewma_expected_runs = np.mean(self.expected_runs_history[-self.num_prev_bars:])
                else:
                    self._ewma_expected_runs = (self.alpha * bar_theta + 
                                              (1 - self.alpha) * self._ewma_expected_runs)
                self.expected_runs = self._ewma_expected_runs
            else:
                # Use simple average
                self.expected_runs = np.mean(self.expected_runs_history[-self.num_prev_bars:])
            
        # Update E[T] (expected number of ticks)
        if len(self.run_tick_statistics['num_ticks_bar']) >= self.num_prev_bars:
            if self.alpha is not None:
                # Use EWMA for E[T]
                if not hasattr(self, '_ewma_num_ticks'):
                    self._ewma_num_ticks = np.mean(self.run_tick_statistics['num_ticks_bar'][-self.num_prev_bars:])
                else:
                    self._ewma_num_ticks = (self.alpha * current_num_ticks + 
                                          (1 - self.alpha) * self._ewma_num_ticks)
                avg_num_ticks = self._ewma_num_ticks
            else:
                # Use simple average
                avg_num_ticks = np.mean(self.run_tick_statistics['num_ticks_bar'][-self.num_prev_bars:])
            
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


class EMARunBars(RunBars):
    """
    EMA Run Bars implementation
    Uses Exponentially Weighted Moving Average for adaptive threshold estimation
    """
    
    def __init__(self, metric: str, alpha: float = 0.3, **kwargs):
        """
        Constructor for EMA Run Bars
        
        :param metric: (str) Type of run bar
        :param alpha: (float) EMA smoothing factor (0 < alpha <= 1)
        """
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
            
        # Pass alpha to parent class
        super().__init__(metric, alpha=alpha, **kwargs)
        self.alpha = alpha


# Public API functions
def get_tick_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                     num_prev_bars: int = 3, expected_runs_window: int = 100,
                     exp_num_ticks_init: int = 20000, exp_num_ticks_constraints: list = None,
                     batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, 
                     output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
    """
    Creates tick run bars
    """
    bars = RunBars(metric='tick_run', num_prev_bars=num_prev_bars,
                  expected_runs_window=expected_runs_window,
                  exp_num_ticks_init=exp_num_ticks_init,
                  exp_num_ticks_constraints=exp_num_ticks_constraints,
                  batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)


def get_volume_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                       num_prev_bars: int = 3, expected_runs_window: int = 100,
                       exp_num_ticks_init: int = 20000, exp_num_ticks_constraints: list = None,
                       batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, 
                       output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
    """
    Creates volume run bars
    """
    bars = RunBars(metric='volume_run', num_prev_bars=num_prev_bars,
                  expected_runs_window=expected_runs_window,
                  exp_num_ticks_init=exp_num_ticks_init,
                  exp_num_ticks_constraints=exp_num_ticks_constraints,
                  batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)


def get_dollar_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                       num_prev_bars: int = 3, expected_runs_window: int = 100,
                       exp_num_ticks_init: int = 20000, exp_num_ticks_constraints: list = None,
                       batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, 
                       output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
    """
    Creates dollar run bars
    """
    bars = RunBars(metric='dollar_run', num_prev_bars=num_prev_bars,
                  expected_runs_window=expected_runs_window,
                  exp_num_ticks_init=exp_num_ticks_init,
                  exp_num_ticks_constraints=exp_num_ticks_constraints,
                  batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)


# EMA variants
def get_ema_tick_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                         alpha: float = 0.3, **kwargs) -> Union[pd.DataFrame, None]:
    """Creates EMA tick run bars"""
    bars = EMARunBars(metric='tick_run', alpha=alpha, **kwargs)
    return bars.batch_run(file_path_or_df, **kwargs)


def get_ema_volume_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                           alpha: float = 0.3, **kwargs) -> Union[pd.DataFrame, None]:
    """Creates EMA volume run bars"""
    bars = EMARunBars(metric='volume_run', alpha=alpha, **kwargs)
    return bars.batch_run(file_path_or_df, **kwargs)


def get_ema_dollar_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                           alpha: float = 0.3, **kwargs) -> Union[pd.DataFrame, None]:
    """Creates EMA dollar run bars"""
    bars = EMARunBars(metric='dollar_run', alpha=alpha, **kwargs)
    return bars.batch_run(file_path_or_df, **kwargs)


# Const variants (using fixed window)
def get_const_tick_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                           **kwargs) -> Union[pd.DataFrame, None]:
    """Creates const tick run bars"""
    return get_tick_run_bars(file_path_or_df, **kwargs)


def get_const_volume_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                             **kwargs) -> Union[pd.DataFrame, None]:
    """Creates const volume run bars"""
    return get_volume_run_bars(file_path_or_df, **kwargs)


def get_const_dollar_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                             **kwargs) -> Union[pd.DataFrame, None]:
    """Creates const dollar run bars"""
    return get_dollar_run_bars(file_path_or_df, **kwargs)
