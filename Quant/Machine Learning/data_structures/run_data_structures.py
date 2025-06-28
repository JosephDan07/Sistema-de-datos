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
except ImportError:
    from base_bars import BaseBars, ewma


class RunBars(BaseBars):
    """
    Contains all of the logic to construct run bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_tick_run_bars which will create an instance of this
    class and then construct the run bars.
    """

    def __init__(self, metric: str, num_prev_bars: int = 3, expected_runs_window: int = 100, 
                 exp_num_ticks_init: int = 20000, exp_num_ticks_constraints: list = None,
                 batch_size: int = 20000000):
        """
        Constructor

        :param metric: (str) Type of run bar to create. Example: "tick_run", "volume_run", "dollar_run"
        :param num_prev_bars: (int) Number of previous bars used for expected runs window
        :param expected_runs_window: (int) Rolling window used for expected runs and number of ticks estimation
        :param exp_num_ticks_init: (int) Initial expected number of ticks per bar
        :param exp_num_ticks_constraints: (list) Constraints on expected number of ticks [min, max]
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        """
        super().__init__(metric, batch_size)
        
        # Run bar specific parameters
        self.metric = metric
        self.num_prev_bars = num_prev_bars
        self.expected_runs_window = expected_runs_window
        self.exp_num_ticks_init = exp_num_ticks_init
        self.exp_num_ticks_constraints = exp_num_ticks_constraints or [10, np.inf]
        
        # Run tracking
        self.run_tick_statistics = {'num_ticks_bar': []}
        self.expected_runs_window = []
        self.runs_array = []
        self.warm_up_flag = False
        
        # Current run tracking
        self.current_run_value = 0
        self.current_run_direction = 0
        self.pos_runs = 0
        self.neg_runs = 0
        
        # Expected values
        self.expected_runs = np.nan
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
        
        :param run_value: (float) Current run value (signed)
        """
        if run_value > 0:
            if self.current_run_direction != 1:
                # Starting new positive run
                if self.current_run_direction == -1:
                    # Ending previous negative run
                    self.neg_runs += abs(self.current_run_value)
                self.current_run_direction = 1
                self.current_run_value = run_value
            else:
                # Continuing positive run
                self.current_run_value += run_value
                
        elif run_value < 0:
            if self.current_run_direction != -1:
                # Starting new negative run
                if self.current_run_direction == 1:
                    # Ending previous positive run
                    self.pos_runs += self.current_run_value
                self.current_run_direction = -1
                self.current_run_value = run_value
            else:
                # Continuing negative run
                self.current_run_value += run_value
                
        # If run_value == 0, continue current run without change
        
    def _should_create_bar(self) -> bool:
        """
        Check if a bar should be created based on run conditions
        
        Following the formula: θT = max{∑bt,i+, ∑bt,i-}
        
        :return: (bool) True if bar should be created
        """
        # Always create first bar after expected number of ticks
        if len(self.run_tick_statistics['num_ticks_bar']) == 0:
            return self.cum_statistics['cum_ticks'] >= self.expected_num_ticks
            
        # Add current run to appropriate total
        total_pos_runs = self.pos_runs
        total_neg_runs = self.neg_runs
        
        if self.current_run_direction == 1:
            total_pos_runs += self.current_run_value
        elif self.current_run_direction == -1:
            total_neg_runs += abs(self.current_run_value)
            
        # Calculate θT = max{positive runs, negative runs}
        current_runs = max(total_pos_runs, total_neg_runs)
        
        # Create bar if runs exceed expected threshold or reached expected number of ticks
        return (current_runs >= self.expected_runs or 
                self.cum_statistics['cum_ticks'] >= self.expected_num_ticks)
                
    def _update_run_statistics(self):
        """
        Update the expected runs and number of ticks for future bars
        """
        # Store number of ticks in this bar
        self.run_tick_statistics['num_ticks_bar'].append(self.cum_statistics['cum_ticks'])
        
        # Calculate final run value for this bar
        total_pos_runs = self.pos_runs
        total_neg_runs = self.neg_runs
        
        if self.current_run_direction == 1:
            total_pos_runs += self.current_run_value
        elif self.current_run_direction == -1:
            total_neg_runs += abs(self.current_run_value)
            
        bar_runs = max(total_pos_runs, total_neg_runs)
        self.expected_runs_window.append(bar_runs)
        
        # Keep only recent window
        if len(self.expected_runs_window) > self.num_prev_bars:
            self.expected_runs_window = self.expected_runs_window[-self.num_prev_bars:]
            
        if len(self.run_tick_statistics['num_ticks_bar']) > self.num_prev_bars:
            self.run_tick_statistics['num_ticks_bar'] = \
                self.run_tick_statistics['num_ticks_bar'][-self.num_prev_bars:]
                
        # Update expected runs (using mean)
        if len(self.expected_runs_window) >= self.num_prev_bars:
            self.expected_runs = np.mean(self.expected_runs_window[-self.num_prev_bars:])
            
        # Update expected number of ticks
        if len(self.run_tick_statistics['num_ticks_bar']) >= self.num_prev_bars:
            self.expected_num_ticks = int(np.mean(self.run_tick_statistics['num_ticks_bar'][-self.num_prev_bars:]))
            
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
    """
    
    def __init__(self, metric: str, alpha: float = 0.3, **kwargs):
        """
        Constructor for EMA Run Bars
        
        :param metric: (str) Type of run bar
        :param alpha: (float) EMA smoothing factor
        """
        super().__init__(metric, **kwargs)
        self.alpha = alpha
        self.ema_expected_runs = None
        self.ema_expected_num_ticks = None
        
    def _update_run_statistics(self):
        """
        Update expected values using EMA
        """
        super()._update_run_statistics()
        
        # Calculate EMA for expected runs
        total_pos_runs = self.pos_runs
        total_neg_runs = self.neg_runs
        
        if self.current_run_direction == 1:
            total_pos_runs += self.current_run_value
        elif self.current_run_direction == -1:
            total_neg_runs += abs(self.current_run_value)
            
        current_runs = max(total_pos_runs, total_neg_runs)
        
        if self.ema_expected_runs is None:
            self.ema_expected_runs = current_runs
        else:
            self.ema_expected_runs = (self.alpha * current_runs + 
                                    (1 - self.alpha) * self.ema_expected_runs)
            
        # Calculate EMA for expected number of ticks
        current_num_ticks = self.cum_statistics['cum_ticks']
        if self.ema_expected_num_ticks is None:
            self.ema_expected_num_ticks = current_num_ticks
        else:
            self.ema_expected_num_ticks = (self.alpha * current_num_ticks + 
                                         (1 - self.alpha) * self.ema_expected_num_ticks)
            
        # Update expected values
        self.expected_runs = self.ema_expected_runs
        self.expected_num_ticks = max(self.exp_num_ticks_constraints[0], 
                                    min(int(self.ema_expected_num_ticks), self.exp_num_ticks_constraints[1]))


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
