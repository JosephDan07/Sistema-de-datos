"""
Logic regarding labeling from chapter 3. In particular the Triple Barrier Method and Meta-Labeling.
Based on Advances in Financial Machine Learning, López de Prado

This module implements the core labeling techniques described in Chapter 3:
- Triple Barrier Method
- Meta-Labeling
- CUSUM Filter for event detection
- Daily volatility estimation
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def mp_pandas_obj(func, pd_obj, num_threads=1, **kwargs):
    """
    Parallelize jobs, return a pandas object
    
    :param func: function to be parallelized. Returns a DataFrame
    :param pd_obj: Name of argument used to pass the molecule
    :param num_threads: Number of cores
    :param kwargs: any other arguments needed by func
    :return: DataFrame
    """
    
    if num_threads == 1:
        out = func(pd_obj[1], **kwargs)
    else:
        tasks = []
        parts = np.array_split(pd_obj[1], num_threads)
        
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            for part in parts:
                task = executor.submit(func, part, **kwargs)
                tasks.append(task)
            
            out = []
            for task in as_completed(tasks):
                out.append(task.result())
        
        if isinstance(out[0], pd.DataFrame):
            out = pd.concat(out, sort=False)
        elif isinstance(out[0], pd.Series):
            out = pd.concat(out, sort=False)
        else:
            # If not pandas objects, concatenate differently
            out = pd.concat([pd.DataFrame(x) for x in out], sort=False)
    
    return out


def apply_pt_sl_on_t1(molecule: List[datetime], close: pd.Series, events: pd.DataFrame, pt_sl: np.array) -> pd.DataFrame:
    """
    Advances in Financial Machine Learning, Snippet 3.2, page 45.

    Triple Barrier Labeling Method

    This function applies the triple-barrier labeling method. It works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.

    Mainly it returns a DataFrame of timestamps regarding the time when the first barriers were reached.

    :param close: (pd.Series) Close prices
    :param events: (pd.DataFrame) Events DataFrame with 't1' and 'trgt' columns
    :param pt_sl: (np.array) Element 0, indicates the profit taking level; Element 1 is stop loss level
    :param molecule: (List[datetime]) A set of datetime index values for processing
    :return: (pd.DataFrame) Timestamps of when first barrier was touched
    """
    
    # Apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs
    
    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs
    
    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # Path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # Path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # Earliest stop loss
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # Earliest profit taking
    
    return out


def add_vertical_barrier(t_events: pd.Series, close: pd.Series, num_days: int = 0, 
                        num_hours: int = 0, num_minutes: int = 0, num_seconds: int = 0) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 3.4 page 49.

    Adding a Vertical Barrier

    For each index in t_events, it finds the timestamp of the next price bar at or immediately after
    a number of days num_days. This vertical barrier can be passed as an optional argument t1 in get_events.

    This function creates a series that has all the timestamps of when the vertical barrier would be reached.

    :param t_events: (pd.Series) Series of events (symmetric CUSUM filter)
    :param close: (pd.Series) Close prices
    :param num_days: (int) Number of days to add for vertical barrier
    :param num_hours: (int) Number of hours to add for vertical barrier
    :param num_minutes: (int) Number of minutes to add for vertical barrier
    :param num_seconds: (int) Number of seconds to add for vertical barrier
    :return: (pd.Series) Timestamps of vertical barriers
    """
    
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days, 
                                                          hours=num_hours,
                                                          minutes=num_minutes, 
                                                          seconds=num_seconds))
    t1 = t1[t1 < close.shape[0]]
    t1 = (pd.Series(close.index[t1], index=t_events[:t1.shape[0]]))
    
    return t1


def get_events(close: pd.Series, t_events: pd.Series, pt_sl: np.array, target: pd.Series, 
               min_ret: float, num_threads: int = 1, vertical_barrier_times: Union[pd.Series, bool] = False,
               side_prediction: Optional[pd.Series] = None, verbose: bool = True) -> pd.DataFrame:
    """
    Advances in Financial Machine Learning, Snippet 3.6 page 50.

    Getting the Time of the First Touch, with Meta Labels

    This function is orchestrator to meta-label the data, in conjunction with the Triple Barrier Method.

    :param close: (pd.Series) Close prices
    :param t_events: (pd.Series) of t_events. These are timestamps that will seed every triple barrier.
        These are the timestamps selected by the sampling procedures discussed in Chapter 2, Section 2.5.
        Eg: CUSUM Filter
    :param pt_sl: (2 element array) Element 0, indicates the profit taking level; Element 1 is stop loss level.
        A non-negative float that sets the width of the two barriers. A 0 value means that the respective
        horizontal barrier (profit taking and/or stop loss) will be disabled.
    :param target: (pd.Series) of values that are used (in conjunction with pt_sl) to determine the width
        of the barrier. In this program this is daily volatility series.
    :param min_ret: (float) The minimum target return required for running a triple barrier search.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param vertical_barrier_times: (pd.Series) A pandas series with the timestamps of the vertical barriers.
        We pass a False when we want to disable vertical barriers.
    :param side_prediction: (pd.Series) Side of the bet (long/short) as decided by the primary model
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.DataFrame) Events
            -events.index is event's starttime
            -events['t1'] is event's endtime
            -events['trgt'] is event's target
            -events['side'] (optional) implies the algo's position side
            -events['pt'] is profit taking multiple
            -events['sl']  is stop loss multiple
    """
    
    # 1) Get target
    target = target.loc[t_events]
    target = target[target > min_ret]  # min_ret
    
    # 2) Get vertical barrier (max holding period)
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events)
    
    # 3) Form events object, apply stop loss on vertical barrier
    if side_prediction is None:
        side_ = pd.Series(1., index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[1]]
    else:
        side_ = side_prediction.loc[target.index]
        pt_sl_ = [pt_sl[0], pt_sl[0]]  # meta-labeling
    
    events = (pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_}, axis=1)
              .dropna(subset=['trgt']))
    
    # Apply triple barrier method (simplified without parallelization for now)
    if num_threads == 1:
        df0 = apply_pt_sl_on_t1(molecule=events.index, close=close, events=events, pt_sl=pt_sl_)
    else:
        # For now, fall back to single thread
        df0 = apply_pt_sl_on_t1(molecule=events.index, close=close, events=events, pt_sl=pt_sl_)
    
    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores NaN
    
    if side_prediction is None:
        events = events.drop('side', axis=1)
    
    # Add profit taking and stop loss multiples for future use
    events['pt'] = pt_sl[0]
    events['sl'] = pt_sl[1]
    
    return events


def barrier_touched(out_df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Advances in Financial Machine Learning, Snippet 3.9, page 55, Question 3.3.

    Adjust the getBins function (Snippet 3.7) to return a 0 whenever the vertical barrier is the one touched first.

    Top horizontal barrier: 1
    Bottom horizontal barrier: -1
    Vertical barrier: 0

    :param out_df: (pd.DataFrame) Returns and target
    :param events: (pd.DataFrame) The original events data frame. Contains the pt sl multiples needed here.
    :return: (pd.DataFrame) Returns, target, and labels
    """
    
    store = []
    for date_time, values in out_df.iterrows():
        
        ret = values['ret']
        target = values['trgt']
        
        pt_level_reached = ret > (events.loc[date_time, 'pt'] * target)
        sl_level_reached = ret < (-events.loc[date_time, 'sl'] * target)
        
        if ret > 0 and pt_level_reached:
            # Top barrier reached
            store.append(1)
        elif ret < 0 and sl_level_reached:
            # Bottom barrier reached  
            store.append(-1)
        else:
            # Vertical barrier reached
            store.append(0)
    
    # Assign the labels
    out_df['bin'] = store
    
    return out_df


def get_bins(triple_barrier_events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    Advances in Financial Machine Learning, Snippet 3.7, page 51.

    Labeling for Side & Size with Meta Labels

    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:

    Now the possible values for labels in out['bin'] are {0,1}, as opposed to whether to take the bet or pass,
    a purely binary prediction. When the predicted label the previous feasible values {-1,0,1}.
    The ML algorithm will be trained to decide is 1, we can use the probability of this secondary prediction
    to derive the size of the bet, where the side (sign) of the position has been set by the primary model.

    :param triple_barrier_events: (pd.DataFrame)
                -events.index is event's starttime
                -events['t1'] is event's endtime
                -events['trgt'] is event's target
                -events['side'] (optional) implies the algo's position side
                Case 1: ('side' not in events): bin in (-1,1) <-label by price action
                Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    :param close: (pd.Series) Close prices
    :return: (pd.DataFrame) Meta-labeled events
    """
    
    # 1) Prices aligned with events
    events_ = triple_barrier_events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    
    # 2) Create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    
    if 'side' in events_:
        out['ret'] *= events_['side']  # meta-labeling
    
    out['trgt'] = events_['trgt']
    out['bin'] = np.sign(out['ret'])
    
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
    
    return out


def drop_labels(events: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
    """
    Advances in Financial Machine Learning, Snippet 3.8 page 54.

    This function recursively eliminates rare observations.

    :param events: (dp.DataFrame) Events.
    :param min_pct: (float) A fraction used to decide if the observation occurs less than that fraction.
    :return: (pd.DataFrame) Events.
    """
    
    # Apply weights, drop labels with insufficient examples
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        
        print('Dropped label:', df0.argmin(), df0.min())
        events = events[events['bin'] != df0.argmin()]
    
    return events


def get_daily_vol(close: pd.Series, span0: int = 100) -> pd.Series:
    """
    Snippet 3.1, page 44, Daily Volatility Estimates
    
    Compute daily volatility estimates using EWMA
    
    :param close: (pd.Series) Close prices
    :param span0: (int) Span for EWMA calculation
    :return: (pd.Series) Daily volatility estimates
    """
    
    # Daily returns
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - 1], 
                     index=close.index[close.shape[0] - df0.shape[0]:]))
    
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # Daily returns
    df0 = df0.ewm(span=span0).std()
    
    return df0


def cusum_filter(raw_time_series: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """
    CUSUM filter for sampling
    
    The CUSUM filter is a quality-control method, designed to detect a shift in the 
    mean value of a measured quantity away from a target value.
    
    :param raw_time_series: (pd.Series) Close prices or other time series
    :param threshold: (float) When the cumulative sum exceeds this threshold, the function captures
    :return: (pd.DatetimeIndex) Filtered events timestamps  
    """
    
    t_events = []
    s_pos = 0
    s_neg = 0
    
    diff = raw_time_series.diff().dropna()
    
    for i in diff.index[1:]:
        pos = float(s_pos + diff.loc[i])
        neg = float(s_neg + diff.loc[i])
        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)
        
        if s_neg < -threshold:
            s_neg = 0
            t_events.append(i)
        elif s_pos > threshold:
            s_pos = 0
            t_events.append(i)
    
    return pd.DatetimeIndex(t_events)


# Main execution
if __name__ == "__main__":
    print("Triple Barrier Labeling Module loaded successfully!")
    print("Use help() on any function for detailed documentation.")
