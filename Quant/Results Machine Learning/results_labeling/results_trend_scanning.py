"""
Implementation of Trend-Scanning labels described in `Advances in Financial Machine Learning: Lecture 3/10
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>`_

Based on López de Prado's methodology for trend-scanning labeling.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import warnings


def get_betas(price_series: pd.Series, add_constant: bool = True) -> np.ndarray:
    """
    Calculate regression betas for trend analysis.
    
    :param price_series: (pd.Series) Price series for regression
    :param add_constant: (bool) Whether to add constant term to regression
    :return: (np.ndarray) Beta coefficients
    """
    if len(price_series) < 2:
        return np.array([np.nan, np.nan]) if add_constant else np.array([np.nan])
    
    y = np.log(price_series).values
    x = np.arange(len(y)).reshape(-1, 1)
    
    if add_constant:
        x = np.column_stack([np.ones(len(y)), x])
    
    try:
        # Use least squares regression
        betas = np.linalg.lstsq(x, y, rcond=None)[0]
        return betas
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan]) if add_constant else np.array([np.nan])


def trend_scanning_labels(price_series: pd.Series, t_events: Optional[List] = None, observation_window: int = 20,
                          look_forward: bool = True, min_sample_length: int = 5, step: int = 1) -> pd.DataFrame:
    """
    `Trend scanning <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419>`_ is both a classification and
    regression labeling technique.

    That can be used in the following ways:

    1. Classification: By taking the sign of t-value for a given observation we can set {-1, 1} labels to define the
       trends as either downward or upward.
    2. Classification: By adding a minimum t-value threshold you can generate {-1, 0, 1} labels for downward, no-trend,
       upward.
    3. The t-values can be used as sample weights in classification problems.
    4. Regression: The t-values can be used in a regression setting to determine the magnitude of the trend.

    The output of this algorithm is a DataFrame with t1 (time stamp for the farthest observation), t-value, returns for
    the trend, and bin.

    This function allows using both forward-looking and backward-looking window (use the look_forward parameter).

    :param price_series: (pd.Series) Close prices used to label the data set
    :param t_events: (List) Filtered events, array of pd.Timestamps
    :param observation_window: (int) Maximum look forward window used to get the trend value
    :param look_forward: (bool) True if using a forward-looking window, False if using a backward-looking one
    :param min_sample_length: (int) Minimum sample length used to fit regression
    :param step: (int) Optimal t-value index is searched every 'step' indices
    :return: (pd.DataFrame) Consists of t1, t-value, ret, bin (label information). t1 - label endtime, tvalue,
        ret - price change %, bin - label value based on price change sign
    """
    
    if t_events is None:
        t_events = price_series.index
    
    # Convert to list if needed
    if isinstance(t_events, pd.DatetimeIndex):
        t_events = t_events.tolist()
    
    labels = []
    
    for event_time in t_events:
        if event_time not in price_series.index:
            continue
            
        event_idx = price_series.index.get_loc(event_time)
        
        if look_forward:
            # Forward-looking window
            start_idx = event_idx
            end_idx = min(event_idx + observation_window, len(price_series))
        else:
            # Backward-looking window
            start_idx = max(event_idx - observation_window, 0)
            end_idx = event_idx + 1
        
        if end_idx - start_idx < min_sample_length:
            continue
        
        # Get the price window
        window_prices = price_series.iloc[start_idx:end_idx]
        
        if len(window_prices) < min_sample_length:
            continue
        
        # Find optimal t-value by testing different endpoints
        max_abs_tvalue = 0
        best_t_value = 0
        best_end_idx = start_idx + min_sample_length - 1
        best_ret = 0
        
        # Search for optimal endpoint
        for end_search_idx in range(start_idx + min_sample_length - 1, end_idx, step):
            sub_window = price_series.iloc[start_idx:end_search_idx + 1]
            
            if len(sub_window) < min_sample_length:
                continue
            
            # Calculate regression coefficients
            betas = get_betas(sub_window, add_constant=True)
            
            if len(betas) < 2 or np.isnan(betas).any():
                continue
            
            # Calculate t-statistic for the slope coefficient
            y = np.log(sub_window).values
            x = np.arange(len(y))
            
            # Predicted values
            y_pred = betas[0] + betas[1] * x
            
            # Residuals and standard error
            residuals = y - y_pred
            mse = np.sum(residuals**2) / (len(y) - 2) if len(y) > 2 else np.inf
            
            if mse == 0 or np.isinf(mse):
                continue
            
            # Standard error of slope
            x_centered = x - np.mean(x)
            se_slope = np.sqrt(mse / np.sum(x_centered**2)) if np.sum(x_centered**2) > 0 else np.inf
            
            if se_slope == 0 or np.isinf(se_slope):
                continue
            
            # t-statistic
            t_value = betas[1] / se_slope
            
            if abs(t_value) > max_abs_tvalue:
                max_abs_tvalue = abs(t_value)
                best_t_value = t_value
                best_end_idx = end_search_idx
                best_ret = (sub_window.iloc[-1] / sub_window.iloc[0]) - 1
        
        # Create label entry
        if max_abs_tvalue > 0:
            labels.append({
                't1': price_series.index[best_end_idx],
                'tvalue': best_t_value,
                'ret': best_ret,
                'bin': np.sign(best_t_value)
            })
    
    if not labels:
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=['t1', 'tvalue', 'ret', 'bin'])
    
    result_df = pd.DataFrame(labels)
    result_df.index = [label_data['t1'] if 't1' in label_data else idx 
                      for idx, label_data in enumerate(labels)]
    
    return result_df
