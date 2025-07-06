"""
Chapter 3.2 Fixed-Time Horizon Method, in Advances in Financial Machine Learning, by M. L. de Prado.

Work "Classification-based Financial Markets Prediction using Deep Neural Networks" by Dixon et al. (2016) describes how
labeling data this way can be used in training deep neural networks to predict price movements.
"""

import warnings
import pandas as pd
import numpy as np
from typing import Union


def fixed_time_horizon(prices: Union[pd.Series, pd.DataFrame], threshold: Union[float, pd.Series] = 0, 
                      resample_by: str = None, lag: bool = True, standardized: bool = False, 
                      window: int = None) -> Union[pd.Series, pd.DataFrame]:
    """
    Fixed-Time Horizon Labeling Method.

    Originally described in the book Advances in Financial Machine Learning, Chapter 3.2, p.43-44.

    Returns 1 if return is greater than the threshold, -1 if less, and 0 if in between. If no threshold is
    provided then it will simply take the sign of the return.

    :param prices: (pd.Series or pd.DataFrame) Time-indexed stock prices used to calculate returns.
    :param threshold: (float or pd.Series) When the absolute value of return exceeds the threshold, the observation is
                    labeled with 1 or -1, depending on the sign of the return. If return is less, it's labeled as 0.
                    Can be dynamic if threshold is inputted as a pd.Series, and threshold.index must match prices.index.
                    If resampling is used, the index of threshold must match the index of prices after resampling.
                    If threshold is negative, then the directionality of the labels will be reversed. If no threshold
                    is provided, it is assumed to be 0 and the sign of the return is returned.
    :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns. 'B' = per
                        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.
                        For full details see `here.
                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
    :param lag: (bool) If True, returns will be lagged to make them forward-looking.
    :param standardized: (bool) Whether returns are scaled by mean and standard deviation.
    :param window: (int) If standardized is True, the rolling window period for calculating the mean and standard
                    deviation of returns.
    :return: (pd.Series or pd.DataFrame) -1, 0, or 1 denoting whether the return for each observation is
                    less/between/greater than the threshold at each corresponding time index. First or last row will be
                    NaN, depending on lag.
    """
    
    # Resample if requested
    if resample_by is not None:
        prices = prices.resample(resample_by).last()
        if isinstance(threshold, pd.Series):
            threshold = threshold.resample(resample_by).last()
    
    # Calculate returns
    returns = prices.pct_change()
    
    # Lag returns if requested (forward-looking)
    if lag:
        returns = returns.shift(-1)
    
    # Standardize returns if requested
    if standardized:
        if window is None:
            warnings.warn("Window parameter must be specified when standardized=True. Using full sample.")
            returns = (returns - returns.mean()) / returns.std()
        else:
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
            returns = (returns - rolling_mean) / rolling_std
    
    # Apply threshold logic
    if isinstance(threshold, (int, float)):
        # Static threshold
        if threshold == 0:
            # Simple sign of returns
            labels = np.sign(returns)
        else:
            # Threshold-based labeling
            labels = pd.Series(index=returns.index, dtype=float)
            labels[returns > threshold] = 1
            labels[returns < -threshold] = -1
            labels[(returns >= -threshold) & (returns <= threshold)] = 0
    else:
        # Dynamic threshold (pd.Series)
        if not isinstance(threshold, pd.Series):
            raise ValueError("Threshold must be a float or pd.Series")
        
        # Align threshold with returns
        threshold = threshold.reindex(returns.index, method='ffill')
        
        labels = pd.Series(index=returns.index, dtype=float)
        labels[returns > threshold] = 1
        labels[returns < -threshold] = -1
        labels[(returns >= -threshold) & (returns <= threshold)] = 0
    
    return labels


def get_daily_vol_simple(close: pd.Series, lookback: int = 100) -> pd.Series:
    """
    Simple daily volatility estimation for use with fixed time horizon labeling.
    
    :param close: (pd.Series) Close prices
    :param lookback: (int) Number of days to use for volatility calculation
    :return: (pd.Series) Daily volatility estimates
    """
    returns = close.pct_change().dropna()
    return returns.rolling(window=lookback).std()


if __name__ == "__main__":
    # Example usage
    print("Fixed Time Horizon Labeling module loaded successfully!")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.02), index=dates)
    
    # Basic labeling
    labels = fixed_time_horizon(prices, threshold=0.01)
    print(f"Sample labels: {labels.dropna().head()}")
