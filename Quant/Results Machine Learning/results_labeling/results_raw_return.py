"""
Labeling Raw Returns.

Most basic form of labeling based on raw return of each observation relative to its previous value.
"""

import numpy as np
import pandas as pd
from typing import Union


def raw_return(prices: Union[pd.Series, pd.DataFrame], binary: bool = False, logarithmic: bool = False, 
               resample_by: str = None, lag: bool = True) -> Union[pd.Series, pd.DataFrame]:
    """
    Raw returns labeling method.

    This is the most basic and ubiquitous labeling method used as a precursor to almost any kind of financial data
    analysis or machine learning. User can specify simple or logarithmic returns, numerical or binary labels, a
    resample period, and whether returns are lagged to be forward looking.

    :param prices: (pd.Series or pd.DataFrame) Time-indexed price data on stocks with which to calculate return.
    :param binary: (bool) If False, will return numerical returns. If True, will return the sign of the raw return.
    :param logarithmic: (bool) If False, will calculate simple returns. If True, will calculate logarithmic returns.
    :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns. 'B' = per
                        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.
                        For full details see `here.
                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
    :param lag: (bool) If True, returns will be lagged to make them forward-looking.
    :return:  (pd.Series or pd.DataFrame) Raw returns on market data. User can specify whether returns will be based on
                simple or logarithmic return, and whether the output will be numerical or categorical.
    """
    
    # Resample if requested
    if resample_by is not None:
        prices = prices.resample(resample_by).last()
    
    # Calculate returns
    if logarithmic:
        returns = np.log(prices).diff()
    else:
        returns = prices.pct_change()
    
    # Lag returns if requested (forward-looking)
    if lag:
        returns = returns.shift(-1)
    
    # Convert to binary if requested
    if binary:
        returns = np.sign(returns)
    
    return returns


if __name__ == "__main__":
    print("Raw Return labeling module loaded successfully!")
    
    # Example usage
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.02), index=dates)
    
    # Raw returns
    raw_rets = raw_return(prices, binary=False, logarithmic=False)
    print(f"Sample raw returns: {raw_rets.dropna().head()}")
    
    # Binary labels
    binary_labels = raw_return(prices, binary=True, logarithmic=False)
    print(f"Sample binary labels: {binary_labels.dropna().head()}")
