"""
Return in excess of median method.

Described in "The benefits of tree-based models for stock selection", Zhu et al. (2012). Data labeled this way can be
used in regression and classification models to predict stock returns over market.
"""
import numpy as np
import pandas as pd
from typing import Union


def excess_over_median(prices: pd.DataFrame, binary: bool = False, resample_by: str = None, 
                      lag: bool = True) -> pd.DataFrame:
    """
    Return in excess of median labeling method. Sourced from "The benefits of tree-based models for stock selection"
    Zhu et al. (2012).

    Returns a DataFrame containing returns of stocks over the median of all stocks in the portfolio, or returns a
    DataFrame containing signs of those returns. In the latter case, an observation may be labeled as 0 if it itself is
    the median.

    :param prices: (pd.DataFrame) Close prices of all stocks in the market that are used to establish the median.
                   Returns on each stock are then compared to the median for the given timestamp.
    :param binary: (bool) If False, the numerical value of excess returns over median will be given. If True, then only
                    the sign of the excess return over median will be given (-1 or 1). A label of 0 will be given if
                    the observation itself is the median. According to Zhu et al., categorical labels can alleviate
                    issues with extreme outliers present with numerical labels.
    :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns. 'B' = per
                        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.
                        For full details see `here.
                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
    :param lag: (bool) If True, returns will be lagged to make them forward-looking.
    :return: (pd.DataFrame) Numerical returns in excess of the market median return, or sign of return depending on
                    whether binary is False or True respectively.
    """
    
    # Resample if requested
    if resample_by is not None:
        prices = prices.resample(resample_by).last()
    
    # Calculate returns for all stocks
    returns = prices.pct_change()
    
    # Lag returns if requested (forward-looking)
    if lag:
        returns = returns.shift(-1)
    
    # Calculate median return for each timestamp
    median_returns = returns.median(axis=1)
    
    # Calculate excess returns over median
    excess_returns = returns.subtract(median_returns, axis=0)
    
    # Convert to binary if requested
    if binary:
        excess_returns = np.sign(excess_returns)
    
    return excess_returns


if __name__ == "__main__":
    print("Excess Over Median labeling module loaded successfully!")
    
    # Example usage
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Generate random price data
    price_data = {}
    for asset in assets:
        price_data[asset] = 100 + np.cumsum(np.random.randn(100) * 0.02)
    
    prices = pd.DataFrame(price_data, index=dates)
    
    # Calculate excess over median
    excess_rets = excess_over_median(prices, binary=False)
    print(f"Sample excess returns: {excess_rets.dropna().head()}")
    
    # Binary labels
    binary_labels = excess_over_median(prices, binary=True)
    print(f"Sample binary labels: {binary_labels.dropna().head()}")
