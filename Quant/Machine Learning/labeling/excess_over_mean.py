"""
Return in excess of mean method.

Chapter 5, Machine Learning for Factor Investing, by Coqueret and Guida, (2020).
"""
import numpy as np
import pandas as pd
from typing import Union


def excess_over_mean(prices: pd.DataFrame, binary: bool = False, resample_by: str = None, 
                    lag: bool = True) -> pd.DataFrame:
    """
    Return in excess of mean labeling method. Sourced from Chapter 5.5.1 of Machine Learning for Factor Investing,
    by Coqueret, G. and Guida, T. (2020).

    Returns a DataFrame containing returns of stocks over the mean of all stocks in the portfolio. Returns a DataFrame
    of signs of the returns if binary is True. In this case, an observation may be labeled as 0 if it itself is the
    mean.

    :param prices: (pd.DataFrame) Close prices of all tickers in the market that are used to establish the mean. NaN
                    values are ok. Returns on each ticker are then compared to the mean for the given timestamp.
    :param binary: (bool) If False, the numerical value of excess returns over mean will be given. If True, then only
                    the sign of the excess return over mean will be given (-1 or 1). A label of 0 will be given if
                    the observation itself equal to the mean.
    :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns. 'B' = per
                        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.
                        For full details see `here.
                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
    :param lag: (bool) If True, returns will be lagged to make them forward-looking.
    :return: (pd.DataFrame) Numerical returns in excess of the market mean return, or sign of return depending on
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
    
    # Calculate mean return for each timestamp (excluding NaN values)
    mean_returns = returns.mean(axis=1, skipna=True)
    
    # Calculate excess returns over mean
    excess_returns = returns.subtract(mean_returns, axis=0)
    
    # Convert to binary if requested
    if binary:
        excess_returns = np.sign(excess_returns)
    
    return excess_returns


if __name__ == "__main__":
    print("Excess Over Mean labeling module loaded successfully!")
    
    # Example usage
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Generate random price data
    price_data = {}
    for asset in assets:
        price_data[asset] = 100 + np.cumsum(np.random.randn(100) * 0.02)
    
    prices = pd.DataFrame(price_data, index=dates)
    
    # Calculate excess over mean
    excess_rets = excess_over_mean(prices, binary=False)
    print(f"Sample excess returns: {excess_rets.dropna().head()}")
    
    # Binary labels
    binary_labels = excess_over_mean(prices, binary=True)
    print(f"Sample binary labels: {binary_labels.dropna().head()}")
