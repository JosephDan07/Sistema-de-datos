"""
Return in excess of a given benchmark.

Chapter 5, Machine Learning for Factor Investing, by Coqueret and Guida, (2020).

Work "Evaluating multiple classifiers for stock price direction prediction" by Ballings et al. (2015) uses this method
to label yearly returns over a predetermined value to compare the performance of several machine learning algorithms.
"""
import numpy as np
import pandas as pd
from typing import Union


def return_over_benchmark(prices: Union[pd.Series, pd.DataFrame], 
                         benchmark: Union[pd.Series, float] = 0, 
                         binary: bool = False, resample_by: str = None, 
                         lag: bool = True) -> Union[pd.Series, pd.DataFrame]:
    """
    Return over benchmark labeling method. Sourced from Chapter 5.5.1 of Machine Learning for Factor Investing,
    by Coqueret, G. and Guida, T. (2020).

    Returns a Series or DataFrame of numerical or categorical returns over a given benchmark. The time index of the
    benchmark must match those of the price observations.

    :param prices: (pd.Series or pd.DataFrame) Time indexed prices to compare returns against a benchmark.
    :param benchmark: (pd.Series or float) Benchmark of returns to compare the returns from prices against for labeling.
                    Can be a constant value, or a Series matching the index of prices. If no benchmark is given, then it
                    is assumed to have a constant value of 0.
    :param binary: (bool) If False, labels are given by their numerical value of return over benchmark. If True,
                labels are given according to the sign of their excess return.
    :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns. 'B' = per
                        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.
                        For full details see `here.
                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
    :param lag: (bool) If True, returns will be lagged to make them forward-looking.
    :return: (pd.Series or pd.DataFrame) Excess returns over benchmark. If binary, the labels are -1 if the
            return is below the benchmark, 1 if above, and 0 if it exactly matches the benchmark.
    """
    
    # Resample if requested
    if resample_by is not None:
        prices = prices.resample(resample_by).last()
        if isinstance(benchmark, pd.Series):
            benchmark = benchmark.resample(resample_by).last()
    
    # Calculate returns
    returns = prices.pct_change()
    
    # Lag returns if requested (forward-looking)
    if lag:
        returns = returns.shift(-1)
    
    # Handle benchmark
    if isinstance(benchmark, pd.Series):
        # Align benchmark with returns index
        benchmark = benchmark.reindex(returns.index, method='ffill')
        excess_returns = returns.subtract(benchmark, axis=0)
    else:
        # Constant benchmark
        excess_returns = returns - benchmark
    
    # Convert to binary if requested
    if binary:
        excess_returns = np.sign(excess_returns)
    
    return excess_returns


if __name__ == "__main__":
    print("Return Over Benchmark labeling module loaded successfully!")
    
    # Example usage
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Single asset
    prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.02), index=dates)
    
    # Calculate excess over constant benchmark
    excess_rets = return_over_benchmark(prices, benchmark=0.001, binary=False)
    print(f"Sample excess returns: {excess_rets.dropna().head()}")
    
    # Binary labels
    binary_labels = return_over_benchmark(prices, benchmark=0.001, binary=True)
    print(f"Sample binary labels: {binary_labels.dropna().head()}")
    
    # Multi-asset example
    assets = ['AAPL', 'GOOGL', 'MSFT']
    price_data = {}
    for asset in assets:
        price_data[asset] = 100 + np.cumsum(np.random.randn(100) * 0.02)
    
    prices_df = pd.DataFrame(price_data, index=dates)
    
    # Create benchmark series
    benchmark_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
    
    excess_multi = return_over_benchmark(prices_df, benchmark=benchmark_returns, binary=False)
    print(f"Sample multi-asset excess returns: {excess_multi.dropna().head()}")
