"""
Various volatility estimators
"""
import pandas as pd
import numpy as np


# pylint: disable=redefined-builtin

def get_daily_vol(close, lookback=100):
    """
    Advances in Financial Machine Learning, Snippet 3.1, page 44.

    Daily Volatility Estimates

    Computes the daily volatility at intraday estimation points.

    In practice we want to set profit taking and stop-loss limits that are a function of the risks involved
    in a bet. Otherwise, sometimes we will be aiming too high (tao ≫ sigma_t_i,0), and sometimes too low
    (tao ≪ sigma_t_i,0 ), considering the prevailing volatility. Snippet 3.1 computes the daily volatility
    at intraday estimation points, applying a span of lookback days to an exponentially weighted moving
    standard deviation.

    See the pandas documentation for details on the pandas.Series.ewm function.
    Note: This function is used to compute dynamic thresholds for profit taking and stop loss limits.

    :param close: (pd.Series) Closing prices
    :param lookback: (int) Lookback period to compute volatility
    :return: (pd.Series) Daily volatility value
    """
    
    # Calculate returns
    returns = close.pct_change().dropna()
    
    # Calculate exponentially weighted moving standard deviation
    # Using span parameter which corresponds to 2/(span+1) decay in terms of alpha
    volatility = returns.ewm(span=lookback).std()
    
    # Annualize volatility (assuming 252 trading days per year)
    volatility = volatility * np.sqrt(252)
    
    return volatility


def get_parksinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Parkinson volatility estimator
    
    The Parkinson volatility estimator is more efficient than the traditional close-to-close estimator
    as it incorporates the intraday high-low range information.

    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Parkinson volatility
    """
    
    # Parkinson volatility formula: sqrt(1/(4*ln(2)) * ln(H/L)^2)
    # Rolling version: sqrt(mean(ln(H/L)^2) / (4*ln(2)))
    
    # Calculate log ratio of high to low
    log_hl_ratio = np.log(high / low)
    
    # Square the log ratio
    log_hl_ratio_squared = log_hl_ratio ** 2
    
    # Calculate rolling mean
    rolling_mean = log_hl_ratio_squared.rolling(window=window).mean()
    
    # Apply Parkinson formula
    parkinson_vol = np.sqrt(rolling_mean / (4 * np.log(2)))
    
    # Annualize (assuming 252 trading days)
    parkinson_vol = parkinson_vol * np.sqrt(252)
    
    return parkinson_vol


def get_garman_class_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                         window: int = 20) -> pd.Series:
    """
    Garman-Class volatility estimator
    
    The Garman-Class estimator uses all four price points (OHLC) to provide a more accurate
    volatility estimate than close-to-close methods.

    :param open: (pd.Series): Open prices
    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param close: (pd.Series): Close prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Garman-Class volatility
    """
    
    # Garman-Class formula:
    # 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
    
    # Calculate log ratios
    log_hl = np.log(high / low)
    log_co = np.log(close / open)
    
    # Apply Garman-Class formula
    garman_klass = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    
    # Calculate rolling mean
    rolling_mean = garman_klass.rolling(window=window).mean()
    
    # Take square root and annualize
    garman_vol = np.sqrt(rolling_mean * 252)
    
    return garman_vol


def get_yang_zhang_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                       window: int = 20) -> pd.Series:
    """
    Yang-Zhang volatility estimator
    
    The Yang-Zhang estimator is the most efficient unbiased estimator that handles both
    opening jumps and intraday price movements.

    :param open: (pd.Series): Open prices
    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param close: (pd.Series): Close prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Yang-Zhang volatility
    """
    
    # Overnight returns (Close[t-1] to Open[t])
    overnight_returns = np.log(open / close.shift(1))
    
    # Open-to-close returns
    open_to_close = np.log(close / open)
    
    # Rogers-Satchell component (intraday volatility without drift)
    log_ho = np.log(high / open)
    log_hc = np.log(high / close)
    log_lo = np.log(low / open)
    log_lc = np.log(low / close)
    
    rogers_satchell = log_ho * log_hc + log_lo * log_lc
    
    # Yang-Zhang estimator components
    # Overnight volatility
    overnight_vol = overnight_returns.rolling(window=window).var()
    
    # Open-to-close volatility  
    open_close_vol = open_to_close.rolling(window=window).var()
    
    # Rogers-Satchell volatility
    rs_vol = rogers_satchell.rolling(window=window).mean()
    
    # Yang-Zhang volatility (weighted combination)
    # k factor for bias correction
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    
    yang_zhang = overnight_vol + k * open_close_vol + (1 - k) * rs_vol
    
    # Take square root and annualize
    yang_zhang_vol = np.sqrt(yang_zhang * 252)
    
    return yang_zhang_vol


def get_rogers_satchell_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                           window: int = 20) -> pd.Series:
    """
    Rogers-Satchell volatility estimator
    
    The Rogers-Satchell estimator is designed to handle drift and doesn't require overnight returns.

    :param open: (pd.Series): Open prices
    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param close: (pd.Series): Close prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Rogers-Satchell volatility
    """
    
    # Rogers-Satchell formula: ln(H/O)*ln(H/C) + ln(L/O)*ln(L/C)
    log_ho = np.log(high / open)
    log_hc = np.log(high / close)
    log_lo = np.log(low / open)
    log_lc = np.log(low / close)
    
    rogers_satchell = log_ho * log_hc + log_lo * log_lc
    
    # Calculate rolling mean
    rolling_mean = rogers_satchell.rolling(window=window).mean()
    
    # Take square root and annualize
    rs_vol = np.sqrt(rolling_mean * 252)
    
    return rs_vol


def get_ewma_vol(returns: pd.Series, alpha: float = 0.94, initial_vol: float = None) -> pd.Series:
    """
    EWMA (RiskMetrics style) volatility estimator
    
    :param returns: (pd.Series): Return series
    :param alpha: (float): Decay factor (default 0.94 as in RiskMetrics)
    :param initial_vol: (float): Initial volatility estimate
    :return: (pd.Series): EWMA volatility
    """
    
    if initial_vol is None:
        initial_vol = returns.std()
    
    # Initialize variance series
    variance = pd.Series(index=returns.index, dtype=float)
    variance.iloc[0] = initial_vol ** 2
    
    # Calculate EWMA variance
    for i in range(1, len(returns)):
        variance.iloc[i] = alpha * variance.iloc[i-1] + (1 - alpha) * (returns.iloc[i-1] ** 2)
    
    # Return volatility (square root of variance)
    return np.sqrt(variance)
