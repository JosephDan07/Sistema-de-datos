"""
Volume classification methods (BVC and tick rule)
"""

from scipy.stats import norm
import pandas as pd
import numpy as np


def get_bvc_buy_volume(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculates the BVC (Bulk Volume Classification) buy volume
    
    BVC is a method for estimating the proportion of volume that is buyer-initiated
    based on the distribution of price changes.

    :param close: (pd.Series): Close prices
    :param volume: (pd.Series): Bar volumes
    :param window: (int): Window for std estimation uses in BVC calculation
    :return: (pd.Series) BVC buy volume
    """
    
    # Calculate price changes
    price_changes = close.diff()
    
    # Calculate rolling standard deviation
    rolling_std = price_changes.rolling(window=window).std()
    
    # Standardize price changes
    standardized_changes = price_changes / rolling_std
    
    # Apply normal CDF to get probability of being a buy
    # .apply(norm.cdf) is used to omit Warning for norm.cdf(pd.Series with NaNs)
    buy_probability = standardized_changes.apply(norm.cdf)
    
    # Calculate buy volume
    buy_volume = buy_probability * volume
    
    return buy_volume


def get_tick_rule_buy_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculates buy volume using the tick rule (Lee-Ready algorithm)
    
    :param close: (pd.Series): Close prices
    :param volume: (pd.Series): Bar volumes
    :return: (pd.Series) Tick rule buy volume
    """
    
    # Calculate price changes
    price_changes = close.diff()
    
    # Apply tick rule
    # Positive change = buy, negative change = sell, no change = previous direction
    tick_direction = np.sign(price_changes)
    
    # Forward fill for zero changes (use previous direction)
    tick_direction = tick_direction.replace(0, np.nan).ffill()
    
    # Fill remaining NaNs with 1 (assume first tick is a buy)
    tick_direction = tick_direction.fillna(1)
    
    # Calculate buy volume (1 for buy, 0 for sell)
    buy_indicator = (tick_direction + 1) / 2  # Convert -1,1 to 0,1
    buy_volume = buy_indicator * volume
    
    return buy_volume


def get_quote_rule_buy_volume(close: pd.Series, bid: pd.Series, ask: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculates buy volume using the quote rule
    
    :param close: (pd.Series): Close prices
    :param bid: (pd.Series): Bid prices
    :param ask: (pd.Series): Ask prices
    :param volume: (pd.Series): Bar volumes
    :return: (pd.Series) Quote rule buy volume
    """
    
    # Calculate midpoint
    midpoint = (bid + ask) / 2
    
    # Classify trades
    # Above midpoint = buy, below midpoint = sell, at midpoint = use tick rule
    buy_indicator = np.where(close > midpoint, 1,
                           np.where(close < midpoint, 0, np.nan))
    
    # For trades at midpoint, use tick rule
    price_changes = close.diff()
    tick_direction = np.sign(price_changes)
    tick_direction = tick_direction.replace(0, np.nan).ffill().fillna(1)
    
    # Fill midpoint trades with tick rule
    buy_indicator = pd.Series(buy_indicator, index=close.index)
    midpoint_mask = buy_indicator.isna()
    buy_indicator[midpoint_mask] = (tick_direction[midpoint_mask] + 1) / 2
    
    # Calculate buy volume
    buy_volume = buy_indicator * volume
    
    return buy_volume
