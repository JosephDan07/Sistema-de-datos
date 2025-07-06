"""
Detection of bull and bear markets.
"""
import numpy as np
import pandas as pd
from typing import Union
from scipy.signal import argrelextrema


def pagan_sossounov(prices: pd.DataFrame, window: int = 8, censor: int = 6, 
                   cycle: int = 16, phase: int = 4, threshold: float = 0.2) -> pd.DataFrame:
    """
    Pagan and Sossounov's labeling method. Sourced from `Pagan, Adrian R., and Kirill A. Sossounov. "A simple framework
    for analysing bull and bear markets." Journal of applied econometrics 18.1 (2003): 23-46.
    <https://onlinelibrary.wiley.com/doi/pdf/10.1002/jae.664>`__

    Returns a DataFrame with labels of 1 for Bull and -1 for Bear.

    :param prices: (pd.DataFrame) Close prices of all tickers in the market.
    :param window: (int) Rolling window length to determine local extrema. Paper suggests 8 months for monthly obs.
    :param censor: (int) Number of months to eliminate for start and end. Paper suggests 6 months for monthly obs.
    :param cycle: (int) Minimum length for a complete cycle. Paper suggests 16 months for monthly obs.
    :param phase: (int) Minimum length for a phase. Paper suggests 4 months for monthly obs.
    :param threshold: (float) Minimum threshold for phase change. Paper suggests 0.2.
    :return: (pd.DataFrame) Labeled pd.DataFrame. 1 for Bull, -1 for Bear.
    """
    
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    
    labels = prices.copy()
    
    for column in prices.columns:
        labels[column] = _apply_pagan_sossounov(prices[column], window, censor, cycle, phase, threshold)
    
    return labels


def _alternation(price: pd.Series) -> pd.Series:
    """
    Helper function to check peak and trough alternation.

    :param price: (pd.Series) Close prices time series.
    :return: (pd.Series) Series indicating peaks (1) and troughs (-1).
    """
    
    # Find local maxima and minima
    maxima_idx = argrelextrema(price.values, np.greater, order=1)[0]
    minima_idx = argrelextrema(price.values, np.less, order=1)[0]
    
    # Create series with extrema
    extrema = pd.Series(index=price.index, dtype=float)
    extrema.iloc[maxima_idx] = 1  # Peaks
    extrema.iloc[minima_idx] = -1  # Troughs
    
    # Ensure alternation
    extrema_values = extrema.dropna()
    
    if len(extrema_values) < 2:
        return extrema
    
    # Remove consecutive same-type extrema, keeping the more extreme one
    i = 1
    while i < len(extrema_values):
        if extrema_values.iloc[i] == extrema_values.iloc[i-1]:
            # Same type of extrema, keep the more extreme one
            if extrema_values.iloc[i] == 1:  # Both are peaks
                if price.loc[extrema_values.index[i]] > price.loc[extrema_values.index[i-1]]:
                    extrema.loc[extrema_values.index[i-1]] = np.nan
                else:
                    extrema.loc[extrema_values.index[i]] = np.nan
            else:  # Both are troughs
                if price.loc[extrema_values.index[i]] < price.loc[extrema_values.index[i-1]]:
                    extrema.loc[extrema_values.index[i-1]] = np.nan
                else:
                    extrema.loc[extrema_values.index[i]] = np.nan
            extrema_values = extrema.dropna()
            i = 1  # Restart check
        else:
            i += 1
    
    return extrema


def _apply_pagan_sossounov(price: pd.Series, window: int, censor: int, 
                          cycle: int, phase: int, threshold: float) -> pd.Series:
    """
    Helper function for Pagan and Sossounov labeling method.

    :param price: (pd.Series) Close prices time series.
    :param window: (int) Rolling window length to determine local extrema.
    :param censor: (int) Number of periods to eliminate for start and end.
    :param cycle: (int) Minimum length for a complete cycle.
    :param phase: (int) Minimum length for a phase.
    :param threshold: (float) Minimum threshold for phase change.
    :return: (pd.Series) Labeled series. 1 for Bull, -1 for Bear.
    """
    
    # Step 1: Find turning points using rolling window
    rolling_max = price.rolling(window=window, center=True).max()
    rolling_min = price.rolling(window=window, center=True).min()
    
    # Identify peaks and troughs
    peaks = (price == rolling_max) & (price.shift(1) != rolling_max)
    troughs = (price == rolling_min) & (price.shift(1) != rolling_min)
    
    # Step 2: Apply alternation rule
    extrema = pd.Series(index=price.index, dtype=float)
    extrema[peaks] = 1
    extrema[troughs] = -1
    extrema = _alternation(price)
    
    # Step 3: Apply censoring rule (remove first and last censor periods)
    if censor > 0:
        extrema.iloc[:censor] = np.nan
        extrema.iloc[-censor:] = np.nan
    
    # Step 4: Apply phase and cycle rules
    extrema_points = extrema.dropna()
    
    if len(extrema_points) < 2:
        return pd.Series(0, index=price.index)
    
    # Check minimum phase length and threshold
    valid_extrema = []
    
    for i in range(len(extrema_points)):
        if i == 0:
            valid_extrema.append(extrema_points.iloc[i])
            continue
        
        # Check phase length
        phase_length = (extrema_points.index[i] - extrema_points.index[i-1]).days
        
        # Check threshold
        price_change = abs(price.loc[extrema_points.index[i]] / price.loc[extrema_points.index[i-1]] - 1)
        
        if phase_length >= phase and price_change >= threshold:
            valid_extrema.append(extrema_points.iloc[i])
    
    # Step 5: Create labels
    labels = pd.Series(0, index=price.index)
    
    if len(valid_extrema) < 2:
        return labels
    
    # Assign bull (1) and bear (-1) labels between turning points
    for i in range(len(valid_extrema) - 1):
        start_idx = extrema_points.index[i]
        end_idx = extrema_points.index[i + 1]
        
        if valid_extrema[i] == -1:  # Trough to peak = bull market
            labels.loc[start_idx:end_idx] = 1
        else:  # Peak to trough = bear market
            labels.loc[start_idx:end_idx] = -1
    
    return labels


def lunde_timmermann(prices: pd.DataFrame, bull_threshold: float = 0.15, 
                    bear_threshold: float = 0.15) -> pd.DataFrame:
    """
    Lunde and Timmermann's labeling method. Sourced from `Lunde, Asger, and Allan Timmermann. "Duration dependence
    in stock prices: An analysis of bull and bear markets." Journal of Business & Economic Statistics 22.3 (2004): 253-273.
    <https://repec.cepr.org/repec/cpr/ceprdp/DP4104.pdf>`__

    Returns a DataFrame with labels of 1 for Bull and -1 for Bear.

    :param prices: (pd.DataFrame) Close prices of all tickers in the market.
    :param bull_threshold: (float) Threshold to identify bull market. Paper suggests 0.15.
    :param bear_threshold: (float) Threshold to identify bear market. Paper suggests 0.15.
    :return: (pd.DataFrame) Labeled pd.DataFrame. 1 for Bull, -1 for Bear.
    """
    
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    
    labels = prices.copy()
    
    for column in prices.columns:
        labels[column] = _apply_lunde_timmermann(prices[column], bull_threshold, bear_threshold)
    
    return labels


def _apply_lunde_timmermann(price: pd.Series, bull_threshold: float, 
                           bear_threshold: float) -> pd.Series:
    """
    Helper function for Lunde and Timmermann labeling method.

    :param price: (pd.Series) Close prices time series.
    :param bull_threshold: (float) Threshold to identify bull market.
    :param bear_threshold: (float) Threshold to identify bear market.
    :return: (pd.Series) Labeled series. 1 for Bull, -1 for Bear.
    """
    
    labels = pd.Series(0, index=price.index)
    
    if len(price) < 2:
        return labels
    
    current_state = 0  # 0: neutral, 1: bull, -1: bear
    reference_price = price.iloc[0]
    reference_idx = 0
    
    for i in range(1, len(price)):
        current_price = price.iloc[i]
        
        # Calculate return from reference point
        return_from_ref = (current_price / reference_price) - 1
        
        if current_state == 0 or current_state == -1:  # Neutral or bear state
            # Check for bull market
            if return_from_ref >= bull_threshold:
                current_state = 1
                # Label all periods from reference to current as bull
                labels.iloc[reference_idx:i+1] = 1
                reference_price = current_price
                reference_idx = i
        
        if current_state == 0 or current_state == 1:  # Neutral or bull state
            # Check for bear market
            if return_from_ref <= -bear_threshold:
                current_state = -1
                # Label all periods from reference to current as bear
                labels.iloc[reference_idx:i+1] = -1
                reference_price = current_price
                reference_idx = i
        
        # Update reference if we hit a new extreme in current direction
        if current_state == 1 and current_price > reference_price:
            reference_price = current_price
            reference_idx = i
        elif current_state == -1 and current_price < reference_price:
            reference_price = current_price
            reference_idx = i
        
        labels.iloc[i] = current_state
    
    return labels


if __name__ == "__main__":
    print("Bull Bear labeling module loaded successfully!")
    
    # Example usage
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Generate price series with bull and bear phases
    trend = np.concatenate([
        np.linspace(100, 150, 100),  # Bull phase
        np.linspace(150, 90, 100)    # Bear phase
    ])
    noise = np.random.randn(200) * 2
    prices = pd.Series(trend + noise, index=dates)
    
    # Apply Pagan-Sossounov method
    ps_labels = pagan_sossounov(prices.to_frame(), window=10, phase=5, threshold=0.1)
    print(f"Pagan-Sossounov labels: {ps_labels.iloc[0].value_counts()}")
    
    # Apply Lunde-Timmermann method
    lt_labels = lunde_timmermann(prices.to_frame(), bull_threshold=0.1, bear_threshold=0.1)
    print(f"Lunde-Timmermann labels: {lt_labels.iloc[0].value_counts()}")
