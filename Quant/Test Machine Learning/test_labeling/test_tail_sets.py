"""
Tail Set Labels implementation.

Tail set labels are a classification labeling technique introduced in the paper:
"Nonlinear support vector machines can systematically identify stocks with high and low future returns."
Algorithmic Finance, 2(1), pp.45-58.

A tail set is defined to be a group of stocks whose volatility-adjusted return is in the highest 
or lowest quantile, for example the highest or lowest 5%.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union


class TailSetLabels:
    """
    Tail set labels classification technique.
    
    A tail set is a group of stocks whose volatility-adjusted return is in the highest 
    or lowest quantile. A classification model is then fit using these labels to determine 
    which stocks to buy and sell in a long/short portfolio.
    """

    def __init__(self, prices: pd.DataFrame, n_bins: int, vol_adj: Optional[str] = None, window: Optional[int] = None):
        """
        Initialize the TailSetLabels class.
        
        :param prices: (pd.DataFrame) Asset prices with assets as columns and dates as index
        :param n_bins: (int) Number of bins to determine quantiles for defining tail sets
        :param vol_adj: (str) Volatility adjustment method: None, 'mean_abs_dev', or 'stdev'
        :param window: (int) Window period for volatility adjustment calculation
        """
        self.prices = prices
        self.n_bins = n_bins
        self.vol_adj = vol_adj
        self.window = window or 20
        
        # Calculate returns
        self.returns = prices.pct_change().dropna()
        
        # Calculate volatility-adjusted returns if specified
        if vol_adj:
            self.vol_adj_returns = self._vol_adjusted_rets()
        else:
            self.vol_adj_returns = self.returns

    def get_tail_sets(self) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Compute the tail sets (positive and negative) and return them.
        
        :return: (tuple) positive set, negative set, full matrix set
        """
        # Calculate quantiles
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        
        # Create the full matrix
        full_matrix = pd.DataFrame(index=self.vol_adj_returns.index, 
                                 columns=self.vol_adj_returns.columns)
        
        # For each time period, assign quantile labels
        for date in self.vol_adj_returns.index:
            returns_cross_section = self.vol_adj_returns.loc[date]
            
            # Calculate quantiles for this cross-section
            quantile_values = returns_cross_section.quantile(quantiles).values
            
            # Assign labels based on quantiles
            labels = np.zeros(len(returns_cross_section))
            
            for i, ret in enumerate(returns_cross_section.values):
                if not np.isnan(ret):
                    # Find which quantile bin this return belongs to
                    bin_idx = np.digitize(ret, quantile_values) - 1
                    bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
                    
                    # Top quantile gets +1, bottom quantile gets -1
                    if bin_idx == self.n_bins - 1:  # Top quantile
                        labels[i] = 1
                    elif bin_idx == 0:  # Bottom quantile
                        labels[i] = -1
                    else:
                        labels[i] = 0
            
            full_matrix.loc[date] = labels
        
        # Create positive and negative sets
        positive_sets = pd.Series(index=full_matrix.index, dtype=object)
        negative_sets = pd.Series(index=full_matrix.index, dtype=object)
        
        for date in full_matrix.index:
            pos_assets = full_matrix.columns[full_matrix.loc[date] == 1].tolist()
            neg_assets = full_matrix.columns[full_matrix.loc[date] == -1].tolist()
            
            positive_sets.loc[date] = pos_assets
            negative_sets.loc[date] = neg_assets
        
        return positive_sets, negative_sets, full_matrix

    def _vol_adjusted_rets(self) -> pd.DataFrame:
        """
        Compute volatility-adjusted returns.
        
        :return: (pd.DataFrame) Volatility-adjusted returns
        """
        if self.vol_adj == 'stdev':
            # Standard deviation adjustment
            vol_measure = self.returns.rolling(window=self.window).std()
        elif self.vol_adj == 'mean_abs_dev':
            # Mean absolute deviation adjustment
            vol_measure = self.returns.rolling(window=self.window).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
        else:
            raise ValueError(f"Unknown volatility adjustment method: {self.vol_adj}")
        
        # Avoid division by zero
        vol_measure = vol_measure.replace(0, np.nan)
        
        # Calculate volatility-adjusted returns
        vol_adj_returns = self.returns / vol_measure
        
        return vol_adj_returns.dropna()


def tail_sets_labels_simple(prices: pd.DataFrame, n_bins: int = 5, 
                           vol_adj: Optional[str] = 'stdev', window: int = 20) -> pd.DataFrame:
    """
    Convenience function for generating tail set labels.
    
    :param prices: (pd.DataFrame) Asset prices
    :param n_bins: (int) Number of quantile bins
    :param vol_adj: (str) Volatility adjustment method
    :param window: (int) Rolling window for volatility calculation
    :return: (pd.DataFrame) Tail set labels matrix
    """
    labeler = TailSetLabels(prices, n_bins, vol_adj, window)
    _, _, full_matrix = labeler.get_tail_sets()
    return full_matrix


if __name__ == "__main__":
    print("Tail Set Labels module loaded successfully!")
