"""
Various useful functions
"""

import pandas as pd
import numpy as np

def crop_data_frame_in_batches(df: pd.DataFrame, chunksize: int):
    # pylint: disable=invalid-name
    """
    Splits df into chunks of chunksize

    :param df: (pd.DataFrame) Dataframe to split
    :param chunksize: (int) Number of rows in chunk
    :return: (list) Chunks (pd.DataFrames)
    """
    
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    
    chunks = []
    for start in range(0, len(df), chunksize):
        end = min(start + chunksize, len(df))
        chunks.append(df.iloc[start:end].copy())
    
    return chunks


def bootstrap_sample_indices(length: int, n_samples: int = None, random_state: int = None) -> np.ndarray:
    """
    Generate bootstrap sample indices
    
    :param length: (int) Length of original dataset
    :param n_samples: (int) Number of bootstrap samples (default: same as length)
    :param random_state: (int) Random seed
    :return: (np.ndarray) Bootstrap indices
    """
    
    if n_samples is None:
        n_samples = length
    
    if random_state is not None:
        np.random.seed(random_state)
    
    return np.random.choice(length, size=n_samples, replace=True)


def get_sample_weights(returns: pd.Series, lookback: int = 252) -> pd.Series:
    """
    Calculate sample weights based on return uniqueness
    
    :param returns: (pd.Series) Return series
    :param lookback: (int) Lookback period for weight calculation
    :return: (pd.Series) Sample weights
    """
    
    weights = pd.Series(index=returns.index, dtype=float)
    
    for i in range(len(returns)):
        if i < lookback:
            # Not enough history, use equal weight
            weights.iloc[i] = 1.0 / (i + 1)
        else:
            # Calculate average uniqueness over lookback period
            recent_returns = returns.iloc[i-lookback+1:i+1]
            unique_returns = len(recent_returns.unique())
            total_returns = len(recent_returns)
            
            # Weight based on uniqueness ratio
            weights.iloc[i] = unique_returns / total_returns
    
    # Normalize weights
    weights = weights / weights.sum()
    
    return weights


def apply_pca_weights(returns: pd.DataFrame, n_components: int = None) -> pd.Series:
    """
    Apply PCA to calculate portfolio weights
    
    :param returns: (pd.DataFrame) Asset returns
    :param n_components: (int) Number of PCA components (default: all)
    :return: (pd.Series) PCA-based weights
    """
    
    from sklearn.decomposition import PCA
    
    if n_components is None:
        n_components = min(returns.shape)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca_returns = pca.fit_transform(returns.fillna(0))
    
    # Use first component loadings as weights
    weights = pd.Series(pca.components_[0], index=returns.columns)
    
    # Take absolute values and normalize
    weights = weights.abs()
    weights = weights / weights.sum()
    
    return weights


def winsorize_series(series: pd.Series, limits: tuple = (0.05, 0.95)) -> pd.Series:
    """
    Winsorize a pandas Series
    
    :param series: (pd.Series) Input series
    :param limits: (tuple) Lower and upper percentile limits
    :return: (pd.Series) Winsorized series
    """
    
    lower_limit = series.quantile(limits[0])
    upper_limit = series.quantile(limits[1])
    
    return series.clip(lower=lower_limit, upper=upper_limit)
