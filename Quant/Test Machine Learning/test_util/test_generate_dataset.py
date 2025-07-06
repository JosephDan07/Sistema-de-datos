'''
This module generates  synthetic classification dataset of INFORMED, REDUNDANT, and NOISE explanatory
variables based on the book Machine Learning for Asset Manager (code snippet 6.1)
'''
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# pylint: disable=invalid-name
def get_classification_data(n_features=100, n_informative=25, n_redundant=25, n_samples=10000, random_state=0, sigma=.0):
    """
    A function to generate synthetic classification datasets

    :param n_features: (int) Total number of features to be generated (i.e. informative + redundant + noisy).
    :param n_informative: (int) Number of informative features.
    :param n_redundant: (int) Number of redundant features.
    :param n_samples: (int) Number of samples (rows) to be generate.
    :param random_state: (int) Random seed.
    :param sigma: (float) This argument is used to introduce substitution effect to the redundant features in
                     the dataset by adding gaussian noise. The lower the  value of  sigma, the  greater the
                     substitution effect.
    :return: (pd.DataFrame, pd.Series)  X and y as features and labels respectively.
    """
    
    # Adjust parameters if they don't fit
    if n_informative + n_redundant > n_features:
        # Adjust proportionally
        total_specified = n_informative + n_redundant
        n_informative = max(1, int((n_informative / total_specified) * (n_features - 1)))
        n_redundant = max(0, n_features - n_informative - 1)  # Leave at least 1 for noise
    
    # Calculate number of noise features
    n_noise = n_features - n_informative - n_redundant
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=1,
        n_classes=2,
        random_state=random_state,
        shuffle=False
    )
    
    # Add noise to redundant features if sigma > 0
    if sigma > 0 and n_redundant > 0:
        np.random.seed(random_state)
        redundant_start = n_informative
        redundant_end = n_informative + n_redundant
        
        # Add Gaussian noise to redundant features
        noise = np.random.normal(0, sigma, (n_samples, n_redundant))
        X[:, redundant_start:redundant_end] += noise
    
    # Create feature names
    feature_names = []
    for i in range(n_informative):
        feature_names.append(f'informative_{i}')
    for i in range(n_redundant):
        feature_names.append(f'redundant_{i}')
    for i in range(n_noise):
        feature_names.append(f'noise_{i}')
    
    # Convert to pandas
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


def get_regression_data(n_features=100, n_informative=25, n_targets=1, n_samples=10000, 
                       noise=0.1, random_state=0):
    """
    Generate synthetic regression datasets
    
    :param n_features: (int) Total number of features
    :param n_informative: (int) Number of informative features
    :param n_targets: (int) Number of target variables
    :param n_samples: (int) Number of samples
    :param noise: (float) Standard deviation of gaussian noise
    :param random_state: (int) Random seed
    :return: (pd.DataFrame, pd.DataFrame) X and y
    """
    
    from sklearn.datasets import make_regression
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        noise=noise,
        random_state=random_state,
        shuffle=False
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Convert to pandas
    X_df = pd.DataFrame(X, columns=feature_names)
    
    if n_targets == 1:
        y_df = pd.Series(y, name='target')
    else:
        target_names = [f'target_{i}' for i in range(n_targets)]
        y_df = pd.DataFrame(y, columns=target_names)
    
    return X_df, y_df


def get_financial_data_synthetic(n_samples=10000, n_assets=5, start_price=100, 
                                volatility=0.02, correlation=0.3, random_state=0):
    """
    Generate synthetic financial time series data
    
    :param n_samples: (int) Number of time periods
    :param n_assets: (int) Number of assets
    :param start_price: (float) Starting price for all assets
    :param volatility: (float) Daily volatility
    :param correlation: (float) Correlation between assets
    :param random_state: (int) Random seed
    :return: (pd.DataFrame) Price data
    """
    
    np.random.seed(random_state)
    
    # Create correlation matrix
    corr_matrix = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate correlated random returns
    returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=corr_matrix * (volatility ** 2),
        size=n_samples
    )
    
    # Convert to prices using geometric Brownian motion
    prices = np.zeros((n_samples + 1, n_assets))
    prices[0, :] = start_price
    
    for t in range(n_samples):
        prices[t + 1, :] = prices[t, :] * np.exp(returns[t, :])
    
    # Create DataFrame
    asset_names = [f'Asset_{i}' for i in range(n_assets)]
    dates = pd.date_range(start='2020-01-01', periods=n_samples + 1, freq='D')
    
    price_df = pd.DataFrame(prices, index=dates, columns=asset_names)
    
    return price_df
