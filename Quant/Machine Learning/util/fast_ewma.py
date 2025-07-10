"""
This module contains an implementation of an exponentially weighted moving average based on sample size.
The inspiration and context for this code was from a blog post by writen by Maksim Ivanov:
https://towardsdatascience.com/financial-machine-learning-part-0-bars-745897d4e4ba
"""

# Imports
import numpy as np
import pandas as pd

# Try to import Numba, fall back to pure Python if not available
try:
    from numba import jit  # type: ignore
    from numba import float64  # type: ignore
    from numba import int64  # type: ignore
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator for compatibility
    def jit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator
    float64 = None  # type: ignore
    int64 = None  # type: ignore


if NUMBA_AVAILABLE:
    # Use lazy compilation to avoid startup delays
    @jit(nopython=True, cache=True)
    def _ewma_numba(arr_in, window):  # pragma: no cover
        """
        Exponentially weighted moving average specified by a decay ``window`` to provide better adjustments for
        small windows via:
            y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
                   (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

        :param arr_in: (np.ndarray) A single dimensional numpy array
        :param window: (int) The decay window, or 'span'
        :return: (np.ndarray) The EWMA vector, same length / shape as ``arr_in``
        """
        
        if len(arr_in) == 0:
            return np.empty(0, dtype=np.float64)
        
        # Calculate the decay factor alpha
        alpha = 2.0 / (window + 1.0)
        
        # Initialize output array
        ewma_out = np.empty(len(arr_in), dtype=np.float64)
        
        # Initialize first value
        ewma_out[0] = arr_in[0]
        
        # Calculate EWMA for the rest of the array
        for i in range(1, len(arr_in)):
            ewma_out[i] = alpha * arr_in[i] + (1 - alpha) * ewma_out[i - 1]
        
        return ewma_out
    
    def ewma(arr_in, window, handle_nans=True):
        """
        EWMA wrapper that properly handles pandas Series and other input types
        
        :param arr_in: Input array or Series
        :param window: Window size for EWMA calculation
        :param handle_nans: If True, propagate NaNs in output; if False, skip NaNs in calculation
        """
        # Convert to numpy array if necessary
        if hasattr(arr_in, 'values'):  # pandas Series/DataFrame
            arr_in = arr_in.values
        
        # Ensure numpy array with proper dtype
        arr_in = np.asarray(arr_in, dtype=np.float64)
        
        if len(arr_in) == 0:
            return np.array([], dtype=np.float64)
        
        # Handle NaN values if requested
        if handle_nans and np.any(np.isnan(arr_in)):
            # Use pandas for better NaN handling when NaNs are present
            s = pd.Series(arr_in)
            alpha = 2.0 / (window + 1.0)
            result = s.ewm(alpha=alpha, adjust=False).mean()
            return result.values.astype(np.float64)
        elif not handle_nans and np.any(np.isnan(arr_in)):
            # Use pandas to skip NaNs in calculation
            s = pd.Series(arr_in)
            alpha = 2.0 / (window + 1.0)
            result = s.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
            return result.values.astype(np.float64)
        
        return _ewma_numba(arr_in, window)
else:
    def ewma(arr_in, window, handle_nans=True):
        """
        Pure Python EWMA fallback when Numba is not available
        
        :param arr_in: Input array or Series
        :param window: Window size for EWMA calculation
        :param handle_nans: If True, propagate NaNs in output; if False, skip NaNs in calculation
        """
        # Use pandas EWMA for efficiency
        s = pd.Series(arr_in)
        alpha = 2.0 / (window + 1.0)
        
        # Handle NaNs based on parameter
        if handle_nans:
            # Pandas ewm naturally handles NaNs by propagating them
            result = s.ewm(alpha=alpha, adjust=False).mean()
        else:
            # Skip NaNs in calculation
            result = s.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        
        return result.values.astype(np.float64)


if NUMBA_AVAILABLE:
    def ewma_vectorized(arr_in, window, handle_nans=True):  # pragma: no cover
        """
        Optimized vectorized version of EWMA with enhanced numerical stability.
        Uses pandas for now to avoid Numba compilation issues.
        
        :param arr_in: (np.ndarray) A single dimensional numpy array
        :param window: (int) The decay window, or 'span'
        :param handle_nans: If True, propagate NaNs in output; if False, skip NaNs in calculation
        :return: (np.ndarray) The EWMA vector, same length / shape as ``arr_in``
        """
        
        # Use pandas EWMA for reliability - it's already optimized
        arr_in = np.asarray(arr_in, dtype=np.float64)
        if len(arr_in) == 0:
            return np.array([], dtype=np.float64)
            
        span = window
        
        # Use pandas ewm for vectorized calculation
        s = pd.Series(arr_in)
        if handle_nans:
            # Pandas ewm naturally handles NaNs by propagating them
            result = s.ewm(span=span, adjust=False).mean()
        else:
            # Skip NaNs in calculation
            result = s.ewm(span=span, adjust=False, ignore_na=True).mean()
        
        return result.values.astype(np.float64)
else:
    def ewma_vectorized(arr_in, window, handle_nans=True):
        """
        Pure Python EWMA vectorized fallback when Numba is not available
        """
        return ewma(arr_in, window, handle_nans)


if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _ewma_alpha_numba(arr_in, alpha):  # pragma: no cover
        """
        EWMA with direct alpha specification for better control.
        
        :param arr_in: (np.ndarray) A single dimensional numpy array
        :param alpha: (float) Smoothing parameter between 0 and 1
        :return: (np.ndarray) The EWMA vector, same length / shape as ``arr_in``
        """
        
        if len(arr_in) == 0:
            return np.empty(0, dtype=np.float64)
        
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("Alpha must be between 0 and 1")
        
        # Initialize output array
        ewma_out = np.empty(len(arr_in), dtype=np.float64)
        
        # Initialize first value
        ewma_out[0] = arr_in[0]
        
        # Calculate EWMA
        alpha_complement = 1.0 - alpha
        for i in range(1, len(arr_in)):
            ewma_out[i] = alpha * arr_in[i] + alpha_complement * ewma_out[i - 1]
        
        return ewma_out
    
    def ewma_alpha(arr_in, alpha, handle_nans=True):
        """
        EWMA alpha wrapper that properly handles pandas Series and other input types
        
        :param arr_in: Input array or Series
        :param alpha: Smoothing parameter between 0 and 1
        :param handle_nans: If True, propagate NaNs in output; if False, skip NaNs in calculation
        """
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("Alpha must be between 0 and 1")
        
        # Convert to numpy array if necessary
        if hasattr(arr_in, 'values'):  # pandas Series/DataFrame
            arr_in = arr_in.values
        
        # Ensure numpy array with proper dtype
        arr_in = np.asarray(arr_in, dtype=np.float64)
        
        if len(arr_in) == 0:
            return np.array([], dtype=np.float64)
        
        # Handle NaN values if requested
        if handle_nans and np.any(np.isnan(arr_in)):
            # Use pandas for better NaN handling when NaNs are present
            s = pd.Series(arr_in)
            result = s.ewm(alpha=alpha, adjust=False).mean()
            return result.values.astype(np.float64)
        elif not handle_nans and np.any(np.isnan(arr_in)):
            # Use pandas to skip NaNs in calculation
            s = pd.Series(arr_in)
            result = s.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
            return result.values.astype(np.float64)
        
        return _ewma_alpha_numba(arr_in, alpha)
else:
    def ewma_alpha(arr_in, alpha, handle_nans=True):
        """
        Pure Python EWMA alpha fallback when Numba is not available
        
        :param arr_in: Input array or Series
        :param alpha: Smoothing parameter between 0 and 1
        :param handle_nans: If True, propagate NaNs in output; if False, skip NaNs in calculation
        """
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("Alpha must be between 0 and 1")
        
        s = pd.Series(arr_in)
        
        # Handle NaNs based on parameter
        if handle_nans:
            # Pandas ewm naturally handles NaNs by propagating them
            result = s.ewm(alpha=alpha, adjust=False).mean()
        else:
            # Skip NaNs in calculation
            result = s.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        
        return result.values.astype(np.float64)


def ewma_halflife(arr_in, halflife, handle_nans=True):
    """
    EWMA using half-life specification (more intuitive for financial data).
    
    :param arr_in: (np.ndarray) Input array
    :param halflife: (float) Half-life in number of periods
    :param handle_nans: If True, propagate NaNs in output; if False, skip NaNs in calculation
    :return: (np.ndarray) The EWMA vector
    """
    
    if halflife <= 0:
        raise ValueError("Half-life must be positive")
    
    # Convert half-life to alpha
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    
    # Ensure proper dtype
    arr_in = np.asarray(arr_in, dtype=np.float64)
    return ewma_alpha(arr_in, alpha, handle_nans)


def ewma_com(arr_in, com, handle_nans=True):
    """
    EWMA using center of mass specification.
    
    :param arr_in: (np.ndarray) Input array
    :param com: (float) Center of mass
    :param handle_nans: If True, propagate NaNs in output; if False, skip NaNs in calculation
    :return: (np.ndarray) The EWMA vector
    """
    
    if com < 0:
        raise ValueError("Center of mass must be non-negative")
    
    # Convert center of mass to alpha
    alpha = 1.0 / (1.0 + com)
    
    # Ensure proper dtype
    arr_in = np.asarray(arr_in, dtype=np.float64)
    return ewma_alpha(arr_in, alpha, handle_nans)


def get_ewma_info():
    """
    Get information about the EWMA implementation being used.
    
    :return: (str) Information about Numba availability and performance
    """
    if NUMBA_AVAILABLE:
        return "EWMA: Using optimized Numba JIT compilation for maximum performance"
    else:
        return "EWMA: Using pure Python/Pandas fallback (Numba not available)"


# Print info when module is loaded
if __name__ != "__main__":
    import warnings
    if not NUMBA_AVAILABLE:
        warnings.warn("Numba not available, using slower pure Python EWMA implementation", 
                     UserWarning, stacklevel=2)
