"""
Labeling techniques used in financial machine learning.

This module provides comprehensive labeling methods for financial time series data,
including triple barrier method, meta-labeling, trend scanning, and various other
techniques described in "Advances in Financial Machine Learning" by López de Prado
and related literature.

All modules are fully self-contained and do not depend on external libraries like mlfinlab.
"""

# Core labeling functions from labeling.py
from .labeling import (
    add_vertical_barrier, 
    apply_pt_sl_on_t1, 
    barrier_touched, 
    drop_labels,
    get_bins, 
    get_events,
    get_daily_vol,
    cusum_filter,
    mp_pandas_obj
)

# Trend scanning
from .trend_scanning import trend_scanning_labels

# Tail set labels
from .tail_sets import TailSetLabels, tail_sets_labels_simple

# Fixed time horizon labeling
from .fixed_time_horizon import fixed_time_horizon

# Matrix flag labeling
from .matrix_flags import MatrixFlagLabels, matrix_flag_labels_simple

# Additional labeling methods (implementations available)
from .excess_over_median import excess_over_median
from .excess_over_mean import excess_over_mean
from .return_vs_benchmark import return_over_benchmark
from .raw_return import raw_return
from .bull_bear import pagan_sossounov, lunde_timmermann

__all__ = [
    # Core labeling
    'add_vertical_barrier', 
    'apply_pt_sl_on_t1', 
    'barrier_touched', 
    'drop_labels',
    'get_bins', 
    'get_events',
    'get_daily_vol',
    'cusum_filter',
    'mp_pandas_obj',
    
    # Trend scanning
    'trend_scanning_labels',
    
    # Tail sets
    'TailSetLabels',
    'tail_sets_labels_simple',
    
    # Fixed time horizon
    'fixed_time_horizon',
    
    # Matrix flags
    'MatrixFlagLabels',
    'matrix_flag_labels_simple',
    
    # Additional methods
    'excess_over_median',
    'excess_over_mean', 
    'return_over_benchmark',
    'raw_return',
    'pagan_sossounov', 
    'lunde_timmermann'
]

# Version info
__version__ = "1.0.0"
__author__ = "Refactored for production use"
__description__ = "Self-contained financial machine learning labeling techniques"
