"""
Functionality relating to the ETF trick and stitching futures contracts together.

This module provides:
- ETFTrick: Implementation of López de Prado's ETF trick for synthetic price series
- FuturesRoll: Rolling futures contracts for continuous time series  
- Convenience functions for both ETF trick and futures rolling
"""

from .etf_trick import ETFTrick, create_etf_trick_from_csv, create_etf_trick_from_dataframes
from .futures_roll import (
    FuturesRoll, 
    prepare_vix_futures_dataset, 
    create_continuous_vix_series, 
    create_continuous_futures_series
)

__all__ = [
    'ETFTrick',
    'create_etf_trick_from_csv',
    'create_etf_trick_from_dataframes',
    'FuturesRoll',
    'prepare_vix_futures_dataset',
    'create_continuous_vix_series', 
    'create_continuous_futures_series'
]
