#!/usr/bin/env python3
"""
Simple test script for labeling modules.
"""

import sys
import os

# Change to the correct directory
os.chdir('/workspaces/Sistema-de-datos/Quant/Machine Learning')
print("Current directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))

try:
    # Test basic import
    from labeling.labeling import get_daily_vol, cusum_filter
    print("✓ Basic labeling imports successful")
    
    # Test creating sample data
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.02), index=dates)
    
    # Test daily volatility
    vol = get_daily_vol(prices, span0=20)
    print(f"✓ Daily volatility calculated, shape: {vol.shape}")
    
    # Test CUSUM filter
    events = cusum_filter(prices, threshold=0.05)
    print(f"✓ CUSUM filter found {len(events)} events")
    
    print("\n✓ ALL BASIC TESTS PASSED!")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
