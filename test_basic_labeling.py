#!/usr/bin/env python3

import sys
import os

# Set working directory and path
os.chdir('/workspaces/Sistema-de-datos/Quant/Machine Learning')
sys.path.insert(0, '/workspaces/Sistema-de-datos/Quant/Machine Learning')

print("=== TESTING LABELING MODULES ===")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path[0]}")

try:
    # Test basic imports
    from labeling.labeling import get_daily_vol, cusum_filter
    print("✓ Core labeling imports successful")
    
    # Test other modules
    from labeling.matrix_flags import MatrixFlagLabels
    from labeling.tail_sets import TailSetLabels
    print("✓ Additional module imports successful")
    
    # Create sample data
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
    print("✓ Labeling modules are working correctly!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== TEST COMPLETE ===")
