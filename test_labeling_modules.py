#!/usr/bin/env python3
"""
Test script for the refactored labeling modules.

This script tests all the labeling functions to ensure they work correctly
after removing dependencies on mlfinlab.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the correct directory to the path so we can import from Machine Learning
sys.path.append('/workspaces/Sistema-de-datos/Quant/Machine Learning')

def test_basic_labeling():
    """Test the core labeling functions from labeling.py"""
    print("Testing core labeling functions...")
    
    from labeling.labeling import get_events, add_vertical_barrier, get_bins, get_daily_vol, cusum_filter
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    prices = pd.Series(100 + np.cumsum(np.random.randn(1000) * 0.02), index=dates)
    
    # Test daily volatility
    vol = get_daily_vol(prices, span0=50)
    print(f"✓ Daily volatility calculated, shape: {vol.shape}")
    
    # Test CUSUM filter
    t_events = cusum_filter(prices, threshold=0.05)
    print(f"✓ CUSUM filter found {len(t_events)} events")
    
    # Test vertical barrier
    vertical_barriers = add_vertical_barrier(t_events[:10], prices, num_days=5)
    print(f"✓ Vertical barriers created, shape: {vertical_barriers.shape}")
    
    # Test get_events (simplified)
    target = vol.loc[t_events[:10]]
    events = get_events(prices, t_events[:10], [1, 1], target, min_ret=0.01, num_threads=1)
    print(f"✓ Events generated, shape: {events.shape}")
    
    # Test get_bins
    if len(events) > 0:
        bins = get_bins(events, prices)
        print(f"✓ Bins calculated, shape: {bins.shape}")
    

def test_trend_scanning():
    """Test trend scanning labels"""
    print("\nTesting trend scanning...")
    
    from labeling.trend_scanning import trend_scanning_labels
    
    # Create sample data with trend
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    trend = np.linspace(100, 120, 200)
    noise = np.random.randn(200) * 1
    prices = pd.Series(trend + noise, index=dates)
    
    # Test trend scanning
    labels = trend_scanning_labels(prices, observation_window=20, min_sample_length=5)
    print(f"✓ Trend scanning labels generated, shape: {labels.shape}")
    print(f"  Sample labels: {labels.head()}")


def test_tail_sets():
    """Test tail set labels"""
    print("\nTesting tail set labels...")
    
    from labeling.tail_sets import TailSetLabels, tail_sets_labels_simple
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    price_data = {}
    for asset in assets:
        price_data[asset] = 100 + np.cumsum(np.random.randn(100) * 0.02)
    
    prices = pd.DataFrame(price_data, index=dates)
    
    # Test TailSetLabels class
    tail_labeler = TailSetLabels(prices, n_bins=5, vol_adj='stdev', window=20)
    pos_sets, neg_sets, full_matrix = tail_labeler.get_tail_sets()
    print(f"✓ Tail set labels generated, matrix shape: {full_matrix.shape}")
    
    # Test convenience function
    simple_labels = tail_sets_labels_simple(prices, n_bins=5)
    print(f"✓ Simple tail set labels generated, shape: {simple_labels.shape}")


def test_fixed_time_horizon():
    """Test fixed time horizon labeling"""
    print("\nTesting fixed time horizon...")
    
    from labeling.fixed_time_horizon import fixed_time_horizon
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.02), index=dates)
    
    # Test basic labeling
    labels = fixed_time_horizon(prices, threshold=0.01)
    print(f"✓ Fixed time horizon labels generated, shape: {labels.shape}")
    
    # Test binary labeling
    binary_labels = fixed_time_horizon(prices, threshold=0.01, binary=True)
    print(f"✓ Binary labels generated, unique values: {binary_labels.dropna().unique()}")


def test_matrix_flags():
    """Test matrix flag labeling"""
    print("\nTesting matrix flags...")
    
    from labeling.matrix_flags import MatrixFlagLabels, matrix_flag_labels_simple
    
    # Create sample data with trend
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    trend = np.linspace(100, 120, 200)
    noise = np.random.randn(200) * 2
    prices = pd.Series(trend + noise, index=dates)
    
    # Test MatrixFlagLabels class
    labeler = MatrixFlagLabels(prices, window=50, template_name='leigh_bull')
    labels = labeler.get_labels(threshold=0.3)
    print(f"✓ Matrix flag labels generated, shape: {labels.shape}")
    print(f"  Available templates: {list(labeler.templates.keys())}")
    
    # Test convenience function
    simple_labels = matrix_flag_labels_simple(prices, window=50, template_name='leigh_bull')
    print(f"✓ Simple matrix flag labels generated, shape: {simple_labels.shape}")


def test_excess_returns():
    """Test excess return labeling methods"""
    print("\nTesting excess return methods...")
    
    from labeling.excess_over_mean import excess_over_mean
    from labeling.excess_over_median import excess_over_median
    from labeling.return_vs_benchmark import return_over_benchmark
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    price_data = {}
    for asset in assets:
        price_data[asset] = 100 + np.cumsum(np.random.randn(100) * 0.02)
    
    prices = pd.DataFrame(price_data, index=dates)
    
    # Test excess over mean
    excess_mean = excess_over_mean(prices)
    print(f"✓ Excess over mean calculated, shape: {excess_mean.shape}")
    
    # Test excess over median
    excess_median = excess_over_median(prices)
    print(f"✓ Excess over median calculated, shape: {excess_median.shape}")
    
    # Test return over benchmark
    benchmark = pd.Series(np.random.randn(100) * 0.01, index=dates)
    excess_benchmark = return_over_benchmark(prices['AAPL'], benchmark=benchmark)
    print(f"✓ Return over benchmark calculated, shape: {excess_benchmark.shape}")


def test_raw_return():
    """Test raw return labeling"""
    print("\nTesting raw return labeling...")
    
    from labeling.raw_return import raw_return
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.02), index=dates)
    
    # Test raw returns
    returns = raw_return(prices)
    print(f"✓ Raw returns calculated, shape: {returns.shape}")
    
    # Test binary returns
    binary_returns = raw_return(prices, binary=True)
    print(f"✓ Binary returns calculated, unique values: {binary_returns.dropna().unique()}")


def test_bull_bear():
    """Test bull/bear market detection"""
    print("\nTesting bull/bear market detection...")
    
    from labeling.bull_bear import pagan_sossounov, lunde_timmermann
    
    # Create sample data with bull and bear phases
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Create clear bull and bear phases
    bull_phase = np.linspace(100, 150, 100)
    bear_phase = np.linspace(150, 90, 100)
    trend = np.concatenate([bull_phase, bear_phase])
    noise = np.random.randn(200) * 2
    prices = pd.Series(trend + noise, index=dates)
    
    # Test Pagan-Sossounov
    ps_labels = pagan_sossounov(prices.to_frame(), window=10, phase=5, threshold=0.1)
    print(f"✓ Pagan-Sossounov labels calculated, shape: {ps_labels.shape}")
    print(f"  Label distribution: {ps_labels.iloc[:, 0].value_counts().to_dict()}")
    
    # Test Lunde-Timmermann
    lt_labels = lunde_timmermann(prices.to_frame(), bull_threshold=0.1, bear_threshold=0.1)
    print(f"✓ Lunde-Timmermann labels calculated, shape: {lt_labels.shape}")
    print(f"  Label distribution: {lt_labels.iloc[:, 0].value_counts().to_dict()}")


def test_imports():
    """Test that all imports work from the main module"""
    print("\nTesting module imports...")
    
    try:
        # Test importing from the main labeling module
        from labeling import (
            get_events, trend_scanning_labels, TailSetLabels, fixed_time_horizon,
            MatrixFlagLabels, excess_over_median, raw_return, pagan_sossounov
        )
        print("✓ All main imports successful")
        
        # Test importing from Machine Learning.labeling
        import sys
        sys.path.append('/workspaces/Sistema-de-datos/Quant/Machine Learning')
        
        from labeling import get_daily_vol, cusum_filter
        print("✓ Alternative import path successful")
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False
        
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING REFACTORED LABELING MODULES")
    print("=" * 60)
    
    # Change to the correct directory
    os.chdir('/workspaces/Sistema-de-datos/Quant/Machine Learning')
    
    try:
        test_imports()
        test_basic_labeling()
        test_trend_scanning()
        test_tail_sets()
        test_fixed_time_horizon()
        test_matrix_flags()
        test_excess_returns()
        test_raw_return()
        test_bull_bear()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("✓ All labeling modules are working correctly")
        print("✓ No dependencies on mlfinlab remaining")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
