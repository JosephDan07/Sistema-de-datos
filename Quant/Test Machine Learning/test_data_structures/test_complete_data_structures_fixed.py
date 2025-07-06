#!/usr/bin/env python3
"""
Complete Test Suite for Data Structures Module
==============================================

Comprehensive testing for all data structures functionality
using the actual functions available in the module.
"""

import os
import sys
import time
import json
import unittest
from datetime import datetime
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from test_config_manager import ConfigurationManager
import pandas as pd
import numpy as np

# Import actual functions from data_structures
from data_structures import (
    get_tick_bars, get_dollar_bars, get_volume_bars, get_time_bars,
    get_tick_imbalance_bars, get_volume_imbalance_bars, get_dollar_imbalance_bars,
    get_tick_run_bars, get_volume_run_bars, get_dollar_run_bars
)

class TestCompleteDataStructures(unittest.TestCase):
    """Complete test suite for data structures"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.config_manager = ConfigurationManager()
        cls.config = cls.config_manager.get_config('data_structures')
        cls.results_path = Path(__file__).parent.parent.parent / "Results Machine Learning" / "results_data_structures"
        cls.results_path.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic data for testing
        cls.test_data = cls._create_test_data()
        
        # Results storage
        cls.test_results = {}
        cls.performance_metrics = {}
        
    @classmethod
    def _create_test_data(cls):
        """Create realistic synthetic financial data"""
        np.random.seed(42)
        
        # Generate 1000 data points
        n_points = 1000
        
        # Create timestamps
        timestamps = pd.date_range(start='2023-01-01', periods=n_points, freq='1min')
        
        # Generate realistic price data (geometric brownian motion)
        initial_price = 100.0
        returns = np.random.normal(0, 0.001, n_points)
        prices = [initial_price]
        
        for i in range(1, n_points):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(new_price)
        
        # Generate volume data (log-normal distribution)
        volumes = np.random.lognormal(mean=10, sigma=1, size=n_points)
        volumes = volumes.astype(int)
        
        # Create DataFrame
        data = pd.DataFrame({
            'date_time': timestamps,
            'price': prices,
            'volume': volumes
        })
        
        return data
    
    def test_tick_bars(self):
        """Test tick bars functionality"""
        print("🧪 Testing Tick Bars...")
        start_time = time.time()
        
        try:
            # Test with different thresholds
            for threshold in [50, 100, 200]:
                bars = get_tick_bars(self.test_data, threshold=threshold)
                
                # Validate results
                self.assertIsInstance(bars, pd.DataFrame)
                self.assertGreater(len(bars), 0)
                self.assertTrue(all(col in bars.columns for col in ['open', 'high', 'low', 'close', 'volume']))
                
            execution_time = time.time() - start_time
            self.test_results['tick_bars'] = {'passed': True, 'execution_time': execution_time}
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results['tick_bars'] = {'passed': False, 'execution_time': execution_time, 'error': str(e)}
            raise
    
    def test_volume_bars(self):
        """Test volume bars functionality"""
        print("🧪 Testing Volume Bars...")
        start_time = time.time()
        
        try:
            # Calculate appropriate threshold
            total_volume = self.test_data['volume'].sum()
            threshold = int(total_volume / 20)  # Create ~20 bars
            
            bars = get_volume_bars(self.test_data, threshold=threshold)
            
            # Validate results
            self.assertIsInstance(bars, pd.DataFrame)
            self.assertGreater(len(bars), 0)
            self.assertTrue(all(col in bars.columns for col in ['open', 'high', 'low', 'close', 'volume']))
            
            execution_time = time.time() - start_time
            self.test_results['volume_bars'] = {'passed': True, 'execution_time': execution_time}
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results['volume_bars'] = {'passed': False, 'execution_time': execution_time, 'error': str(e)}
            raise
    
    def test_dollar_bars(self):
        """Test dollar bars functionality"""
        print("🧪 Testing Dollar Bars...")
        start_time = time.time()
        
        try:
            # Calculate appropriate threshold
            total_value = (self.test_data['price'] * self.test_data['volume']).sum()
            threshold = int(total_value / 15)  # Create ~15 bars
            
            bars = get_dollar_bars(self.test_data, threshold=threshold)
            
            # Validate results
            self.assertIsInstance(bars, pd.DataFrame)
            self.assertGreater(len(bars), 0)
            self.assertTrue(all(col in bars.columns for col in ['open', 'high', 'low', 'close', 'volume']))
            
            execution_time = time.time() - start_time
            self.test_results['dollar_bars'] = {'passed': True, 'execution_time': execution_time}
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results['dollar_bars'] = {'passed': False, 'execution_time': execution_time, 'error': str(e)}
            raise
    
    def test_time_bars(self):
        """Test time bars functionality"""
        print("🧪 Testing Time Bars...")
        start_time = time.time()
        
        try:
            # Test with different intervals
            for interval in ['5min', '10min', '15min']:
                bars = get_time_bars(self.test_data, interval=interval)
                
                # Validate results
                self.assertIsInstance(bars, pd.DataFrame)
                self.assertGreater(len(bars), 0)
                self.assertTrue(all(col in bars.columns for col in ['open', 'high', 'low', 'close', 'volume']))
                
            execution_time = time.time() - start_time
            self.test_results['time_bars'] = {'passed': True, 'execution_time': execution_time}
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results['time_bars'] = {'passed': False, 'execution_time': execution_time, 'error': str(e)}
            raise
    
    def test_imbalance_bars(self):
        """Test imbalance bars functionality"""
        print("🧪 Testing Imbalance Bars...")
        start_time = time.time()
        
        try:
            # Test tick imbalance bars
            tick_imbalance_bars = get_tick_imbalance_bars(self.test_data, num_ticks_init=20)
            self.assertIsInstance(tick_imbalance_bars, pd.DataFrame)
            self.assertGreater(len(tick_imbalance_bars), 0)
            
            # Test volume imbalance bars
            volume_imbalance_bars = get_volume_imbalance_bars(self.test_data, num_ticks_init=20)
            self.assertIsInstance(volume_imbalance_bars, pd.DataFrame)
            self.assertGreater(len(volume_imbalance_bars), 0)
            
            # Test dollar imbalance bars
            dollar_imbalance_bars = get_dollar_imbalance_bars(self.test_data, num_ticks_init=20)
            self.assertIsInstance(dollar_imbalance_bars, pd.DataFrame)
            self.assertGreater(len(dollar_imbalance_bars), 0)
            
            execution_time = time.time() - start_time
            self.test_results['imbalance_bars'] = {'passed': True, 'execution_time': execution_time}
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results['imbalance_bars'] = {'passed': False, 'execution_time': execution_time, 'error': str(e)}
            raise
    
    def test_run_bars(self):
        """Test run bars functionality"""
        print("🧪 Testing Run Bars...")
        start_time = time.time()
        
        try:
            # Test tick run bars
            tick_run_bars = get_tick_run_bars(self.test_data, num_ticks_init=20)
            self.assertIsInstance(tick_run_bars, pd.DataFrame)
            self.assertGreater(len(tick_run_bars), 0)
            
            # Test volume run bars
            volume_run_bars = get_volume_run_bars(self.test_data, num_ticks_init=20)
            self.assertIsInstance(volume_run_bars, pd.DataFrame)
            self.assertGreater(len(volume_run_bars), 0)
            
            # Test dollar run bars
            dollar_run_bars = get_dollar_run_bars(self.test_data, num_ticks_init=20)
            self.assertIsInstance(dollar_run_bars, pd.DataFrame)
            self.assertGreater(len(dollar_run_bars), 0)
            
            execution_time = time.time() - start_time
            self.test_results['run_bars'] = {'passed': True, 'execution_time': execution_time}
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results['run_bars'] = {'passed': False, 'execution_time': execution_time, 'error': str(e)}
            raise
    
    def test_data_quality(self):
        """Test data quality and validation"""
        print("🧪 Testing Data Quality...")
        start_time = time.time()
        
        try:
            # Test with missing data
            data_with_nans = self.test_data.copy()
            data_with_nans.loc[10:15, 'price'] = np.nan
            
            bars = get_tick_bars(data_with_nans, threshold=50)
            self.assertIsInstance(bars, pd.DataFrame)
            
            # Test with zero prices
            data_with_zeros = self.test_data.copy()
            data_with_zeros.loc[20:25, 'price'] = 0
            
            bars = get_tick_bars(data_with_zeros, threshold=50)
            self.assertIsInstance(bars, pd.DataFrame)
            
            execution_time = time.time() - start_time
            self.test_results['data_quality'] = {'passed': True, 'execution_time': execution_time}
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results['data_quality'] = {'passed': False, 'execution_time': execution_time, 'error': str(e)}
            raise
    
    def test_performance_metrics(self):
        """Test performance and resource usage"""
        print("🧪 Testing Performance Metrics...")
        start_time = time.time()
        
        try:
            # Test with larger dataset
            large_data = self._create_large_test_data(10000)
            
            # Measure performance for different bar types
            performance_results = {}
            
            # Test tick bars performance
            tick_start = time.time()
            tick_bars = get_tick_bars(large_data, threshold=500)
            performance_results['tick_bars'] = {
                'execution_time': time.time() - tick_start,
                'bars_created': len(tick_bars),
                'memory_usage': tick_bars.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            }
            
            # Test volume bars performance
            volume_start = time.time()
            volume_bars = get_volume_bars(large_data, threshold=large_data['volume'].sum() // 20)
            performance_results['volume_bars'] = {
                'execution_time': time.time() - volume_start,
                'bars_created': len(volume_bars),
                'memory_usage': volume_bars.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            }
            
            self.performance_metrics = performance_results
            
            execution_time = time.time() - start_time
            self.test_results['performance_metrics'] = {'passed': True, 'execution_time': execution_time}
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results['performance_metrics'] = {'passed': False, 'execution_time': execution_time, 'error': str(e)}
            raise
    
    def _create_large_test_data(self, n_points):
        """Create large test dataset for performance testing"""
        np.random.seed(42)
        
        timestamps = pd.date_range(start='2023-01-01', periods=n_points, freq='1min')
        
        initial_price = 100.0
        returns = np.random.normal(0, 0.001, n_points)
        prices = [initial_price]
        
        for i in range(1, n_points):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(new_price)
        
        volumes = np.random.lognormal(mean=10, sigma=1, size=n_points)
        volumes = volumes.astype(int)
        
        return pd.DataFrame({
            'date_time': timestamps,
            'price': prices,
            'volume': volumes
        })
    
    @classmethod
    def tearDownClass(cls):
        """Save test results"""
        # Calculate summary statistics
        total_tests = len(cls.test_results)
        passed_tests = len([r for r in cls.test_results.values() if r['passed']])
        total_time = sum(r['execution_time'] for r in cls.test_results.values())
        
        # Create comprehensive results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'module': 'data_structures',
            'total_tests': total_tests,
            'tests_passed': passed_tests,
            'tests_failed': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'total_execution_time': total_time,
            'test_details': cls.test_results,
            'performance_metrics': cls.performance_metrics,
            'test_data_summary': {
                'data_points': len(cls.test_data),
                'date_range': f"{cls.test_data['date_time'].min()} to {cls.test_data['date_time'].max()}",
                'price_range': f"{cls.test_data['price'].min():.2f} to {cls.test_data['price'].max():.2f}",
                'volume_range': f"{cls.test_data['volume'].min()} to {cls.test_data['volume'].max()}"
            }
        }
        
        # Save results
        results_file = cls.results_path / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("📊 DATA STRUCTURES TEST SUMMARY")
        print("="*50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {final_results['success_rate']:.1f}%")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Results saved to: {results_file}")
        print("="*50)

if __name__ == '__main__':
    unittest.main(verbosity=2)
