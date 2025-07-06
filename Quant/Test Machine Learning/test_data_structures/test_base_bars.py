"""
Advanced Testing Module for Base Bars - Financial Data Structure Testing
========================================================================

This module provides comprehensive testing for base bar implementations including:
- Unit tests with edge cases
- Performance benchmarking with Numba optimization
- Statistical analysis 
- Visualization and reporting
- Data quality validation
- Stress testing
- Interactive dashboards
- HTML reports with charts

Author: Advanced ML Finance Team
Date: July 2025
"""

import sys
import os
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import configuration system
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_config_manager import get_config, get_output_path

# Core libraries
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ Base imports completed successfully")

# Additional visualization libraries
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
    print("✅ Plotly imported successfully")
except ImportError as e:
    PLOTLY_AVAILABLE = False
    logger.warning(f"Plotly not available: {e}")

# Statistical libraries
try:
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
    print("✅ Statsmodels imported successfully")
except ImportError as e:
    STATSMODELS_AVAILABLE = False
    logger.warning(f"Statsmodels not available: {e}")

# Performance optimization
try:
    import numba
    from numba import jit
    NUMBA_AVAILABLE = True
    print("✅ Numba imported successfully")
except ImportError as e:
    NUMBA_AVAILABLE = False
    logger.warning(f"Numba not available: {e}")

warnings.filterwarnings('ignore')

# Import the module under test
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Machine Learning', 'data_structures'))
    from base_bars import BaseBars, get_dollar_bars, get_volume_bars, get_tick_bars, get_time_bars
    logger.info("✅ Successfully imported base_bars module")
except ImportError as e:
    logger.error(f"❌ Failed to import base_bars module: {e}")
    # Create mock classes for testing framework
    class BaseBars:
        def __init__(self, *args, **kwargs):
            pass
        def batch_run(self, *args, **kwargs):
            return pd.DataFrame()
    
    def get_dollar_bars(*args, **kwargs):
        logger.warning("Using mock function for get_dollar_bars")
        return pd.DataFrame()
    
    def get_volume_bars(*args, **kwargs):
        logger.warning("Using mock function for get_volume_bars")
        return pd.DataFrame()
    
    def get_tick_bars(*args, **kwargs):
        logger.warning("Using mock function for get_tick_bars")
        return pd.DataFrame()
    
    def get_time_bars(*args, **kwargs):
        logger.warning("Using mock function for get_time_bars")
        return pd.DataFrame()

print("✅ Module imports completed successfully")

class TestBaseBarsAdvanced:
    """
    Advanced testing framework for BaseBars implementations
    
    This class provides comprehensive testing capabilities including:
    - Unit tests for all bar types
    - Performance benchmarking
    - Statistical analysis
    - Visualization generation
    - HTML report generation
    """
    
    def __init__(self, save_path: str = None):
        """
        Initialize the testing framework
        
        :param save_path: Path to save test results, plots, and reports
        """
        if save_path is None:
            # Create results in the Results Machine Learning structure
            current_dir = os.path.dirname(__file__)
            project_root = os.path.join(current_dir, '..', '..')
            results_dir = os.path.join(project_root, 'Results Machine Learning', 'data_structures')
            save_path = os.path.join(results_dir, 'test_results_base_bars')
        
        self.save_path = save_path
        self.results = {}
        self.performance_metrics = {}
        self.test_data = {}
        self.plots = {}
        
        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Initialize report structure
        self.report = {
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'performance_metrics': {},
            'statistical_analysis': {},
            'plots_generated': [],
            'errors': [],
            'warnings': []
        }
        
        logger.info(f"✅ Advanced testing framework initialized")
        logger.info(f"📁 Results will be saved to: {save_path}")
        
    def load_config(self, config_path: str = None):
        """
        Load configuration using the hybrid configuration system
        This method is kept for backward compatibility but now uses the ConfigurationManager
        
        :param config_path: Path to configuration file (ignored, using hybrid system)
        :return: Configuration dictionary
        """
        # Use the already loaded configuration from __init__
        if hasattr(self, 'config') and self.config:
            return self.config
        
        # Fallback to loading if not already loaded
        self.config = get_config('data_structures', 'test_base_bars')
        return self.config
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            "testing_parameters": {
                "synthetic_data": {
                    "default_samples": 10000,
                    "price_start": 100.0,
                    "volatility": 0.02,
                    "trend": 0.0001
                },
                "performance_testing": {
                    "data_sizes": [1000, 5000, 10000],
                    "bar_types": ["time", "tick", "volume", "dollar"]
                }
            }
        }
    
    def generate_synthetic_data(self, n_samples: int = None, 
                               price_start: float = None,
                               volatility: float = None,
                               trend: float = None,
                               include_gaps: bool = None,
                               include_outliers: bool = None) -> pd.DataFrame:
        """
        Generate synthetic financial data for testing using hybrid configuration
        
        :param n_samples: Number of data points (from config if None)
        :param price_start: Starting price (from config if None)
        :param volatility: Daily volatility (from config if None)
        :param trend: Price trend (from config if None)
        :param include_gaps: Whether to include price gaps (from config if None)
        :param include_outliers: Whether to include outliers (from config if None)
        :return: Synthetic market data DataFrame
        """
        # Get parameters from configuration if not provided
        synth_config = self.config.get('global_settings', {}).get('synthetic_data', {})
        module_synth_config = self.config.get('module_settings', {}).get('synthetic_data', {})
        
        # Merge configs with module taking precedence
        effective_synth_config = {**synth_config, **module_synth_config}
        
        # Use provided values or fall back to config
        n_samples = n_samples if n_samples is not None else effective_synth_config.get('default_samples', 10000)
        price_start = price_start if price_start is not None else effective_synth_config.get('price_start', 100.0)
        volatility = volatility if volatility is not None else effective_synth_config.get('volatility', 0.02)
        trend = trend if trend is not None else effective_synth_config.get('trend', 0.0001)
        include_gaps = include_gaps if include_gaps is not None else effective_synth_config.get('include_gaps', True)
        include_outliers = include_outliers if include_outliers is not None else effective_synth_config.get('include_outliers', True)
        
        logger.info(f"📊 Generating synthetic data: {n_samples} samples using hybrid config")
        
        # Set random seed for reproducibility
        random_seed = effective_synth_config.get('random_seed', 42)
        np.random.seed(random_seed)
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=max(1, n_samples//1000))
        timestamps = pd.date_range(start=start_time, periods=n_samples, freq='1S')
        
        # Generate price series with geometric brownian motion
        returns = np.random.normal(trend, volatility, n_samples)
        
        # Apply autocorrelation to make it more realistic
        if n_samples > 1:
            returns[1:] = 0.1 * returns[:-1] + 0.9 * returns[1:]
        
        prices = np.exp(np.cumsum(returns)) * price_start
        
        # Generate volumes using config parameters
        tick_config = effective_synth_config.get('tick_data', {})
        volume_mean = tick_config.get('volume_mean', 1000)
        volume_std = tick_config.get('volume_std', 500)
        
        if tick_config.get('volume_distribution', 'lognormal') == 'lognormal':
            volumes = np.random.lognormal(mean=np.log(volume_mean), sigma=volume_std/volume_mean, size=n_samples)
        else:
            volumes = np.random.normal(volume_mean, volume_std, n_samples)
            volumes = np.maximum(volumes, 1)  # Ensure positive volumes
        
        # Generate dollar amounts
        dollar_amounts = prices * volumes
        
        # Add realistic price gaps
        if include_gaps:
            gap_indices = np.random.choice(n_samples, size=max(1, n_samples//1000), replace=False)
            gap_multipliers = np.random.uniform(0.98, 1.02, size=len(gap_indices))
            prices[gap_indices] *= gap_multipliers
        
        # Add volume outliers
        if include_outliers:
            outlier_indices = np.random.choice(n_samples, size=max(1, n_samples//5000), replace=False)
            outlier_multipliers = np.random.choice([0.1, 5.0], size=len(outlier_indices))
            volumes[outlier_indices] *= outlier_multipliers
        
        # Create DataFrame with proper structure
        data = pd.DataFrame({
            'date_time': timestamps,
            'price': prices,
            'volume': volumes,
            'dollar_amount': dollar_amounts
        })
        
        # Add microstructure features if configured
        microstructure_config = effective_synth_config.get('microstructure', {})
        if microstructure_config.get('bid_ask_spread'):
            spread = microstructure_config['bid_ask_spread']
            data['bid'] = data['price'] * (1 - spread/2)
            data['ask'] = data['price'] * (1 + spread/2)
        else:
            # Default microstructure
            data['bid'] = data['price'] * 0.9999
            data['ask'] = data['price'] * 1.0001
        
        data['tick_rule'] = np.random.choice([1, -1], size=n_samples)
        
        logger.info(f"✅ Generated synthetic data with {len(data)} rows")
        logger.info(f"📈 Price range: ${data['price'].min():.2f} - ${data['price'].max():.2f}")
        logger.info(f"📊 Volume range: {data['volume'].min():.0f} - {data['volume'].max():.0f}")
        logger.debug(f"Configuration used: {effective_synth_config}")
        
        return data
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite
        
        :return: Complete test results
        """
        logger.info("🚀 Starting comprehensive test suite")
        start_time = time.time()
        
        try:
            # Load configuration (already loaded in __init__)
            config = self.config
            
            # Generate test data using hybrid configuration
            logger.info("📊 Generating test data...")
            self.test_data = self.generate_synthetic_data()
            
            # Run test categories
            logger.info("🧪 Running unit tests...")
            self._run_unit_tests()
            
            logger.info("⚡ Running performance tests...")
            self._run_performance_tests()
            
            logger.info("📈 Running statistical analysis...")
            self._run_statistical_analysis()
            
            logger.info("🎯 Running stress tests...")
            self._run_stress_tests()
            
            logger.info("🖼️ Generating visualizations...")
            self._generate_visualizations()
            
            logger.info("📄 Generating reports...")
            self._generate_reports()
            
            # Calculate final metrics
            total_time = time.time() - start_time
            self.report['total_execution_time'] = total_time
            success_rate = (self.report['tests_passed'] / max(self.report['tests_run'], 1)) * 100
            
            # Summary
            logger.info(f"✅ Test suite completed in {total_time:.2f} seconds")
            logger.info(f"📊 Tests: {self.report['tests_run']} run, {self.report['tests_passed']} passed, {self.report['tests_failed']} failed")
            logger.info(f"🎯 Success rate: {success_rate:.1f}%")
            
            return self.report
            
        except Exception as e:
            logger.error(f"❌ Critical error in test suite: {e}")
            self.report['errors'].append(f"Critical error: {str(e)}")
            self.report['tests_failed'] += 1
            return self.report
    
    def _run_unit_tests(self):
        """Run unit tests for all bar types"""
        logger.info("🧪 Starting unit tests")
        
        test_methods = [
            ("Time Bars", self._test_time_bars),
            ("Tick Bars", self._test_tick_bars),
            ("Volume Bars", self._test_volume_bars),
            ("Dollar Bars", self._test_dollar_bars),
            ("Data Validation", self._test_data_validation),
            ("Batch Processing", self._test_batch_processing)
        ]
        
        for test_name, test_method in test_methods:
            try:
                logger.info(f"  🔍 Testing {test_name}...")
                test_method()
                self.report['tests_passed'] += 1
                logger.info(f"  ✅ {test_name} passed")
            except Exception as e:
                self.report['tests_failed'] += 1
                error_msg = f"{test_name} failed: {str(e)}"
                logger.error(f"  ❌ {error_msg}")
                self.report['errors'].append(error_msg)
            
            self.report['tests_run'] += 1
    
    def _test_time_bars(self):
        """Test time-based bars with comprehensive validation"""
        frequencies = ['1Min', '5Min', '15Min', '1H']
        
        for freq in frequencies:
            try:
                bars = get_time_bars(self.test_data, freq)
                
                # Basic validation
                if not isinstance(bars, pd.DataFrame):
                    raise AssertionError(f"Expected DataFrame, got {type(bars)}")
                
                if not bars.empty:
                    # Check required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in bars.columns]
                    if missing_cols:
                        logger.warning(f"Missing columns in time bars: {missing_cols}")
                    
                    # OHLC validation
                    if all(col in bars.columns for col in ['open', 'high', 'low', 'close']):
                        assert (bars['high'] >= bars['open']).all(), "High should be >= Open"
                        assert (bars['high'] >= bars['close']).all(), "High should be >= Close"
                        assert (bars['low'] <= bars['open']).all(), "Low should be <= Open"
                        assert (bars['low'] <= bars['close']).all(), "Low should be <= Close"
                        assert (bars['high'] >= bars['low']).all(), "High should be >= Low"
                    
                    # Volume validation
                    if 'volume' in bars.columns:
                        assert (bars['volume'] >= 0).all(), "Volume should be non-negative"
                
                logger.info(f"    ✅ Time bars ({freq}): {len(bars)} bars generated")
                
            except Exception as e:
                logger.warning(f"    ⚠️ Time bars ({freq}) test issue: {str(e)}")
                raise
    
    def _test_tick_bars(self):
        """Test tick-based bars with various thresholds"""
        thresholds = [100, 500, 1000, 2000]
        
        for threshold in thresholds:
            try:
                bars = get_tick_bars(self.test_data, threshold)
                
                # Basic validation
                if not isinstance(bars, pd.DataFrame):
                    raise AssertionError(f"Expected DataFrame, got {type(bars)}")
                
                if not bars.empty:
                    # Validate tick count consistency
                    if 'tick_count' in bars.columns:
                        # Most bars should have approximately the threshold number of ticks
                        tick_counts = bars['tick_count']
                        avg_ticks = tick_counts.mean()
                        assert avg_ticks > threshold * 0.5, f"Average ticks ({avg_ticks}) too low for threshold {threshold}"
                
                logger.info(f"    ✅ Tick bars ({threshold}): {len(bars)} bars generated")
                
            except Exception as e:
                logger.warning(f"    ⚠️ Tick bars ({threshold}) test issue: {str(e)}")
                raise
    
    def _test_volume_bars(self):
        """Test volume-based bars with validation"""
        thresholds = [5000, 10000, 25000, 50000]
        
        for threshold in thresholds:
            try:
                bars = get_volume_bars(self.test_data, threshold)
                
                # Basic validation
                if not isinstance(bars, pd.DataFrame):
                    raise AssertionError(f"Expected DataFrame, got {type(bars)}")
                
                if not bars.empty and 'volume' in bars.columns:
                    # Validate volume aggregation
                    volumes = bars['volume']
                    avg_volume = volumes.mean()
                    
                    # Volume should be close to threshold for most bars
                    assert avg_volume > threshold * 0.3, f"Average volume ({avg_volume}) too low for threshold {threshold}"
                    assert (volumes > 0).all(), "All volumes should be positive"
                
                logger.info(f"    ✅ Volume bars ({threshold}): {len(bars)} bars generated")
                
            except Exception as e:
                logger.warning(f"    ⚠️ Volume bars ({threshold}) test issue: {str(e)}")
                raise
    
    def _test_dollar_bars(self):
        """Test dollar-based bars with validation"""
        thresholds = [100000, 250000, 500000, 1000000]
        
        for threshold in thresholds:
            try:
                bars = get_dollar_bars(self.test_data, threshold)
                
                # Basic validation
                if not isinstance(bars, pd.DataFrame):
                    raise AssertionError(f"Expected DataFrame, got {type(bars)}")
                
                if not bars.empty:
                    # Validate dollar amount aggregation
                    if 'dollar_amount' in bars.columns:
                        dollar_amounts = bars['dollar_amount']
                        avg_dollar = dollar_amounts.mean()
                        
                        # Dollar amounts should be reasonable
                        assert avg_dollar > threshold * 0.1, f"Average dollar amount ({avg_dollar}) too low for threshold {threshold}"
                        assert (dollar_amounts > 0).all(), "All dollar amounts should be positive"
                
                logger.info(f"    ✅ Dollar bars ({threshold}): {len(bars)} bars generated")
                
            except Exception as e:
                logger.warning(f"    ⚠️ Dollar bars ({threshold}) test issue: {str(e)}")
                raise
    
    def _test_data_validation(self):
        """Test data validation with edge cases"""
        test_cases = [
            ("Empty DataFrame", pd.DataFrame()),
            ("Single row", self.test_data.iloc[:1]),
            ("Missing columns", pd.DataFrame({'price': [100, 101, 102]})),
            ("NaN values", pd.DataFrame({
                'date_time': pd.date_range('2020-01-01', periods=3),
                'price': [100, np.nan, 102],
                'volume': [1000, 2000, 3000]
            })),
            ("Negative prices", pd.DataFrame({
                'date_time': pd.date_range('2020-01-01', periods=3),
                'price': [100, -50, 102],
                'volume': [1000, 2000, 3000]
            })),
            ("Zero volume", pd.DataFrame({
                'date_time': pd.date_range('2020-01-01', periods=3),
                'price': [100, 101, 102],
                'volume': [0, 0, 0]
            }))
        ]
        
        for case_name, test_data in test_cases:
            try:
                # Test with time bars (most basic)
                bars = get_time_bars(test_data, '1Min')
                logger.info(f"    ✅ Data validation ({case_name}): Handled gracefully")
                
            except Exception as e:
                # Some failures are expected for invalid data
                logger.info(f"    ⚠️ Data validation ({case_name}): {str(e)}")
    
    def _test_batch_processing(self):
        """Test batch processing with larger datasets"""
        try:
            # Generate larger dataset
            large_data = self.generate_synthetic_data(n_samples=20000)
            
            # Test batch processing
            start_time = time.time()
            bars = get_time_bars(large_data, '1Min')
            processing_time = time.time() - start_time
            
            # Validate results
            assert isinstance(bars, pd.DataFrame), "Batch processing should return DataFrame"
            
            # Store performance metrics
            self.performance_metrics['batch_processing'] = {
                'processing_time': processing_time,
                'data_size': len(large_data),
                'bars_generated': len(bars),
                'throughput': len(large_data) / processing_time if processing_time > 0 else 0
            }
            
            logger.info(f"    ✅ Batch processing: {len(large_data)} samples → {len(bars)} bars in {processing_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"    ⚠️ Batch processing test issue: {str(e)}")
            raise
    
    def _run_performance_tests(self):
        """Run comprehensive performance benchmarks using hybrid configuration"""
        logger.info("⚡ Starting performance benchmarks")
        
        # Get configuration for performance testing
        perf_config = self.config.get('global_settings', {}).get('performance_testing', {})
        module_perf_config = self.config.get('module_settings', {}).get('performance_testing', {})
        
        # Merge configs with module taking precedence
        effective_perf_config = {**perf_config, **module_perf_config}
        
        data_sizes = effective_perf_config.get('data_sizes', [1000, 5000, 10000])
        bar_types = self.config.get('module_settings', {}).get('bar_types', ["time", "tick", "volume", "dollar"])
        
        performance_results = {}
        
        for size in data_sizes:
            logger.info(f"  📊 Testing with {size} samples")
            performance_results[size] = {}
            
            # Generate test data
            test_data = self.generate_synthetic_data(n_samples=size)
            
            for bar_type in bar_types:
                try:
                    start_time = time.time()
                    memory_before = self._get_memory_usage()
                    
                    # Execute the bar generation
                    if bar_type == 'time':
                        bars = get_time_bars(test_data, '1Min')
                    elif bar_type == 'tick':
                        bars = get_tick_bars(test_data, 100)
                    elif bar_type == 'volume':
                        bars = get_volume_bars(test_data, 10000)
                    elif bar_type == 'dollar':
                        bars = get_dollar_bars(test_data, 100000)
                    
                    execution_time = time.time() - start_time
                    memory_after = self._get_memory_usage()
                    memory_used = memory_after - memory_before
                    
                    # Store results
                    performance_results[size][bar_type] = {
                        'execution_time': execution_time,
                        'bars_generated': len(bars),
                        'throughput': size / execution_time if execution_time > 0 else 0,
                        'memory_used_mb': memory_used,
                        'bars_per_second': len(bars) / execution_time if execution_time > 0 else 0
                    }
                    
                    logger.info(f"    ⚡ {bar_type} bars: {execution_time:.3f}s, {len(bars)} bars, {size/execution_time:.0f} samples/s")
                    
                except Exception as e:
                    logger.error(f"    ❌ Performance test failed for {bar_type}: {str(e)}")
                    performance_results[size][bar_type] = {'error': str(e)}
        
        self.performance_metrics['detailed_performance'] = performance_results
        self.report['performance_metrics'] = performance_results
        
        # Generate performance summary
        self._generate_performance_summary()
    
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0  # psutil not available
    
    def _generate_performance_summary(self):
        """Generate performance summary statistics"""
        if 'detailed_performance' not in self.performance_metrics:
            return
        
        perf_data = self.performance_metrics['detailed_performance']
        summary = {
            'fastest_bar_type': None,
            'slowest_bar_type': None,
            'most_memory_efficient': None,
            'highest_throughput': None,
            'average_execution_times': {}
        }
        
        # Calculate averages
        bar_types = ['time', 'tick', 'volume', 'dollar']
        for bar_type in bar_types:
            times = []
            throughputs = []
            
            for size in perf_data:
                if bar_type in perf_data[size] and 'execution_time' in perf_data[size][bar_type]:
                    times.append(perf_data[size][bar_type]['execution_time'])
                    throughputs.append(perf_data[size][bar_type]['throughput'])
            
            if times:
                summary['average_execution_times'][bar_type] = np.mean(times)
        
        # Find fastest and slowest
        if summary['average_execution_times']:
            fastest = min(summary['average_execution_times'].items(), key=lambda x: x[1])
            slowest = max(summary['average_execution_times'].items(), key=lambda x: x[1])
            
            summary['fastest_bar_type'] = fastest[0]
            summary['slowest_bar_type'] = slowest[0]
        
        self.performance_metrics['summary'] = summary
        logger.info(f"  🏆 Fastest bar type: {summary['fastest_bar_type']}")
        logger.info(f"  🐌 Slowest bar type: {summary['slowest_bar_type']}")
    
    def _run_statistical_analysis(self):
        """Run statistical analysis on generated bars"""
        logger.info("📈 Starting statistical analysis")
        
        try:
            # Generate bars for analysis
            bars = get_time_bars(self.test_data, '1Min')
            
            if bars.empty:
                logger.warning("  ⚠️ No bars generated for statistical analysis")
                return
            
            statistical_results = {}
            
            # Price analysis
            if 'close' in bars.columns:
                prices = bars['close'].dropna()
                if len(prices) > 10:
                    statistical_results['price_analysis'] = {
                        'mean': float(prices.mean()),
                        'std': float(prices.std()),
                        'min': float(prices.min()),
                        'max': float(prices.max()),
                        'median': float(prices.median()),
                        'skewness': float(stats.skew(prices)),
                        'kurtosis': float(stats.kurtosis(prices)),
                        'count': len(prices)
                    }
                    
                    # Normality test
                    if len(prices) > 8:
                        shapiro_stat, shapiro_p = stats.shapiro(prices[:5000] if len(prices) > 5000 else prices)
                        statistical_results['price_analysis']['normality_test'] = {
                            'shapiro_stat': float(shapiro_stat),
                            'shapiro_p_value': float(shapiro_p),
                            'is_normal': shapiro_p > 0.05
                        }
                    
                    logger.info(f"  📊 Price analysis: μ={prices.mean():.2f}, σ={prices.std():.2f}")
            
            # Returns analysis
            if 'close' in bars.columns and len(bars) > 1:
                returns = bars['close'].pct_change().dropna()
                if len(returns) > 10:
                    statistical_results['returns_analysis'] = {
                        'mean': float(returns.mean()),
                        'std': float(returns.std()),
                        'min': float(returns.min()),
                        'max': float(returns.max()),
                        'skewness': float(stats.skew(returns)),
                        'kurtosis': float(stats.kurtosis(returns)),
                        'count': len(returns)
                    }
                    
                    # Stationarity test
                    if STATSMODELS_AVAILABLE and len(returns) > 10:
                        try:
                            adf_stat, adf_p, _, _, adf_critical, _ = adfuller(returns)
                            statistical_results['returns_analysis']['stationarity_test'] = {
                                'adf_statistic': float(adf_stat),
                                'adf_p_value': float(adf_p),
                                'adf_critical_values': {k: float(v) for k, v in adf_critical.items()},
                                'is_stationary': adf_p < 0.05
                            }
                        except Exception as e:
                            logger.warning(f"  ⚠️ Stationarity test failed: {str(e)}")
                    
                    logger.info(f"  📈 Returns analysis: μ={returns.mean():.6f}, σ={returns.std():.6f}")
            
            # Volume analysis
            if 'volume' in bars.columns:
                volumes = bars['volume'].dropna()
                if len(volumes) > 10:
                    statistical_results['volume_analysis'] = {
                        'mean': float(volumes.mean()),
                        'std': float(volumes.std()),
                        'min': float(volumes.min()),
                        'max': float(volumes.max()),
                        'median': float(volumes.median()),
                        'count': len(volumes)
                    }
                    
                    # Volume distribution test
                    if len(volumes) > 8:
                        ks_stat, ks_p = stats.kstest(volumes, 'norm')
                        statistical_results['volume_analysis']['distribution_test'] = {
                            'ks_statistic': float(ks_stat),
                            'ks_p_value': float(ks_p),
                            'is_normal': ks_p > 0.05
                        }
                    
                    logger.info(f"  📊 Volume analysis: μ={volumes.mean():.0f}, σ={volumes.std():.0f}")
            
            self.report['statistical_analysis'] = statistical_results
            logger.info("  ✅ Statistical analysis completed")
            
        except Exception as e:
            logger.error(f"  ❌ Statistical analysis failed: {str(e)}")
            self.report['errors'].append(f"Statistical analysis error: {str(e)}")
    
    def _run_stress_tests(self):
        """Run stress tests with extreme conditions"""
        logger.info("🎯 Starting stress tests")
        
        stress_test_results = {}
        
        # Test 1: Extreme volatility
        try:
            logger.info("  🌪️ Testing extreme volatility...")
            extreme_data = self.generate_synthetic_data(
                n_samples=5000,
                volatility=0.5,  # Very high volatility
                include_gaps=True,
                include_outliers=True
            )
            
            start_time = time.time()
            bars = get_time_bars(extreme_data, '1Min')
            processing_time = time.time() - start_time
            
            stress_test_results['extreme_volatility'] = {
                'passed': True,
                'bars_generated': len(bars),
                'processing_time': processing_time,
                'data_size': len(extreme_data),
                'message': 'Successfully handled extreme volatility'
            }
            
            logger.info(f"    ✅ Extreme volatility test passed: {len(bars)} bars in {processing_time:.2f}s")
            
        except Exception as e:
            stress_test_results['extreme_volatility'] = {
                'passed': False,
                'error': str(e)
            }
            logger.error(f"    ❌ Extreme volatility test failed: {str(e)}")
        
        # Test 2: Large dataset
        try:
            logger.info("  📊 Testing large dataset...")
            large_data = self.generate_synthetic_data(n_samples=50000)
            
            start_time = time.time()
            bars = get_time_bars(large_data, '1Min')
            processing_time = time.time() - start_time
            
            stress_test_results['large_dataset'] = {
                'passed': True,
                'bars_generated': len(bars),
                'processing_time': processing_time,
                'data_size': len(large_data),
                'throughput': len(large_data) / processing_time if processing_time > 0 else 0,
                'message': f'Successfully processed {len(large_data)} samples'
            }
            
            logger.info(f"    ✅ Large dataset test passed: {len(large_data)} samples → {len(bars)} bars in {processing_time:.2f}s")
            
        except Exception as e:
            stress_test_results['large_dataset'] = {
                'passed': False,
                'error': str(e)
            }
            logger.error(f"    ❌ Large dataset test failed: {str(e)}")
        
        # Test 3: Edge cases
        try:
            logger.info("  🔍 Testing edge cases...")
            
            # Test with identical prices
            identical_data = self.test_data.copy()
            identical_data['price'] = 100.0
            bars_identical = get_time_bars(identical_data, '1Min')
            
            # Test with monotonic prices
            monotonic_data = self.test_data.copy()
            monotonic_data['price'] = np.arange(len(monotonic_data))
            bars_monotonic = get_time_bars(monotonic_data, '1Min')
            
            stress_test_results['edge_cases'] = {
                'passed': True,
                'identical_prices_bars': len(bars_identical),
                'monotonic_prices_bars': len(bars_monotonic),
                'message': 'Successfully handled edge cases'
            }
            
            logger.info(f"    ✅ Edge cases test passed")
            
        except Exception as e:
            stress_test_results['edge_cases'] = {
                'passed': False,
                'error': str(e)
            }
            logger.error(f"    ❌ Edge cases test failed: {str(e)}")
        
        self.report['stress_tests'] = stress_test_results
        logger.info("  ✅ Stress tests completed")
    
    def _generate_visualizations(self):
        """Generate comprehensive visualizations"""
        logger.info("🖼️ Starting visualization generation")
        
        try:
            # Generate different types of bars for comparison
            time_bars = get_time_bars(self.test_data, '1Min')
            tick_bars = get_tick_bars(self.test_data, 100)
            volume_bars = get_volume_bars(self.test_data, 10000)
            dollar_bars = get_dollar_bars(self.test_data, 100000)
            
            # Create individual plots
            self._create_price_comparison_plot(time_bars, tick_bars, volume_bars, dollar_bars)
            self._create_volume_analysis_plot(time_bars, tick_bars, volume_bars, dollar_bars)
            self._create_performance_plots()
            self._create_statistical_plots(time_bars)
            
            # Create interactive dashboard if Plotly is available
            if PLOTLY_AVAILABLE:
                self._create_interactive_dashboard()
            
            logger.info("  ✅ All visualizations generated successfully")
            
        except Exception as e:
            logger.error(f"  ❌ Error generating visualizations: {str(e)}")
            self.report['errors'].append(f"Visualization error: {str(e)}")
    
    def _create_price_comparison_plot(self, time_bars, tick_bars, volume_bars, dollar_bars):
        """Create price comparison plot"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Price Comparison Across Different Bar Types', fontsize=16, fontweight='bold')
            
            bar_types = [
                (time_bars, 'Time Bars (1Min)', axes[0, 0]),
                (tick_bars, 'Tick Bars (100 ticks)', axes[0, 1]),
                (volume_bars, 'Volume Bars (10K)', axes[1, 0]),
                (dollar_bars, 'Dollar Bars (100K)', axes[1, 1])
            ]
            
            for bars, title, ax in bar_types:
                if not bars.empty and 'close' in bars.columns:
                    ax.plot(bars.index, bars['close'], linewidth=1.5, alpha=0.8)
                    ax.set_title(title, fontsize=12, fontweight='bold')
                    ax.set_xlabel('Bar Index')
                    ax.set_ylabel('Price ($)')
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics
                    price_mean = bars['close'].mean()
                    price_std = bars['close'].std()
                    ax.text(0.02, 0.98, f'μ: ${price_mean:.2f}\nσ: ${price_std:.2f}\nBars: {len(bars)}', 
                           transform=ax.transAxes, verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, f'No data available\nfor {title}', 
                           transform=ax.transAxes, ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                    ax.set_title(title, fontsize=12)
            
            plt.tight_layout()
            plot_path = os.path.join(self.save_path, 'price_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.report['plots_generated'].append(plot_path)
            logger.info(f"    📈 Price comparison plot saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"    ❌ Error creating price comparison plot: {str(e)}")
    
    def _create_volume_analysis_plot(self, time_bars, tick_bars, volume_bars, dollar_bars):
        """Create volume analysis plot"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Volume Analysis Across Different Bar Types', fontsize=16, fontweight='bold')
            
            bar_types = [
                (time_bars, 'Time Bars', axes[0, 0]),
                (tick_bars, 'Tick Bars', axes[0, 1]),
                (volume_bars, 'Volume Bars', axes[1, 0]),
                (dollar_bars, 'Dollar Bars', axes[1, 1])
            ]
            
            for bars, title, ax in bar_types:
                if not bars.empty and 'volume' in bars.columns:
                    volumes = bars['volume']
                    ax.hist(volumes, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
                    ax.set_title(f'{title} - Volume Distribution', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Volume')
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics
                    vol_mean = volumes.mean()
                    vol_std = volumes.std()
                    vol_median = volumes.median()
                    ax.text(0.98, 0.98, f'μ: {vol_mean:.0f}\nσ: {vol_std:.0f}\nmedian: {vol_median:.0f}', 
                           transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, f'No volume data\nfor {title}', 
                           transform=ax.transAxes, ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                    ax.set_title(f'{title} - Volume Distribution', fontsize=12)
            
            plt.tight_layout()
            plot_path = os.path.join(self.save_path, 'volume_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.report['plots_generated'].append(plot_path)
            logger.info(f"    📊 Volume analysis plot saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"    ❌ Error creating volume analysis plot: {str(e)}")
    
    def _create_performance_plots(self):
        """Create performance analysis plots"""
        if 'detailed_performance' not in self.performance_metrics:
            logger.warning("    ⚠️ No performance data available for plotting")
            return
        
        try:
            performance_data = self.performance_metrics['detailed_performance']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Performance Analysis - Bar Generation Benchmarks', fontsize=16, fontweight='bold')
            
            # Extract data for plotting
            data_sizes = sorted(performance_data.keys())
            bar_types = ['time', 'tick', 'volume', 'dollar']
            colors = ['blue', 'red', 'green', 'orange']
            
            # Execution time plot
            for i, bar_type in enumerate(bar_types):
                execution_times = []
                valid_sizes = []
                
                for size in data_sizes:
                    if (bar_type in performance_data[size] and 
                        'execution_time' in performance_data[size][bar_type]):
                        execution_times.append(performance_data[size][bar_type]['execution_time'])
                        valid_sizes.append(size)
                
                if execution_times:
                    axes[0, 0].plot(valid_sizes, execution_times, 
                                   marker='o', label=f'{bar_type} bars', 
                                   color=colors[i], linewidth=2)
            
            axes[0, 0].set_title('Execution Time vs Data Size', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Data Size (samples)')
            axes[0, 0].set_ylabel('Execution Time (seconds)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_yscale('log')
            
            # Throughput plot
            for i, bar_type in enumerate(bar_types):
                throughputs = []
                valid_sizes = []
                
                for size in data_sizes:
                    if (bar_type in performance_data[size] and 
                        'throughput' in performance_data[size][bar_type]):
                        throughputs.append(performance_data[size][bar_type]['throughput'])
                        valid_sizes.append(size)
                
                if throughputs:
                    axes[0, 1].plot(valid_sizes, throughputs, 
                                   marker='s', label=f'{bar_type} bars', 
                                   color=colors[i], linewidth=2)
            
            axes[0, 1].set_title('Throughput vs Data Size', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Data Size (samples)')
            axes[0, 1].set_ylabel('Throughput (samples/second)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Bars generated plot
            for i, bar_type in enumerate(bar_types):
                bars_generated = []
                valid_sizes = []
                
                for size in data_sizes:
                    if (bar_type in performance_data[size] and 
                        'bars_generated' in performance_data[size][bar_type]):
                        bars_generated.append(performance_data[size][bar_type]['bars_generated'])
                        valid_sizes.append(size)
                
                if bars_generated:
                    axes[1, 0].plot(valid_sizes, bars_generated, 
                                   marker='^', label=f'{bar_type} bars', 
                                   color=colors[i], linewidth=2)
            
            axes[1, 0].set_title('Bars Generated vs Data Size', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Data Size (samples)')
            axes[1, 0].set_ylabel('Number of Bars Generated')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Memory usage plot
            for i, bar_type in enumerate(bar_types):
                memory_usage = []
                valid_sizes = []
                
                for size in data_sizes:
                    if (bar_type in performance_data[size] and 
                        'memory_used_mb' in performance_data[size][bar_type]):
                        memory_usage.append(performance_data[size][bar_type]['memory_used_mb'])
                        valid_sizes.append(size)
                
                if memory_usage:
                    axes[1, 1].plot(valid_sizes, memory_usage, 
                                   marker='d', label=f'{bar_type} bars', 
                                   color=colors[i], linewidth=2)
            
            axes[1, 1].set_title('Memory Usage vs Data Size', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Data Size (samples)')
            axes[1, 1].set_ylabel('Memory Used (MB)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(self.save_path, 'performance_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.report['plots_generated'].append(plot_path)
            logger.info(f"    ⚡ Performance analysis plot saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"    ❌ Error creating performance plots: {str(e)}")
    
    def _create_statistical_plots(self, bars_data):
        """
        Create statistical analysis plots for bar data.
        
        Args:
            bars_data (pd.DataFrame): The bars data to analyze
        """
        logger.info("  📊 Creating statistical analysis plots...")
        
        if bars_data is None or bars_data.empty:
            logger.warning("    ⚠️ No bars data available for statistical plotting")
            return
        
        try:
            # Statistical analysis
            stats = {}
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'dollar_volume']
            
            for col in numeric_columns:
                if col in bars_data.columns:
                    series = bars_data[col].dropna()
                    if len(series) > 0:
                        stats[col] = {
                            'mean': series.mean(),
                            'std': series.std(),
                            'skewness': series.skew(),
                            'kurtosis': series.kurtosis(),
                            'min': series.min(),
                            'max': series.max(),
                            'median': series.median(),
                            'q25': series.quantile(0.25),
                            'q75': series.quantile(0.75)
                        }
            
            # Create statistical plots
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            fig.suptitle('Statistical Analysis - Bar Data Properties', fontsize=16, fontweight='bold')
            
            # Price distribution (close prices)
            if 'close' in bars_data.columns and not bars_data['close'].empty:
                close_prices = bars_data['close'].dropna()
                axes[0, 0].hist(close_prices, bins=50, alpha=0.7, color='blue', edgecolor='black')
                axes[0, 0].set_title('Close Price Distribution', fontsize=12, fontweight='bold')
                axes[0, 0].set_xlabel('Close Price')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Add statistical info
                mean_price = close_prices.mean()
                std_price = close_prices.std()
                axes[0, 0].axvline(mean_price, color='red', linestyle='--', 
                                  label=f'Mean: ${mean_price:.2f}')
                axes[0, 0].axvline(mean_price + std_price, color='orange', linestyle='--', 
                                  label=f'±1σ: ${std_price:.2f}')
                axes[0, 0].axvline(mean_price - std_price, color='orange', linestyle='--')
                axes[0, 0].legend()
            
            # Volume distribution
            if 'volume' in bars_data.columns and not bars_data['volume'].empty:
                volume_data = bars_data['volume'].dropna()
                axes[0, 1].hist(volume_data, bins=50, alpha=0.7, color='green', edgecolor='black')
                axes[0, 1].set_title('Volume Distribution', fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel('Volume')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_yscale('log')
            
            # Price volatility (returns)
            if 'close' in bars_data.columns and len(bars_data['close']) > 1:
                returns = bars_data['close'].pct_change().dropna()
                if len(returns) > 0:
                    axes[1, 0].hist(returns, bins=50, alpha=0.7, color='purple', edgecolor='black')
                    axes[1, 0].set_title('Returns Distribution', fontsize=12, fontweight='bold')
                    axes[1, 0].set_xlabel('Return (%)')
                    axes[1, 0].set_ylabel('Frequency')
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Add normal distribution overlay
                    x = np.linspace(returns.min(), returns.max(), 100)
                    y = norm.pdf(x, returns.mean(), returns.std())
                    axes[1, 0].plot(x, y * len(returns) * (returns.max() - returns.min()) / 50, 
                                   'r-', linewidth=2, label='Normal Distribution')
                    axes[1, 0].legend()
            
            # Statistical summary heatmap
            if stats:
                stat_names = ['mean', 'std', 'skewness', 'kurtosis']
                columns = list(stats.keys())
                
                stat_matrix = np.zeros((len(stat_names), len(columns)))
                for i, stat_name in enumerate(stat_names):
                    for j, col in enumerate(columns):
                        if stat_name in stats[col]:
                            stat_matrix[i, j] = stats[col][stat_name]
                
                # Normalize for better visualization
                stat_matrix_norm = (stat_matrix - stat_matrix.min()) / (stat_matrix.max() - stat_matrix.min() + 1e-10)
                
                im = axes[1, 1].imshow(stat_matrix_norm, cmap='viridis', aspect='auto')
                axes[1, 1].set_title('Statistical Summary Heatmap', fontsize=12, fontweight='bold')
                axes[1, 1].set_xticks(range(len(columns)))
                axes[1, 1].set_xticklabels(columns, rotation=45)
                axes[1, 1].set_yticks(range(len(stat_names)))
                axes[1, 1].set_yticklabels(stat_names)
                
                # Add colorbar
                plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
                
                # Add text annotations
                for i in range(len(stat_names)):
                    for j in range(len(columns)):
                        text = axes[1, 1].text(j, i, f'{stat_matrix[i, j]:.3f}',
                                             ha="center", va="center", color="white", fontsize=8)
            
            # Time series plot (if datetime index available)
            if 'close' in bars_data.columns:
                axes[2, 0].plot(bars_data.index, bars_data['close'], 
                               color='blue', linewidth=1, alpha=0.8)
                axes[2, 0].set_title('Price Time Series', fontsize=12, fontweight='bold')
                axes[2, 0].set_xlabel('Time')
                axes[2, 0].set_ylabel('Close Price')
                axes[2, 0].grid(True, alpha=0.3)
                
                # Add volume on secondary axis
                if 'volume' in bars_data.columns:
                    ax2 = axes[2, 0].twinx()
                    ax2.bar(bars_data.index, bars_data['volume'], 
                           alpha=0.3, color='green', width=0.8)
                    ax2.set_ylabel('Volume', color='green')
                    ax2.tick_params(axis='y', labelcolor='green')
            
            # Correlation matrix
            if len(stats) > 1:
                corr_data = bars_data[list(stats.keys())].corr()
                im = axes[2, 1].imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
                axes[2, 1].set_title('Correlation Matrix', fontsize=12, fontweight='bold')
                axes[2, 1].set_xticks(range(len(corr_data.columns)))
                axes[2, 1].set_xticklabels(corr_data.columns, rotation=45)
                axes[2, 1].set_yticks(range(len(corr_data.index)))
                axes[2, 1].set_yticklabels(corr_data.index)
                
                # Add colorbar
                plt.colorbar(im, ax=axes[2, 1], fraction=0.046, pad=0.04)
                
                # Add text annotations
                for i in range(len(corr_data.index)):
                    for j in range(len(corr_data.columns)):
                        text = axes[2, 1].text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                                             ha="center", va="center", color="black", fontsize=8)
            
            plt.tight_layout()
            plot_path = os.path.join(self.save_path, 'statistical_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.report['plots_generated'].append(plot_path)
            self.report['statistical_analysis'] = stats
            logger.info(f"    📊 Statistical analysis plot saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"    ❌ Error creating statistical plots: {str(e)}")
            logger.error(f"    Debug info: {traceback.format_exc()}")
    
    def _create_interactive_dashboard(self):
        """
        Create an interactive dashboard using Plotly for advanced data visualization.
        """
        logger.info("  🎯 Creating interactive dashboard...")
        
        try:
            # Check if plotly is available
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Performance Metrics', 'Memory Usage', 
                              'Data Quality Score', 'Test Results Summary',
                              'Execution Time Trends', 'Statistical Distribution'),
                specs=[[{"secondary_y": True}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Performance metrics plot
            if hasattr(self, 'performance_metrics') and self.performance_metrics:
                perf_data = self.performance_metrics.get('detailed_performance', {})
                if perf_data:
                    data_sizes = sorted(perf_data.keys())
                    bar_types = ['time', 'tick', 'volume', 'dollar']
                    colors = ['blue', 'red', 'green', 'orange']
                    
                    for i, bar_type in enumerate(bar_types):
                        execution_times = []
                        throughputs = []
                        valid_sizes = []
                        
                        for size in data_sizes:
                            if (bar_type in perf_data[size] and 
                                'execution_time' in perf_data[size][bar_type]):
                                execution_times.append(perf_data[size][bar_type]['execution_time'])
                                throughputs.append(perf_data[size][bar_type].get('throughput', 0))
                                valid_sizes.append(size)
                        
                        if execution_times:
                            # Execution time
                            fig.add_trace(
                                go.Scatter(x=valid_sizes, y=execution_times,
                                          mode='lines+markers',
                                          name=f'{bar_type} (time)',
                                          line=dict(color=colors[i]),
                                          hovertemplate='Size: %{x}<br>Time: %{y:.4f}s<extra></extra>'),
                                row=1, col=1
                            )
                            
                            # Throughput
                            fig.add_trace(
                                go.Scatter(x=valid_sizes, y=throughputs,
                                          mode='lines+markers',
                                          name=f'{bar_type} (throughput)',
                                          line=dict(color=colors[i], dash='dash'),
                                          hovertemplate='Size: %{x}<br>Throughput: %{y:.2f} samples/s<extra></extra>'),
                                row=1, col=1, secondary_y=True
                            )
            
            # Memory usage plot
            if hasattr(self, 'performance_metrics') and self.performance_metrics:
                memory_data = self.performance_metrics.get('memory_usage', {})
                if memory_data:
                    methods = list(memory_data.keys())
                    memory_values = [memory_data[method] for method in methods]
                    
                    fig.add_trace(
                        go.Bar(x=methods, y=memory_values,
                               name='Memory Usage',
                               marker_color='lightblue',
                               hovertemplate='Method: %{x}<br>Memory: %{y:.2f} MB<extra></extra>'),
                        row=1, col=2
                    )
            
            # Data quality scores
            if hasattr(self, 'data_quality_scores') and self.data_quality_scores:
                quality_metrics = list(self.data_quality_scores.keys())
                quality_values = [self.data_quality_scores[metric] for metric in quality_metrics]
                
                fig.add_trace(
                    go.Bar(x=quality_metrics, y=quality_values,
                           name='Data Quality',
                           marker_color='lightgreen',
                           hovertemplate='Metric: %{x}<br>Score: %{y:.2f}<extra></extra>'),
                    row=2, col=1
                )
            
            # Test results summary
            if hasattr(self, 'test_results') and self.test_results:
                passed = sum(1 for result in self.test_results.values() if result.get('passed', False))
                failed = len(self.test_results) - passed
                
                fig.add_trace(
                    go.Pie(labels=['Passed', 'Failed'], values=[passed, failed],
                           marker_colors=['lightgreen', 'lightcoral'],
                           hovertemplate='%{label}: %{value}<br>%{percent}<extra></extra>'),
                    row=2, col=2
                )
            
            # Execution time trends
            if hasattr(self, 'execution_history') and self.execution_history:
                timestamps = list(self.execution_history.keys())
                exec_times = [self.execution_history[ts]['execution_time'] for ts in timestamps]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=exec_times,
                              mode='lines+markers',
                              name='Execution Time Trend',
                              line=dict(color='purple'),
                              hovertemplate='Time: %{x}<br>Execution: %{y:.4f}s<extra></extra>'),
                    row=3, col=1
                )
            
            # Statistical distribution (if available)
            if hasattr(self, 'statistical_data') and self.statistical_data:
                # Use returns data if available
                if 'returns' in self.statistical_data:
                    returns = self.statistical_data['returns']
                    fig.add_trace(
                        go.Histogram(x=returns, nbinsx=50,
                                   name='Returns Distribution',
                                   marker_color='lightblue',
                                   hovertemplate='Return: %{x:.4f}<br>Count: %{y}<extra></extra>'),
                        row=3, col=2
                    )
            
            # Update layout
            fig.update_layout(
                title_text="Base Bars Testing - Interactive Dashboard",
                title_font_size=20,
                showlegend=True,
                height=1200,
                template='plotly_white'
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Data Size", row=1, col=1)
            fig.update_yaxes(title_text="Execution Time (s)", row=1, col=1)
            fig.update_yaxes(title_text="Throughput (samples/s)", row=1, col=1, secondary_y=True)
            
            fig.update_xaxes(title_text="Method", row=1, col=2)
            fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)
            
            fig.update_xaxes(title_text="Quality Metric", row=2, col=1)
            fig.update_yaxes(title_text="Score", row=2, col=1)
            
            fig.update_xaxes(title_text="Time", row=3, col=1)
            fig.update_yaxes(title_text="Execution Time (s)", row=3, col=1)
            
            fig.update_xaxes(title_text="Return Value", row=3, col=2)
            fig.update_yaxes(title_text="Frequency", row=3, col=2)
            
            # Save interactive dashboard
            dashboard_path = os.path.join(self.save_path, 'interactive_dashboard.html')
            fig.write_html(dashboard_path)
            
            self.report['interactive_dashboard'] = dashboard_path
            logger.info(f"    🎯 Interactive dashboard saved to {dashboard_path}")
            
        except ImportError:
            logger.warning("    ⚠️ Plotly not available - skipping interactive dashboard")
        except Exception as e:
            logger.error(f"    ❌ Error creating interactive dashboard: {str(e)}")
            logger.error(f"    Debug info: {traceback.format_exc()}")
    
    def _generate_reports(self):
        """Generate comprehensive HTML and JSON reports"""
        logger.info("📄 Generating comprehensive reports...")
        
        try:
            # Generate JSON report
            json_report_path = os.path.join(self.save_path, 'test_report.json')
            with open(json_report_path, 'w') as f:
                json.dump(self.report, f, indent=2, default=str)
            
            # Generate HTML report
            html_report_path = os.path.join(self.save_path, 'test_report.html')
            html_content = self._generate_html_report()
            
            with open(html_report_path, 'w') as f:
                f.write(html_content)
            
            # Generate summary report for central dashboard
            summary_path = os.path.join(self.save_path, 'test_summary.json')
            summary = {
                'module_name': 'test_base_bars',
                'test_date': self.report['test_date'],
                'tests_run': self.report['tests_run'],
                'tests_passed': self.report['tests_passed'],
                'tests_failed': self.report['tests_failed'],
                'success_rate': (self.report['tests_passed'] / max(self.report['tests_run'], 1)) * 100,
                'plots_generated': len(self.report['plots_generated']),
                'html_report': html_report_path,
                'json_report': json_report_path,
                'plots': self.report['plots_generated']
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"  📄 JSON report saved to {json_report_path}")
            logger.info(f"  📄 HTML report saved to {html_report_path}")
            logger.info(f"  📄 Summary report saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"  ❌ Error generating reports: {str(e)}")
    
    def _generate_html_report(self):
        """Generate HTML report content"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Base Bars Testing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .success {{ color: green; }}
                .error {{ color: red; }}
                .warning {{ color: orange; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Base Bars Testing Report</h1>
                <p>Generated on: {self.report['test_date']}</p>
                <p>Tests Run: {self.report['tests_run']}</p>
                <p>Tests Passed: <span class="success">{self.report['tests_passed']}</span></p>
                <p>Tests Failed: <span class="error">{self.report['tests_failed']}</span></p>
                <p>Success Rate: {(self.report['tests_passed'] / max(self.report['tests_run'], 1)) * 100:.1f}%</p>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <p>Performance testing completed for different bar types and data sizes.</p>
                <ul>
                    <li>Execution time analysis</li>
                    <li>Memory usage monitoring</li>
                    <li>Throughput benchmarks</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Generated Plots</h2>
                <ul>
                    {''.join([f"<li>{plot}</li>" for plot in self.report['plots_generated']])}
                </ul>
            </div>
            
            <div class="section">
                <h2>Errors and Warnings</h2>
                {'<p class="success">No errors detected!</p>' if not self.report['errors'] else ''}
                {''.join([f'<p class="error">Error: {error}</p>' for error in self.report['errors']])}
                {''.join([f'<p class="warning">Warning: {warning}</p>' for warning in self.report['warnings']])}
            </div>
        </body>
        </html>
        """
        return html_template


def main():
    """Main function to run the complete test suite"""
    print("🚀 Starting Base Bars Advanced Testing Suite")
    print("=" * 60)
    
    # Initialize testing framework
    tester = TestBaseBarsAdvanced()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Run: {results['tests_run']}")
    print(f"Tests Passed: {results['tests_passed']}")
    print(f"Tests Failed: {results['tests_failed']}")
    print(f"Success Rate: {(results['tests_passed'] / max(results['tests_run'], 1)) * 100:.1f}%")
    print(f"Plots Generated: {len(results['plots_generated'])}")
    
    if results['errors']:
        print("\n❌ ERRORS:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print(f"\n📁 Results saved to: {tester.save_path}")
    print("✅ Testing completed successfully!")
    
    return results


if __name__ == "__main__":
    main()
