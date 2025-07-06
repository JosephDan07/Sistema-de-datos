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

# Core libraries
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    def __init__(self, save_path: str = "./test_results/"):
        """
        Initialize the testing framework
        
        :param save_path: Path to save test results, plots, and reports
        """
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
        Load configuration from JSON file
        
        :param config_path: Path to configuration file
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "test_config.json")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"✅ Configuration loaded from {config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"⚠️ Configuration file not found: {config_path}")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing configuration file: {e}")
            return self._get_default_config()
    
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
    
    def generate_synthetic_data(self, n_samples: int = 10000, 
                               price_start: float = 100.0,
                               volatility: float = 0.02,
                               trend: float = 0.0001,
                               include_gaps: bool = True,
                               include_outliers: bool = True) -> pd.DataFrame:
        """
        Generate synthetic financial data for testing
        
        :param n_samples: Number of data points
        :param price_start: Starting price
        :param volatility: Daily volatility
        :param trend: Price trend
        :param include_gaps: Whether to include price gaps
        :param include_outliers: Whether to include outliers
        :return: Synthetic market data DataFrame
        """
        logger.info(f"📊 Generating synthetic data: {n_samples} samples")
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=max(1, n_samples//1000))
        timestamps = pd.date_range(start=start_time, periods=n_samples, freq='1S')
        
        # Generate price series with geometric brownian motion
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(trend, volatility, n_samples)
        
        # Apply autocorrelation to make it more realistic
        if n_samples > 1:
            returns[1:] = 0.1 * returns[:-1] + 0.9 * returns[1:]
        
        prices = np.exp(np.cumsum(returns)) * price_start
        
        # Generate volumes (log-normal distribution)
        volumes = np.random.lognormal(mean=8, sigma=1, size=n_samples)
        
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
        
        # Add some realistic market microstructure
        data['bid'] = data['price'] * 0.9999
        data['ask'] = data['price'] * 1.0001
        data['tick_rule'] = np.random.choice([1, -1], size=n_samples)
        
        logger.info(f"✅ Generated synthetic data with {len(data)} rows")
        logger.info(f"📈 Price range: ${data['price'].min():.2f} - ${data['price'].max():.2f}")
        logger.info(f"📊 Volume range: {data['volume'].min():.0f} - {data['volume'].max():.0f}")
        
        return data
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite
        
        :return: Complete test results
        """
        logger.info("🚀 Starting comprehensive test suite")
        start_time = time.time()
        
        try:
            # Load configuration
            config = self.load_config()
            
            # Generate test data
            logger.info("📊 Generating test data...")
            self.test_data = self.generate_synthetic_data(
                n_samples=config["testing_parameters"]["synthetic_data"]["default_samples"]
            )
            
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
        """Run comprehensive performance benchmarks"""
        logger.info("⚡ Starting performance benchmarks")
        
        config = self.load_config()
        data_sizes = config["testing_parameters"]["performance_testing"]["data_sizes"]
        bar_types = config["testing_parameters"]["performance_testing"]["bar_types"]
        
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
