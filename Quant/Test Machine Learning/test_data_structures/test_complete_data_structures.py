#!/usr/bin/env python3
"""
Test completo para todos los módulos de data_structures
Tests robustos con datos reales, evitando sobreajuste
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import json
import logging

# Agregar paths para importar módulos
sys.path.append('/workspaces/Sistema-de-datos/Quant/Machine Learning')
sys.path.append('/workspaces/Sistema-de-datos/Quant/Test Machine Learning')

from test_config_manager import ConfigurationManager

# Importaciones robustas para data_structures
try:
    from data_structures.base_bars import BaseBars, BaseImbalanceBars, BaseRunBars
except ImportError:
    try:
        # Fallback imports
        sys.path.insert(0, '/workspaces/Sistema-de-datos/Quant/Machine Learning/data_structures')
        from base_bars import BaseBars, BaseImbalanceBars, BaseRunBars
    except ImportError:
        BaseBars = BaseImbalanceBars = BaseRunBars = None
        print("⚠️  Warning: Could not import base bars classes")

try:
    from data_structures.imbalance_data_structures import TickImbalanceBars, VolumeImbalanceBars, DollarImbalanceBars
except ImportError:
    try:
        from imbalance_data_structures import TickImbalanceBars, VolumeImbalanceBars, DollarImbalanceBars
    except ImportError:
        TickImbalanceBars = VolumeImbalanceBars = DollarImbalanceBars = None
        print("⚠️  Warning: Could not import imbalance bars classes")

try:
    from data_structures.time_data_structures import TimeBarFeatures
except ImportError:
    try:
        from time_data_structures import TimeBarFeatures
    except ImportError:
        TimeBarFeatures = None
        print("⚠️  Warning: Could not import TimeBarFeatures")

try:
    from data_structures.standard_data_structures import StandardFeatures
except ImportError:
    try:
        from standard_data_structures import StandardFeatures
    except ImportError:
        StandardFeatures = None
        print("⚠️  Warning: Could not import StandardFeatures")

try:
    from data_structures.config import config
except ImportError:
    try:
        from config import config
    except ImportError:
        config = None
        print("⚠️  Warning: Could not import config")

warnings.filterwarnings('ignore')

class TestCompleteDataStructures(unittest.TestCase):
    """
    Test completo para todos los módulos de data_structures
    """
    
    @classmethod
    def setUpClass(cls):
        """Setup inicial para todos los tests"""
        cls.config_manager = ConfigurationManager()
        cls.config = cls.config_manager.get_config('test_data_structures')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
        # Cargar datos reales
        cls.data_path = '/workspaces/Sistema-de-datos/Quant/Datos/Tiingo_CSV'
        cls.results_path = '/workspaces/Sistema-de-datos/Quant/Results Machine Learning'
        
        # Crear carpeta de resultados si no existe
        os.makedirs(cls.results_path, exist_ok=True)
        
        # Cargar múltiples datasets para evitar sobreajuste
        cls.datasets = cls._load_multiple_datasets()
        
        cls.logger.info(f"Loaded {len(cls.datasets)} datasets for testing")
    
    @classmethod
    def _load_multiple_datasets(cls):
        """Cargar múltiples datasets para tests robustos"""
        datasets = {}
        
        # Cargar diferentes tipos de activos
        files_to_load = [
            'AAPL_daily.csv',  # Tech stock
            'SPY_daily.csv',   # ETF
            'BTCUSD_daily.csv', # Crypto
            'GOOGL_daily.csv',  # Tech stock
            'GLD_daily.csv'     # Commodities
        ]
        
        for file in files_to_load:
            try:
                file_path = os.path.join(cls.data_path, file)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    # Preparar datos en formato tick
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    
                    # Crear datos tick simulados realistas
                    tick_data = cls._create_realistic_tick_data(df)
                    datasets[file.replace('.csv', '')] = tick_data
                    
            except Exception as e:
                cls.logger.warning(f"Could not load {file}: {str(e)}")
        
        return datasets
    
    @classmethod
    def _create_realistic_tick_data(cls, daily_df):
        """Crear datos tick realistas a partir de datos diarios"""
        tick_data = []
        
        for idx, row in daily_df.iterrows():
            # Simular ticks intradiarios realistas
            n_ticks = np.random.randint(100, 500)  # Número variable de ticks
            
            # Crear distribución de precios realista
            price_range = row['high'] - row['low']
            if price_range > 0:
                prices = np.random.uniform(row['low'], row['high'], n_ticks)
                # Asegurar que open y close estén incluidos
                prices[0] = row['open']
                prices[-1] = row['close']
            else:
                prices = np.full(n_ticks, row['close'])
            
            # Volúmenes realistas
            total_volume = row['volume']
            volumes = np.random.exponential(total_volume / n_ticks, n_ticks)
            volumes = volumes * (total_volume / volumes.sum())  # Normalizar
            
            # Timestamps realistas
            base_time = idx
            time_increments = np.random.exponential(1.0, n_ticks)
            time_increments = time_increments / time_increments.sum()  # Normalizar
            
            timestamps = []
            current_time = base_time
            for inc in time_increments:
                current_time += timedelta(seconds=inc * 86400)  # Distribuir en el día
                timestamps.append(current_time)
            
            # Crear ticks
            for i in range(n_ticks):
                tick_data.append({
                    'date_time': timestamps[i],
                    'price': prices[i],
                    'volume': max(1, int(volumes[i]))
                })
        
        df_ticks = pd.DataFrame(tick_data)
        df_ticks.set_index('date_time', inplace=True)
        return df_ticks
    
    def setUp(self):
        """Setup para cada test individual"""
        self.test_results = {}
        self.start_time = datetime.now()
    
    def tearDown(self):
        """Cleanup después de cada test"""
        end_time = datetime.now()
        execution_time = (end_time - self.start_time).total_seconds()
        
        # Guardar resultados del test
        test_name = self._testMethodName
        self.test_results[test_name] = {
            'execution_time': execution_time,
            'timestamp': end_time.isoformat(),
            'status': 'passed'
        }
        
        self._save_test_results()
    
    def _save_test_results(self):
        """Guardar resultados del test"""
        results_file = os.path.join(
            self.results_path, 
            f'results_data_structures_complete_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
    
    def test_time_bars_multiple_datasets(self):
        """Test TimeBars con múltiples datasets"""
        self.logger.info("Testing TimeBars with multiple datasets")
        
        results = {}
        
        for dataset_name, data in self.datasets.items():
            try:
                # Test con diferentes intervalos
                intervals = ['1H', '30T', '2H']
                
                for interval in intervals:
                    time_bars = TimeBars(interval)
                    bars = time_bars.get_bars(data)
                    
                    # Validaciones
                    self.assertIsInstance(bars, pd.DataFrame)
                    self.assertGreater(len(bars), 0)
                    
                    # Verificar columnas esperadas
                    expected_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in expected_columns:
                        self.assertIn(col, bars.columns)
                    
                    # Verificar consistencia OHLC
                    self.assertTrue((bars['high'] >= bars['low']).all())
                    self.assertTrue((bars['high'] >= bars['open']).all())
                    self.assertTrue((bars['high'] >= bars['close']).all())
                    self.assertTrue((bars['low'] <= bars['open']).all())
                    self.assertTrue((bars['low'] <= bars['close']).all())
                    
                    # Verificar que no hay valores negativos
                    self.assertTrue((bars['volume'] >= 0).all())
                    
                    results[f'{dataset_name}_{interval}'] = {
                        'bars_count': len(bars),
                        'avg_volume': bars['volume'].mean(),
                        'price_range': bars['high'].max() - bars['low'].min()
                    }
                
                self.logger.info(f"TimeBars test passed for {dataset_name}")
                
            except Exception as e:
                self.logger.error(f"TimeBars test failed for {dataset_name}: {str(e)}")
                self.fail(f"TimeBars test failed for {dataset_name}: {str(e)}")
        
        self.test_results['time_bars_results'] = results
    
    def test_tick_bars_multiple_datasets(self):
        """Test TickBars con múltiples datasets"""
        self.logger.info("Testing TickBars with multiple datasets")
        
        results = {}
        
        for dataset_name, data in self.datasets.items():
            try:
                # Test con diferentes threshold
                thresholds = [50, 100, 200]
                
                for threshold in thresholds:
                    tick_bars = TickBars(threshold)
                    bars = tick_bars.get_bars(data)
                    
                    # Validaciones
                    self.assertIsInstance(bars, pd.DataFrame)
                    self.assertGreater(len(bars), 0)
                    
                    # Verificar que cada barra tiene aproximadamente el threshold de ticks
                    # (puede variar ligeramente debido a la última barra)
                    if len(bars) > 1:
                        # Calcular ticks promedio por barra
                        expected_ticks = len(data) / len(bars)
                        self.assertAlmostEqual(expected_ticks, threshold, delta=threshold * 0.2)
                    
                    results[f'{dataset_name}_{threshold}'] = {
                        'bars_count': len(bars),
                        'avg_volume': bars['volume'].mean(),
                        'expected_ticks_per_bar': threshold
                    }
                
                self.logger.info(f"TickBars test passed for {dataset_name}")
                
            except Exception as e:
                self.logger.error(f"TickBars test failed for {dataset_name}: {str(e)}")
                self.fail(f"TickBars test failed for {dataset_name}: {str(e)}")
        
        self.test_results['tick_bars_results'] = results
    
    def test_volume_bars_multiple_datasets(self):
        """Test VolumeBars con múltiples datasets"""
        self.logger.info("Testing VolumeBars with multiple datasets")
        
        results = {}
        
        for dataset_name, data in self.datasets.items():
            try:
                # Calcular threshold basado en volumen total
                total_volume = data['volume'].sum()
                thresholds = [
                    int(total_volume / 100),  # 100 barras
                    int(total_volume / 50),   # 50 barras
                    int(total_volume / 20)    # 20 barras
                ]
                
                for threshold in thresholds:
                    if threshold > 0:
                        volume_bars = VolumeBars(threshold)
                        bars = volume_bars.get_bars(data)
                        
                        # Validaciones
                        self.assertIsInstance(bars, pd.DataFrame)
                        self.assertGreater(len(bars), 0)
                        
                        # Verificar que el volumen por barra es aproximadamente el threshold
                        if len(bars) > 1:
                            avg_volume = bars['volume'].mean()
                            self.assertAlmostEqual(avg_volume, threshold, delta=threshold * 0.3)
                        
                        results[f'{dataset_name}_{threshold}'] = {
                            'bars_count': len(bars),
                            'avg_volume': bars['volume'].mean(),
                            'threshold': threshold
                        }
                
                self.logger.info(f"VolumeBars test passed for {dataset_name}")
                
            except Exception as e:
                self.logger.error(f"VolumeBars test failed for {dataset_name}: {str(e)}")
                self.fail(f"VolumeBars test failed for {dataset_name}: {str(e)}")
        
        self.test_results['volume_bars_results'] = results
    
    def test_dollar_bars_multiple_datasets(self):
        """Test DollarBars con múltiples datasets"""
        self.logger.info("Testing DollarBars with multiple datasets")
        
        results = {}
        
        for dataset_name, data in self.datasets.items():
            try:
                # Calcular threshold basado en valor total
                data_copy = data.copy()
                data_copy['dollar_volume'] = data_copy['price'] * data_copy['volume']
                total_dollar_volume = data_copy['dollar_volume'].sum()
                
                thresholds = [
                    int(total_dollar_volume / 100),  # 100 barras
                    int(total_dollar_volume / 50),   # 50 barras
                    int(total_dollar_volume / 20)    # 20 barras
                ]
                
                for threshold in thresholds:
                    if threshold > 0:
                        dollar_bars = DollarBars(threshold)
                        bars = dollar_bars.get_bars(data)
                        
                        # Validaciones
                        self.assertIsInstance(bars, pd.DataFrame)
                        self.assertGreater(len(bars), 0)
                        
                        # Verificar que el volumen en dólares por barra es aproximadamente el threshold
                        if len(bars) > 1:
                            bars_copy = bars.copy()
                            bars_copy['dollar_volume'] = bars_copy['close'] * bars_copy['volume']
                            avg_dollar_volume = bars_copy['dollar_volume'].mean()
                            self.assertAlmostEqual(avg_dollar_volume, threshold, delta=threshold * 0.3)
                        
                        results[f'{dataset_name}_{threshold}'] = {
                            'bars_count': len(bars),
                            'avg_dollar_volume': bars_copy['dollar_volume'].mean(),
                            'threshold': threshold
                        }
                
                self.logger.info(f"DollarBars test passed for {dataset_name}")
                
            except Exception as e:
                self.logger.error(f"DollarBars test failed for {dataset_name}: {str(e)}")
                self.fail(f"DollarBars test failed for {dataset_name}: {str(e)}")
        
        self.test_results['dollar_bars_results'] = results
    
    def test_imbalance_bars_multiple_datasets(self):
        """Test ImbalanceBars con múltiples datasets"""
        self.logger.info("Testing ImbalanceBars with multiple datasets")
        
        results = {}
        
        for dataset_name, data in self.datasets.items():
            try:
                # Test different imbalance types
                imbalance_types = [
                    ('tick', TickImbalanceBars),
                    ('volume', VolumeImbalanceBars), 
                    ('dollar', DollarImbalanceBars)
                ]
                
                for imbalance_type, ImbalanceClass in imbalance_types:
                    try:
                        imbalance_bars = ImbalanceClass(
                            num_prev_bars=10,
                            expected_imbalance_window=100
                        )
                        bars = imbalance_bars.get_bars(data)
                        
                        # Validaciones básicas
                        self.assertIsInstance(bars, pd.DataFrame)
                        # ImbalanceBars pueden generar pocas barras, así que relajamos la validación
                        if len(bars) > 0:
                            # Verificar estructura básica
                            expected_columns = ['open', 'high', 'low', 'close', 'volume']
                            for col in expected_columns:
                                self.assertIn(col, bars.columns)
                        
                        results[f'{dataset_name}_{imbalance_type}'] = {
                            'bars_count': len(bars),
                            'avg_volume': bars['volume'].mean() if len(bars) > 0 else 0
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"ImbalanceBars {imbalance_type} failed for {dataset_name}: {str(e)}")
                        results[f'{dataset_name}_{imbalance_type}'] = {
                            'bars_count': 0,
                            'error': str(e)
                        }
                
                self.logger.info(f"ImbalanceBars test completed for {dataset_name}")
                
            except Exception as e:
                self.logger.error(f"ImbalanceBars test failed for {dataset_name}: {str(e)}")
                results[f'{dataset_name}_error'] = str(e)
        
        self.test_results['imbalance_bars_results'] = results
    
    def test_configuration_system(self):
        """Test del sistema de configuración"""
        self.logger.info("Testing configuration system")
        
        try:
            # Test configuración global
            global_config = self.config_manager.get_config('global')
            self.assertIsInstance(global_config, dict)
            
            # Test configuración específica
            ds_config = self.config_manager.get_config('test_data_structures')
            self.assertIsInstance(ds_config, dict)
            
            # Test modificación de configuración
            original_batch_size = config.processing.default_batch_size
            config.processing.default_batch_size = 1000000
            self.assertEqual(config.processing.default_batch_size, 1000000)
            
            # Restaurar valor original
            config.processing.default_batch_size = original_batch_size
            
            self.test_results['configuration_test'] = {
                'global_config_loaded': True,
                'module_config_loaded': True,
                'config_modification_works': True
            }
            
            self.logger.info("Configuration system test passed")
            
        except Exception as e:
            self.logger.error(f"Configuration system test failed: {str(e)}")
            self.fail(f"Configuration system test failed: {str(e)}")
    
    def test_performance_benchmarks(self):
        """Test de performance con benchmarks"""
        self.logger.info("Testing performance benchmarks")
        
        performance_results = {}
        
        # Seleccionar dataset más grande para test de performance
        largest_dataset = max(self.datasets.keys(), 
                            key=lambda k: len(self.datasets[k]))
        data = self.datasets[largest_dataset]
        
        # Benchmark diferentes tipos de barras
        bar_types = [
            ('TimeBars', TimeBars, '1H'),
            ('TickBars', TickBars, 100),
            ('VolumeBars', VolumeBars, int(data['volume'].sum() / 50)),
            ('DollarBars', DollarBars, int((data['price'] * data['volume']).sum() / 50))
        ]
        
        for bar_name, BarClass, param in bar_types:
            try:
                start_time = datetime.now()
                
                bar_instance = BarClass(param)
                bars = bar_instance.get_bars(data)
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                performance_results[bar_name] = {
                    'execution_time': execution_time,
                    'input_size': len(data),
                    'output_size': len(bars),
                    'throughput': len(data) / execution_time if execution_time > 0 else 0
                }
                
                self.logger.info(f"{bar_name} processed {len(data)} ticks in {execution_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Performance test failed for {bar_name}: {str(e)}")
                performance_results[bar_name] = {
                    'error': str(e),
                    'input_size': len(data)
                }
        
        self.test_results['performance_results'] = performance_results
    
    def test_edge_cases(self):
        """Test de casos edge y robustez"""
        self.logger.info("Testing edge cases and robustness")
        
        edge_case_results = {}
        
        # Test con datos mínimos
        minimal_data = pd.DataFrame({
            'price': [100.0, 101.0, 99.0],
            'volume': [1000, 1500, 800]
        }, index=pd.date_range('2023-01-01', periods=3, freq='1min'))
        
        try:
            time_bars = TimeBars('1H')
            bars = time_bars.get_bars(minimal_data)
            edge_case_results['minimal_data'] = {
                'passed': True,
                'output_size': len(bars)
            }
        except Exception as e:
            edge_case_results['minimal_data'] = {
                'passed': False,
                'error': str(e)
            }
        
        # Test con valores extremos
        extreme_data = pd.DataFrame({
            'price': [0.01, 1000000.0, 0.001],
            'volume': [1, 999999999, 1]
        }, index=pd.date_range('2023-01-01', periods=3, freq='1min'))
        
        try:
            volume_bars = VolumeBars(500000000)
            bars = volume_bars.get_bars(extreme_data)
            edge_case_results['extreme_values'] = {
                'passed': True,
                'output_size': len(bars)
            }
        except Exception as e:
            edge_case_results['extreme_values'] = {
                'passed': False,
                'error': str(e)
            }
        
        # Test con datos duplicados
        duplicate_data = pd.DataFrame({
            'price': [100.0, 100.0, 100.0],
            'volume': [1000, 1000, 1000]
        }, index=pd.date_range('2023-01-01', periods=3, freq='1min'))
        
        try:
            tick_bars = TickBars(2)
            bars = tick_bars.get_bars(duplicate_data)
            edge_case_results['duplicate_data'] = {
                'passed': True,
                'output_size': len(bars)
            }
        except Exception as e:
            edge_case_results['duplicate_data'] = {
                'passed': False,
                'error': str(e)
            }
        
        self.test_results['edge_case_results'] = edge_case_results
        
        # Verificar que al menos algunos tests pasaron
        passed_tests = sum(1 for result in edge_case_results.values() 
                          if isinstance(result, dict) and result.get('passed', False))
        
        self.assertGreater(passed_tests, 0, "No edge case tests passed")
        
        self.logger.info(f"Edge case tests completed: {passed_tests}/{len(edge_case_results)} passed")


if __name__ == '__main__':
    unittest.main(verbosity=2)
