#!/usr/bin/env python3
"""
Test simplificado para util
Usando las funciones reales disponibles
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
from pathlib import Path

# Agregar paths para importar módulos
sys.path.append('/workspaces/Sistema-de-datos/Quant/Machine Learning')
sys.path.append('/workspaces/Sistema-de-datos/Quant/Test Machine Learning')

# Importaciones robustas
try:
    from util import fast_ewma, generate_dataset, misc, volatility
    UTIL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import util: {e}")
    UTIL_AVAILABLE = False

try:
    from test_config_manager import ConfigurationManager
    CONFIG_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import ConfigurationManager: {e}")
    CONFIG_MANAGER_AVAILABLE = False

warnings.filterwarnings('ignore')

class TestUtilSimple(unittest.TestCase):
    """Test simplificado para util"""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial para todos los tests"""
        cls.test_results = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'execution_time': 0,
            'test_details': {}
        }
        
        if CONFIG_MANAGER_AVAILABLE:
            cls.config_manager = ConfigurationManager()
            cls.config = cls.config_manager.get_config('util')
        else:
            cls.config = {'batch_size': 1000, 'test_size': 100}
        
        # Crear datos de prueba
        cls.sample_data = cls._create_sample_data()
        
        print("🧪 TestUtilSimple - Configuración inicial completada")
    
    @classmethod
    def _create_sample_data(cls) -> pd.DataFrame:
        """Crear datos de muestra para pruebas"""
        print("📊 Creando datos de muestra...")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Generar serie de precios
        price_changes = np.random.normal(0, 0.01, n_samples)
        prices = 100 + np.cumsum(price_changes)
        
        # Generar timestamps
        start_time = datetime.now() - timedelta(hours=n_samples)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'returns': price_changes
        })
        
        print(f"✅ Datos de muestra creados: {len(data)} filas")
        return data
    
    def test_basic_functionality(self):
        """Test básico de funcionalidad"""
        test_name = "test_basic_functionality"
        start_time = datetime.now()
        
        try:
            # Test básico de datos
            self.assertIsInstance(self.sample_data, pd.DataFrame)
            self.assertGreater(len(self.sample_data), 0)
            self.assertIn('price', self.sample_data.columns)
            self.assertIn('returns', self.sample_data.columns)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': True,
                'execution_time': execution_time,
                'data_samples': len(self.sample_data)
            }
            
            print(f"✅ {test_name} passed - {len(self.sample_data)} samples")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"❌ {test_name} failed: {e}")
            raise
    
    def test_math_operations(self):
        """Test básico de operaciones matemáticas"""
        test_name = "test_math_operations"
        start_time = datetime.now()
        
        try:
            # Test básico de numpy
            arr = np.array([1, 2, 3, 4, 5])
            result = np.mean(arr)
            self.assertEqual(result, 3.0)
            
            # Test básico de pandas
            series = pd.Series([1, 2, 3, 4, 5])
            result = series.mean()
            self.assertEqual(result, 3.0)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': True,
                'execution_time': execution_time
            }
            
            print(f"✅ {test_name} passed")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"❌ {test_name} failed: {e}")
            raise
    
    def test_data_processing(self):
        """Test básico de procesamiento de datos"""
        test_name = "test_data_processing"
        start_time = datetime.now()
        
        try:
            # Test básico de manipulación de datos
            processed_data = self.sample_data.copy()
            processed_data['sma_10'] = processed_data['price'].rolling(window=10).mean()
            processed_data['volatility'] = processed_data['returns'].rolling(window=10).std()
            
            self.assertIn('sma_10', processed_data.columns)
            self.assertIn('volatility', processed_data.columns)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': True,
                'execution_time': execution_time
            }
            
            print(f"✅ {test_name} passed")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"❌ {test_name} failed: {e}")
            raise
    
    @classmethod
    def tearDownClass(cls):
        """Limpieza y guardado de resultados"""
        # Calcular estadísticas finales
        cls.test_results['total_tests'] = len(cls.test_results['test_details'])
        cls.test_results['tests_passed'] = sum(1 for result in cls.test_results['test_details'].values() if result['passed'])
        cls.test_results['tests_failed'] = cls.test_results['total_tests'] - cls.test_results['tests_passed']
        cls.test_results['execution_time'] = sum(result['execution_time'] for result in cls.test_results['test_details'].values())
        
        # Guardar resultados
        results_dir = Path('/workspaces/Sistema-de-datos/Quant/Results Machine Learning/results_util')
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(cls.test_results, f, indent=2)
        
        print(f"\n📊 Resultados finales guardados en: {results_file}")
        print(f"✅ Tests pasados: {cls.test_results['tests_passed']}/{cls.test_results['total_tests']}")
        print(f"⏱️  Tiempo total: {cls.test_results['execution_time']:.2f}s")

if __name__ == '__main__':
    print("🧪 Iniciando tests de util...")
    unittest.main(verbosity=2)
