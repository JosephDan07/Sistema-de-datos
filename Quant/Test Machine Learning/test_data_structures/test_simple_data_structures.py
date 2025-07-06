#!/usr/bin/env python3
"""
Test simplificado para data_structures
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
    from data_structures import (
        get_tick_bars, get_dollar_bars, get_volume_bars,
        get_time_bars, get_tick_imbalance_bars, 
        get_volume_imbalance_bars, get_dollar_imbalance_bars
    )
    DATA_STRUCTURES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import data_structures: {e}")
    DATA_STRUCTURES_AVAILABLE = False

try:
    from test_config_manager import ConfigurationManager
    CONFIG_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import ConfigurationManager: {e}")
    CONFIG_MANAGER_AVAILABLE = False

warnings.filterwarnings('ignore')

class TestDataStructuresSimple(unittest.TestCase):
    """Test simplificado para data_structures"""
    
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
            cls.config = cls.config_manager.get_config('data_structures')
        else:
            cls.config = {'batch_size': 1000, 'test_size': 100}
        
        # Crear datos de prueba
        cls.sample_data = cls._create_sample_data()
        
        print("🧪 TestDataStructuresSimple - Configuración inicial completada")
    
    @classmethod
    def _create_sample_data(cls) -> pd.DataFrame:
        """Crear datos de muestra para pruebas"""
        print("📊 Creando datos de muestra...")
        
        # Datos más realistas
        np.random.seed(42)
        n_samples = 1000
        
        # Generar precios con random walk
        price_changes = np.random.normal(0, 0.001, n_samples)
        prices = 100 + np.cumsum(price_changes)
        
        # Generar volumen con distribución lognormal
        volumes = np.random.lognormal(mean=10, sigma=1, size=n_samples)
        
        # Generar timestamps
        start_time = datetime.now() - timedelta(hours=n_samples)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
        
        data = pd.DataFrame({
            'date_time': timestamps,
            'price': prices,
            'volume': volumes.astype(int)
        })
        
        # NO usar set_index para mantener date_time como columna
        
        print(f"✅ Datos de muestra creados: {len(data)} filas")
        return data
    
    def test_time_bars(self):
        """Test para time bars"""
        test_name = "test_time_bars"
        start_time = datetime.now()
        
        try:
            if not DATA_STRUCTURES_AVAILABLE:
                raise ImportError("Data structures not available")
            
            # Test básico de time bars
            time_bars = get_time_bars(
                file_path_or_df=self.sample_data,
                resolution='MIN',
                num_units=5  # 5 minutos
            )
            
            self.assertIsInstance(time_bars, pd.DataFrame)
            self.assertGreater(len(time_bars), 0)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': True,
                'execution_time': execution_time,
                'bars_created': len(time_bars)
            }
            
            print(f"✅ {test_name} passed - {len(time_bars)} bars created")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"❌ {test_name} failed: {e}")
            raise
    
    def test_tick_bars(self):
        """Test para tick bars"""
        test_name = "test_tick_bars"
        start_time = datetime.now()
        
        try:
            if not DATA_STRUCTURES_AVAILABLE:
                raise ImportError("Data structures not available")
            
            # Test básico de tick bars
            tick_bars = get_tick_bars(
                file_path_or_df=self.sample_data,
                threshold=100
            )
            
            self.assertIsInstance(tick_bars, pd.DataFrame)
            self.assertGreater(len(tick_bars), 0)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': True,
                'execution_time': execution_time,
                'bars_created': len(tick_bars)
            }
            
            print(f"✅ {test_name} passed - {len(tick_bars)} bars created")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"❌ {test_name} failed: {e}")
            raise
    
    def test_volume_bars(self):
        """Test para volume bars"""
        test_name = "test_volume_bars"
        start_time = datetime.now()
        
        try:
            if not DATA_STRUCTURES_AVAILABLE:
                raise ImportError("Data structures not available")
            
            # Calcular threshold dinámico
            total_volume = self.sample_data['volume'].sum()
            threshold = int(total_volume / 10)  # 10 bars aproximadamente
            
            volume_bars = get_volume_bars(
                file_path_or_df=self.sample_data,
                threshold=threshold
            )
            
            self.assertIsInstance(volume_bars, pd.DataFrame)
            self.assertGreater(len(volume_bars), 0)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': True,
                'execution_time': execution_time,
                'bars_created': len(volume_bars)
            }
            
            print(f"✅ {test_name} passed - {len(volume_bars)} bars created")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"❌ {test_name} failed: {e}")
            raise
    
    def test_dollar_bars(self):
        """Test para dollar bars"""
        test_name = "test_dollar_bars"
        start_time = datetime.now()
        
        try:
            if not DATA_STRUCTURES_AVAILABLE:
                raise ImportError("Data structures not available")
            
            # Calcular threshold dinámico
            total_dollar_volume = (self.sample_data['price'] * self.sample_data['volume']).sum()
            threshold = int(total_dollar_volume / 10)  # 10 bars aproximadamente
            
            dollar_bars = get_dollar_bars(
                file_path_or_df=self.sample_data,
                threshold=threshold
            )
            
            self.assertIsInstance(dollar_bars, pd.DataFrame)
            self.assertGreater(len(dollar_bars), 0)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': True,
                'execution_time': execution_time,
                'bars_created': len(dollar_bars)
            }
            
            print(f"✅ {test_name} passed - {len(dollar_bars)} bars created")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"❌ {test_name} failed: {e}")
            raise
    
    def test_basic_functionality(self):
        """Test básico de funcionalidad"""
        test_name = "test_basic_functionality"
        start_time = datetime.now()
        
        try:
            # Test básico de datos
            self.assertIsInstance(self.sample_data, pd.DataFrame)
            self.assertGreater(len(self.sample_data), 0)
            self.assertIn('price', self.sample_data.columns)
            self.assertIn('volume', self.sample_data.columns)
            
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
    
    @classmethod
    def tearDownClass(cls):
        """Limpieza y guardado de resultados"""
        # Calcular estadísticas finales
        cls.test_results['total_tests'] = len(cls.test_results['test_details'])
        cls.test_results['tests_passed'] = sum(1 for result in cls.test_results['test_details'].values() if result['passed'])
        cls.test_results['tests_failed'] = cls.test_results['total_tests'] - cls.test_results['tests_passed']
        cls.test_results['execution_time'] = sum(result['execution_time'] for result in cls.test_results['test_details'].values())
        
        # Guardar resultados
        results_dir = Path('/workspaces/Sistema-de-datos/Quant/Results Machine Learning/results_data_structures')
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(cls.test_results, f, indent=2)
        
        print(f"\n📊 Resultados finales guardados en: {results_file}")
        print(f"✅ Tests pasados: {cls.test_results['tests_passed']}/{cls.test_results['total_tests']}")
        print(f"⏱️  Tiempo total: {cls.test_results['execution_time']:.2f}s")

if __name__ == '__main__':
    print("🧪 Iniciando tests de data_structures...")
    unittest.main(verbosity=2)
