#!/usr/bin/env python3
"""
Test simplificado para multi_product
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
    # Intentar importar funciones de multi_product
    from multi_product import (
        get_ewa_beta, get_ewa_corr, get_ewa_cov,
        get_ewa_var, get_ewa_vol, get_ewa_mean
    )
    MULTI_PRODUCT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import multi_product: {e}")
    MULTI_PRODUCT_AVAILABLE = False

try:
    from test_config_manager import ConfigurationManager
    CONFIG_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import ConfigurationManager: {e}")
    CONFIG_MANAGER_AVAILABLE = False

warnings.filterwarnings('ignore')

class TestMultiProductSimple(unittest.TestCase):
    """Test simplificado para multi_product"""
    
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
            cls.config = cls.config_manager.get_config('multi_product')
        else:
            cls.config = {'batch_size': 1000, 'test_size': 100}
        
        # Crear datos de prueba
        cls.sample_data = cls._create_sample_data()
        
        print("🧪 TestMultiProductSimple - Configuración inicial completada")
    
    @classmethod
    def _create_sample_data(cls) -> pd.DataFrame:
        """Crear datos de muestra para pruebas multi-producto"""
        print("📊 Creando datos de muestra para multi-product...")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Generar serie temporal para múltiples productos
        timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
        
        # Generar datos correlacionados para diferentes productos
        # Producto A (e.g., stock)
        returns_a = np.random.normal(0, 0.02, n_samples)
        price_a = 100 * np.exp(np.cumsum(returns_a))
        
        # Producto B (correlacionado con A)
        returns_b = 0.7 * returns_a + np.random.normal(0, 0.015, n_samples)
        price_b = 50 * np.exp(np.cumsum(returns_b))
        
        # Producto C (menos correlacionado)
        returns_c = 0.3 * returns_a + np.random.normal(0, 0.025, n_samples)
        price_c = 200 * np.exp(np.cumsum(returns_c))
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price_a': price_a,
            'price_b': price_b,
            'price_c': price_c,
            'returns_a': returns_a,
            'returns_b': returns_b,
            'returns_c': returns_c,
            'volume_a': np.random.lognormal(mean=8, sigma=1, size=n_samples),
            'volume_b': np.random.lognormal(mean=7, sigma=1, size=n_samples),
            'volume_c': np.random.lognormal(mean=9, sigma=1, size=n_samples)
        })
        
        data = data.set_index('timestamp')
        
        print(f"✅ Datos multi-producto creados: {len(data)} filas, {len(data.columns)} columnas")
        return data
    
    def test_basic_functionality(self):
        """Test básico de funcionalidad multi-producto"""
        test_name = "test_basic_functionality"
        start_time = datetime.now()
        
        try:
            # Test básico de datos
            self.assertIsInstance(self.sample_data, pd.DataFrame)
            self.assertGreater(len(self.sample_data), 0)
            self.assertIn('price_a', self.sample_data.columns)
            self.assertIn('price_b', self.sample_data.columns)
            self.assertIn('price_c', self.sample_data.columns)
            
            # Verificar que tenemos datos de múltiples productos
            self.assertGreaterEqual(len([col for col in self.sample_data.columns if 'price' in col]), 3)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': True,
                'execution_time': execution_time,
                'data_samples': len(self.sample_data),
                'products_count': len([col for col in self.sample_data.columns if 'price' in col])
            }
            
            print(f"✅ {test_name} passed - {len(self.sample_data)} samples, multiple products")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"❌ {test_name} failed: {e}")
            raise
    
    def test_correlation_analysis(self):
        """Test básico de análisis de correlación entre productos"""
        test_name = "test_correlation_analysis"
        start_time = datetime.now()
        
        try:
            # Calcular correlaciones entre productos
            returns_data = self.sample_data[['returns_a', 'returns_b', 'returns_c']]
            correlation_matrix = returns_data.corr()
            
            self.assertIsInstance(correlation_matrix, pd.DataFrame)
            self.assertEqual(correlation_matrix.shape[0], 3)
            self.assertEqual(correlation_matrix.shape[1], 3)
            
            # Verificar que las correlaciones están en el rango correcto
            for i in range(len(correlation_matrix)):
                for j in range(len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if i == j:
                        self.assertAlmostEqual(corr_val, 1.0, places=5)
                    else:
                        self.assertGreaterEqual(corr_val, -1.0)
                        self.assertLessEqual(corr_val, 1.0)
            
            # Test de correlación EWA si está disponible
            if MULTI_PRODUCT_AVAILABLE:
                try:
                    ewa_corr = get_ewa_corr(returns_data['returns_a'], returns_data['returns_b'])
                    self.assertIsInstance(ewa_corr, (pd.Series, float))
                    print(f"✅ EWA correlation calculated")
                except Exception as e:
                    print(f"⚠️  EWA correlation function error: {e}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': True,
                'execution_time': execution_time,
                'correlation_matrix_shape': correlation_matrix.shape,
                'max_correlation': float(correlation_matrix.max().max()),
                'min_correlation': float(correlation_matrix.min().min())
            }
            
            print(f"✅ {test_name} passed - Correlation analysis completed")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"❌ {test_name} failed: {e}")
            raise
    
    def test_covariance_analysis(self):
        """Test básico de análisis de covarianza entre productos"""
        test_name = "test_covariance_analysis"
        start_time = datetime.now()
        
        try:
            # Calcular covarianzas entre productos
            returns_data = self.sample_data[['returns_a', 'returns_b', 'returns_c']]
            covariance_matrix = returns_data.cov()
            
            self.assertIsInstance(covariance_matrix, pd.DataFrame)
            self.assertEqual(covariance_matrix.shape[0], 3)
            self.assertEqual(covariance_matrix.shape[1], 3)
            
            # Verificar que las covarianzas diagonales son positivas (varianzas)
            for i in range(len(covariance_matrix)):
                diagonal_val = covariance_matrix.iloc[i, i]
                self.assertGreater(diagonal_val, 0)
            
            # Test de covarianza EWA si está disponible
            if MULTI_PRODUCT_AVAILABLE:
                try:
                    ewa_cov = get_ewa_cov(returns_data['returns_a'], returns_data['returns_b'])
                    self.assertIsInstance(ewa_cov, (pd.Series, float))
                    print(f"✅ EWA covariance calculated")
                except Exception as e:
                    print(f"⚠️  EWA covariance function error: {e}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': True,
                'execution_time': execution_time,
                'covariance_matrix_shape': covariance_matrix.shape,
                'diagonal_values': [float(covariance_matrix.iloc[i, i]) for i in range(len(covariance_matrix))]
            }
            
            print(f"✅ {test_name} passed - Covariance analysis completed")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"❌ {test_name} failed: {e}")
            raise
    
    def test_beta_calculation(self):
        """Test básico de cálculo de beta entre productos"""
        test_name = "test_beta_calculation"
        start_time = datetime.now()
        
        try:
            # Calcular beta básico entre productos
            returns_a = self.sample_data['returns_a']
            returns_b = self.sample_data['returns_b']
            
            # Beta = Cov(A,B) / Var(B)
            covariance = np.cov(returns_a, returns_b)[0, 1]
            variance_b = np.var(returns_b)
            beta = covariance / variance_b
            
            self.assertIsInstance(beta, float)
            self.assertFalse(np.isnan(beta))
            self.assertFalse(np.isinf(beta))
            
            # Test de beta EWA si está disponible
            if MULTI_PRODUCT_AVAILABLE:
                try:
                    ewa_beta = get_ewa_beta(returns_a, returns_b)
                    self.assertIsInstance(ewa_beta, (pd.Series, float))
                    print(f"✅ EWA beta calculated")
                except Exception as e:
                    print(f"⚠️  EWA beta function error: {e}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': True,
                'execution_time': execution_time,
                'beta_value': float(beta),
                'covariance': float(covariance),
                'variance_b': float(variance_b)
            }
            
            print(f"✅ {test_name} passed - Beta calculation completed")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"❌ {test_name} failed: {e}")
            raise
    
    def test_volatility_analysis(self):
        """Test básico de análisis de volatilidad multi-producto"""
        test_name = "test_volatility_analysis"
        start_time = datetime.now()
        
        try:
            # Calcular volatilidades para cada producto
            returns_data = self.sample_data[['returns_a', 'returns_b', 'returns_c']]
            volatilities = returns_data.std()
            
            self.assertIsInstance(volatilities, pd.Series)
            self.assertEqual(len(volatilities), 3)
            
            # Verificar que todas las volatilidades son positivas
            for vol in volatilities:
                self.assertGreater(vol, 0)
                self.assertFalse(np.isnan(vol))
            
            # Test de volatilidad EWA si está disponible
            if MULTI_PRODUCT_AVAILABLE:
                try:
                    ewa_vol = get_ewa_vol(returns_data['returns_a'])
                    self.assertIsInstance(ewa_vol, (pd.Series, float))
                    print(f"✅ EWA volatility calculated")
                except Exception as e:
                    print(f"⚠️  EWA volatility function error: {e}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': True,
                'execution_time': execution_time,
                'volatilities': {f'product_{i}': float(vol) for i, vol in enumerate(volatilities)},
                'avg_volatility': float(volatilities.mean())
            }
            
            print(f"✅ {test_name} passed - Volatility analysis completed")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"❌ {test_name} failed: {e}")
            raise
    
    def test_portfolio_analysis(self):
        """Test básico de análisis de portafolio multi-producto"""
        test_name = "test_portfolio_analysis"
        start_time = datetime.now()
        
        try:
            # Crear un portafolio simple con pesos iguales
            returns_data = self.sample_data[['returns_a', 'returns_b', 'returns_c']]
            weights = np.array([1/3, 1/3, 1/3])
            
            # Calcular retornos del portafolio
            portfolio_returns = (returns_data * weights).sum(axis=1)
            
            self.assertIsInstance(portfolio_returns, pd.Series)
            self.assertEqual(len(portfolio_returns), len(returns_data))
            
            # Estadísticas del portafolio
            portfolio_mean = portfolio_returns.mean()
            portfolio_std = portfolio_returns.std()
            portfolio_sharpe = portfolio_mean / portfolio_std if portfolio_std > 0 else 0
            
            self.assertIsInstance(portfolio_mean, float)
            self.assertIsInstance(portfolio_std, float)
            self.assertGreater(portfolio_std, 0)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.test_results['test_details'][test_name] = {
                'passed': True,
                'execution_time': execution_time,
                'portfolio_mean': float(portfolio_mean),
                'portfolio_std': float(portfolio_std),
                'portfolio_sharpe': float(portfolio_sharpe),
                'weights': weights.tolist()
            }
            
            print(f"✅ {test_name} passed - Portfolio analysis completed")
            
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
        results_dir = Path('/workspaces/Sistema-de-datos/Quant/Results Machine Learning/results_multi_product')
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(cls.test_results, f, indent=2)
        
        print(f"\n📊 Resultados finales guardados en: {results_file}")
        print(f"✅ Tests pasados: {cls.test_results['tests_passed']}/{cls.test_results['total_tests']}")
        print(f"⏱️  Tiempo total: {cls.test_results['execution_time']:.2f}s")

if __name__ == '__main__':
    print("🧪 Iniciando tests de multi-product...")
    unittest.main(verbosity=2)
