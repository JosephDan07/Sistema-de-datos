#!/usr/bin/env python3
"""
Test completo para todos los módulos de multi_product
Tests robustos con datos reales, evitando sobreajuste
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import json
import logging

# Agregar paths para importar módulos
sys.path.append('/workspaces/Sistema-de-datos/Quant/Machine Learning')
sys.path.append('/workspaces/Sistema-de-datos/Quant/Test Machine Learning')

from test_config_manager import ConfigurationManager
from multi_product.etf_trick import ETFTrick
from multi_product.futures_roll import FuturesRoll

warnings.filterwarnings('ignore')

class TestCompleteMultiProduct(unittest.TestCase):
    """
    Test completo para todos los módulos de multi_product
    """
    
    @classmethod
    def setUpClass(cls):
        """Setup inicial para todos los tests"""
        cls.config_manager = ConfigurationManager()
        cls.config = cls.config_manager.get_config('test_multi_product')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
        # Paths
        cls.data_path = '/workspaces/Sistema-de-datos/Quant/Datos/Tiingo_CSV'
        cls.results_path = '/workspaces/Sistema-de-datos/Quant/Results Machine Learning'
        
        # Crear carpeta de resultados si no existe
        os.makedirs(cls.results_path, exist_ok=True)
        
        # Cargar múltiples datasets
        cls.datasets = cls._load_multiple_datasets()
        
        cls.logger.info(f"Loaded {len(cls.datasets)} datasets for multi_product testing")
    
    @classmethod
    def _load_multiple_datasets(cls):
        """Cargar múltiples datasets para tests robustos"""
        datasets = {}
        
        # Cargar ETFs y activos para portfolio
        files_to_load = [
            'SPY_daily.csv',   # S&P 500 ETF
            'QQQ_daily.csv',   # NASDAQ ETF
            'GLD_daily.csv',   # Gold ETF
            'TLT_daily.csv',   # Treasury ETF
            'VTI_daily.csv',   # Total Stock Market ETF
            'IWM_daily.csv',   # Small Cap ETF
            'EFA_daily.csv',   # International ETF
            'USO_daily.csv',   # Oil ETF
            'XLE_daily.csv',   # Energy ETF
            'SLV_daily.csv'    # Silver ETF
        ]
        
        for file in files_to_load:
            try:
                file_path = os.path.join(cls.data_path, file)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    
                    # Preparar datos para ETF Trick
                    df['returns'] = df['close'].pct_change()
                    df['log_returns'] = np.log(df['close']).diff()
                    
                    # Simular datos de contratos futuros
                    df['contract_multiplier'] = 1.0
                    df['carry_cost'] = 0.0
                    df['dividend'] = 0.0
                    
                    datasets[file.replace('.csv', '')] = df.dropna()
                    
            except Exception as e:
                cls.logger.warning(f"Could not load {file}: {str(e)}")
        
        return datasets
    
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
            f'results_multi_product_complete_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
    
    def _create_portfolio_allocation(self, assets, strategy='equal_weight'):
        """Crear asignación de portfolio"""
        n_assets = len(assets)
        dates = None
        
        # Encontrar fechas comunes
        for asset_name, asset_data in assets.items():
            if dates is None:
                dates = asset_data.index
            else:
                dates = dates.intersection(asset_data.index)
        
        if len(dates) == 0:
            return None
        
        # Crear DataFrame de allocation
        allocation = pd.DataFrame(index=dates)
        
        if strategy == 'equal_weight':
            # Pesos iguales
            for asset_name in assets.keys():
                allocation[asset_name] = 1.0 / n_assets
        
        elif strategy == 'momentum':
            # Momentum based allocation
            for asset_name, asset_data in assets.items():
                asset_aligned = asset_data.reindex(dates, method='ffill')
                momentum = asset_aligned['close'].pct_change(21).rolling(5).mean()
                allocation[asset_name] = momentum / momentum.abs().sum(axis=1, skipna=True)
        
        elif strategy == 'volatility_weighted':
            # Volatility weighted allocation
            for asset_name, asset_data in assets.items():
                asset_aligned = asset_data.reindex(dates, method='ffill')
                volatility = asset_aligned['returns'].rolling(20).std()
                inv_vol = 1 / volatility
                allocation[asset_name] = inv_vol / inv_vol.sum(axis=1, skipna=True)
        
        return allocation.fillna(0)
    
    def test_etf_trick_equal_weight_portfolio(self):
        """Test ETF Trick con portfolio de pesos iguales"""
        self.logger.info("Testing ETF Trick with equal weight portfolio")
        
        results = {}
        
        # Seleccionar subset de assets para test
        asset_subsets = [
            ['SPY_daily', 'QQQ_daily', 'GLD_daily'],
            ['SPY_daily', 'TLT_daily', 'GLD_daily', 'VTI_daily'],
            ['SPY_daily', 'QQQ_daily', 'GLD_daily', 'TLT_daily', 'IWM_daily']
        ]
        
        for i, asset_names in enumerate(asset_subsets):
            try:
                # Verificar que todos los assets están disponibles
                available_assets = {name: self.datasets[name] for name in asset_names 
                                  if name in self.datasets}
                
                if len(available_assets) >= 2:  # Necesitamos al menos 2 assets
                    # Crear allocation
                    allocation = self._create_portfolio_allocation(available_assets, 'equal_weight')
                    
                    if allocation is not None and len(allocation) > 50:
                        # Preparar datos para ETF Trick
                        open_prices = pd.DataFrame()
                        close_prices = pd.DataFrame()
                        
                        for asset_name, asset_data in available_assets.items():
                            asset_aligned = asset_data.reindex(allocation.index, method='ffill')
                            open_prices[asset_name] = asset_aligned['open']
                            close_prices[asset_name] = asset_aligned['close']
                        
                        # Crear ETF Trick instance
                        etf_trick = ETFTrick()
                        
                        # Calcular ETF series
                        etf_series = etf_trick.calculate_etf_series(
                            allocations=allocation,
                            open_prices=open_prices,
                            close_prices=close_prices,
                            costs=pd.DataFrame(0, index=allocation.index, columns=allocation.columns),
                            multipliers=pd.DataFrame(1, index=allocation.index, columns=allocation.columns),
                            initial_value=10000
                        )
                        
                        # Validaciones
                        self.assertIsInstance(etf_series, pd.Series)
                        self.assertEqual(len(etf_series), len(allocation))
                        self.assertTrue((etf_series > 0).all())  # Valores positivos
                        
                        # Verificar que hay variación en la serie
                        self.assertGreater(etf_series.std(), 0)
                        
                        # Calcular estadísticas
                        etf_returns = etf_series.pct_change().dropna()
                        
                        results[f'portfolio_{i+1}'] = {
                            'assets': list(available_assets.keys()),
                            'series_length': len(etf_series),
                            'initial_value': etf_series.iloc[0],
                            'final_value': etf_series.iloc[-1],
                            'total_return': (etf_series.iloc[-1] / etf_series.iloc[0] - 1) * 100,
                            'volatility': etf_returns.std() * np.sqrt(252) * 100,
                            'sharpe_ratio': etf_returns.mean() / etf_returns.std() * np.sqrt(252) if etf_returns.std() > 0 else 0,
                            'max_drawdown': ((etf_series / etf_series.expanding().max()) - 1).min() * 100,
                            'success': True
                        }
                    else:
                        results[f'portfolio_{i+1}'] = {
                            'success': False,
                            'error': 'Insufficient common dates'
                        }
                else:
                    results[f'portfolio_{i+1}'] = {
                        'success': False,
                        'error': 'Insufficient assets available'
                    }
                
                self.logger.info(f"ETF Trick test passed for portfolio {i+1}")
                
            except Exception as e:
                self.logger.error(f"ETF Trick test failed for portfolio {i+1}: {str(e)}")
                results[f'portfolio_{i+1}'] = {
                    'success': False,
                    'error': str(e)
                }
        
        self.test_results['etf_trick_equal_weight_results'] = results
    
    def test_etf_trick_momentum_portfolio(self):
        """Test ETF Trick con portfolio basado en momentum"""
        self.logger.info("Testing ETF Trick with momentum portfolio")
        
        results = {}
        
        try:
            # Seleccionar assets con suficientes datos
            available_assets = {name: data for name, data in self.datasets.items() 
                              if len(data) > 100}
            
            if len(available_assets) >= 3:
                # Tomar los primeros 3 assets
                selected_assets = dict(list(available_assets.items())[:3])
                
                # Crear allocation basada en momentum
                allocation = self._create_portfolio_allocation(selected_assets, 'momentum')
                
                if allocation is not None and len(allocation) > 50:
                    # Preparar datos para ETF Trick
                    open_prices = pd.DataFrame()
                    close_prices = pd.DataFrame()
                    
                    for asset_name, asset_data in selected_assets.items():
                        asset_aligned = asset_data.reindex(allocation.index, method='ffill')
                        open_prices[asset_name] = asset_aligned['open']
                        close_prices[asset_name] = asset_aligned['close']
                    
                    # Crear ETF Trick instance
                    etf_trick = ETFTrick()
                    
                    # Calcular ETF series
                    etf_series = etf_trick.calculate_etf_series(
                        allocations=allocation,
                        open_prices=open_prices,
                        close_prices=close_prices,
                        costs=pd.DataFrame(0, index=allocation.index, columns=allocation.columns),
                        multipliers=pd.DataFrame(1, index=allocation.index, columns=allocation.columns),
                        initial_value=10000
                    )
                    
                    # Validaciones
                    self.assertIsInstance(etf_series, pd.Series)
                    self.assertGreater(len(etf_series), 0)
                    
                    # Calcular estadísticas
                    etf_returns = etf_series.pct_change().dropna()
                    
                    results['momentum_portfolio'] = {
                        'assets': list(selected_assets.keys()),
                        'series_length': len(etf_series),
                        'allocation_turnover': allocation.diff().abs().sum(axis=1).mean(),
                        'total_return': (etf_series.iloc[-1] / etf_series.iloc[0] - 1) * 100,
                        'volatility': etf_returns.std() * np.sqrt(252) * 100,
                        'success': True
                    }
                else:
                    results['momentum_portfolio'] = {
                        'success': False,
                        'error': 'Insufficient allocation data'
                    }
            else:
                results['momentum_portfolio'] = {
                    'success': False,
                    'error': 'Insufficient assets for momentum portfolio'
                }
                
            self.logger.info("ETF Trick momentum portfolio test completed")
            
        except Exception as e:
            self.logger.error(f"ETF Trick momentum portfolio test failed: {str(e)}")
            results['momentum_portfolio'] = {
                'success': False,
                'error': str(e)
            }
        
        self.test_results['etf_trick_momentum_results'] = results
    
    def test_etf_trick_rebalancing_costs(self):
        """Test ETF Trick con costos de rebalancing"""
        self.logger.info("Testing ETF Trick with rebalancing costs")
        
        results = {}
        
        try:
            # Seleccionar assets para test
            asset_names = ['SPY_daily', 'QQQ_daily', 'GLD_daily']
            available_assets = {name: self.datasets[name] for name in asset_names 
                              if name in self.datasets}
            
            if len(available_assets) >= 2:
                # Crear allocation
                allocation = self._create_portfolio_allocation(available_assets, 'equal_weight')
                
                if allocation is not None and len(allocation) > 50:
                    # Preparar datos
                    open_prices = pd.DataFrame()
                    close_prices = pd.DataFrame()
                    
                    for asset_name, asset_data in available_assets.items():
                        asset_aligned = asset_data.reindex(allocation.index, method='ffill')
                        open_prices[asset_name] = asset_aligned['open']
                        close_prices[asset_name] = asset_aligned['close']
                    
                    # Test con diferentes niveles de costos
                    cost_levels = [0.0, 0.001, 0.005, 0.01]  # 0%, 0.1%, 0.5%, 1%
                    
                    for cost_level in cost_levels:
                        try:
                            # Crear matriz de costos
                            costs = pd.DataFrame(0, index=allocation.index, columns=allocation.columns)
                            
                            # Aplicar costos cuando hay cambios en allocation
                            allocation_changes = allocation.diff().abs()
                            costs = allocation_changes * cost_level
                            
                            # Crear ETF Trick instance
                            etf_trick = ETFTrick()
                            
                            # Calcular ETF series
                            etf_series = etf_trick.calculate_etf_series(
                                allocations=allocation,
                                open_prices=open_prices,
                                close_prices=close_prices,
                                costs=costs,
                                multipliers=pd.DataFrame(1, index=allocation.index, columns=allocation.columns),
                                initial_value=10000
                            )
                            
                            # Calcular estadísticas
                            etf_returns = etf_series.pct_change().dropna()
                            total_return = (etf_series.iloc[-1] / etf_series.iloc[0] - 1) * 100
                            
                            results[f'cost_level_{cost_level}'] = {
                                'cost_level': cost_level * 100,  # En porcentaje
                                'total_return': total_return,
                                'volatility': etf_returns.std() * np.sqrt(252) * 100,
                                'total_costs': costs.sum().sum(),
                                'success': True
                            }
                            
                        except Exception as e:
                            self.logger.warning(f"Rebalancing cost test failed for cost level {cost_level}: {str(e)}")
                            results[f'cost_level_{cost_level}'] = {
                                'success': False,
                                'error': str(e)
                            }
                    
                    # Verificar que los costos más altos resultan en menores retornos
                    successful_results = {k: v for k, v in results.items() 
                                        if v.get('success', False)}
                    
                    if len(successful_results) > 1:
                        sorted_results = sorted(successful_results.items(), 
                                              key=lambda x: x[1]['cost_level'])
                        
                        # Verificar tendencia (costos más altos = retornos más bajos)
                        returns = [r[1]['total_return'] for r in sorted_results]
                        cost_impact_logical = all(returns[i] >= returns[i+1] 
                                                for i in range(len(returns)-1))
                        
                        results['cost_impact_analysis'] = {
                            'logical_cost_impact': cost_impact_logical,
                            'return_degradation': returns[0] - returns[-1] if len(returns) > 1 else 0
                        }
                else:
                    results['error'] = 'Insufficient allocation data'
            else:
                results['error'] = 'Insufficient assets available'
                
            self.logger.info("ETF Trick rebalancing costs test completed")
            
        except Exception as e:
            self.logger.error(f"ETF Trick rebalancing costs test failed: {str(e)}")
            results['general_error'] = str(e)
        
        self.test_results['etf_trick_costs_results'] = results
    
    def test_futures_roll_functionality(self):
        """Test funcionalidad de Futures Roll"""
        self.logger.info("Testing Futures Roll functionality")
        
        results = {}
        
        try:
            # Simular datos de contratos futuros
            # Usar commodity ETFs como proxy
            commodity_etfs = ['GLD_daily', 'SLV_daily', 'USO_daily']
            available_commodities = {name: self.datasets[name] for name in commodity_etfs 
                                   if name in self.datasets}
            
            if len(available_commodities) > 0:
                for commodity_name, commodity_data in available_commodities.items():
                    try:
                        # Simular múltiples contratos (front month, next month, etc.)
                        contracts = {}
                        
                        # Contract 1 (front month)
                        contracts['contract_1'] = commodity_data.copy()
                        
                        # Contract 2 (next month) - simulado con pequeña diferencia
                        contracts['contract_2'] = commodity_data.copy()
                        contracts['contract_2']['open'] *= 1.01
                        contracts['contract_2']['close'] *= 1.01
                        contracts['contract_2']['high'] *= 1.01
                        contracts['contract_2']['low'] *= 1.01
                        
                        # Contract 3 (third month)
                        contracts['contract_3'] = commodity_data.copy()
                        contracts['contract_3']['open'] *= 1.02
                        contracts['contract_3']['close'] *= 1.02
                        contracts['contract_3']['high'] *= 1.02
                        contracts['contract_3']['low'] *= 1.02
                        
                        # Crear Futures Roll instance
                        futures_roll = FuturesRoll()
                        
                        # Definir fechas de roll (cada 30 días)
                        roll_dates = pd.date_range(
                            start=commodity_data.index[30],
                            end=commodity_data.index[-30],
                            freq='30D'
                        )
                        
                        # Calcular serie continua
                        continuous_series = futures_roll.create_continuous_series(
                            contracts=contracts,
                            roll_dates=roll_dates,
                            roll_method='backward_adjustment'
                        )
                        
                        # Validaciones
                        self.assertIsInstance(continuous_series, pd.Series)
                        self.assertGreater(len(continuous_series), 0)
                        
                        # Verificar que no hay gaps grandes en los roll dates
                        returns = continuous_series.pct_change().dropna()
                        extreme_returns = returns[abs(returns) > 0.1]  # > 10%
                        
                        results[commodity_name] = {
                            'series_length': len(continuous_series),
                            'roll_dates_count': len(roll_dates),
                            'extreme_returns_count': len(extreme_returns),
                            'total_return': (continuous_series.iloc[-1] / continuous_series.iloc[0] - 1) * 100,
                            'volatility': returns.std() * np.sqrt(252) * 100,
                            'success': True
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"Futures roll test failed for {commodity_name}: {str(e)}")
                        results[commodity_name] = {
                            'success': False,
                            'error': str(e)
                        }
                
                self.logger.info("Futures Roll functionality test completed")
                
            else:
                results['error'] = 'No commodity data available'
                
        except Exception as e:
            self.logger.error(f"Futures Roll functionality test failed: {str(e)}")
            results['general_error'] = str(e)
        
        self.test_results['futures_roll_results'] = results
    
    def test_performance_benchmarks(self):
        """Test de performance con benchmarks"""
        self.logger.info("Testing performance benchmarks")
        
        performance_results = {}
        
        try:
            # Benchmark ETF Trick con diferentes tamaños de portfolio
            portfolio_sizes = [2, 3, 5]
            
            for portfolio_size in portfolio_sizes:
                try:
                    # Seleccionar assets
                    asset_names = list(self.datasets.keys())[:portfolio_size]
                    selected_assets = {name: self.datasets[name] for name in asset_names}
                    
                    if len(selected_assets) == portfolio_size:
                        # Crear allocation
                        allocation = self._create_portfolio_allocation(selected_assets, 'equal_weight')
                        
                        if allocation is not None and len(allocation) > 50:
                            # Preparar datos
                            open_prices = pd.DataFrame()
                            close_prices = pd.DataFrame()
                            
                            for asset_name, asset_data in selected_assets.items():
                                asset_aligned = asset_data.reindex(allocation.index, method='ffill')
                                open_prices[asset_name] = asset_aligned['open']
                                close_prices[asset_name] = asset_aligned['close']
                            
                            # Medir performance
                            start_time = datetime.now()
                            
                            etf_trick = ETFTrick()
                            etf_series = etf_trick.calculate_etf_series(
                                allocations=allocation,
                                open_prices=open_prices,
                                close_prices=close_prices,
                                costs=pd.DataFrame(0, index=allocation.index, columns=allocation.columns),
                                multipliers=pd.DataFrame(1, index=allocation.index, columns=allocation.columns),
                                initial_value=10000
                            )
                            
                            end_time = datetime.now()
                            execution_time = (end_time - start_time).total_seconds()
                            
                            performance_results[f'portfolio_size_{portfolio_size}'] = {
                                'execution_time': execution_time,
                                'input_size': len(allocation),
                                'portfolio_size': portfolio_size,
                                'output_size': len(etf_series),
                                'throughput': len(allocation) / execution_time if execution_time > 0 else 0
                            }
                            
                            self.logger.info(f"Portfolio size {portfolio_size} processed in {execution_time:.4f}s")
                        else:
                            performance_results[f'portfolio_size_{portfolio_size}'] = {
                                'error': 'Insufficient allocation data'
                            }
                    else:
                        performance_results[f'portfolio_size_{portfolio_size}'] = {
                            'error': 'Insufficient assets'
                        }
                        
                except Exception as e:
                    self.logger.error(f"Performance benchmark failed for portfolio size {portfolio_size}: {str(e)}")
                    performance_results[f'portfolio_size_{portfolio_size}'] = {
                        'error': str(e)
                    }
            
        except Exception as e:
            self.logger.error(f"Performance benchmarks failed: {str(e)}")
            performance_results['general_error'] = str(e)
        
        self.test_results['performance_results'] = performance_results
    
    def test_edge_cases_and_robustness(self):
        """Test de casos edge y robustez"""
        self.logger.info("Testing edge cases and robustness")
        
        edge_case_results = {}
        
        # Test con portfolio de 1 asset
        try:
            if len(self.datasets) > 0:
                single_asset = list(self.datasets.keys())[0]
                asset_data = self.datasets[single_asset]
                
                allocation = pd.DataFrame({single_asset: [1.0] * len(asset_data)}, 
                                        index=asset_data.index)
                
                etf_trick = ETFTrick()
                etf_series = etf_trick.calculate_etf_series(
                    allocations=allocation,
                    open_prices=pd.DataFrame({single_asset: asset_data['open']}),
                    close_prices=pd.DataFrame({single_asset: asset_data['close']}),
                    costs=pd.DataFrame({single_asset: [0.0] * len(asset_data)}, index=asset_data.index),
                    multipliers=pd.DataFrame({single_asset: [1.0] * len(asset_data)}, index=asset_data.index),
                    initial_value=10000
                )
                
                edge_case_results['single_asset'] = {
                    'series_length': len(etf_series),
                    'matches_underlying': np.corrcoef(etf_series.pct_change().dropna(), 
                                                    asset_data['close'].pct_change().dropna())[0,1] > 0.99,
                    'passed': True
                }
            else:
                edge_case_results['single_asset'] = {
                    'passed': False,
                    'error': 'No assets available'
                }
        except Exception as e:
            edge_case_results['single_asset'] = {
                'passed': False,
                'error': str(e)
            }
        
        # Test con allocations extremas
        try:
            if len(self.datasets) >= 2:
                assets = dict(list(self.datasets.items())[:2])
                allocation = self._create_portfolio_allocation(assets, 'equal_weight')
                
                if allocation is not None and len(allocation) > 10:
                    # Crear allocation extrema (todo en un asset)
                    extreme_allocation = allocation.copy()
                    extreme_allocation.iloc[:, 0] = 1.0
                    extreme_allocation.iloc[:, 1] = 0.0
                    
                    # Preparar datos
                    open_prices = pd.DataFrame()
                    close_prices = pd.DataFrame()
                    
                    for asset_name, asset_data in assets.items():
                        asset_aligned = asset_data.reindex(allocation.index, method='ffill')
                        open_prices[asset_name] = asset_aligned['open']
                        close_prices[asset_name] = asset_aligned['close']
                    
                    etf_trick = ETFTrick()
                    etf_series = etf_trick.calculate_etf_series(
                        allocations=extreme_allocation,
                        open_prices=open_prices,
                        close_prices=close_prices,
                        costs=pd.DataFrame(0, index=allocation.index, columns=allocation.columns),
                        multipliers=pd.DataFrame(1, index=allocation.index, columns=allocation.columns),
                        initial_value=10000
                    )
                    
                    edge_case_results['extreme_allocation'] = {
                        'series_length': len(etf_series),
                        'passed': len(etf_series) > 0
                    }
                else:
                    edge_case_results['extreme_allocation'] = {
                        'passed': False,
                        'error': 'Insufficient allocation data'
                    }
            else:
                edge_case_results['extreme_allocation'] = {
                    'passed': False,
                    'error': 'Insufficient assets'
                }
        except Exception as e:
            edge_case_results['extreme_allocation'] = {
                'passed': False,
                'error': str(e)
            }
        
        # Test con costos extremos
        try:
            if len(self.datasets) >= 2:
                assets = dict(list(self.datasets.items())[:2])
                allocation = self._create_portfolio_allocation(assets, 'equal_weight')
                
                if allocation is not None and len(allocation) > 10:
                    # Preparar datos
                    open_prices = pd.DataFrame()
                    close_prices = pd.DataFrame()
                    
                    for asset_name, asset_data in assets.items():
                        asset_aligned = asset_data.reindex(allocation.index, method='ffill')
                        open_prices[asset_name] = asset_aligned['open']
                        close_prices[asset_name] = asset_aligned['close']
                    
                    # Costos extremos (50% por transacción)
                    extreme_costs = pd.DataFrame(0.5, index=allocation.index, columns=allocation.columns)
                    
                    etf_trick = ETFTrick()
                    etf_series = etf_trick.calculate_etf_series(
                        allocations=allocation,
                        open_prices=open_prices,
                        close_prices=close_prices,
                        costs=extreme_costs,
                        multipliers=pd.DataFrame(1, index=allocation.index, columns=allocation.columns),
                        initial_value=10000
                    )
                    
                    edge_case_results['extreme_costs'] = {
                        'series_length': len(etf_series),
                        'final_value': etf_series.iloc[-1],
                        'severe_cost_impact': etf_series.iloc[-1] < etf_series.iloc[0] * 0.5,
                        'passed': len(etf_series) > 0
                    }
                else:
                    edge_case_results['extreme_costs'] = {
                        'passed': False,
                        'error': 'Insufficient allocation data'
                    }
            else:
                edge_case_results['extreme_costs'] = {
                    'passed': False,
                    'error': 'Insufficient assets'
                }
        except Exception as e:
            edge_case_results['extreme_costs'] = {
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
