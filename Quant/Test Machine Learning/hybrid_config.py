"""
Sistema de Configuración Híbrida - ML Finance Testing Framework
================================================================

Este módulo implementa un sistema de configuración híbrida que permite:
1. Configuración Global (global_config.yml)
2. Configuración por Módulo (module_config.yml en cada carpeta)
3. Configuración Específica (test_config.json para cada test)

Author: Advanced ML Finance Team
Date: July 2025
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import copy

logger = logging.getLogger(__name__)

class HybridConfigManager:
    """
    Gestor de configuración híbrida que combina configuraciones globales,
    por módulo y específicas de test.
    """
    
    def __init__(self, base_path: str = None):
        """
        Inicializar el gestor de configuración
        
        :param base_path: Ruta base del proyecto
        """
        self.base_path = base_path or os.path.dirname(os.path.abspath(__file__))
        self.global_config = {}
        self.module_configs = {}
        self.test_configs = {}
        
        # Cargar configuración global
        self._load_global_config()
        
    def _load_global_config(self):
        """Cargar configuración global desde global_config.yml"""
        global_config_path = os.path.join(self.base_path, "global_config.yml")
        
        try:
            if os.path.exists(global_config_path):
                with open(global_config_path, 'r', encoding='utf-8') as f:
                    self.global_config = yaml.safe_load(f) or {}
                logger.info(f"✅ Global config loaded from {global_config_path}")
            else:
                logger.warning(f"⚠️ Global config not found: {global_config_path}")
                self.global_config = self._get_default_global_config()
                
        except Exception as e:
            logger.error(f"❌ Error loading global config: {e}")
            self.global_config = self._get_default_global_config()
    
    def _get_default_global_config(self) -> Dict[str, Any]:
        """Configuración global por defecto"""
        return {
            "project": {
                "name": "ML Finance Testing Framework",
                "version": "2.0.0"
            },
            "testing": {
                "performance": {
                    "default_data_sizes": [1000, 5000, 10000],
                    "timeout_seconds": 300
                },
                "synthetic_data": {
                    "default_samples": 10000,
                    "price_start": 100.0,
                    "volatility": 0.02,
                    "random_seed": 42
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def load_module_config(self, module_name: str, module_path: str = None) -> Dict[str, Any]:
        """
        Cargar configuración específica de un módulo
        
        :param module_name: Nombre del módulo
        :param module_path: Ruta del módulo (opcional)
        :return: Configuración combinada
        """
        if module_name in self.module_configs:
            return self.module_configs[module_name]
        
        # Determinar ruta del módulo
        if module_path is None:
            module_path = os.path.join(self.base_path, f"test_{module_name}")
        
        module_config_path = os.path.join(module_path, "module_config.yml")
        module_config = {}
        
        try:
            if os.path.exists(module_config_path):
                with open(module_config_path, 'r', encoding='utf-8') as f:
                    module_config = yaml.safe_load(f) or {}
                logger.info(f"✅ Module config loaded for {module_name}")
            else:
                logger.info(f"ℹ️ No module config found for {module_name}, using defaults")
                
        except Exception as e:
            logger.error(f"❌ Error loading module config for {module_name}: {e}")
        
        # Combinar con configuración global
        combined_config = self._merge_configs(self.global_config, module_config)
        self.module_configs[module_name] = combined_config
        
        return combined_config
    
    def load_test_config(self, test_name: str, test_path: str = None, module_name: str = None) -> Dict[str, Any]:
        """
        Cargar configuración específica de un test
        
        :param test_name: Nombre del test
        :param test_path: Ruta del test (opcional)
        :param module_name: Nombre del módulo padre (opcional)
        :return: Configuración combinada final
        """
        config_key = f"{module_name}_{test_name}" if module_name else test_name
        
        if config_key in self.test_configs:
            return self.test_configs[config_key]
        
        # Cargar configuración del módulo padre si existe
        base_config = self.global_config
        if module_name:
            base_config = self.load_module_config(module_name)
        
        # Determinar ruta del test
        if test_path is None and module_name:
            test_path = os.path.join(self.base_path, f"test_{module_name}")
        elif test_path is None:
            test_path = self.base_path
            
        test_config_path = os.path.join(test_path, f"{test_name}_config.json")
        test_config = {}
        
        try:
            if os.path.exists(test_config_path):
                with open(test_config_path, 'r', encoding='utf-8') as f:
                    test_config = json.load(f)
                logger.info(f"✅ Test config loaded for {test_name}")
            else:
                # Buscar configuración genérica test_config.json
                generic_config_path = os.path.join(test_path, "test_config.json")
                if os.path.exists(generic_config_path):
                    with open(generic_config_path, 'r', encoding='utf-8') as f:
                        test_config = json.load(f)
                    logger.info(f"✅ Generic test config loaded for {test_name}")
                else:
                    logger.info(f"ℹ️ No test config found for {test_name}, using module/global defaults")
                    
        except Exception as e:
            logger.error(f"❌ Error loading test config for {test_name}: {e}")
        
        # Combinar todas las configuraciones
        combined_config = self._merge_configs(base_config, test_config)
        self.test_configs[config_key] = combined_config
        
        return combined_config
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fusionar configuraciones de manera recursiva
        
        :param base_config: Configuración base
        :param override_config: Configuración que sobrescribe
        :return: Configuración fusionada
        """
        merged = copy.deepcopy(base_config)
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        
        return merged
    
    def get_config(self, config_type: str = "global", **kwargs) -> Dict[str, Any]:
        """
        Obtener configuración según el tipo
        
        :param config_type: "global", "module", o "test"
        :param kwargs: Argumentos adicionales (module_name, test_name, etc.)
        :return: Configuración solicitada
        """
        if config_type == "global":
            return self.global_config
        elif config_type == "module":
            module_name = kwargs.get("module_name")
            module_path = kwargs.get("module_path")
            if not module_name:
                raise ValueError("module_name is required for module config")
            return self.load_module_config(module_name, module_path)
        elif config_type == "test":
            test_name = kwargs.get("test_name")
            test_path = kwargs.get("test_path")
            module_name = kwargs.get("module_name")
            if not test_name:
                raise ValueError("test_name is required for test config")
            return self.load_test_config(test_name, test_path, module_name)
        else:
            raise ValueError(f"Unknown config_type: {config_type}")
    
    def save_config(self, config: Dict[str, Any], config_type: str, **kwargs):
        """
        Guardar configuración en el archivo correspondiente
        
        :param config: Configuración a guardar
        :param config_type: "global", "module", o "test"
        :param kwargs: Argumentos adicionales
        """
        try:
            if config_type == "global":
                config_path = os.path.join(self.base_path, "global_config.yml")
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                    
            elif config_type == "module":
                module_name = kwargs.get("module_name")
                module_path = kwargs.get("module_path") or os.path.join(self.base_path, f"test_{module_name}")
                os.makedirs(module_path, exist_ok=True)
                config_path = os.path.join(module_path, "module_config.yml")
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                    
            elif config_type == "test":
                test_name = kwargs.get("test_name")
                test_path = kwargs.get("test_path")
                module_name = kwargs.get("module_name")
                
                if test_path is None and module_name:
                    test_path = os.path.join(self.base_path, f"test_{module_name}")
                elif test_path is None:
                    test_path = self.base_path
                    
                os.makedirs(test_path, exist_ok=True)
                config_path = os.path.join(test_path, f"{test_name}_config.json")
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ {config_type} config saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving {config_type} config: {e}")
            raise
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validar que la configuración tenga la estructura mínima requerida
        
        :param config: Configuración a validar
        :return: True si es válida
        """
        required_sections = ["testing", "logging"]
        
        for section in required_sections:
            if section not in config:
                logger.warning(f"⚠️ Missing required config section: {section}")
                return False
        
        return True
    
    def get_effective_config(self, test_name: str, module_name: str = None) -> Dict[str, Any]:
        """
        Obtener la configuración efectiva final para un test específico
        (combina global + módulo + test)
        
        :param test_name: Nombre del test
        :param module_name: Nombre del módulo
        :return: Configuración efectiva
        """
        return self.load_test_config(test_name, module_name=module_name)
    
    def list_available_configs(self) -> Dict[str, Any]:
        """
        Listar todas las configuraciones disponibles
        
        :return: Diccionario con información de configuraciones
        """
        return {
            "global_config": bool(self.global_config),
            "module_configs": list(self.module_configs.keys()),
            "test_configs": list(self.test_configs.keys()),
            "base_path": self.base_path
        }


# Instancia global del gestor de configuración
config_manager = HybridConfigManager()

def get_config(config_type: str = "global", **kwargs) -> Dict[str, Any]:
    """
    Función de conveniencia para obtener configuración
    
    :param config_type: Tipo de configuración
    :param kwargs: Argumentos adicionales
    :return: Configuración solicitada
    """
    return config_manager.get_config(config_type, **kwargs)

def get_test_config(test_name: str, module_name: str = None) -> Dict[str, Any]:
    """
    Función de conveniencia para obtener configuración de test
    
    :param test_name: Nombre del test
    :param module_name: Nombre del módulo
    :return: Configuración del test
    """
    return config_manager.get_effective_config(test_name, module_name)
