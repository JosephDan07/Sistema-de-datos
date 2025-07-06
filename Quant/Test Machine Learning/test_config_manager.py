"""
Advanced Configuration Management System for Test Machine Learning
================================================================

This module provides a sophisticated configuration management system that supports:
- Global configuration (test_global_config.yml)
- Module-specific configuration (e.g., test_data_structures_config.yml)
- Test-specific configuration (e.g., test_base_bars_config.json)
- Environment overrides
- Runtime parameter injection

Author: Advanced ML Finance Team
Date: July 2025
"""

import os
import sys
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Hierarchical configuration management system following the pattern:
    Global Config -> Module Config -> Test Config -> Runtime Overrides
    """
    
    def __init__(self, test_machine_learning_path: Optional[str] = None):
        """
        Initialize the configuration manager
        
        :param test_machine_learning_path: Path to Test Machine Learning directory
        """
        if test_machine_learning_path is None:
            # Auto-detect path
            current_path = Path(__file__).parent
            test_machine_learning_path = current_path
        
        self.base_path = Path(test_machine_learning_path)
        self.global_config_path = self.base_path / "test_global_config.yml"
        
        # Configuration hierarchy
        self.global_config: Dict[str, Any] = {}
        self.module_configs: Dict[str, Dict[str, Any]] = {}
        self.test_configs: Dict[str, Dict[str, Any]] = {}
        self.runtime_overrides: Dict[str, Any] = {}
        
        # Environment detection
        self.environment = os.getenv('TEST_ENVIRONMENT', 'development')
        
        # Load configurations
        self._load_global_config()
        
        logger.info(f"Configuration manager initialized for environment: {self.environment}")
    
    def _load_global_config(self) -> None:
        """Load global configuration from YAML file"""
        try:
            if self.global_config_path.exists():
                with open(self.global_config_path, 'r', encoding='utf-8') as f:
                    self.global_config = yaml.safe_load(f) or {}
                logger.info(f"Global configuration loaded from {self.global_config_path}")
            else:
                logger.warning(f"Global config file not found: {self.global_config_path}")
                self.global_config = self._get_default_global_config()
        except Exception as e:
            logger.error(f"Error loading global config: {e}")
            self.global_config = self._get_default_global_config()
    
    def _get_default_global_config(self) -> Dict[str, Any]:
        """Get default global configuration if file is missing"""
        return {
            "global_settings": {
                "max_execution_time": 300,
                "memory_limit_mb": 4096,
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "synthetic_data": {
                    "default_samples": 10000,
                    "random_seed": 42
                }
            }
        }
    
    def load_module_config(self, module_name: str) -> Dict[str, Any]:
        """
        Load module-specific configuration
        
        :param module_name: Name of the module (e.g., 'data_structures')
        :return: Module configuration dictionary
        """
        config_file = self.base_path / f"test_{module_name}" / f"test_{module_name}_config.yml"
        
        try:
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    module_config = yaml.safe_load(f) or {}
                self.module_configs[module_name] = module_config
                logger.debug(f"Module config loaded for {module_name}")
            else:
                logger.debug(f"No module config found for {module_name}, using defaults")
                self.module_configs[module_name] = {}
        except Exception as e:
            logger.error(f"Error loading module config for {module_name}: {e}")
            self.module_configs[module_name] = {}
        
        return self.module_configs[module_name]
    
    def load_test_config(self, module_name: str, test_name: str) -> Dict[str, Any]:
        """
        Load test-specific configuration
        
        :param module_name: Name of the module
        :param test_name: Name of the test
        :return: Test configuration dictionary
        """
        config_file = self.base_path / f"test_{module_name}" / f"{test_name}_config.json"
        config_key = f"{module_name}.{test_name}"
        
        try:
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    test_config = json.load(f)
                self.test_configs[config_key] = test_config
                logger.debug(f"Test config loaded for {config_key}")
            else:
                logger.debug(f"No test config found for {config_key}")
                self.test_configs[config_key] = {}
        except Exception as e:
            logger.error(f"Error loading test config for {config_key}: {e}")
            self.test_configs[config_key] = {}
        
        return self.test_configs[config_key]
    
    def get_config(self, module_name: str, test_name: Optional[str] = None, **runtime_overrides) -> Dict[str, Any]:
        """
        Get hierarchical configuration for a specific test
        
        :param module_name: Name of the module
        :param test_name: Name of the test (optional)
        :param runtime_overrides: Runtime parameter overrides
        :return: Final merged configuration
        """
        # Start with global config
        config = copy.deepcopy(self.global_config)
        
        # Apply environment-specific overrides
        if 'environments' in config and self.environment in config['environments']:
            env_overrides = config['environments'][self.environment]
            config = self._deep_merge(config, env_overrides)
        
        # Apply module defaults from global config
        if 'module_defaults' in config and module_name in config['module_defaults']:
            module_defaults = {'module_settings': config['module_defaults'][module_name]}
            config = self._deep_merge(config, module_defaults)
        
        # Load and apply module-specific config
        module_config = self.load_module_config(module_name)
        if module_config:
            config = self._deep_merge(config, module_config)
        
        # Load and apply test-specific config
        if test_name:
            test_config = self.load_test_config(module_name, test_name)
            if test_config:
                config = self._deep_merge(config, test_config)
        
        # Apply runtime overrides
        if runtime_overrides:
            config = self._deep_merge(config, runtime_overrides)
        
        # Store final config for debugging
        config['_metadata'] = {
            'module_name': module_name,
            'test_name': test_name,
            'environment': self.environment,
            'loaded_at': datetime.now().isoformat(),
            'config_sources': self._get_config_sources(module_name, test_name)
        }
        
        return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override taking precedence
        
        :param base: Base dictionary
        :param override: Override dictionary
        :return: Merged dictionary
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _get_config_sources(self, module_name: str, test_name: Optional[str] = None) -> List[str]:
        """Get list of configuration sources that were loaded"""
        sources = []
        
        if self.global_config_path.exists():
            sources.append(str(self.global_config_path))
        
        module_config_path = self.base_path / f"test_{module_name}" / f"test_{module_name}_config.yml"
        if module_config_path.exists():
            sources.append(str(module_config_path))
        
        if test_name:
            test_config_path = self.base_path / f"test_{module_name}" / f"{test_name}_config.json"
            if test_config_path.exists():
                sources.append(str(test_config_path))
        
        return sources
    
    def save_effective_config(self, config: Dict[str, Any], output_path: str) -> None:
        """
        Save the effective configuration to a file for debugging/auditing
        
        :param config: Configuration to save
        :param output_path: Path to save the configuration
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, default=str)
            logger.debug(f"Effective configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving effective configuration: {e}")
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration for common issues
        
        :param config: Configuration to validate
        :return: List of validation warnings/errors
        """
        issues = []
        
        # Check required global settings
        if 'global_settings' not in config:
            issues.append("Missing 'global_settings' section")
        else:
            global_settings = config['global_settings']
            
            # Check memory limits
            memory_limit = global_settings.get('memory_limit_mb', 0)
            if memory_limit > 8192:
                issues.append(f"Memory limit ({memory_limit}MB) may be too high")
            
            # Check execution time
            exec_time = global_settings.get('max_execution_time', 0)
            if exec_time > 600:
                issues.append(f"Max execution time ({exec_time}s) may be too high")
        
        # Check synthetic data settings
        if 'synthetic_data' in config.get('global_settings', {}):
            synth_data = config['global_settings']['synthetic_data']
            samples = synth_data.get('default_samples', 0)
            if samples > 100000:
                issues.append(f"Default samples ({samples}) may cause performance issues")
        
        return issues
    
    def get_output_path(self, module_name: str, test_name: Optional[str] = None) -> Path:
        """
        Get standardized output path for test results
        
        :param module_name: Name of the module
        :param test_name: Name of the test (optional)
        :return: Path object for output directory
        """
        # Get base directory from config
        config = self.get_config(module_name, test_name)
        base_dir = config.get('global_settings', {}).get('output', {}).get('base_directory', '../Results Machine Learning')
        
        # Build path
        output_path = Path(base_dir) / f"results_{module_name}"
        
        if test_name:
            output_path = output_path / f"test_results_{test_name}"
        
        # Create directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        return output_path


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance (singleton pattern)"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

def get_config(module_name: str, test_name: Optional[str] = None, **runtime_overrides) -> Dict[str, Any]:
    """
    Convenience function to get configuration
    
    :param module_name: Name of the module
    :param test_name: Name of the test (optional)
    :param runtime_overrides: Runtime parameter overrides
    :return: Configuration dictionary
    """
    manager = get_config_manager()
    return manager.get_config(module_name, test_name, **runtime_overrides)

def get_output_path(module_name: str, test_name: Optional[str] = None) -> Path:
    """
    Convenience function to get output path
    
    :param module_name: Name of the module
    :param test_name: Name of the test (optional)
    :return: Path object for output directory
    """
    manager = get_config_manager()
    return manager.get_output_path(module_name, test_name)


if __name__ == "__main__":
    # Example usage and testing
    print("🔧 Testing Configuration Management System")
    print("=" * 50)
    
    # Test configuration loading
    config = get_config('data_structures', 'test_base_bars')
    print(f"✅ Configuration loaded for data_structures.test_base_bars")
    print(f"📁 Output path: {get_output_path('data_structures', 'test_base_bars')}")
    print(f"🔍 Config sources: {config.get('_metadata', {}).get('config_sources', [])}")
    
    # Validate configuration
    manager = get_config_manager()
    issues = manager.validate_config(config)
    if issues:
        print(f"⚠️ Configuration issues found: {issues}")
    else:
        print("✅ Configuration validation passed")
    
    print("🎯 Configuration system ready!")
