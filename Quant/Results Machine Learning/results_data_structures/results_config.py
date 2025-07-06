"""
Configuration module for data structures
Centralizes all configurable parameters for better maintainability
"""
import logging
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class DataQualityConfig:
    """Configuration for data quality validation"""
    remove_missing_data: bool = True
    remove_zero_prices: bool = True
    remove_negative_prices: bool = True
    remove_negative_volume: bool = True
    allow_zero_volume: bool = True
    outlier_detection_method: str = "mad"  # "mad" or "std"
    outlier_threshold: float = 10.0
    min_data_points_for_outlier_detection: int = 10
    max_data_loss_warning_threshold: float = 10.0  # percentage


@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    default_batch_size: int = 20000000
    max_batch_size: int = 50000000
    default_warm_up_period: int = 100
    enable_detailed_logging: bool = False
    log_level: str = "INFO"
    save_processing_metadata: bool = True


@dataclass
class BarConfig:
    """Configuration for bar creation"""
    include_microstructural_features: bool = True
    include_tick_statistics: bool = True
    include_volume_statistics: bool = True
    validate_ohlc_consistency: bool = True
    fix_ohlc_inconsistencies: bool = True


class ConfigManager:
    """
    Centralized configuration manager for data structures
    """
    
    def __init__(self):
        self.data_quality = DataQualityConfig()
        self.processing = ProcessingConfig()
        self.bar = BarConfig()
        
    def load_from_dict(self, config_dict: Dict[str, Any]):
        """
        Load configuration from dictionary
        
        :param config_dict: Configuration dictionary
        """
        if 'data_quality' in config_dict:
            for key, value in config_dict['data_quality'].items():
                if hasattr(self.data_quality, key):
                    setattr(self.data_quality, key, value)
                    
        if 'processing' in config_dict:
            for key, value in config_dict['processing'].items():
                if hasattr(self.processing, key):
                    setattr(self.processing, key, value)
                    
        if 'bar' in config_dict:
            for key, value in config_dict['bar'].items():
                if hasattr(self.bar, key):
                    setattr(self.bar, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration to dictionary
        
        :return: Configuration dictionary
        """
        return {
            'data_quality': {
                'remove_missing_data': self.data_quality.remove_missing_data,
                'remove_zero_prices': self.data_quality.remove_zero_prices,
                'remove_negative_prices': self.data_quality.remove_negative_prices,
                'remove_negative_volume': self.data_quality.remove_negative_volume,
                'allow_zero_volume': self.data_quality.allow_zero_volume,
                'outlier_detection_method': self.data_quality.outlier_detection_method,
                'outlier_threshold': self.data_quality.outlier_threshold,
                'min_data_points_for_outlier_detection': self.data_quality.min_data_points_for_outlier_detection,
                'max_data_loss_warning_threshold': self.data_quality.max_data_loss_warning_threshold
            },
            'processing': {
                'default_batch_size': self.processing.default_batch_size,
                'max_batch_size': self.processing.max_batch_size,
                'default_warm_up_period': self.processing.default_warm_up_period,
                'enable_detailed_logging': self.processing.enable_detailed_logging,
                'log_level': self.processing.log_level,
                'save_processing_metadata': self.processing.save_processing_metadata
            },
            'bar': {
                'include_microstructural_features': self.bar.include_microstructural_features,
                'include_tick_statistics': self.bar.include_tick_statistics,
                'include_volume_statistics': self.bar.include_volume_statistics,
                'validate_ohlc_consistency': self.bar.validate_ohlc_consistency,
                'fix_ohlc_inconsistencies': self.bar.fix_ohlc_inconsistencies
            }
        }
    
    def setup_logging(self):
        """
        Setup logging based on configuration
        """
        log_level = getattr(logging, self.processing.log_level.upper())
        
        # Configure logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Get or create logger
        logger = logging.getLogger('mlfinlab.data_structures')
        logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if self.processing.enable_detailed_logging:
            logger.info("Detailed logging enabled")
        
        return logger


# Global configuration instance
config = ConfigManager()
