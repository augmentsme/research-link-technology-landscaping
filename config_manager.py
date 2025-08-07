"""
Configuration management for Research Link Technology Landscaping
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the research analysis system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from YAML file"""
        if config_path is None:
            # Default to config.yaml in project root
            project_root = Path(__file__).parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'data.base_dir')"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_data_dir(self) -> Path:
        """Get the base data directory path"""
        data_dir = self.get('data.base_dir')
        if not data_dir:
            raise ValueError("data.base_dir not configured")
        return Path(data_dir)
    
    def get_data_file_path(self, file_key: str) -> Path:
        """Get full path to a data file (e.g., 'grants_file' -> '/path/to/data/active_grants.json')"""
        base_dir = self.get_data_dir()
        filename = self.get(f'data.{file_key}')
        if not filename:
            raise ValueError(f"data.{file_key} not configured")
        return base_dir / filename
    
    def get_logs_dir(self) -> Path:
        """Get the logs directory path"""
        logs_dir = self.get('logs.base_dir')
        if not logs_dir:
            raise ValueError("logs.base_dir not configured")
        return Path(logs_dir)
    
    def get_modeling_config(self) -> Dict[str, Any]:
        """Get modeling configuration"""
        return self.get('modeling', {})
    
    def get_web_config(self) -> Dict[str, Any]:
        """Get web dashboard configuration"""
        return self.get('web', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.get('api', {})
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.get('dev.debug', False)
    
    def get_sample_size(self) -> int:
        """Get sample size for development/testing"""
        return self.get('dev.sample_size', 10)
    
    def update_config(self, key: str, value: Any) -> None:
        """Update configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the final key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    def save_config(self) -> None:
        """Save current configuration back to YAML file"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self._config, f, default_flow_style=False, indent=2)
    
    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"


# Global configuration instance
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance"""
    global _global_config
    
    if _global_config is None or config_path is not None:
        _global_config = Config(config_path)
    
    return _global_config


def reload_config() -> Config:
    """Reload configuration from file"""
    global _global_config
    if _global_config is not None:
        _global_config.load_config()
        return _global_config
    else:
        return get_config()


# Convenience functions for common config access patterns
def get_data_dir() -> Path:
    """Get the base data directory"""
    return get_config().get_data_dir()


def get_data_file_path(file_key: str) -> Path:
    """Get full path to a data file"""
    return get_config().get_data_file_path(file_key)


def get_logs_dir() -> Path:
    """Get the logs directory"""
    return get_config().get_logs_dir()


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print(f"Data directory: {config.get_data_dir()}")
    print(f"Grants file: {config.get_data_file_path('grants_file')}")
    print(f"Default model: {config.get('modeling.default_model')}")
    print(f"Debug mode: {config.is_debug_mode()}")
