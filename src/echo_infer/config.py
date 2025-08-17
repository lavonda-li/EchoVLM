"""Configuration management for EchoPrime inference."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML config file
        overrides: Optional dict of config overrides
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
    
    # Apply environment variable overrides
    config = _apply_env_overrides(config)
    
    return config


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config."""
    env_mappings = {
        'ECHO_DEVICE': 'model.device',
        'ECHO_BATCH_SIZE': 'data.batch_size',
        'ECHO_INPUT_DIR': 'data.input_dir',
        'ECHO_OUTPUT_DIR': 'output.dir',
        'ECHO_LOG_LEVEL': 'logging.level',
    }
    
    for env_var, config_path in env_mappings.items():
        if env_var in os.environ:
            keys = config_path.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            value = os.environ[env_var]
            # Try to convert to appropriate type
            if config_path.endswith('batch_size') or config_path.endswith('num_workers'):
                value = int(value)
            elif config_path.endswith('fp16'):
                value = value.lower() in ('true', '1', 'yes')
            current[keys[-1]] = value
    
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'model': {
            'name': 'echoprime',
            'weights_path': 'modules/EchoPrime/model_data/weights/echo_prime_encoder.pt',
            'device': 'cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
        },
        'data': {
            'input_dir': 'data/raw',
            'pattern': '*.dcm',
            'batch_size': 16,
            'num_workers': 4,
        },
        'infer': {
            'fp16': True,
            'num_threads': 4,
            'save_probs': True,
        },
        'output': {
            'dir': 'outputs/',
        },
        'logging': {
            'level': 'INFO',
        }
    }
