"""
Configuration management with YAML support and CLI override.

Provides a unified configuration system that loads defaults,
merges with YAML config file, and allows CLI argument overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from .defaults import DEFAULTS


class objectview:
    """Simple object that allows dict keys to be accessed as attributes."""
    
    def __init__(self, d: Dict[str, Any]):
        self.__dict__ = d
    
    def __repr__(self) -> str:
        return f"objectview({self.__dict__})"


class Config:
    """
    Hierarchical configuration manager.
    
    Priority (highest to lowest):
        1. CLI arguments
        2. YAML config file
        3. Default values
    
    Attributes:
        All configuration keys are accessible as attributes.
    
    Example:
        config = Config.from_args(args)
        print(config.learning_rate)  # 0.001
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with optional config dictionary."""
        self._config = DEFAULTS.copy()
        if config_dict:
            self._config.update(config_dict)
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            return self._config[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._config[name] = value
    
    def __repr__(self) -> str:
        return f"Config({self._config})"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with optional default."""
        return self._config.get(key, default)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update config with dictionary."""
        self._config.update(updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return config as dictionary."""
        return self._config.copy()
    
    def to_objectview(self) -> objectview:
        """Convert to an objectview (attribute access to the keys)."""
        return objectview(self._config.copy())
    
    def save(self, path: Path) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load config from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}
        
        return cls(yaml_config)
    
    @classmethod
    def from_args(cls, args) -> "Config":
        """
        Create config from argparse Namespace.
        
        CLI arguments override YAML config and defaults.
        """
        config = cls()
        
        # Load YAML config if specified
        if hasattr(args, "config") and args.config:
            yaml_config = cls.from_yaml(args.config)
            config.update(yaml_config.to_dict())
        
        # Override with CLI arguments (non-None values only)
        cli_mapping = {
            "mode": "mode",
            "checkpoint": "checkpoint_path",
            "dataset": "dataset_path",
            "output_dir": "output_dir",
            "device": "device",
            "batch_size": "batch_size",
            "epochs": "epochs",
            "learning_rate": "learning_rate",
            "hidden_dim": "hidden_dim",
            "num_layers": "num_layers",
            "dropout": "dropout_rate",
            "weight_decay": "weight_decay",
            "patience": "patience",
            "attention_freq": "attention_freq",
            "damage_loss_weight": "damage_loss_weight",
            "local_attention_enabled": "local_attention_enabled",
            "alpha_global": "alpha_global",
            "global_attention_enabled": "global_attention_enabled",
        }
        
        for cli_key, config_key in cli_mapping.items():
            if hasattr(args, cli_key):
                value = getattr(args, cli_key)
                if value is not None:
                    config._config[config_key] = value
        
        # Boolean flags: for argparse.BooleanOptionalAction we get None when
        # not passed, and explicit True/False when passed on CLI. Apply only
        # explicit values so YAML/default behavior is preserved.
        bool_flags = {
            "local_attention_enabled": "local_attention_enabled",
            # alpha_global (a float) must not be listed here: this loop
            # applies bool(value); it is handled in cli_mapping above
            "global_attention_enabled": "global_attention_enabled",
            "two_stage": "two_stage_enabled",
            "time_injection": "time_injection_enabled",
        }
        for cli_key, config_key in bool_flags.items():
            if hasattr(args, cli_key):
                value = getattr(args, cli_key)
                if value is not None:
                    config._config[config_key] = bool(value)
        
        return config


def load_config(
    config_path: Optional[Path] = None,
    cli_args: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Load configuration with proper priority.
    
    Args:
        config_path: Path to YAML config file
        cli_args: Dictionary of CLI argument overrides
    
    Returns:
        Merged Config object
    """
    config = Config()
    
    if config_path:
        yaml_config = Config.from_yaml(config_path)
        config.update(yaml_config.to_dict())
    
    if cli_args:
        # Filter None values
        cli_args = {k: v for k, v in cli_args.items() if v is not None}
        config.update(cli_args)
    
    return config


def get_project_paths() -> Dict[str, str]:
    """Get project paths (call at runtime, not module level)."""
    root_dir = os.getcwd()
    return {
        "root_dir": root_dir,
        "dataset_dir": os.path.join(root_dir, "datasets"),
        "data_path": os.path.join(root_dir, "data", "processed"),
        "results_dir": os.path.join(root_dir, "results"),
    }
