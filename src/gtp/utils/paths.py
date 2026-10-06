"""
Path management utilities.

Handles results folder creation and project path resolution.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ..config import DEFAULTS


def create_results_folder(prefix: str = None, mode: str = "train") -> Path:
    """
    Create a unique results folder with standardized subdirectory structure.
    
    Directory Structure:
        ./results/{prefix}_{timestamp}/
            ├── best_models/
            ├── hyperparameter_optimization_results/
            │   └── best_hyperparameters.json
            ├── post_training_results/
            ├── test_results/
            └── training_results/
    
    Args:
        prefix: Optional prefix for folder name (defaults to config value)
        mode: Execution mode ('train' or 'test')
    
    Returns:
        Path to the created folder
    """
    if prefix is None:
        prefix = DEFAULTS["results_folder_prefix"]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"{prefix}_{timestamp}"
    full_path = Path(os.getcwd()) / "results" / result_dir
    
    # Create main directory
    full_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        "best_models",
        "hyperparameter_optimization_results",
        "post_training_results",
        "test_results",
        "training_results",
    ]
    
    for subdir in subdirs:
        (full_path / subdir).mkdir(exist_ok=True)
    
    return full_path


def save_hyperparameters(
    output_dir: Path,
    config,
    additional_info: Optional[Dict] = None,
    filename: str = "best_hyperparameters.json"
) -> Path:
    """
    Save hyperparameters to JSON file in the results directory.
    
    Args:
        output_dir: Results directory path
        config: Configuration object with hyperparameters
        additional_info: Optional additional information to include
        filename: Name of the output file
        
    Returns:
        Path to the saved file
    """
    hp_dir = output_dir / "hyperparameter_optimization_results"
    hp_dir.mkdir(exist_ok=True)
    
    output_path = hp_dir / filename
    
    # Extract hyperparameters from config
    hyperparams = {
        "model": {
            "num_layers": getattr(config, 'num_layers', DEFAULTS['num_layers']),
            "hidden_dim": getattr(config, 'hidden_dim', DEFAULTS['hidden_dim']),
            "dropout_rate": getattr(config, 'dropout_rate', DEFAULTS['dropout_rate']),
            "attention_freq": getattr(config, 'attention_freq', DEFAULTS['attention_freq']),
        },
        "training": {
            "epochs": getattr(config, 'epochs', DEFAULTS['epochs']),
            "batch_size": getattr(config, 'batch_size', DEFAULTS['batch_size']),
            "learning_rate": getattr(config, 'learning_rate', DEFAULTS['learning_rate']),
            "weight_decay": getattr(config, 'weight_decay', DEFAULTS['weight_decay']),
            "optimizer": getattr(config, 'optimizer', DEFAULTS['optimizer']),
            "scheduler": getattr(config, 'scheduler', DEFAULTS['scheduler']),
            "patience": getattr(config, 'patience', DEFAULTS['patience']),
        },
        "data": {
            "train_ratio": DEFAULTS['train_ratio'],
            "val_ratio": DEFAULTS['val_ratio'],
            "test_ratio": DEFAULTS['test_ratio'],
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
        }
    }
    
    if additional_info:
        hyperparams["additional"] = additional_info
    
    with open(output_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    return output_path


def load_hyperparameters(path: Path) -> Dict:
    """
    Load hyperparameters from JSON file.
    
    Args:
        path: Path to hyperparameters file or directory containing it
        
    Returns:
        Dictionary of hyperparameters
    """
    path = Path(path)
    
    # If path is a directory, look for the file
    if path.is_dir():
        hp_file = path / "hyperparameter_optimization_results" / "best_hyperparameters.json"
        if not hp_file.exists():
            hp_file = path / "best_hyperparameters.json"
    else:
        hp_file = path
    
    if not hp_file.exists():
        raise FileNotFoundError(f"Hyperparameters file not found: {hp_file}")
    
    with open(hp_file, 'r') as f:
        return json.load(f)


def get_project_paths() -> Dict[str, str]:
    """
    Get project paths (call at runtime, not module level).
    
    Returns:
        Dictionary with project paths
    """
    root_dir = os.getcwd()
    return {
        "root_dir": root_dir,
        "dataset_dir": os.path.join(root_dir, "datasets"),
        "data_path": os.path.join(root_dir, "data", "processed"),
        "results_dir": os.path.join(root_dir, "results"),
    }
