"""
GTP: Graph Transformer Propagator for Damage Prediction.

Local-global graph transformer that propagates failure-index and damage rates
on finite-element meshes of composite plies and laminates, trained on
progressive-damage simulations and evaluated by autoregressive rollout.

Modules:
    config: Configuration management (YAML/CLI)
    data: Dataset loading and normalization
    models: Neural network architectures
    training: Training loops (single-GPU and DDP)
    evaluation: Testing and metrics
    visualization: Plotting and animation generation
    utils: Utilities (logging, GPU, paths)

Usage:
    # Training with DDP (4 GPUs):
    torchrun --nproc_per_node=4 --standalone -m src.gtp --mode train

    # Single GPU training:
    python -m src.gtp --mode train --epochs 100

    # Testing:
    python -m src.gtp --mode test --checkpoint path/to/model.pt

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

from .config import Config, load_config
from .models import FCLGA_GraphTransformer
from .training import Trainer
from .evaluation import Evaluator

__version__ = "2.0.0"
__all__ = [
    "Config",
    "load_config",
    "FCLGA_GraphTransformer",
    "Trainer",
    "Evaluator",
]
