"""
GTP (Graph Transformer Propagator) - Damage Evolution Prediction in Mesh-Based Systems.

src.gtp holds the model, training and evaluation code; src.preprocessing holds
the finite-element pipeline that generates and processes the training data.

Usage:
    # Training with DDP (4 GPUs):
    torchrun --nproc_per_node=4 --standalone -m src.gtp --mode train

    # Single GPU training:
    python -m src.gtp --mode train --epochs 100

    # Testing:
    python -m src.gtp --mode test --checkpoint results/xxx/best_models/model.pt
"""

__version__ = "2.0.0"
__author__ = "Luca Patrignani"
__email__ = "l.patrignani@imperial.ac.uk"

# Main module is src.gtp
from .gtp import *

__all__ = ['gtp']
