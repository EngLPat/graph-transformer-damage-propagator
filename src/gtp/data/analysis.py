"""
Dataset analysis utilities.

Functions for analyzing node features and classifying damage states.
"""

import torch


def analyze_node_features(dataset, rank: int = 0, verbose: bool = False) -> None:
    """
    Analyze structure and statistics of node features for time-dependent damage data.
    
    Args:
        dataset: Collection of PyTorch Geometric Data objects
        rank: DDP rank (only rank 0 prints)
        verbose: Whether to print detailed analysis
    """
    if rank != 0:
        return
    
    sample_data = dataset[0]
    
    if verbose:
        print(f"\\nNode features: {sample_data.x.shape}, Target: {sample_data.y.shape}")
        
        if hasattr(sample_data, "delta_t"):
            print(f"Time: delta_t={sample_data.delta_t.item():.6f}", end="")
            if hasattr(sample_data, "current_time"):
                print(f", current_time={sample_data.current_time.item():.6f}")
            else:
                print()


def classify_damage_states(accumulated_damage: torch.Tensor) -> torch.Tensor:
    """
    2-state classification using accumulated damage.
    
    Args:
        accumulated_damage: [N] cumulative damage in [0, 1]
    
    Returns:
        Tensor [N] with values {0=active, 1=failed}
    """
    if accumulated_damage.dim() > 1:
        accumulated_damage = accumulated_damage.squeeze(-1)
    
    threshold = 0.95  # Damage threshold for active/failed classification
    states = (accumulated_damage > threshold).long()
    
    return states
