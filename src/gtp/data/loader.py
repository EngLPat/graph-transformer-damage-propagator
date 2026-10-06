"""
Dataset loading utilities.

Handles loading PyTorch Geometric datasets and creating train/val/test splits.
"""

from pathlib import Path
from typing import List, Tuple, Optional

import torch


def load_dataset(path: str, rank: int = 0, verbose: bool = True) -> List:
    """
    Load dataset from file.
    
    Args:
        path: Path to the .pt dataset file
        rank: DDP rank (only rank 0 prints)
        verbose: Whether to print loading info
    
    Returns:
        List of PyTorch Geometric Data objects
    
    Raises:
        FileNotFoundError: If dataset file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    
    dataset = torch.load(path, weights_only=False)
    
    if rank == 0 and verbose:
        print(f"Dataset: {len(dataset)} samples, {dataset[0].x.shape[1]} node features, {dataset[0].edge_attr.shape[1]} edge features")
    
    return dataset


def get_data_splits(
    dataset: List,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    interleave_step: Optional[int] = 3,
    rank: int = 0
) -> Tuple[List, List, List]:
    """
    Split dataset into train/val/test sets.

    Splitting strategy (in order of priority):
      1. Multi-simulation dataset (>= 3 sims via sample_idx)
         → simulation-level split (whole sims stay together)
      2. Single-sim with interleave_step → time-series interleaved
      3. Fallback → sequential split by ratio

    Args:
        dataset: Full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        interleave_step: Step size for interleaved sampling (single-sim only)
        rank: DDP rank (only rank 0 prints)
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    import random as _rng
    from collections import defaultdict
    
    total_size = len(dataset)

    # ── 1. Multi-simulation → simulation-level split ─────────────────
    # Group transitions by sample_idx.  If >= 3 sims exist, whole
    # simulations are assigned to train/val/test so that the test set
    # contains complete trajectories (needed for autoregressive rollout)
    # and no data leakage between splits.
    sim_groups = defaultdict(list)
    for i, d in enumerate(dataset):
        sid = int(d.sample_idx.item()) if hasattr(d, 'sample_idx') else 0
        sim_groups[sid].append(i)
    
    sim_ids = sorted(sim_groups.keys())
    n_sims = len(sim_ids)
    
    if n_sims >= 3:
        _rng.seed(42)
        shuffled = list(sim_ids)
        _rng.shuffle(shuffled)
        
        n_train = max(1, int(n_sims * train_ratio))
        n_val = max(1, int(n_sims * val_ratio))
        if n_train + n_val >= n_sims:
            n_val = max(1, n_sims - n_train - 1)
        
        train_sims = sorted(shuffled[:n_train])
        val_sims = sorted(shuffled[n_train:n_train + n_val])
        test_sims = sorted(shuffled[n_train + n_val:])
        if not test_sims:
            test_sims = [val_sims[-1]]
        
        train_dataset = [dataset[i] for sid in train_sims for i in sim_groups[sid]]
        val_dataset   = [dataset[i] for sid in val_sims   for i in sim_groups[sid]]
        test_dataset  = [dataset[i] for sid in test_sims  for i in sim_groups[sid]]
        
        if rank == 0:
            print("\n=== SIMULATION-LEVEL SPLITTING ===")
            print(f"  Simulations: {n_sims}")
            print(f"  Train sims : {train_sims} ({len(train_dataset)} transitions)")
            print(f"  Val sims   : {val_sims} ({len(val_dataset)} transitions)")
            print(f"  Test sims  : {test_sims} ({len(test_dataset)} transitions)")
        
        return train_dataset, val_dataset, test_dataset
    
    # ── 2. Single-sim with interleave ────────────────────────────────
    if interleave_step:
        # Time-series interleaved sampling
        step = interleave_step
        train_indices = list(range(0, total_size, step))
        val_indices = list(range(1, total_size, step))
        test_indices = list(range(2, total_size, step))
        
        min_samples = min(len(train_indices), len(val_indices), len(test_indices))
        train_indices = train_indices[:min(len(train_indices), min_samples * 2)]
        val_indices = val_indices[:min_samples // 2]
        test_indices = test_indices[:min_samples // 2]
        
        if rank == 0:
            print(f"Data splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)} (interleave={step})")
        
        train_dataset = [dataset[i] for i in train_indices]
        val_dataset = [dataset[i] for i in val_indices]
        test_dataset = [dataset[i] for i in test_indices]
    else:
        # ── 3. Sequential splitting ──────────────────────────────────
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:train_size + val_size]
        test_dataset = dataset[train_size + val_size:]
        
        if rank == 0:
            print(f"Data splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset
