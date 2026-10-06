"""
Normalization utilities for damage prediction.

Handles various normalization schemes for node features, edge features,
and damage rate targets with special handling for catastrophic failure patterns.
"""

import torch
from typing import List

from ..config import DEFAULTS


def get_stats(data_list: List) -> List[torch.Tensor]:
    """
    Compute mean and std statistics for node features, edge features, and outputs.
    
    Args:
        data_list: List of PyTorch Geometric Data objects
    
    Returns:
        List of tensors: [mean_x, std_x, mean_edge, std_edge, mean_y, std_y]
    """
    mean_vec_x = torch.zeros(data_list[0].x.shape[1:])
    std_vec_x = torch.zeros(data_list[0].x.shape[1:])
    mean_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])
    std_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])
    mean_vec_y = torch.zeros(data_list[0].y.shape[1:])
    std_vec_y = torch.zeros(data_list[0].y.shape[1:])

    max_accumulations = DEFAULTS["normalization_max_accumulations"]
    eps = torch.tensor(DEFAULTS["normalization_variance_epsilon"])

    num_accs_x = 0
    num_accs_edge = 0
    num_accs_y = 0

    for dp in data_list:
        mean_vec_x += torch.sum(dp.x, dim=0)
        std_vec_x += torch.sum(dp.x ** 2, dim=0)
        num_accs_x += dp.x.shape[0]

        mean_vec_edge += torch.sum(dp.edge_attr, dim=0)
        std_vec_edge += torch.sum(dp.edge_attr ** 2, dim=0)
        num_accs_edge += dp.edge_attr.shape[0]

        mean_vec_y += torch.sum(dp.y, dim=0)
        std_vec_y += torch.sum(dp.y ** 2, dim=0)
        num_accs_y += dp.y.shape[0]

        if (num_accs_x > max_accumulations or
            num_accs_edge > max_accumulations or
            num_accs_y > max_accumulations):
            break

    mean_vec_x = mean_vec_x / num_accs_x
    std_vec_x = torch.maximum(torch.sqrt(std_vec_x / num_accs_x - mean_vec_x ** 2), eps)
    mean_vec_edge = mean_vec_edge / num_accs_edge
    std_vec_edge = torch.maximum(torch.sqrt(std_vec_edge / num_accs_edge - mean_vec_edge ** 2), eps)
    mean_vec_y = mean_vec_y / num_accs_y
    std_vec_y = torch.maximum(torch.sqrt(std_vec_y / num_accs_y - mean_vec_y ** 2), eps)

    return [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y]


def get_stats_improved(data_list: List, max_samples: int = None, rank: int = 0) -> List[torch.Tensor]:
    """
    Damage evolution specialized normalization for catastrophic failure patterns.
    
    Uses log-scale normalization to handle exponential damage growth.
    
    Args:
        data_list: List of PyTorch Geometric Data objects
        max_samples: Maximum samples to use (for memory efficiency)
        rank: DDP rank (only rank 0 prints)
    
    Returns:
        List of 8 tensors: stats for x, edge, progressive_y, catastrophic_y
    """
    all_x_values = []
    all_edge_values = []
    all_y_values = []
    
    if max_samples is not None and len(data_list) > max_samples:
        step = len(data_list) // max_samples
        sampled_data = data_list[::step][:max_samples]
    else:
        sampled_data = data_list
    
    for dp in sampled_data:
        x_sample = dp.x
        all_x_values.append(x_sample.flatten())
        all_edge_values.append(dp.edge_attr.flatten())
        all_y_values.append(dp.y.flatten())

    all_x = torch.cat(all_x_values)
    all_edge = torch.cat(all_edge_values)
    all_y = torch.cat(all_y_values)
    
    # Analyze damage distribution
    # torch.quantile() has a size limit (~2^24 elements); subsample if too large
    _MAX_QUANTILE_ELEMS = 10_000_000  # 10M is safe
    if all_y.numel() > _MAX_QUANTILE_ELEMS:
        _idx = torch.randperm(all_y.numel())[:_MAX_QUANTILE_ELEMS]
        _y_sample = all_y[_idx]
    else:
        _y_sample = all_y
    y_percentiles = torch.quantile(_y_sample, torch.tensor([0.50, 0.90, 0.95, 0.99, 0.999]))
    near_zero = torch.sum(torch.abs(all_y) < 1.0).item()
    catastrophic = torch.sum(all_y > y_percentiles[3]).item()
    
    if rank == 0:
        print(f"Damage rates: range=[{torch.min(all_y):.2f}, {torch.max(all_y):.2f}], "
              f"near-zero={100*near_zero/len(all_y):.1f}%, catastrophic={100*catastrophic/len(all_y):.2f}%")
    
    # Log-scale normalization
    epsilon = DEFAULTS["normalization_epsilon"]
    all_y_safe = torch.maximum(all_y, torch.tensor(epsilon))
    
    log_y = torch.log(all_y_safe + 1.0)
    log_mean = torch.mean(log_y)
    log_std = torch.std(log_y)
    
    p95 = y_percentiles[2]
    
    # Regime analysis
    progressive_values = all_y[all_y <= p95]
    high_damage_values = all_y[all_y > p95]
    
    mean_vec_y = log_mean.expand(data_list[0].y.shape[1:])
    std_vec_y = torch.maximum(log_std.expand(data_list[0].y.shape[1:]), torch.tensor(1e-8))
    
    # Node and edge stats
    mean_vec_x = torch.mean(all_x.view(-1, data_list[0].x.shape[1]), dim=0)
    std_vec_x = torch.maximum(
        torch.std(all_x.view(-1, data_list[0].x.shape[1]), dim=0),
        torch.tensor(1e-8)
    )
    mean_vec_edge = torch.mean(all_edge.view(-1, data_list[0].edge_attr.shape[1]), dim=0)
    std_vec_edge = torch.maximum(
        torch.std(all_edge.view(-1, data_list[0].edge_attr.shape[1]), dim=0),
        torch.tensor(1e-8)
    )
    
    # Dual-regime stats
    mean_progressive = torch.mean(progressive_values)
    std_progressive = torch.maximum(torch.std(progressive_values), torch.tensor(1e-8))
    
    if len(high_damage_values) > 0:
        mean_catastrophic = torch.mean(high_damage_values)
        std_catastrophic = torch.maximum(torch.std(high_damage_values), torch.tensor(1e-8))
    else:
        mean_catastrophic = mean_progressive
        std_catastrophic = std_progressive
    
    mean_vec_y_progressive = mean_progressive.expand(data_list[0].y.shape[1:])
    std_vec_y_progressive = std_progressive.expand(data_list[0].y.shape[1:])
    mean_vec_y_catastrophic = mean_catastrophic.expand(data_list[0].y.shape[1:])
    std_vec_y_catastrophic = std_catastrophic.expand(data_list[0].y.shape[1:])
    
    # Return dummy stats for y (using log-space loss)
    dummy_y_mean = torch.zeros(data_list[0].y.shape[1:])
    dummy_y_std = torch.ones(data_list[0].y.shape[1:])
    
    if rank == 0:
        print("Normalization: log-space for targets, standard for features")
    
    return [
        mean_vec_x, std_vec_x,
        mean_vec_edge, std_vec_edge,
        dummy_y_mean, dummy_y_std,
        dummy_y_mean, dummy_y_std
    ]


def normalize(to_normalize: torch.Tensor, mean_vec: torch.Tensor, std_vec: torch.Tensor) -> torch.Tensor:
    """Normalize tensor by subtracting mean and dividing by std."""
    return (to_normalize - mean_vec) / std_vec


def unnormalize(to_unnormalize: torch.Tensor, mean_vec: torch.Tensor, std_vec: torch.Tensor) -> torch.Tensor:
    """Reverse normalization to recover original scale."""
    return to_unnormalize * std_vec + mean_vec


def normalize_damage_rate(damage_rate: torch.Tensor, mean_vec: torch.Tensor, std_vec: torch.Tensor) -> torch.Tensor:
    """Normalize damage rate using log-scale transformation."""
    epsilon = 1e-6
    damage_rate_safe = torch.maximum(damage_rate, torch.tensor(epsilon, device=damage_rate.device))
    log_damage = torch.log(damage_rate_safe + 1.0)
    return (log_damage - mean_vec) / std_vec


def unnormalize_damage_rate(normalized_damage: torch.Tensor, mean_vec: torch.Tensor, std_vec: torch.Tensor) -> torch.Tensor:
    """Invert damage rate normalization to recover physical values."""
    log_damage = normalized_damage * std_vec + mean_vec
    damage_rate = torch.exp(log_damage) - 1.0
    return torch.maximum(damage_rate, torch.tensor(0.0, device=damage_rate.device))


def normalize_damage_rate_progressive(damage_rate: torch.Tensor, mean_vec: torch.Tensor, std_vec: torch.Tensor) -> torch.Tensor:
    """Linear normalization for progressive regime."""
    eps = 1e-8
    return (damage_rate - mean_vec) / (std_vec + eps)


def unnormalize_damage_rate_progressive(normalized_damage: torch.Tensor, mean_vec: torch.Tensor, std_vec: torch.Tensor) -> torch.Tensor:
    """Inverse of linear normalization for progressive regime."""
    damage_rate = normalized_damage * std_vec + mean_vec
    return torch.maximum(damage_rate, torch.tensor(0.0, device=damage_rate.device))


def normalize_damage_rate_catastrophic(damage_rate: torch.Tensor, mean_vec: torch.Tensor, std_vec: torch.Tensor) -> torch.Tensor:
    """Linear normalization for catastrophic regime."""
    return (damage_rate - mean_vec) / std_vec


def unnormalize_damage_rate_catastrophic(normalized_damage: torch.Tensor, mean_vec: torch.Tensor, std_vec: torch.Tensor) -> torch.Tensor:
    """Unnormalize catastrophic regime predictions."""
    damage_rate = normalized_damage * std_vec + mean_vec
    return torch.maximum(damage_rate, torch.tensor(0.0, device=damage_rate.device))


def normalize_damage_rate_catastrophic_binary(damage_rate: torch.Tensor, threshold: float = 100.0) -> torch.Tensor:
    """Binary normalization for catastrophic decoder."""
    return (damage_rate > threshold).float()


def unnormalize_damage_rate_catastrophic_binary(logits: torch.Tensor, threshold: float = 100.0) -> torch.Tensor:
    """Convert catastrophic decoder logits to damage rate estimate."""
    probs = torch.sigmoid(logits)
    return torch.where(
        probs > 0.5,
        torch.full_like(probs, threshold + 50.0),
        torch.zeros_like(probs)
    )
