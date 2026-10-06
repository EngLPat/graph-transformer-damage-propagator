"""
Loss functions for damage prediction.

Contains the log-space Huber loss used for the rate heads.
"""

import torch


def huber_log_loss(
    pred_rate: torch.Tensor,
    actual_rates: torch.Tensor,
    delta: float = 1.0,
    epsilon: float = 1e-6,
    mask: torch.Tensor = None,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Huber loss in log-space for robust training with damage rates.

    Args:
        pred_rate: Predicted damage rates
        actual_rates: Ground truth damage rates
        delta: Huber loss threshold
        epsilon: Small value for numerical stability
        mask: Optional boolean mask. If provided, only compute loss where mask=True.
        weights: Optional per-node weights (same shape as pred_rate).
                 If provided, compute weighted mean instead of uniform mean.
                 Useful for upweighting active nodes.

    Returns:
        Scalar loss value
    """
    log_pred = torch.log(pred_rate + epsilon)
    log_actual = torch.log(actual_rates + epsilon)
    log_diff = log_pred - log_actual

    loss = torch.where(
        torch.abs(log_diff) <= delta,
        0.5 * log_diff ** 2,
        delta * (torch.abs(log_diff) - 0.5 * delta)
    )
    
    # Apply mask if provided
    if mask is not None:
        if mask.sum() > 0:
            if weights is not None:
                # Weighted mean over masked nodes
                w = weights[mask]
                return (loss[mask] * w).sum() / w.sum()
            return loss[mask].mean()
        else:
            # All nodes are masked out - return 0 loss
            return torch.tensor(0.0, device=loss.device)
    
    if weights is not None:
        return (loss * weights).sum() / weights.sum()
    
    return loss.mean()
