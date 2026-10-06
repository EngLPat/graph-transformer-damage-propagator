"""
Evaluation metrics for damage prediction.

Standard regression metrics for model evaluation with a regime-aware breakdown.
"""

import torch
from typing import Dict, Optional


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Root Mean Square Error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        RMSE value
    """
    if len(predictions) == 0:
        return float('nan')
    mse = torch.mean((predictions - targets) ** 2)
    return torch.sqrt(mse).item()


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        MAE value
    """
    if len(predictions) == 0:
        return float('nan')
    return torch.mean(torch.abs(predictions - targets)).item()


def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate R² (coefficient of determination).
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        R² score
    """
    if len(predictions) == 0:
        return float('nan')
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    if ss_tot < 1e-12:
        return float('nan')
    return (1 - ss_res / ss_tot).item()


def relative_error(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> float:
    """
    Calculate mean relative error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        epsilon: Small value to avoid division by zero
    
    Returns:
        Mean relative error
    """
    if len(predictions) == 0:
        return float('nan')
    rel_err = torch.abs((predictions - targets) / (targets + epsilon))
    return torch.mean(rel_err).item()


def log_space_rmse(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    RMSE in log-space, matching the training loss formulation.
    
    Args:
        predictions: Predicted rate values
        targets: Ground truth rate values
        epsilon: Small value for numerical stability
    
    Returns:
        Log-space RMSE
    """
    if len(predictions) == 0:
        return float('nan')
    log_pred = torch.log(predictions.clamp(min=0) + epsilon)
    log_targ = torch.log(targets.clamp(min=0) + epsilon)
    return torch.sqrt(torch.mean((log_pred - log_targ) ** 2)).item()


def compute_all_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        Dictionary of metric names and values
    """
    return {
        "rmse": rmse(predictions, targets),
        "mae": mae(predictions, targets),
        "r2_score": r2_score(predictions, targets),
        "relative_error": relative_error(predictions, targets),
    }


def compute_regime_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    shock_threshold: float = 100.0,
    accumulated_damage: Optional[torch.Tensor] = None,
    delta_ts: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics broken down by damage regime.
    
    When *accumulated_damage* and *delta_ts* are provided and
    ``DEFAULTS["shock_gt_mode"] == "damage"``, the "shock" regime is defined
    as nodes whose damage transitions to >=0.99 during this timestep
    (crack-front definition).  Otherwise falls back to rate > shock_threshold.
    
    Regimes:
        - all:           Every node (original metrics)
        - steady:        Non-shock nodes  (matches training loss)
        - active_steady: Steady nodes with GT rate > 0.001      (non-trivial predictions)
        - shock:         Crack-front nodes or rate > shock_threshold
        - zero:          Nodes with GT rate < 0.001              (quiescent)
    
    Args:
        predictions: Predicted rate values
        targets: Ground truth rate values
        shock_threshold: Rate threshold separating steady from shock
        accumulated_damage: [N] Current damage D(t) for damage-based GT
        delta_ts: [B] or [N] per-batch delta_t values
    
    Returns:
        Nested dict: {regime_name: {metric_name: value}}
    """
    from ..config.defaults import DEFAULTS
    results = {}
    
    # ── Determine shock mask ──────────────────────────────────────────
    shock_gt_mode = DEFAULTS.get("shock_gt_mode", "damage")
    damage_thresh = DEFAULTS.get("shock_damage_threshold", 0.90)
    
    if shock_gt_mode == "damage" and accumulated_damage is not None and len(accumulated_damage) == len(targets) and delta_ts is not None and len(delta_ts) > 0:
        dt = float(delta_ts.view(-1)[0].item())
        D_next = (accumulated_damage + targets * dt).clamp(max=1.0)
        shock_mask = (D_next >= damage_thresh) & (accumulated_damage < damage_thresh)
    else:
        shock_mask = targets > shock_threshold
    
    # Define masks
    masks = {
        "all": torch.ones(len(targets), dtype=torch.bool),
        "steady": ~shock_mask,
        "active_steady": (~shock_mask) & (targets > 0.001),
        "shock": shock_mask,
        "zero": targets < 0.001,
    }
    
    for regime_name, mask in masks.items():
        p = predictions[mask]
        t = targets[mask]
        n = mask.sum().item()
        
        regime_metrics = {
            "n_nodes": n,
            "frac": n / len(targets) if len(targets) > 0 else 0.0,
        }
        
        if n > 0:
            regime_metrics["rmse"] = rmse(p, t)
            regime_metrics["mae"] = mae(p, t)
            regime_metrics["r2_score"] = r2_score(p, t)
            regime_metrics["log_rmse"] = log_space_rmse(p, t)
            regime_metrics["pred_mean"] = p.mean().item()
            regime_metrics["pred_std"] = p.std().item() if n > 1 else 0.0
            regime_metrics["target_mean"] = t.mean().item()
            regime_metrics["target_std"] = t.std().item() if n > 1 else 0.0
        else:
            for k in ["rmse", "mae", "r2_score", "log_rmse",
                       "pred_mean", "pred_std", "target_mean", "target_std"]:
                regime_metrics[k] = float('nan')

        results[regime_name] = regime_metrics

    return results


def _flat(x):
    """Flatten to 1D; object arrays (per-frame arrays of different sizes) are
    concatenated as float32."""
    import numpy as np
    arr = np.asarray(x)
    if arr.dtype == object:
        return np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in arr])
    return arr.ravel()


def _rollout_row(pred, tgt, name, clip=None):
    import numpy as np
    pred, tgt = _flat(pred), _flat(tgt)
    if clip is not None:
        pred = np.clip(pred, -clip, clip)
        tgt = np.clip(tgt, -clip, clip)
    ss_res = float(np.sum((tgt - pred) ** 2))
    ss_tot = float(np.sum((tgt - tgt.mean()) ** 2))
    return {
        "name": name,
        "n": int(pred.size),
        "r2": 1.0 - ss_res / (ss_tot + 1e-12),
        "rmse": float(np.sqrt(np.mean((tgt - pred) ** 2))),
        "mae": float(np.mean(np.abs(tgt - pred))),
        "tgt_mean": float(tgt.mean()),
        "tgt_std": float(tgt.std()),
    }


def rollout_metrics(rollout, clip=None):
    """R², RMSE and MAE of the autoregressive rollout.

    ``rollout`` maps the rollout fields (as returned by the evaluator or saved
    in rollout_arrays.npz) to arrays. Rows: the failure index over all frames
    and at the final frame of each simulation, and the damage variable over all
    frames (two-stage models).
    """
    import numpy as np
    rows = []
    if "predicted_damage" in rollout and "ground_truth_damage" in rollout:
        pred = np.asarray(rollout["predicted_damage"])
        tgt = np.asarray(rollout["ground_truth_damage"])
        rows.append(_rollout_row(pred, tgt, "FI integrated (all)", clip))
        if "sim_ids_per_frame" in rollout:
            sim_ids = np.asarray(rollout["sim_ids_per_frame"]).ravel()
            last = np.zeros_like(sim_ids, dtype=bool)
            seen = set()
            for i in range(len(sim_ids) - 1, -1, -1):
                if sim_ids[i] not in seen:
                    last[i] = True
                    seen.add(sim_ids[i])
            rows.append(_rollout_row(pred[last], tgt[last], "FI integrated (final frame)", clip))
    if "predicted_matrix_damage" in rollout and "ground_truth_matrix_damage" in rollout:
        rows.append(_rollout_row(np.asarray(rollout["predicted_matrix_damage"]),
                                 np.asarray(rollout["ground_truth_matrix_damage"]),
                                 "damage ω (all)", clip))
    return rows


def format_rollout_metrics(rows):
    """One line per rollout_metrics row."""
    return [f"{r['name']:<35s} N={r['n']:>10d}  R²={r['r2']:+.4f}  "
            f"RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}" for r in rows]
