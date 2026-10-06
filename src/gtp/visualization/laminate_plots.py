"""
Per-ply (through-thickness) and regression analysis plots for laminate GNN.

Adds to the standard test visualization:
  1. Per-z-level spatial damage comparison (pred vs GT per ply surface)
  2. Regression scatter plots: predicted vs target rates, coloured by regime
  3. Per-ply error metrics summary bar chart

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict

from .plotting import safe_to_numpy


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _get_z_levels(pos_3d: np.ndarray, tol: float = 1e-5):
    """Return sorted unique z-values and per-node z-level index."""
    z = pos_3d[:, 2]
    z_unique = np.unique(np.round(z / tol) * tol)
    z_idx = np.zeros(len(z), dtype=int)
    for li, zv in enumerate(z_unique):
        mask = np.abs(z - zv) < tol
        z_idx[mask] = li
    return z_unique, z_idx


# ─────────────────────────────────────────────────────────────────────
# Plot 3: Regression scatter of the rates (overall + per-ply)
# ─────────────────────────────────────────────────────────────────────

def plot_rate_regression(
    predictions: np.ndarray,
    targets: np.ndarray,
    mesh_pos: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    title: str = "Rate Regression",
    rate_cap: float = 10.0,
) -> Dict[str, float]:
    """Multi-panel regression scatter for damage rates.

    Panel layout (2x2):
      [0,0] All rates (clipped to rate_cap): scatter with density
      [0,1] Active rates only (GT > threshold): linear scale
      [1,0] Per-z-level R² bar chart (if mesh_pos provided)
      [1,1] Residual vs predicted (bias check)

    Returns dict of per-ply and overall R² scores.
    """
    predictions = safe_to_numpy(predictions)
    targets = safe_to_numpy(targets)

    # Clip to rate_cap, as the training loss does, so that the largest rates
    # do not dominate the scale
    predictions = np.clip(predictions, -rate_cap, rate_cap)
    targets = np.clip(targets, -rate_cap, rate_cap)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    results = {}

    # ── (0,0) All rates scatter ──
    ax = axes[0, 0]
    # Use hexbin for density
    valid = np.isfinite(predictions) & np.isfinite(targets)
    p, t = predictions[valid], targets[valid]
    ax.hexbin(t, p, gridsize=80, cmap='Blues', mincnt=1, linewidths=0.2)
    lims = [min(t.min(), p.min()), max(t.max(), p.max())]
    ax.plot(lims, lims, 'r--', lw=1.5, label='y=x')
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2_all = 1 - ss_res / (ss_tot + 1e-10)
    results['R2_all'] = r2_all
    ax.set_xlabel('Target rate'); ax.set_ylabel('Predicted rate')
    ax.set_title(f'All nodes  R²={r2_all:.4f}  N={len(t)}', fontsize=10)
    ax.legend(fontsize=9)

    # ── (0,1) Active rates only ──
    ax = axes[0, 1]
    threshold = 0.001
    active = np.abs(targets) > threshold
    if active.sum() > 0:
        pa, ta = predictions[active], targets[active]
        ax.hexbin(ta, pa, gridsize=60, cmap='Oranges', mincnt=1, linewidths=0.2)
        lims_a = [min(ta.min(), pa.min()), max(ta.max(), pa.max())]
        ax.plot(lims_a, lims_a, 'r--', lw=1.5)
        ss_res_a = np.sum((ta - pa) ** 2)
        ss_tot_a = np.sum((ta - ta.mean()) ** 2)
        r2_active = 1 - ss_res_a / (ss_tot_a + 1e-10)
        results['R2_active'] = r2_active
        mae_active = float(np.mean(np.abs(ta - pa)))
        results['MAE_active'] = mae_active
        ax.set_title(f'Active nodes (|rate|>{threshold})  R²={r2_active:.4f}\n'
                     f'MAE={mae_active:.4f}  N={len(ta)}', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No active nodes', transform=ax.transAxes, ha='center')
    ax.set_xlabel('Target rate'); ax.set_ylabel('Predicted rate')

    # ── (1,0) Per-z-level R² ──
    ax = axes[1, 0]
    if mesh_pos is not None:
        z_unique, z_idx = _get_z_levels(mesh_pos)
        r2_per_z = []
        labels = []
        for li, zv in enumerate(z_unique):
            mask = z_idx == li
            tl, pl = targets[mask], predictions[mask]
            ss_r = np.sum((tl - pl) ** 2)
            ss_t = np.sum((tl - tl.mean()) ** 2)
            r2_l = 1 - ss_r / (ss_t + 1e-10) if ss_t > 0 else 0.0
            r2_per_z.append(r2_l)
            labels.append(f"z={zv:.4g}")
            results[f'R2_z{li}'] = r2_l

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(r2_per_z)))
        bars = ax.bar(range(len(r2_per_z)), r2_per_z, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, fontsize=9)
        ax.set_ylabel('R²')
        ax.set_title('R² per z-level (ply surface)', fontsize=10)
        ax.set_ylim(min(0, min(r2_per_z) - 0.05), 1.05)
        ax.axhline(r2_all, color='red', ls='--', lw=1, label=f'Overall R²={r2_all:.3f}')
        ax.legend(fontsize=9)
        for bar, val in zip(bars, r2_per_z):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No mesh_pos available', transform=ax.transAxes, ha='center')

    # ── (1,1) Residual vs predicted ──
    ax = axes[1, 1]
    residual = predictions - targets
    ax.hexbin(predictions, residual, gridsize=60, cmap='Purples', mincnt=1, linewidths=0.2)
    ax.axhline(0, color='red', ls='--', lw=1.5)
    ax.set_xlabel('Predicted rate'); ax.set_ylabel('Residual (pred - target)')
    ax.set_title('Residual analysis (bias check)', fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return results


# ─────────────────────────────────────────────────────────────────────
# Plot 4: Damage regression (accumulated damage, not rates)
# ─────────────────────────────────────────────────────────────────────

def plot_damage_regression(
    pred_damage: np.ndarray,
    true_damage: np.ndarray,
    mesh_pos: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    title: str = "Accumulated Damage Regression",
) -> Dict[str, float]:
    """Scatter of accumulated damage: pred vs GT, with per-ply breakdown."""
    pred_damage = safe_to_numpy(pred_damage)
    true_damage = safe_to_numpy(true_damage)

    has_z = mesh_pos is not None
    n_cols = 3 if has_z else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5.5))
    results = {}

    # Auto-detect range (supports FI > 1)
    val_max = max(1.0, float(np.max(true_damage)), float(np.max(pred_damage)))
    lim_lo, lim_hi = -0.05 * val_max, 1.05 * val_max

    # ── Overall scatter ──
    ax = axes[0]
    ax.hexbin(true_damage, pred_damage, gridsize=80, cmap='Blues', mincnt=1, linewidths=0.2)
    ax.plot([0, val_max], [0, val_max], 'r--', lw=1.5, label='y=x')
    ss_res = np.sum((true_damage - pred_damage) ** 2)
    ss_tot = np.sum((true_damage - true_damage.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    results['R2_damage'] = r2
    ax.set_xlabel('GT'); ax.set_ylabel('Pred')
    ax.set_title(f'All nodes  R²={r2:.4f}', fontsize=10)
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect('equal')
    ax.legend()

    # ── Active nodes only ──
    ax = axes[1]
    active = true_damage > 0.01
    if active.sum() > 0:
        pd_a = pred_damage[active]
        td_a = true_damage[active]
        ax.hexbin(td_a, pd_a, gridsize=60, cmap='Oranges', mincnt=1, linewidths=0.2)
        a_max = max(float(td_a.max()), float(pd_a.max()), 1.0)
        ax.plot([0, a_max], [0, a_max], 'r--', lw=1.5)
        ss_r = np.sum((td_a - pd_a) ** 2)
        ss_t = np.sum((td_a - td_a.mean()) ** 2)
        r2_a = 1 - ss_r / (ss_t + 1e-10)
        results['R2_damage_active'] = r2_a
        ax.set_title(f'Active nodes (>0.01)  R²={r2_a:.4f}  N={active.sum()}', fontsize=10)
    ax.set_xlabel('GT'); ax.set_ylabel('Pred')
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect('equal')

    # ── Per-ply R² ──
    if has_z:
        ax = axes[2]
        z_unique, z_idx = _get_z_levels(mesh_pos)
        r2_per_z = []
        labels = []
        for li, zv in enumerate(z_unique):
            mask = z_idx == li
            tl = true_damage[mask]
            pl = pred_damage[mask]
            ss_r = np.sum((tl - pl) ** 2)
            ss_t = np.sum((tl - tl.mean()) ** 2)
            r2_l = 1 - ss_r / (ss_t + 1e-10) if ss_t > 0 else 0.0
            r2_per_z.append(r2_l)
            labels.append(f"z={zv:.4g}")
            results[f'R2_damage_z{li}'] = r2_l
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(r2_per_z)))
        bars = ax.bar(range(len(r2_per_z)), r2_per_z, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, fontsize=9)
        ax.set_ylabel('R²')
        ax.set_title('R² per z-level', fontsize=10)
        ax.set_ylim(min(0, min(r2_per_z) - 0.05), 1.05)
        ax.axhline(r2, color='red', ls='--', lw=1, label=f'Overall R²={r2:.3f}')
        ax.legend(fontsize=9)
        for bar, val in zip(bars, r2_per_z):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return results


# ─────────────────────────────────────────────────────────────────────
# Plot 5: Per-ply error metrics summary
# ─────────────────────────────────────────────────────────────────────

def plot_per_ply_metrics_summary(
    predictions: np.ndarray,
    targets: np.ndarray,
    mesh_pos: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Per-Ply Error Metrics",
    rate_cap: float = 10.0,
) -> Dict[str, float]:
    """Bar charts of RMSE, MAE, and R² per z-level."""
    predictions = np.clip(safe_to_numpy(predictions), -rate_cap, rate_cap)
    targets = np.clip(safe_to_numpy(targets), -rate_cap, rate_cap)

    z_unique, z_idx = _get_z_levels(mesh_pos)
    n_levels = len(z_unique)

    rmse_per_z, mae_per_z, r2_per_z = [], [], []
    labels = []
    results = {}

    for li, zv in enumerate(z_unique):
        mask = z_idx == li
        tl, pl = targets[mask], predictions[mask]
        err = pl - tl
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        ss_r = np.sum(err ** 2)
        ss_t = np.sum((tl - tl.mean()) ** 2)
        r2 = 1 - ss_r / (ss_t + 1e-10) if ss_t > 0 else 0.0
        rmse_per_z.append(rmse)
        mae_per_z.append(mae)
        r2_per_z.append(r2)
        labels.append(f"z={zv:.4g}")
        results[f'RMSE_z{li}'] = rmse
        results[f'MAE_z{li}'] = mae
        results[f'R2_z{li}'] = r2

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_levels))
    x = range(n_levels)

    for ax, vals, ylabel, name in [
        (axes[0], rmse_per_z, 'RMSE', 'RMSE'),
        (axes[1], mae_per_z, 'MAE', 'MAE'),
        (axes[2], r2_per_z, 'R²', 'R²'),
    ]:
        bars = ax.bar(x, vals, color=colors)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{name} per z-level', fontsize=11)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.4f}', ha='center', va='bottom', fontsize=8)
        if name == 'R²':
            ax.set_ylim(min(0, min(vals) - 0.05), 1.05)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return results


# ─────────────────────────────────────────────────────────────────────
# Master function: generate all laminate analysis plots
# ─────────────────────────────────────────────────────────────────────

def generate_laminate_test_plots(
    test_dataset: list,
    predictions: np.ndarray,
    targets: np.ndarray,
    rollout_results: Optional[dict],
    output_dir: Path,
    n_spatial_frames: int = 3,
) -> Dict[str, float]:
    """Generate all laminate-specific test plots.

    Called from ``_run_testing`` after the standard plots.

    Parameters
    ----------
    test_dataset : list of PyG Data
        Test samples (need mesh_pos, cells).
    predictions : np.ndarray [total_nodes]
        Concatenated predicted rates across all test samples.
    targets : np.ndarray [total_nodes]
        Concatenated target rates.
    rollout_results : dict or None
        From ``evaluator.evaluate_with_visualization``.
    output_dir : Path
        Where to save plots.
    n_spatial_frames : int
        Number of rollout frames to make per-ply spatial plots for.
    """
    laminate_dir = output_dir / "laminate_analysis"
    laminate_dir.mkdir(parents=True, exist_ok=True)
    print("\n[Laminate Through-Thickness Analysis]")
    print(f"  Output: {laminate_dir}")

    all_metrics = {}

    # Read rate_cap from env (same source as training loss)
    import os
    rate_cap = float(os.environ.get("GTP_RATE_CAP", "10.0"))
    print(f"  Rate cap: {rate_cap}")

    # Field type: cohesive-laminate datasets carry n_ply_nodes and predict FI
    sample_0 = test_dataset[0]
    is_fi = hasattr(sample_0, 'n_ply_nodes') and sample_0.n_ply_nodes is not None
    field_name = "FI" if is_fi else "damage"
    print(f"  Field type: {field_name}")

    # ── Check if we have multiple z-levels ──
    pos_0 = sample_0.mesh_pos.cpu().numpy() if torch.is_tensor(sample_0.mesh_pos) else sample_0.mesh_pos
    z_unique, _ = _get_z_levels(pos_0)
    if len(z_unique) < 2:
        print("  Single z-level; skipping laminate analysis")
        return all_metrics

    print(f"  Detected {len(z_unique)} z-levels: {z_unique}")

    # ── Rollout-based spatial plots ──
    # Per-ply spatial frames are generated as element-hex GIFs in the
    # per-simulation damage_evolution/ directory (fi/, fi_rate/, damage/,
    # damage_rate/ sub-folders).  Only regression and summary plots are
    # generated here.
    if rollout_results is not None:
        pred_dmg = rollout_results.get('predicted_damage')
        gt_dmg = rollout_results.get('ground_truth_damage')

        if pred_dmg is not None and gt_dmg is not None:
            # ── FI / accumulated-field regression (last frame) ──
            try:
                d_metrics = plot_damage_regression(
                    pred_dmg[-1], gt_dmg[-1], mesh_pos=pos_0,
                    save_path=laminate_dir / f"{field_name.lower()}_regression_last_frame.png",
                    title=f"Accumulated {field_name} Regression, Last Frame",
                )
                all_metrics.update({f'dmg_{k}': v for k, v in d_metrics.items()})
                print(f"  {field_name} regression (last frame): "
                      f"R²={d_metrics.get('R2_damage', 0):.4f}")
            except Exception as e:
                print(f"  {field_name} regression failed: {e}")

        # ── Two-stage: matrix damage regression ─────────────────────────
        pred_mat_dmg = rollout_results.get('predicted_matrix_damage')
        gt_mat_dmg = rollout_results.get('ground_truth_matrix_damage')

        if pred_mat_dmg is not None and gt_mat_dmg is not None:
            # ── Matrix damage regression (last frame) ──
            try:
                md_metrics = plot_damage_regression(
                    pred_mat_dmg[-1], gt_mat_dmg[-1], mesh_pos=pos_0,
                    save_path=laminate_dir / "matrix_damage_regression_last_frame.png",
                    title="Matrix Damage Regression, Last Frame",
                )
                all_metrics.update({f'mat_dmg_{k}': v for k, v in md_metrics.items()})
                print(f"  Matrix damage regression (last frame): "
                      f"R²={md_metrics.get('R2_damage', 0):.4f}, "
                      f"R²_active={md_metrics.get('R2_damage_active', 0):.4f}")
            except Exception as e:
                print(f"  Matrix damage regression failed: {e}")

            # ── Matrix damage per-ply metrics summary ──
            try:
                md_ply = plot_per_ply_metrics_summary(
                    pred_mat_dmg[-1], gt_mat_dmg[-1], pos_0,
                    save_path=laminate_dir / "per_ply_matrix_damage_metrics.png",
                    title="Per-Ply Matrix Damage Error Metrics, Last Frame",
                    rate_cap=1.0,
                )
                all_metrics.update({f'mat_dmg_ply_{k}': v for k, v in md_ply.items()})
                print("  Per-ply matrix damage metrics saved")
            except Exception as e:
                print(f"  Per-ply matrix damage metrics failed: {e}")

    # ── Save all metrics to JSON ──
    import json
    # Convert numpy float32 to Python float for JSON serialization
    serializable = {k: float(v) for k, v in all_metrics.items()}
    metrics_path = laminate_dir / "laminate_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    return all_metrics
