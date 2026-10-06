#!/usr/bin/env python3
"""Resample an existing GTP dataset to uniform time steps.

==========================================================================
PURPOSE
==========================================================================

The Abaqus/Explicit solver uses adaptive time stepping (Courant limit),
producing output frames at highly irregular intervals.  In our datasets,
delta_t can vary by **>100x** between consecutive transitions.

This script resamples every simulation onto a **uniform** time grid so
that every transition has exactly the same delta_t.  This has profound
benefits:

  1. **Standard problem**: all five reviewed papers in the literature
     use fixed delta_t.  Uniform stepping makes our results directly
     comparable.
  2. **Simpler target**: with constant delta_t, the damage *increment*
     (delta_D = D(t+1) - D(t)) becomes a well-conditioned target.
     No need for the rate (dD/dt) which spans 6 orders of magnitude.
  3. **History features**: a sliding window of m past increments is
     meaningful only when each step covers the same physical time.
  4. **Curriculum rollout**: N-step unrolling during training is
     consistent in physical time per step.

==========================================================================
HOW IT WORKS
==========================================================================

For each simulation (identified by ``sample_idx``):

  1. Reconstruct the ordered timeline:
       t_0, t_1, ..., t_{T-1}   (original irregular times)
       D_0, D_1, ..., D_{T-1}   (damage fields, [N] per frame)

  2. Choose a uniform grid with ``n_intervals`` steps:
       tau_0 = t_0,  tau_1 = t_0 + dt_uni,  ...,  tau_K = t_{T-1}
       dt_uni = (t_{T-1} - t_0) / n_intervals

  3. For each uniform time tau_k, find the two original frames that
     bracket it:  t_n <= tau_k < t_{n+1}.

  4. Linearly interpolate the damage field:
       D(tau_k) = (1 - alpha) * D_n + alpha * D_{n+1}
       alpha    = (tau_k - t_n) / (t_{n+1} - t_n)

     This is physically valid because D evolves continuously (the
     material subroutine updates D per increment).  Shocks (rapid D jumps)
     are preserved: they are smeared across at most one uniform step
     rather than being split across a tiny original step.

  5. Rebuild Data objects for each consecutive pair on the uniform
     grid:  (tau_k, tau_{k+1}) with target = damage rate or
     damage increment.

==========================================================================
CHOOSING n_intervals
==========================================================================

The ``n_intervals`` parameter controls how many uniform steps each
simulation is divided into.  Guidelines:

  - **Too few** (e.g. 50):  fast damage events (shocks) get smoothed
    out. The interpolated damage field may miss the crack front
    entirely if it initiates and completes between two uniform frames.

  - **Too many** (e.g. 2000):  most transitions have near-zero
    increments (the damage hasn't changed yet).  Training becomes
    slow and dominated by trivial samples.

  - **Good range**: 150-300 intervals.  This typically gives:
    * Median original dt roughly matches the uniform dt
    * Shock events are captured within 1-2 uniform steps
    * Training set size is manageable (~150-300 transitions per sim)

  The script prints detailed statistics (original dt distribution,
  chosen uniform dt, fraction of active transitions) to help you
  tune this parameter.

==========================================================================
USAGE
==========================================================================

    # Analyse the dataset (dry run, statistics only):
    python src/preprocessing/resample_uniform_dt.py \\
        --input data/processed/synthetic_50sims.pt \\
        --analyse-only

    # Resample with 200 intervals per simulation:
    python src/preprocessing/resample_uniform_dt.py \\
        --input data/processed/synthetic_50sims.pt \\
        --n-intervals 200 \\
        --output data/processed/synthetic_50sims_uniform200.pt

    # Rate target (default):
    python src/preprocessing/resample_uniform_dt.py \\
        --input data/processed/synthetic_50sims.pt \\
        --target rate

    # Increment target:
    python src/preprocessing/resample_uniform_dt.py \\
        --input data/processed/synthetic_50sims.pt \\
        --target increment

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


# =========================================================================
# ANALYSIS of the original dataset
# =========================================================================

def analyse_dataset(dataset: list[Data]) -> dict:
    """Print and return detailed statistics about the original dataset.

    This helps choose a good ``n_intervals`` value.
    """
    # --- Group by simulation ---
    sims = defaultdict(list)
    for d in dataset:
        sid = int(d.sample_idx.item())
        sims[sid].append(d)

    # Sort each sim by current_time
    for sid in sims:
        sims[sid].sort(key=lambda d: d.current_time.item())

    # --- Collect statistics ---
    all_dt = torch.cat([d.delta_t for d in dataset])
    all_rates = torch.cat([d.y for d in dataset], dim=0)
    n_active = int((all_rates.abs() > 1e-4).sum())
    n_total = all_rates.numel()

    sim_durations = []
    sim_n_frames = []
    sim_dt_ranges = []
    for sid in sorted(sims.keys()):
        trans = sims[sid]
        t_start = trans[0].current_time.item()
        t_end = trans[-1].next_time.item()
        duration = t_end - t_start
        sim_durations.append(duration)
        sim_n_frames.append(len(trans) + 1)
        dts = torch.tensor([t.delta_t.item() for t in trans])
        sim_dt_ranges.append((dts.min().item(), dts.median().item(),
                              dts.max().item(), dts.std().item()))

    all_dt_np = all_dt.numpy()
    percentiles = np.percentile(all_dt_np, [5, 25, 50, 75, 95])

    print("=" * 72)
    print("DATASET ANALYSIS")
    print("=" * 72)
    print(f"  Simulations      : {len(sims)}")
    print(f"  Total transitions: {len(dataset)}")
    print(f"  Nodes per graph  : {dataset[0].x.shape[0]}")
    print()

    print("  delta_t distribution (across all transitions):")
    print(f"    min    = {all_dt.min().item():.6e}")
    print(f"    5th %  = {percentiles[0]:.6e}")
    print(f"    25th % = {percentiles[1]:.6e}")
    print(f"    median = {percentiles[2]:.6e}")
    print(f"    75th % = {percentiles[3]:.6e}")
    print(f"    95th % = {percentiles[4]:.6e}")
    print(f"    max    = {all_dt.max().item():.6e}")
    print(f"    ratio max/min = {all_dt.max().item() / max(all_dt.min().item(), 1e-30):.1f}x")
    print()

    print("  Per-simulation breakdown:")
    for i, sid in enumerate(sorted(sims.keys())):
        dur = sim_durations[i]
        nf = sim_n_frames[i]
        dt_min, dt_med, dt_max, dt_std = sim_dt_ranges[i]
        print(f"    sim {sid:>3d}: duration={dur:.6e}s, "
              f"frames={nf}, dt=[{dt_min:.2e}, {dt_med:.2e}, {dt_max:.2e}]")
    print()

    print("  Damage rate statistics:")
    print(f"    range  = [{all_rates.min():.6f}, {all_rates.max():.6f}]")
    print(f"    active = {n_active}/{n_total} ({100*n_active/max(n_total,1):.1f}%) "
          f"above 1e-4")
    print()

    # --- Suggest --dt value ---
    min_duration = min(sim_durations)
    max_duration = max(sim_durations)
    p95_dt = percentiles[4]   # 95th percentile of original dt
    suggested_dt = p95_dt
    suggested_n_short = int(min_duration / suggested_dt)
    suggested_n_long = int(max_duration / suggested_dt)
    print("  Suggested --dt value:")
    print(f"    95th-percentile original dt = {p95_dt:.4e}")
    print(f"    Recommended --dt {suggested_dt:.4e}")
    print(f"    Shortest sim ({min_duration:.4f}s) -> {suggested_n_short} intervals")
    print(f"    Longest  sim ({max_duration:.4f}s) -> {suggested_n_long} intervals")
    print("    All simulations will share the SAME dt.")
    print("=" * 72)

    return {
        "n_sims": len(sims),
        "sims": sims,
        "all_dt": all_dt,
        "percentiles": percentiles,
        "sim_durations": sim_durations,
        "suggested_dt": suggested_dt,
    }


# =========================================================================
# RESAMPLING CORE
# =========================================================================

def reconstruct_simulation(transitions: list[Data], damage_col: int = -1) -> tuple:
    """Reconstruct the ordered time series for a single simulation.

    Parameters
    ----------
    transitions : list of Data
        All transitions for one ``sample_idx``, already sorted by time.
    damage_col : int
        Column index of the damage feature in ``x``.  Default -1 (last).

    Returns
    -------
    times : ndarray [T]
        Frame times (T = len(transitions) + 1).
    damage_fields : ndarray [T, N]
        FI values at each frame.
    template : Data
        A template Data object to clone mesh-static attributes from.
    dmg_fields : ndarray [T, N] or None
        Matrix damage at each frame (None if dataset lacks ``y_dmg``).
    """
    T = len(transitions) + 1
    N = transitions[0].x.shape[0]

    times = np.empty(T, dtype=np.float64)
    damage = np.empty((T, N), dtype=np.float32)

    has_dmg_channel = hasattr(transitions[0], 'dmg_now') and transitions[0].dmg_now is not None
    dmg_fields = np.empty((T, N), dtype=np.float32) if has_dmg_channel else None

    for i, d in enumerate(transitions):
        times[i] = d.current_time.item()
        damage[i] = d.x[:, damage_col].numpy()
        if has_dmg_channel:
            dmg_fields[i] = d.dmg_now.squeeze(-1).numpy()

    # Last frame: compute from the last transition's target
    last = transitions[-1]
    times[T - 1] = last.next_time.item()
    dt_last = last.delta_t.item()
    dmg_last = last.x[:, damage_col].numpy()
    rate_last = last.y[:, 0].numpy()
    damage[T - 1] = dmg_last + rate_last * dt_last

    if has_dmg_channel:
        dmg_last_val = last.dmg_now.squeeze(-1).numpy()
        dmg_rate_last = last.y_dmg[:, 0].numpy()
        dmg_fields[T - 1] = np.clip(dmg_last_val + dmg_rate_last * dt_last, 0.0, 1.0)

    return times, damage, transitions[0], dmg_fields


def interpolate_uniform(
    times: np.ndarray,
    damage: np.ndarray,
    n_intervals: int,
) -> tuple:
    """Interpolate damage fields onto a uniform time grid.

    Parameters
    ----------
    times : ndarray [T_orig]
    damage : ndarray [T_orig, N]
    n_intervals : int
        Number of uniform intervals (produces n_intervals + 1 frames).

    Returns
    -------
    uni_times : ndarray [K+1]   where K = n_intervals
    uni_damage : ndarray [K+1, N]
    """
    t_start = times[0]
    t_end = times[-1]
    uni_times = np.linspace(t_start, t_end, n_intervals + 1)

    N = damage.shape[1]
    uni_damage = np.empty((n_intervals + 1, N), dtype=np.float32)

    # First and last frames are exact (no interpolation needed)
    uni_damage[0] = damage[0]
    uni_damage[-1] = damage[-1]

    # For each interior uniform frame, find bracketing original frames
    # and interpolate.  Use a pointer to avoid O(K*T) search.
    orig_idx = 0
    T_orig = len(times)

    for k in range(1, n_intervals):
        tau = uni_times[k]

        # Advance pointer until times[orig_idx+1] >= tau
        while orig_idx < T_orig - 2 and times[orig_idx + 1] < tau:
            orig_idx += 1

        t_lo = times[orig_idx]
        t_hi = times[orig_idx + 1]
        span = t_hi - t_lo

        if span <= 0:
            # Degenerate interval: copy the lower frame
            uni_damage[k] = damage[orig_idx]
        else:
            alpha = (tau - t_lo) / span
            alpha = np.clip(alpha, 0.0, 1.0)
            uni_damage[k] = (1 - alpha) * damage[orig_idx] + alpha * damage[orig_idx + 1]

    return uni_times, uni_damage


def _interp_on_grid(
    times: np.ndarray,
    damage: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Interpolate damage fields at arbitrary query times.

    Uses the same pointer-based linear interpolation as
    ``interpolate_uniform``, but on an explicitly supplied grid.
    """
    K = len(query_times)
    N = damage.shape[1]
    out = np.empty((K, N), dtype=np.float32)
    T_orig = len(times)

    # First and last: exact copy
    out[0] = damage[0]
    out[-1] = damage[min(T_orig - 1, K - 1)]   # safety clamp

    orig_idx = 0
    for k in range(1, K - 1):
        tau = query_times[k]

        while orig_idx < T_orig - 2 and times[orig_idx + 1] < tau:
            orig_idx += 1

        t_lo = times[orig_idx]
        t_hi = times[orig_idx + 1]
        span = t_hi - t_lo

        if span <= 0:
            out[k] = damage[orig_idx]
        else:
            alpha = np.clip((tau - t_lo) / span, 0.0, 1.0)
            out[k] = (1 - alpha) * damage[orig_idx] + alpha * damage[orig_idx + 1]

    return out


def build_uniform_transitions(
    uni_times: np.ndarray,
    uni_damage: np.ndarray,
    template: Data,
    sample_idx: int,
    target_mode: str = "rate",
    damage_col: int = -1,
    uni_dmg: np.ndarray | None = None,
    dmg_in_features: bool = False,
) -> list[Data]:
    """Build Data objects from uniformly-sampled frames.

    Parameters
    ----------
    target_mode : str
        "rate":      y = (D_{k+1} - D_k) / dt_uni   (the training target)
        "increment": y = D_{k+1} - D_k            (bounded in [0, 1])
    damage_col : int
        Column index of the damage feature in ``x``.  Default -1 (last).
    uni_dmg : ndarray [K+1, N] or None
        If present, matrix damage interpolated on the same uniform grid.
        Used to compute ``y_dmg`` (damage rate) and ``dmg_now`` per transition.
    dmg_in_features : bool
        If True, the template x layout is [static, D_now, FI_now] (built with
        --include-damage-feature): BOTH state columns are stripped from the
        static block and rebuilt per frame from the interpolated fields
        (``uni_dmg`` for D, ``uni_damage`` for FI). Requires ``uni_dmg``.
    """
    K = len(uni_times) - 1  # number of transitions
    dt_uni = float(uni_times[1] - uni_times[0])
    n_feat = template.x.shape[1]
    # Resolve negative index
    dc = damage_col if damage_col >= 0 else n_feat + damage_col
    if dmg_in_features:
        if uni_dmg is None:
            raise RuntimeError(
                "dmg_in_features=True but the dataset carries no matrix-"
                "damage field (uni_dmg is None); rebuild the raw dataset "
                "with --include-damage-feature from NPZs with SDV_DMG_1."
            )
        if dc != n_feat - 1:
            raise RuntimeError(
                "dmg_in_features=True expects the FI state in the LAST "
                f"feature column (got damage_col={dc} of {n_feat})."
            )
        # Strip the two per-frame state columns (D_now at -2, FI_now at -1)
        node_features_static = template.x[:, :dc - 1]
    else:
        # Static features = everything except the damage column
        node_features_static = torch.cat(
            [template.x[:, :dc], template.x[:, dc+1:]], dim=1
        ) if dc < n_feat - 1 else template.x[:, :dc]

    # Clamp FI at cap: values beyond this are physically meaningless.
    # Import the canonical cap from the dataset builder.
    from .build_cohesive_dataset import FI_CAP
    uni_damage = np.clip(uni_damage, None, FI_CAP)

    transitions = []
    for k in range(K):
        dmg_now = torch.from_numpy(uni_damage[k]).unsqueeze(1)      # [N, 1]
        dmg_next = torch.from_numpy(uni_damage[k + 1]).unsqueeze(1)  # [N, 1]
        delta_D = dmg_next - dmg_now                                  # [N, 1]
        delta_D = delta_D.clamp(min=0.0)  # FI rate ≥ 0

        if target_mode == "rate":
            y = delta_D / dt_uni
        elif target_mode == "increment":
            y = delta_D
        else:
            raise ValueError(f"Unknown target_mode: {target_mode!r}")

        if dmg_in_features:
            d_now_feat = torch.from_numpy(uni_dmg[k]).unsqueeze(1)  # [N, 1]
            x = torch.cat([node_features_static, d_now_feat, dmg_now], dim=1)
        else:
            x = torch.cat([node_features_static, dmg_now], dim=1)

        data_obj = Data(
            x=x,
            edge_index=template.edge_index,
            edge_attr=template.edge_attr,
            y=y,
            mesh_pos=template.mesh_pos,
            delta_t=torch.tensor([dt_uni], dtype=torch.float),
            current_time=torch.tensor([uni_times[k]], dtype=torch.float),
            next_time=torch.tensor([uni_times[k + 1]], dtype=torch.float),
            sample_idx=torch.tensor([sample_idx], dtype=torch.long),
        )
        if hasattr(template, 'cells') and template.cells is not None:
            data_obj.cells = template.cells
        # Preserve cohesive laminate metadata (ply/interface node counts)
        for attr in ('n_ply_nodes', 'n_iface_nodes', 'displacement', 'sim_id',
                     'iface_trainable'):
            if hasattr(template, attr) and getattr(template, attr) is not None:
                setattr(data_obj, attr, getattr(template, attr))

        # Two-stage damage channel
        if uni_dmg is not None:
            d_now = torch.from_numpy(uni_dmg[k]).unsqueeze(1)       # [N, 1]
            d_next = torch.from_numpy(uni_dmg[k + 1]).unsqueeze(1)
            delta_d = (d_next - d_now).clamp(min=0.0)
            if target_mode == "rate":
                y_dmg = delta_d / dt_uni
            else:
                y_dmg = delta_d
            data_obj.y_dmg   = y_dmg    # [N, 1] matrix damage rate
            data_obj.dmg_now = d_now     # [N, 1] current matrix damage

        transitions.append(data_obj)

    return transitions


# =========================================================================
# VALIDATION
# =========================================================================

_REQUIRED_ATTRS = ['sample_idx', 'mesh_pos', 'current_time', 'delta_t', 'y']


def _validate_resampled(transitions: list, allow_single_sim: bool = False) -> None:
    """Fail fast if resampled dataset has missing metadata.

    allow_single_sim: skip the sample_idx-uniqueness guard.  Legitimate for
    single-geometry extrapolation studies (inference-only, no splitting);
    the guard otherwise protects training builds from a broken sample_idx
    column (e.g. 400 sims all carrying idx 0).
    """
    errors = []
    for i, d in enumerate(transitions):
        for attr in _REQUIRED_ATTRS:
            val = getattr(d, attr, None)
            if val is None:
                errors.append(f"transition {i}: missing '{attr}'")
        # delta_t must be positive
        dt = getattr(d, 'delta_t', None)
        if dt is not None and dt.item() <= 0:
            errors.append(f"transition {i}: delta_t <= 0 ({dt.item():.6e})")

    # Check sample_idx uniqueness
    sids = set()
    for d in transitions:
        sid = getattr(d, 'sample_idx', None)
        if sid is not None:
            sids.add(int(sid.item()))
    if len(sids) < 2 and len(transitions) > 50 and not allow_single_sim:
        errors.append(
            f"Only {len(sids)} unique sample_idx in {len(transitions)} "
            f"transitions; simulation splitting will fail "
            f"(pass --allow-single-sim for single-geometry test studies)"
        )

    if errors:
        msg = (f"RESAMPLED DATASET VALIDATION FAILED: {len(errors)} error(s):\n"
               + "\n".join(f"  - {e}" for e in errors[:20]))
        if len(errors) > 20:
            msg += f"\n  ... and {len(errors) - 20} more"
        raise ValueError(msg)

    print(f"\n  Validation PASSED: {len(sids)} sims, "
          f"{len(transitions)} transitions")


# =========================================================================
# MAIN ENTRY POINT
# =========================================================================

def resample_dataset(
    input_path: str | Path,
    output_path: str | Path | None = None,
    n_intervals: int | None = None,
    dt: float | None = None,
    target_mode: str = "rate",
    analyse_only: bool = False,
    dmg_in_features: bool = False,
    allow_single_sim: bool = False,
) -> list[Data] | None:
    """Resample a GTP dataset to uniform time steps.

    Parameters
    ----------
    input_path : path
        Path to the original dataset .pt file.
    output_path : path or None
        Where to save the resampled dataset.  If None, auto-generated.
    n_intervals : int or None
        Number of uniform intervals per simulation; dt then differs between
        simulations of different duration. Mutually exclusive with *dt*.
    dt : float or None
        Fixed physical time step (seconds) shared by ALL simulations.
        Each simulation gets ``floor(duration / dt)`` intervals.
        Mutually exclusive with *n_intervals*.
    target_mode : str
        "rate" or "increment".
    analyse_only : bool
        If True, only print statistics, do not resample.
    """
    if dt is None and n_intervals is None:
        n_intervals = 200  # default when neither is given
    if dt is not None and n_intervals is not None:
        raise ValueError("Specify --dt or --n-intervals, not both.")

    use_fixed_dt = dt is not None

    t0 = time.time()
    input_path = Path(input_path)

    print("=" * 72)
    print("RESAMPLE DATASET TO UNIFORM TIME STEPS")
    print("=" * 72)
    print(f"  Input  : {input_path}")
    if use_fixed_dt:
        print(f"  dt (fixed)  : {dt:.6e} s")
    else:
        print(f"  n_intervals : {n_intervals}")
    print(f"  target_mode : {target_mode}")
    print()

    # --- Load ---
    print("  Loading dataset ...")
    dataset = torch.load(str(input_path), map_location="cpu", weights_only=False)
    print(f"  Loaded {len(dataset)} transitions")
    print()

    # --- Analyse ---
    stats = analyse_dataset(dataset)

    if analyse_only:
        print()
        print("  [analyse-only mode: not resampling]")
        return None

    sims = stats["sims"]

    # --- Resample each simulation ---
    print()
    print("-" * 72)
    if use_fixed_dt:
        print(f"Resampling {len(sims)} simulation(s) with fixed dt = {dt:.6e} s ...")
    else:
        print(f"Resampling {len(sims)} simulation(s) to {n_intervals} uniform intervals ...")
    print("-" * 72)

    # Auto-detect damage column: last feature column
    damage_col = -1

    skipped = []
    all_new_transitions = []
    for sid in sorted(sims.keys()):
        trans_orig = sims[sid]

        # Reconstruct time series
        times, damage, template, dmg_fields = reconstruct_simulation(
            trans_orig, damage_col=damage_col)
        duration = times[-1] - times[0]

        # Determine number of intervals for this simulation
        if use_fixed_dt:
            n_int = int(duration / dt)
            if n_int < 2:
                skipped.append((sid, duration))
                print(f"  sim {sid:>3d}: duration={duration:.4e}s < 2·dt; SKIPPED")
                continue
            dt_uni = dt
        else:
            n_int = n_intervals
            dt_uni = duration / n_int

        # Interpolate onto uniform grid
        if use_fixed_dt:
            # Build exact grid: t_0, t_0+dt, t_0+2*dt, ..., t_0+n_int*dt
            uni_times = times[0] + np.arange(n_int + 1) * dt
            # Clip to sim range (last point may overshoot slightly)
            uni_times_clip = np.minimum(uni_times, times[-1])
            uni_damage = _interp_on_grid(times, damage, uni_times_clip)
            uni_dmg = _interp_on_grid(times, dmg_fields, uni_times_clip) if dmg_fields is not None else None
        else:
            uni_times, uni_damage = interpolate_uniform(times, damage, n_int)
            uni_dmg = _interp_on_grid(times, dmg_fields, uni_times) if dmg_fields is not None else None

        # Build new Data objects
        new_trans = build_uniform_transitions(
            uni_times, uni_damage, template, sample_idx=sid,
            target_mode=target_mode, damage_col=damage_col,
            uni_dmg=uni_dmg,
            dmg_in_features=dmg_in_features,
        )
        all_new_transitions.extend(new_trans)

        # Per-sim stats
        new_rates = torch.cat([d.y for d in new_trans], dim=0)
        n_active = int((new_rates.abs() > 1e-4).sum())
        n_total = new_rates.numel()
        print(f"  sim {sid:>3d}: {len(trans_orig)} orig -> {len(new_trans)} uniform "
              f"(dt_uni={dt_uni:.4e}), "
              f"active={n_active}/{n_total} ({100*n_active/max(n_total,1):.1f}%)")

    if skipped:
        print(f"\n  WARNING: {len(skipped)} simulation(s) skipped (duration < 2·dt):")
        for sid, dur in skipped:
            print(f"    sim {sid}: duration={dur:.4e}s")

    # --- Output statistics ---
    print()
    print("-" * 72)
    print("Resampled dataset statistics:")
    total = len(all_new_transitions)
    d0 = all_new_transitions[0]
    print(f"  Total transitions : {total}")
    print(f"  x shape           : {list(d0.x.shape)}")
    print(f"  y shape           : {list(d0.y.shape)}  ({target_mode})")

    all_dt_new = torch.cat([d.delta_t for d in all_new_transitions])
    print(f"  delta_t (uniform) : min={all_dt_new.min().item():.6e}, "
          f"max={all_dt_new.max().item():.6e}")
    print(f"  delta_t std       : {all_dt_new.std().item():.2e} "
          f"(should be ~0 when using --dt)")

    all_y = torch.cat([d.y for d in all_new_transitions], dim=0)
    print(f"  y range           : [{all_y.min():.6f}, {all_y.max():.6f}]")
    n_active = int((all_y.abs() > 1e-4).sum())
    n_total = all_y.numel()
    print(f"  active targets    : {n_active}/{n_total} ({100*n_active/max(n_total,1):.1f}%)")

    dc = all_new_transitions[0].x.shape[1] - 1  # damage is always last column
    all_dmg = torch.cat([d.x[:, dc:dc+1] for d in all_new_transitions], dim=0)
    print(f"  damage (feat {dc})   : [{all_dmg.min():.4f}, {all_dmg.max():.4f}]")
    print("-" * 72)

    # --- Save ---
    if output_path is None:
        stem = input_path.stem
        if use_fixed_dt:
            # e.g. dt=0.01 -> "_uniform_dt0.01.pt"
            dt_str = f"{dt:g}"
            output_path = input_path.parent / f"{stem}_uniform_dt{dt_str}.pt"
        else:
            output_path = input_path.parent / f"{stem}_uniform{n_intervals}.pt"
    else:
        output_path = Path(output_path)

    # ── Fail-fast: validate resampled dataset ─────────────────────────
    _validate_resampled(all_new_transitions, allow_single_sim=allow_single_sim)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_new_transitions, str(output_path))

    elapsed = time.time() - t0
    print()
    print("=" * 72)
    print("RESAMPLING COMPLETE")
    print("=" * 72)
    print(f"  Saved {total} transitions -> {output_path}")
    print(f"  File size : {output_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Elapsed   : {elapsed:.1f}s")
    print("=" * 72)

    return all_new_transitions


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Resample a GTP dataset to uniform time steps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyse only (no resampling):
  python src/preprocessing/resample_uniform_dt.py \\
      --input data/processed/processed_data_3d_time_prop_1000_steps.pt \\
      --analyse-only

  # Resample with fixed dt=0.01 s (recommended: same dt for all sims):
  python src/preprocessing/resample_uniform_dt.py \\
      --input data/processed/processed_data_3d_time_prop_1000_steps.pt \\
      --dt 0.01

  # Resample with 200 intervals per simulation (dt differs between sims):
  python src/preprocessing/resample_uniform_dt.py \\
      --input data/processed/processed_data_3d_time_prop_1000_steps.pt \\
      --n-intervals 200
""",
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to the original dataset .pt file."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path.  Default: auto-generated from input name."
    )

    # Mutually exclusive: --dt vs --n-intervals
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument(
        "--dt", type=float, default=None,
        help="Fixed physical time step (seconds) shared by ALL simulations. "
             "Each sim gets floor(duration/dt) intervals.  RECOMMENDED."
    )
    step_group.add_argument(
        "--n-intervals", "-n", type=int, default=None,
        help="Number of uniform intervals per simulation; dt differs "
             "between simulations of different duration."
    )

    parser.add_argument(
        "--target", type=str, choices=["rate", "increment"], default="rate",
        help="Target type: 'rate' (dD/dt, default) or "
             "'increment' (delta_D, bounded in [0,1])."
    )
    parser.add_argument(
        "--analyse-only", action="store_true",
        help="Only print statistics about the original dataset, "
             "do not resample."
    )
    parser.add_argument(
        "--allow-single-sim", action="store_true",
        help="Skip the sample_idx-uniqueness guard. Use for single-geometry "
             "extrapolation studies (inference-only, no train/val/test split)."
    )
    parser.add_argument(
        "--dmg-in-features", action="store_true",
        help="Dataset carries D_now as an input feature (15-dim x built with "
             "--include-damage-feature): resample it alongside FI. Without "
             "this flag on such a dataset, the D_now column would stay frozen "
             "at its t=0 value."
    )
    args = parser.parse_args()

    resample_dataset(
        input_path=args.input,
        output_path=args.output,
        n_intervals=args.n_intervals,
        dt=args.dt,
        target_mode=args.target,
        analyse_only=args.analyse_only,
        dmg_in_features=args.dmg_in_features,
        allow_single_sim=args.allow_single_sim,
    )


if __name__ == "__main__":
    main()
