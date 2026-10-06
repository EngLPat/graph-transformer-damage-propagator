#!/usr/bin/env python3
"""Rollout R² of an already-tested run, from its saved rollout arrays.

Computes, with the same function test mode uses (src/gtp/evaluation/metrics.py),
the R², RMSE and MAE of the failure index (all frames, and the final frame of
each simulation) and of the damage variable, and writes them to
``test_results/rescored_metrics.txt``.

Usage
-----
    python tools/rescore_r2.py results/<run>

Reads the ``test_results/rollout_*.npz`` written by ``--mode test``, and stops
with a message if no saved rollout exists.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gtp.evaluation.metrics import format_rollout_metrics, rollout_metrics

KEYS = ("predicted_damage", "ground_truth_damage", "sim_ids_per_frame",
        "predicted_matrix_damage", "ground_truth_matrix_damage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path,
                    help="Run folder (contains test_results/ with rollout npz)")
    ap.add_argument("--rollout-npz", type=Path, default=None,
                    help="Path to rollout .npz (default: search test_results/).")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    test_dir = run_dir / "test_results"
    if not test_dir.is_dir():
        sys.exit(f"No test_results/ under {run_dir}")

    npz_path = args.rollout_npz
    if npz_path is None:
        candidates = sorted(test_dir.glob("rollout*.npz"))
        if not candidates:
            sys.exit(
                f"No rollout*.npz found under {test_dir}.\n"
                "Re-run --mode test with rollout-saving enabled, or pass --rollout-npz."
            )
        npz_path = candidates[0]
    print(f"Loading: {npz_path}")
    # allow_pickle=True: per-frame arrays of different sizes are stored as objects
    data = np.load(npz_path, allow_pickle=True)
    rows = rollout_metrics({k: data[k] for k in KEYS if k in data.files})

    print()
    print(f"  {'metric':<32s}  {'N':>10s}  {'R²':>8s}  {'RMSE':>8s}  {'MAE':>8s}")
    for r in rows:
        print(f"  {r['name']:<32s}  {r['n']:>10d}  {r['r2']:>8.4f}  "
              f"{r['rmse']:>8.3f}  {r['mae']:>8.3f}")

    out = test_dir / "rescored_metrics.txt"
    with out.open("w") as fh:
        fh.write("=" * 80 + "\n")
        fh.write("TEST RESULTS (autoregressive rollout)\n")
        fh.write("=" * 80 + "\n\n")
        for line in format_rollout_metrics(rows):
            fh.write(line + "\n")
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
