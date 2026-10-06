#!/usr/bin/env python
"""Pooled Dice of a rollout, with the Dice definition of the paper.

Pooled over all element-time samples of all test simulations, as the R^2 is.
An element-time sample counts as damaged when the field exceeds the threshold
(strict >): damage variable > 0.05, or FI > 1 (initiation) for the failure
index.

Keys of rollout_arrays.npz: predicted_damage / ground_truth_damage hold the
integrated failure index; predicted_matrix_damage / ground_truth_matrix_damage
hold the damage variable omega.

Usage: dice_pooled.py <rollout_arrays.npz> [label]
"""
import sys
import numpy as np


def flat(a):
    a = np.asarray(a, dtype=object) if getattr(a, "dtype", None) == object else np.asarray(a)
    if a.dtype == object:
        return np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in a])
    return a.astype(np.float64).ravel()


def dice(pred, true):
    tp = np.logical_and(pred, true).sum()
    den = pred.sum() + true.sum()
    return 2.0 * tp / den if den else float("nan")


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    d = np.load(path, allow_pickle=True)
    out = [label]
    for name, pk, gk, thrs in (
        ("damage", "predicted_matrix_damage", "ground_truth_matrix_damage", (0.05, 0.5)),
        ("FI", "predicted_damage", "ground_truth_damage", (1.0,)),
    ):
        p, g = flat(d[pk]), flat(d[gk])
        m = np.isfinite(p) & np.isfinite(g)
        p, g = p[m], g[m]
        for t in thrs:
            P, G = p > t, g > t
            out.append(f"{name}>{t}: Dice {dice(P, G):.4f}  area pred {P.mean()*100:.1f}%  "
                       f"true {G.mean()*100:.1f}%  (N={len(p)})")
    print("\n  ".join(out))


if __name__ == "__main__":
    main()
