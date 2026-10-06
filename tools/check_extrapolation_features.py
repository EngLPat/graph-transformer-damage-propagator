"""Report how far an extrapolation dataset sits outside the training stats.

For each node feature (and edge feature), computes the dataset's min/max/
mean, normalizes with the TRAINING normalization_stats.pt, and flags
features whose normalized range exceeds what the model saw in training
(|z| > ~3 is suspicious, |z| > 5 is strongly out-of-distribution).

This answers the question "will inference on the new geometry break the
normalization?" BEFORE burning GPU time.

Usage
-----
    python tools/check_extrapolation_features.py \\
        --dataset data/processed/extrap_stitched/extrap_stitched_dmgstate_dataset_uniform_dt0.01.pt \\
        --stats   results/laminate/normalization_stats.pt
"""

from __future__ import annotations

import argparse

import torch

FEAT_NAMES_13 = ["x", "y", "z", "cos_a1", "sin_a1", "cos_a2", "sin_a2",
                 "layer_norm", "is_hole", "is_fixed", "is_disp", "disp_amt",
                 "node_type"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stats", required=True)
    args = ap.parse_args()

    data_list = torch.load(args.dataset, weights_only=False)
    stats = torch.load(args.stats, map_location="cpu", weights_only=False)
    mean_x, std_x = stats[0].float(), stats[1].float()
    mean_e, std_e = stats[2].float(), stats[3].float()

    x = torch.cat([d.x for d in data_list], dim=0).float()
    e = torch.cat([d.edge_attr for d in data_list], dim=0).float()

    n_feat = x.shape[1]
    names = list(FEAT_NAMES_13)
    if n_feat == 14:
        names += ["FI_now"]
    elif n_feat == 15:
        names += ["D_now", "FI_now"]
    else:
        names += [f"f{k}" for k in range(len(names), n_feat)]

    print(f"dataset: {args.dataset}")
    print(f"  {len(data_list)} transitions, {x.shape[0]} node rows, "
          f"{n_feat} features")
    print(f"stats:   {args.stats}  (mean_x/std_x dims: {len(mean_x)})")
    if len(mean_x) != n_feat:
        print(f"  *** FEATURE-DIM MISMATCH: dataset has {n_feat}, stats have "
              f"{len(mean_x)}; wrong dataset variant for this checkpoint ***")
        return

    print(f"\n{'feature':>12} {'raw min':>10} {'raw max':>10} "
          f"{'z min':>8} {'z max':>8}  flag")
    for k in range(n_feat):
        raw_min, raw_max = x[:, k].min().item(), x[:, k].max().item()
        sd = std_x[k].item() if std_x[k].item() > 1e-12 else 1.0
        z_min = (raw_min - mean_x[k].item()) / sd
        z_max = (raw_max - mean_x[k].item()) / sd
        worst = max(abs(z_min), abs(z_max))
        flag = ("  <-- OOD (|z|>5)" if worst > 5
                else "  <-- borderline (|z|>3)" if worst > 3 else "")
        print(f"{names[k]:>12} {raw_min:>10.4f} {raw_max:>10.4f} "
              f"{z_min:>8.2f} {z_max:>8.2f}{flag}")

    print(f"\n{'edge feat':>12} {'raw min':>10} {'raw max':>10} "
          f"{'z min':>8} {'z max':>8}")
    e_names = ["dx", "dy", "dz", "dist"]
    for k in range(e.shape[1]):
        raw_min, raw_max = e[:, k].min().item(), e[:, k].max().item()
        sd = std_e[k].item() if std_e[k].item() > 1e-12 else 1.0
        z_min = (raw_min - mean_e[k].item()) / sd
        z_max = (raw_max - mean_e[k].item()) / sd
        print(f"{e_names[k] if k < 4 else f'e{k}':>12} "
              f"{raw_min:>10.4f} {raw_max:>10.4f} {z_min:>8.2f} {z_max:>8.2f}")


if __name__ == "__main__":
    main()
