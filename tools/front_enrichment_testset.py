#!/usr/bin/env python3
"""Region-enrichment table over the test split, at several frames per simulation.

Applies the measurement of influence_on_front.py (influence share per damage
region, divided by the exact expectation of a hop-shell-matched random draw) to
the test simulations of the training dataset, at several frames of each, and
reports the mean with a between-simulation standard error. Hop distances are
computed once per simulation, since the mesh is constant within a simulation;
each (simulation, frame, node, model) costs one backward pass.

Usage:
  python3 tools/front_enrichment_testset.py [--sims 12] [--frames 6] [--nodes 40]
"""
import argparse, json, os, random, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code
ROOT = os.environ.get("GTP_ROOT", REPO)                              # data/ and results/
os.chdir(ROOT)
sys.path.insert(0, REPO)
from tools.influence_on_front import (load, predict, hop_distances, shell_index,
                                      enrichment_for_node, build_regions,
                                      REGIONS, MODELS)

DATA = ("data/processed/laminate/"
        "laminate_dmgstate_dataset_uniform_dt0.01.pt")


def test_split_sims(dataset):
    """Reproduce the simulation-level test split exactly as _run_testing does.

    Same seed (42), same shuffle, same ratios, so the simulations measured here
    are the ones the reported R2 is computed on -- not a different sample that
    happens to be convenient.
    """
    from src.gtp.config import DEFAULTS
    groups = {}
    for i, d in enumerate(dataset):
        sid = int(d.sample_idx.item()) if hasattr(d, "sample_idx") else 0
        groups.setdefault(sid, []).append(i)
    sim_ids = sorted(groups)
    n = len(sim_ids)
    rng = random.Random()
    rng.seed(42)
    shuffled = list(sim_ids)
    rng.shuffle(shuffled)
    n_train = max(1, int(n * DEFAULTS["train_ratio"]))
    n_val = max(1, int(n * DEFAULTS["val_ratio"]))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    return sorted(shuffled[n_train + n_val:]), groups


def main(n_sims, n_frames, n_nodes, ply, out):
    print("loading %s ..." % DATA)
    ds = torch.load(DATA, map_location="cpu", weights_only=False)
    seq = ds["data"] if isinstance(ds, dict) and "data" in ds else ds
    test_sims, groups = test_split_sims(seq)
    print("test split: %d simulations" % len(test_sims))

    rng = random.Random(0)
    sims = sorted(rng.sample(test_sims, min(n_sims, len(test_sims))))
    print("measuring on %d of them: %s" % (len(sims), sims))

    models = {}
    for run, lab, col, r2 in MODELS:
        m, st, _cfg = load(run)
        models[run] = (m, st, lab, col, r2)

    # per arm -> per region -> list of PER-SIMULATION means
    acc = {run: {r: [] for r in REGIONS} for run in models}
    n_used = 0

    for si, sid in enumerate(sims):
        idxs = groups[sid]
        # frames spread over the damaging part of the simulation
        picks = [idxs[int(len(idxs) * f)]
                 for f in np.linspace(0.30, 0.85, n_frames)]
        d_first = seq[picks[0]]
        N = d_first.x.shape[0]
        deg = torch.bincount(d_first.edge_index[0], minlength=N)

        zs = sorted(set(d_first.x[:, 2].tolist()))
        zsel = zs[min(ply, len(zs) - 1)]
        plymask = (d_first.x[:, 2] - zsel).abs() < 1e-6
        cand = torch.where(plymask & (deg >= 3))[0]
        g = torch.Generator().manual_seed(sid)
        nodes = cand[torch.randperm(cand.numel(), generator=g)[:n_nodes]].tolist()

        # mesh is fixed within a simulation -> one BFS for all its frames
        Dh = hop_distances(d_first.edge_index, N, nodes)
        shells = [shell_index(Dh[k]) for k in range(len(nodes))]

        per_sim = {run: {r: [] for r in REGIONS} for run in models}
        for fi_, fidx in enumerate(picks):
            d = seq[fidx]
            regions = build_regions(d.x[:, -2].numpy(), d.x[:, -1].numpy())
            if regions["failed"].sum() < 10 or regions["tip"].sum() < 10:
                continue                      # nothing to measure yet
            n_used += 1
            for run, (model, stats, lab, col, r2) in models.items():
                for k, i in enumerate(nodes):
                    dd = d.clone()
                    dd.x = d.x.clone().requires_grad_(True)
                    y = predict(model, stats, dd)
                    gr = torch.autograd.grad(y[int(i)], dd.x,
                                             retain_graph=False)[0]
                    infl = gr.norm(dim=1).detach().numpy().astype(np.float64)
                    res = enrichment_for_node(infl, Dh[k], regions, shells[k])
                    if not res:
                        continue
                    for r in REGIONS:
                        if r in res:
                            per_sim[run][r].append(res[r]["enrichment"])
        for run in models:
            for r in REGIONS:
                v = per_sim[run][r]
                if v:
                    acc[run][r].append(float(np.mean(v)))
        print("  sim %3d  (%d/%d)  frames used so far: %d"
              % (sid, si + 1, len(sims), n_used))

    results = {}
    for run, (model, stats, lab, col, r2) in models.items():
        S = dict(label=lab, colour=col, r2=r2, n_sims=len(acc[run][REGIONS[0]]))
        for r in REGIONS:
            a = np.asarray(acc[run][r], dtype=float)
            S[r] = dict(mean=float(a.mean()), sd=float(a.std(ddof=1)),
                        se=float(a.std(ddof=1) / np.sqrt(a.size)),
                        n_sims=int(a.size), per_sim=a.tolist())
        results[run] = S

    os.makedirs(out, exist_ok=True)
    json.dump(dict(meta=dict(data=DATA, sims=sims, n_frames=n_frames,
                             n_nodes=n_nodes, ply=ply,
                             note="error bars are BETWEEN-SIMULATION"),
                   arms=results),
              open(os.path.join(out, "front_enrichment_testset.json"), "w"),
              indent=2)

    print("\n" + "=" * 100)
    print("INFLUENCE ENRICHMENT BY DAMAGE REGION -- TEST SPLIT")
    print("%d simulations x %d frames x %d nodes; +- is between-simulation s.e."
          % (len(sims), n_frames, n_nodes))
    print("=" * 100)
    print("%-24s %14s %14s %14s %14s %8s"
          % ("arm", "tip", "process", "failed", "intact", "FI R2"))
    print("-" * 100)
    for run, lab, col, r2 in MODELS:
        if run not in results:
            continue
        S = results[run]
        print("%-24s" % lab + "".join(
            " %8.2f+-%.2f" % (S[r]["mean"], S[r]["se"]) for r in REGIONS)
            + " %8.3f" % r2)
    print("=" * 100)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=12)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--nodes", type=int, default=40)
    ap.add_argument("--ply", type=int, default=0)
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    main(a.sims, a.frames, a.nodes, a.ply, a.out)
