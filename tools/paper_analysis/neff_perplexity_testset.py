#!/usr/bin/env python3
"""N_eff and perplexity of the influence rows on the test split, from one sample.

Both measures are computed from the same normalised influence row Ihat_i,

    N_eff(i) = 1 / sum_j Ihat_ij^2               (participation ratio)
    P(i)     = exp(-sum_j Ihat_ij log Ihat_ij)   (perplexity)

using the sampling of tools/neff_testset.py (same simulation draw, node draw,
frames and degree filter), so the N_eff column reproduces that tool's output.
"""
import os, random, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # code
ROOT = os.environ.get("GTP_ROOT", REPO)                              # data/ and results/
os.chdir(ROOT)
sys.path.insert(0, REPO)
from tools.influence_on_front import load, predict
from tools.front_enrichment_testset import test_split_sims, DATA
from tools.neff_testset import ARMS

N_SIMS, N_FRAMES, N_NODES, PLY = 60, 6, 25, 0


def measures_for_nodes(model, stats, d, nodes):
    """Both concentration measures of the influence row of each sampled element."""
    neff, perp = [], []
    for i in nodes:
        dd = d.clone()
        dd.x = d.x.clone().requires_grad_(True)
        y = predict(model, stats, dd)
        g = torch.autograd.grad(y[int(i)], dd.x, retain_graph=False)[0]
        I = g.norm(dim=1).detach().double()
        s = I.sum()
        if s <= 0:
            continue
        Ih = I / s
        neff.append(float(1.0 / (Ih ** 2).sum().clamp_min(1e-30)))
        nz = Ih[Ih > 0]
        perp.append(float(torch.exp(-(nz * nz.log()).sum())))
    return neff, perp


def main():
    print("loading %s ..." % DATA, flush=True)
    ds = torch.load(DATA, map_location="cpu", weights_only=False)
    seq = ds["data"] if isinstance(ds, dict) and "data" in ds else ds
    test_sims, groups = test_split_sims(seq)
    rng = random.Random(0)
    sims = sorted(rng.sample(test_sims, min(N_SIMS, len(test_sims))))
    print("test split %d sims; measuring %d" % (len(test_sims), len(sims)), flush=True)

    taus = np.linspace(0.10, 0.95, N_FRAMES)
    models = {run: load(run)[:2] for run, _, _ in ARMS}
    acc = {run: (np.full((len(sims), N_FRAMES), np.nan),
                 np.full((len(sims), N_FRAMES), np.nan)) for run, _, _ in ARMS}

    for si, sid in enumerate(sims):
        idxs = groups[sid]
        d0 = seq[idxs[0]]
        N = d0.x.shape[0]
        deg = torch.bincount(d0.edge_index[0], minlength=N)
        zs = sorted(set(d0.x[:, 2].tolist()))
        zsel = zs[min(PLY, len(zs) - 1)]
        cand = torch.where(((d0.x[:, 2] - zsel).abs() < 1e-6) & (deg >= 3))[0]
        g = torch.Generator().manual_seed(sid)
        nodes = cand[torch.randperm(cand.numel(), generator=g)[:N_NODES]].tolist()

        for ti, tau in enumerate(taus):
            d = seq[idxs[min(int(len(idxs) * tau), len(idxs) - 1)]]
            for run, lab, col in ARMS:
                m, st = models[run]
                ne, pe = measures_for_nodes(m, st, d, nodes)
                if ne:
                    acc[run][0][si, ti] = float(np.mean(ne))
                    acc[run][1][si, ti] = float(np.mean(pe))
        print("   sim %3d (%d/%d)" % (sid, si + 1, len(sims)), flush=True)

    def between_sim(a):
        per = [r[np.isfinite(r)].mean() for r in a if np.isfinite(r).any()]
        per = np.asarray(per)
        return per.mean(), per.std(ddof=1) / np.sqrt(per.size)

    print("\n" + "=" * 78)
    print("N_eff AND PERPLEXITY, %d test simulations x %d frames x %d elements"
          % (len(sims), N_FRAMES, N_NODES))
    print("=" * 78)
    print("%-16s %18s %20s" % ("arm", "N_eff", "perplexity"))
    res = {}
    for run, lab, col in ARMS:
        nm, ns = between_sim(acc[run][0])
        pm, ps = between_sim(acc[run][1])
        res[lab] = (nm, pm)
        print("%-16s %10.1f +- %-5.1f %12.1f +- %-6.1f" % (lab, nm, ns, pm, ps))

    print("\ncontrasts on adding local attention (ratio = more selective by):")
    for a, b in [("attentionless", "local only"), ("global only", "local-global")]:
        if a in res and b in res:
            print("  %-14s -> %-14s  N_eff %5.2fx (%+.0f%%)   perplexity %5.2fx (%+.0f%%)"
                  % (a, b,
                     res[a][0] / res[b][0], 100 * (res[b][0] / res[a][0] - 1),
                     res[a][1] / res[b][1], 100 * (res[b][1] / res[a][1] - 1)))
    print("=" * 78)


if __name__ == "__main__":
    main()
