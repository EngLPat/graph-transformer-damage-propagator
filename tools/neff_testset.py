#!/usr/bin/env python3
"""N_eff of the influence rows over the test split, and its evolution in the rollout.

For each sampled output element i,

    I_ij     = || d yhat_i / d x_j ||_2       (Jacobian influence)
    Ihat_ij  = I_ij / sum_k I_ik
    N_eff(i) = 1 / sum_j Ihat_ij^2           (participation ratio)

N_eff is m when a prediction draws equally on m elements and 1 when one element
dominates. It is measured on the test simulations at several frames of each,
with between-simulation standard errors, and reported against normalised
pseudo-time and against damage extent. The sampled elements are chosen once per
simulation, since the mesh is constant within a simulation; each (simulation,
frame, element, model) costs one backward pass.

Usage:
  python3 tools/neff_testset.py [--sims 60] [--frames 6] [--nodes 25]
"""
import argparse, csv, json, os, random, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code
ROOT = os.environ.get("GTP_ROOT", REPO)                              # data/ and results/
os.chdir(ROOT)
sys.path.insert(0, REPO)
from tools.influence_on_front import load, predict
from tools.front_enrichment_testset import test_split_sims, DATA

# Same arms and colours as fig_neff, so the two figures agree.
ARMS = [("ablation_no_attention", "attentionless",  "#999999"),
        ("ablation_local_only",   "local only",     "#E69F00"),
        ("ablation_global_only",  "global only",    "#009E73"),
        ("ablation_reference",    "local-global",   "#0072B2")]
R2 = {"ablation_no_attention": 0.502, "ablation_local_only": 0.576,
      "ablation_global_only": 0.699, "ablation_reference": 0.728}
PAIRS = [("ablation_no_attention", "ablation_local_only",
          "no global branch"),
         ("ablation_global_only", "ablation_reference",
          "with a global branch")]
N_BINS = 6


def n_eff_for_nodes(model, stats, d, nodes):
    """Participation ratio of the influence row of each sampled element."""
    out = []
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
        out.append(float(1.0 / (Ih ** 2).sum().clamp_min(1e-30)))
    return out


def main(n_sims, n_frames, n_nodes, ply, out, one_sim=None, tag=""):
    print("loading %s ..." % DATA)
    ds = torch.load(DATA, map_location="cpu", weights_only=False)
    seq = ds["data"] if isinstance(ds, dict) and "data" in ds else ds
    test_sims, groups = test_split_sims(seq)
    if one_sim is not None:
        if one_sim not in test_sims:
            raise SystemExit("simulation %d is not in the test split" % one_sim)
        sims = [one_sim]
    else:
        rng = random.Random(0)
        sims = sorted(rng.sample(test_sims, min(n_sims, len(test_sims))))
    print("test split %d sims; measuring %d" % (len(test_sims), len(sims)))

    taus = np.linspace(0.10, 0.95, n_frames)
    models = {run: load(run)[:2] for run, _, _ in ARMS}

    # per arm: (sim, frame) grids of N_eff and of the damage-state coordinate
    neff = {run: np.full((len(sims), n_frames), np.nan) for run, _, _ in ARMS}
    extent = np.full((len(sims), n_frames), np.nan)

    for si, sid in enumerate(sims):
        idxs = groups[sid]
        d0 = seq[idxs[0]]
        N = d0.x.shape[0]
        deg = torch.bincount(d0.edge_index[0], minlength=N)
        zs = sorted(set(d0.x[:, 2].tolist()))
        zsel = zs[min(ply, len(zs) - 1)]
        cand = torch.where(((d0.x[:, 2] - zsel).abs() < 1e-6) & (deg >= 3))[0]
        g = torch.Generator().manual_seed(sid)
        nodes = cand[torch.randperm(cand.numel(), generator=g)[:n_nodes]].tolist()

        for ti, tau in enumerate(taus):
            d = seq[idxs[min(int(len(idxs) * tau), len(idxs) - 1)]]
            extent[si, ti] = float((d.x[:, -2].numpy() > 0.05).mean())
            for run, lab, col in ARMS:
                m, st = models[run]
                v = n_eff_for_nodes(m, st, d, nodes)
                if v:
                    neff[run][si, ti] = float(np.mean(v))
        print("   sim %3d (%d/%d)  extent %.3f -> %.3f"
              % (sid, si + 1, len(sims), extent[si, 0], extent[si, -1]))

    # ---------------- aggregate: overall, vs time, vs damage extent ----------
    def between_sim(a, mask=None):
        """Mean and s.e. taken ACROSS simulations, never across elements."""
        per_sim = []
        for si in range(len(sims)):
            row = a[si] if mask is None else a[si][mask[si]]
            row = row[np.isfinite(row)]
            if row.size:
                per_sim.append(row.mean())
        if len(per_sim) < 2:
            return float("nan"), float("nan"), len(per_sim)
        p = np.asarray(per_sim)
        return float(p.mean()), float(p.std(ddof=1) / np.sqrt(p.size)), int(p.size)

    edges = np.linspace(np.nanmin(extent), np.nanmax(extent), N_BINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    which = np.clip(np.digitize(extent, edges) - 1, 0, N_BINS - 1)

    res = {}
    for run, lab, col in ARMS:
        mu, se, n = between_sim(neff[run])
        if one_sim is not None:
            # With one simulation there is no between-simulation spread to
            # take, and between_sim correctly refuses to invent one. Report
            # the mean over that simulation's own frames instead, with the
            # frame-to-frame standard error, and say so in the label.
            row = neff[run][0][np.isfinite(neff[run][0])]
            mu = float(row.mean()) if row.size else float("nan")
            se = (float(row.std(ddof=1) / np.sqrt(row.size))
                  if row.size > 1 else float("nan"))
            n = int(row.size)
        if one_sim is not None:
            vs_t = [(float(neff[run][0, ti]), 0.0, 1) for ti in range(n_frames)]
        else:
            vs_t = [between_sim(neff[run], mask=np.tile(
                np.arange(n_frames) == ti, (len(sims), 1)))
                for ti in range(n_frames)]
        vs_e = [between_sim(neff[run], mask=(which == b)) for b in range(N_BINS)]
        res[run] = dict(label=lab, colour=col, r2=R2[run],
                        n_eff=mu, n_eff_se=se, n_sims=n,
                        taus=taus.tolist(),
                        vs_time=[v[0] for v in vs_t],
                        vs_time_se=[v[1] for v in vs_t],
                        extent_centres=centres.tolist(),
                        vs_extent=[v[0] for v in vs_e],
                        vs_extent_se=[v[1] for v in vs_e])

    os.makedirs(out, exist_ok=True)
    json.dump(dict(meta=dict(data=DATA, sims=sims, n_frames=n_frames,
                             n_nodes=n_nodes, ply=ply,
                             note="error bars are BETWEEN-SIMULATION"),
                   arms=res),
              open(os.path.join(out, "neff_testset%s.json" % tag), "w"), indent=2)
    with open(os.path.join(out, "neff_testset%s.csv" % tag), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "N_eff", "N_eff_se", "n_sims", "FI_R2"])
        for run, lab, col in ARMS:
            S = res[run]
            w.writerow([lab, "%.2f" % S["n_eff"], "%.2f" % S["n_eff_se"],
                        S["n_sims"], S["r2"]])

    # ------------------------------- figure ---------------------------------
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 200,
                         "savefig.bbox": "tight"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    for run, lab, col in ARMS:
        S = res[run]
        m = np.asarray(S["vs_time"]); e = np.asarray(S["vs_time_se"])
        ok = np.isfinite(m)
        a1.plot(taus[ok], m[ok], "o-", color=col, ms=3.6, lw=1.8, label=lab)
        a1.fill_between(taus[ok], (m - e)[ok], (m + e)[ok], color=col,
                        alpha=0.18, lw=0)
        m = np.asarray(S["vs_extent"]); e = np.asarray(S["vs_extent_se"])
        ok = np.isfinite(m)
        a2.plot(centres[ok], m[ok], "o-", color=col, ms=3.6, lw=1.8, label=lab)
        a2.fill_between(centres[ok], (m - e)[ok], (m + e)[ok], color=col,
                        alpha=0.18, lw=0)
    a1.set_xlabel("normalised pseudo-time")
    a2.set_xlabel("damage extent   (fraction of elements with $\\omega>0.05$)")
    for a in (a1, a2):
        a.set_ylabel("$N_{\\mathrm{eff}}$")
    a1.set_title("Concentration through the rollout", loc="left", fontsize=10)
    a2.set_title("The same, against damage state", loc="left", fontsize=10)
    a1.legend(frameon=False, fontsize=8)
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(out, "fig_neff_testset%s.%s" % (tag, e)))
    plt.close(fig)
    print("\nwrote %s/fig_neff_testset%s.png" % (out, tag))

    print("\n" + "=" * 78)
    print("N_eff OVER THE TEST SPLIT  (%d sims x %d frames x %d elements)"
          % (len(sims), n_frames, n_nodes))
    print("error bars are BETWEEN-SIMULATION")
    print("=" * 78)
    print("%-24s %16s %9s" % ("arm", "N_eff", "FI R2"))
    print("-" * 78)
    for run, lab, col in ARMS:
        S = res[run]
        print("%-24s %8.1f +- %-5.1f %9.3f" % (lab, S["n_eff"], S["n_eff_se"],
                                               S["r2"]))
    print("-" * 78)
    for a, b, ctx in PAIRS:
        na, nb = res[a]["n_eff"], res[b]["n_eff"]
        print("  + local attention, %-22s  N_eff %.1f -> %.1f  (%+.0f%%)   "
              "dR2 %+.3f" % (ctx, na, nb, 100.0 * (nb - na) / na,
                             R2[b] - R2[a]))
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--nodes", type=int, default=25)
    ap.add_argument("--ply", type=int, default=0)
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--sim", type=int, default=None,
                    help="one test simulation, traced through its own rollout")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.sims, a.frames, a.nodes, a.ply, a.out, one_sim=a.sim, tag=a.tag)
