#!/usr/bin/env python3
"""Global attention received by boundary and damage categories, by damage state.

For the dense global attention matrix W (rows sum to one), the attention
received by element j is the column sum A_j = sum_i W_ij, averaged over the
global blocks of the processor (or taken from one block, with --block). Its
uniform value is exactly 1, and the enrichment of a category c is

    enrichment(c) = mean_{j in c} A_j / mean_j A_j .

Each frame is placed by the damage extent of its own simulation (fraction of
elements with D > 0.05) rather than by time, and frames are pooled over the
test simulations into extent bins, with between-simulation standard errors.
The categories are the hole, the fixed and displaced edges, and the undamaged,
front and failed elements.

Usage: python3 tools/global_attention_by_state.py [--sims 60] [--frames 16]
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
from tools.attention_by_category import geometric_categories, damage_categories

# ref is the proposed model; globalonly is a cross-check
ARMS = {"ref": ("ablation_reference", "local-global (proposed)"),
        "globalonly": ("ablation_global_only", "global only")}

CATS = ["hole", "fixed (left)", "displaced (right)",
        "undamaged", "front", "failed"]
COL = {"hole": "#CC79A7", "fixed (left)": "#009E73",
       "displaced (right)": "#0072B2", "undamaged": "#999999",
       "front": "#56B4E9", "failed": "#D55E00"}
PHASES = ["pristine", "initiation", "propagation"]
N_BINS = 10
MIN_MEMBERS = 20        # smallest category population that gives a usable mean


class _CollectAll(dict):
    """Dict that keeps every global-attention map of a forward pass.

    The shared OptimizedGlobalAttention module stores each call under
    _attention_weights[batch_idx], so a plain dict keeps only the last block.
    This subclass also appends every assignment to .maps.
    """

    def __init__(self):
        super().__init__()
        self.maps = []

    def __setitem__(self, key, value):
        self.maps.append(value)
        super().__setitem__(key, value)


def attention_maps(model, stats, d):
    """Every global attention matrix produced by one forward pass.

    Returns a list of [N, N] tensors, one per attention block, or None when the
    arm has no global branch so the caller can skip it rather than crash.
    """
    if not getattr(model, "global_attention_enabled", False):
        return None
    model.enable_attention_analysis(True)
    collector = _CollectAll()
    model.global_attention._attention_weights = collector
    try:
        with torch.no_grad():
            predict(model, stats, d)
        maps = list(collector.maps)
    finally:
        model.enable_attention_analysis(False)
    if not maps:
        raise RuntimeError("no global attention captured")
    return maps


def received_global(model, stats, d, block=None):
    """A_j, the global attention received by each element (column sum of W).

    block=None averages over all attention blocks; block=k returns block k
    alone, counting from 0 at the shallowest. The blocks differ substantially,
    so a figure of the first global-attention call must use block=0.
    """
    maps = attention_maps(model, stats, d)
    if maps is None:
        return None
    if block is not None:
        if block >= len(maps):
            raise SystemExit("block %d requested but only %d were captured"
                             % (block, len(maps)))
        maps = [maps[block]]
    A = sum(W.sum(dim=0) for W in maps) / float(len(maps))
    return A.detach().cpu().numpy().astype(np.float64)


def main(arm_key, n_sims, n_frames, out, raw=False, one_sim=None, tag="", block=None):
    """raw: report A_j itself, not A_j divided by its mean over the plate.
    one_sim: restrict to a single simulation and use pseudo-time as the x
    axis, which shows the whole history including the pristine stage instead
    of pooling it into one coarse damage-extent bin."""
    run, label = ARMS[arm_key]
    print("loading dataset ...")
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
    print("test split %d sims; measuring %d %s"
          % (len(test_sims), len(sims), "(raw)" if raw else "(enrichment)"))

    taus = np.linspace(0.01, 0.99, n_frames)
    model, stats, _ = load(run)

    # (sim, frame) -> extent, phase, and the per-category enrichment
    extent = np.full((len(sims), n_frames), np.nan)
    phase = np.empty((len(sims), n_frames), dtype=object)
    enr = {c: np.full((len(sims), n_frames), np.nan) for c in CATS}

    for si, sid in enumerate(sims):
        idxs = groups[sid]
        for ti, tau in enumerate(taus):
            d = seq[idxs[min(int(len(idxs) * tau), len(idxs) - 1)]]
            D = d.x[:, -2].numpy()
            extent[si, ti] = float((D > 0.05).mean())
            mx = float(D.max())
            phase[si, ti] = ("pristine" if mx < 0.05
                             else "initiation" if mx < 0.95 else "propagation")
            A = received_global(model, stats, d, block=block)
            if A is None:
                raise SystemExit("arm %s has no global branch" % run)
            base = 1.0 if raw else A.mean()
            if not np.isfinite(base) or base <= 0:
                continue
            cats = dict(geometric_categories(d))
            cats.update(damage_categories(d))
            for c in CATS:
                m = cats[c]
                if m.sum() < MIN_MEMBERS:
                    continue
                enr[c][si, ti] = float(A[m].mean() / base)
        print("   sim %3d (%d/%d)" % (sid, si + 1, len(sims)))

    # ---- bin by damage extent, between-simulation error bars ----
    edges = np.linspace(np.nanmin(extent), np.nanmax(extent), N_BINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    which = np.clip(np.digitize(extent, edges) - 1, 0, N_BINS - 1)

    def by_bin(a):
        """Mean per extent bin, s.e. taken ACROSS simulations."""
        mu, se = np.full(N_BINS, np.nan), np.full(N_BINS, np.nan)
        for b in range(N_BINS):
            per_sim = []
            for si in range(len(sims)):
                v = a[si][(which[si] == b) & np.isfinite(a[si])]
                if v.size:
                    per_sim.append(v.mean())
            if len(per_sim) >= 2:
                p = np.asarray(per_sim)
                mu[b] = p.mean()
                se[b] = p.std(ddof=1) / np.sqrt(p.size)
            elif per_sim:
                mu[b] = per_sim[0]
        return mu, se

    def by_phase(a):
        o = {}
        for ph in PHASES:
            per_sim = []
            for si in range(len(sims)):
                v = a[si][(phase[si] == ph) & np.isfinite(a[si])]
                if v.size:
                    per_sim.append(v.mean())
            if len(per_sim) >= 2:
                p = np.asarray(per_sim)
                o[ph] = dict(mean=float(p.mean()),
                             se=float(p.std(ddof=1) / np.sqrt(p.size)),
                             n_sims=int(p.size))
            else:
                o[ph] = dict(mean=float("nan"), se=float("nan"),
                             n_sims=len(per_sim))
        return o

    if one_sim is not None:
        # One simulation: plot against pseudo-time, without extent binning
        centres = taus
        binned = {c: (enr[c][0], np.zeros(n_frames)) for c in CATS}
    else:
        binned = {c: by_bin(enr[c]) for c in CATS}
    phased = {c: by_phase(enr[c]) for c in CATS}

    os.makedirs(out, exist_ok=True)
    json.dump(dict(meta=dict(data=DATA, run=run, sims=sims,
                             n_frames=n_frames,
                             note="received global attention, column sum of "
                                  "the dense attention matrix; uniform = 1.0"),
                   extent_centres=centres.tolist(),
                   binned={c: dict(mean=binned[c][0].tolist(),
                                   se=binned[c][1].tolist()) for c in CATS},
                   phase=phased),
              open(os.path.join(out, "global_attention_by_state%s.json" % tag), "w"),
              indent=2)

    with open(os.path.join(out, "global_attention_by_state%s.csv" % tag), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phase"] + CATS + [c + "_se" for c in CATS])
        for ph in PHASES:
            w.writerow([ph] + ["%.4f" % phased[c][ph]["mean"] for c in CATS]
                       + ["%.4f" % phased[c][ph]["se"] for c in CATS])

    # ---------------------------- the figure ----------------------------
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 200,
                         "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for c in CATS:
        mu, se = binned[c]
        ok = np.isfinite(mu)
        ax.plot(centres[ok], mu[ok], "o-", color=COL[c], ms=3.6, lw=1.9,
                label=c)
        ax.fill_between(centres[ok], (mu - se)[ok], (mu + se)[ok],
                        color=COL[c], alpha=0.18, lw=0)
    # Log scale: the quantity is a ratio against uniform
    ax.set_yscale("log")
    if not raw:
        ax.axhline(1.0, color="#888888", ls=":", lw=1.1)
        ax.annotate("uniform attention", (centres[-1], 1.0),
                    textcoords="offset points", xytext=(-2, 5), fontsize=7.5,
                    ha="right", color="#888888")
    if one_sim is not None:
        ax.set_xlabel("normalised pseudo-time")
    else:
        ax.set_xlabel("damage extent   (fraction of elements with "
                      "$\\omega > 0.05$)"
                      "\nundamaged $\\longrightarrow$ fully cracked")
    ax.set_ylabel("received global attention $A_j$" if raw
                  else "received global attention / uniform")
    ax.set_title(("Received global attention, simulation %d" % one_sim)
                 if one_sim is not None else
                 "Where global attention goes as damage develops",
                 loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")
    ax.text(0.5, -0.30,
            "Received global attention is the column sum of the dense attention\n"
            "matrix, whose rows sum to one, so uniform is exactly 1.0. Bands are\n"
            "between-simulation standard errors over %d test simulations; the\n"
            "attentionless model returns exactly 1.0 everywhere by construction."
            % len(sims), transform=ax.transAxes, fontsize=7.4, color="#555555",
            ha="center", va="top")
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(out, "fig_global_attention_by_state%s.%s" % (tag, e)))
    plt.close(fig)
    print("\nwrote %s/fig_global_attention_by_state%s.png" % (out, tag))

    print("\n" + "=" * 84)
    print("RECEIVED GLOBAL ATTENTION BY CATEGORY -- %s" % label)
    print("%d test simulations x %d frames; uniform = 1.000; +- between-sim s.e."
          % (len(sims), n_frames))
    print("=" * 84)
    print("%-14s" % "phase" + "".join("%18s" % c for c in CATS))
    print("-" * 84)
    for ph in PHASES:
        print("%-14s" % ph + "".join(
            "  %8.3f+-%.3f" % (phased[c][ph]["mean"], phased[c][ph]["se"])
            for c in CATS))
    print("=" * 84)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ref", choices=sorted(ARMS))
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--raw", action="store_true",
                    help="plot A_j itself rather than A_j divided by its mean "
                         "over the plate")
    ap.add_argument("--sim", type=int, default=None,
                    help="restrict to one test simulation and use pseudo-time "
                         "as the x axis, so the pristine stage is visible")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filenames, so variants coexist")
    ap.add_argument("--block", type=int, default=None,
                    help="report one attention block (0 = the first global "
                         "call) instead of the mean over all blocks")
    a = ap.parse_args()
    main(a.arm, a.sims, a.frames, a.out, raw=a.raw, one_sim=a.sim,
         tag=a.tag, block=a.block)
