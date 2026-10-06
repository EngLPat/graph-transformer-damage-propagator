#!/usr/bin/env python3
"""Local attention received by geometric and damage categories, by damage state.

The processor aggregates into edge_index[0], so for an edge (i, j) node i is the
receiver and j the neighbour weighted by alpha_ij. The local attention received
by element j is A_j = sum_{i : (i,j) in E} alpha_ij, averaged over the processor
layers. As A_j grows with the in-degree k_j, each category c is scored against
the uniform-attention baseline,

    enrichment(c) = ( sum_{j in c} A_j / sum_j A_j ) / ( sum_{j in c} k_j / sum_j k_j ) ,

which is exactly 1 when all alpha are equal, so the attentionless model is a
flat control. Two partitions are reported, since they overlap: geometry (hole,
fixed edge, displaced edge, bulk) and damage state (undamaged, front, process,
failed).

Each frame is placed by the damage extent of its own simulation (fraction of
elements with D > 0.05), and frames are pooled over the test simulations into
extent bins, with between-simulation standard errors. The table uses three
phases: pristine (max D < 0.05), initiation (0.05 <= max D < 0.95) and
propagation (max D >= 0.95).

Usage: python3 tools/attention_by_category.py [--sims 12] [--frames 12]
"""
import argparse, json, os, random, sys
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

# Plotted models; the attentionless model is a control that comes out flat at 1.0
ARMS = [("ablation_reference",    "local-global", "#0072B2"),
        ("ablation_local_only",   "local only",   "#E69F00"),
        ("ablation_no_attention", "attentionless (control)", "#999999")]

PHASES = ["pristine", "initiation", "propagation"]
N_EXTENT_BINS = 12
GEO = ["hole", "fixed (left)", "displaced (right)", "bulk"]
DMG = ["undamaged", "front", "process", "failed", "high FI"]
GEO_COL = {"hole": "#CC79A7", "fixed (left)": "#009E73",
           "displaced (right)": "#0072B2", "bulk": "#999999"}
DMG_COL = {"undamaged": "#999999", "front": "#56B4E9",
           "process": "#E69F00", "failed": "#D55E00",
           "high FI": "#CC79A7"}


# Node-feature layout, from src/preprocessing/build_cohesive_dataset.py:
#   8 is_hole, 9 is_fixed, 10 is_disp
I_HOLE, I_FIXED, I_DISP = 8, 9, 10


def geometric_categories(d):
    """Geometric partition: hole, fixed edge, displaced edge, bulk.

    Read from the dataset's own is_hole, is_fixed and is_disp node features,
    so the partition is the one the model itself sees.
    """
    x = d.x
    hole = x[:, I_HOLE] > 0.5
    fixed = x[:, I_FIXED] > 0.5          # left edge,  x ~ x_min, restrained
    disp = x[:, I_DISP] > 0.5            # right edge, x ~ x_max, loaded
    outer = fixed | disp
    # The fixed and displaced edges are reported separately; the free top and
    # bottom edges carry no boundary condition and belong to the bulk
    return {"fixed (left)": fixed.numpy(),
            "displaced (right)": disp.numpy(),
            "hole": (hole & ~outer).numpy(),
            "bulk": (~(hole | outer)).numpy()}


FRONT_HOPS = int(os.environ.get("GTP_FRONT_HOPS", "1"))


def damage_categories(d):
    """Damage-state partition: four disjoint classes plus one diagnostic.

    The front is the undamaged material within GTP_FRONT_HOPS mesh steps
    (default 1) of a damaged element. The four classes cover every element:
        undamaged   omega < 0.05 and not adjacent to damage
        front       omega < 0.05 and adjacent to damage
        process     0.05 <= omega < 0.95, partially degraded
        failed      omega >= 0.95
    "high FI" (undamaged, FI > 0.5) is a diagnostic only, not plotted: an
    alternative front definition that overlaps undamaged and front.
    """
    D = d.x[:, -2].numpy(); FI = d.x[:, -1].numpy()
    undamaged = D < 0.05
    # Elements within FRONT_HOPS of damage. edge_index[0] is the receiver and
    # edge_index[1] the neighbour, so scattering a neighbour flag into
    # edge_index[0] marks every element with a marked neighbour: one such pass
    # per hop.
    rec, snd = d.edge_index[0], d.edge_index[1]
    reach = torch.as_tensor(~undamaged).clone()
    for _ in range(max(FRONT_HOPS, 1)):
        nxt = reach.clone()
        nxt[rec[reach[snd]]] = True
        reach = nxt
    near = reach.numpy() & undamaged
    return {"undamaged": undamaged & ~near,
            "front":     near,
            "process":   (D > 0.05) & (D < 0.95),
            "failed":    (D >= 0.95),
            "high FI":   undamaged & (FI > 0.5)}


def received_attention(model, stats, d):
    """A_j per element, averaged over processor layers, plus in-degree k_j."""
    for L in model.processor:
        L.store_attention = True
    with torch.no_grad():
        predict(model, stats, d)
    a = [L._edge_attention.squeeze(-1) for L in model.processor
         if getattr(L, "_edge_attention", None) is not None]
    if not a:
        raise RuntimeError("no attention captured -- store_attention unsupported")
    alpha = torch.stack(a).mean(0)
    snd = d.edge_index[1]
    N = d.x.shape[0]
    A = torch.zeros(N).index_add_(0, snd, alpha)
    k = torch.bincount(snd, minlength=N).float()
    return A.numpy(), k.numpy()


def enrichment(A, k, mask):
    """Share of attention received, divided by share of incoming edges."""
    ta, tk = A.sum(), k.sum()
    if ta <= 0 or tk <= 0 or mask.sum() == 0 or k[mask].sum() <= 0:
        return np.nan
    return (A[mask].sum() / ta) / (k[mask].sum() / tk)


def main(n_sims, n_frames, out):
    print("loading dataset ...")
    ds = torch.load(DATA, map_location="cpu", weights_only=False)
    seq = ds["data"] if isinstance(ds, dict) and "data" in ds else ds
    test_sims, groups = test_split_sims(seq)
    rng = random.Random(0)
    sims = sorted(rng.sample(test_sims, min(n_sims, len(test_sims))))
    print("test split %d sims; measuring %d: %s" % (len(test_sims), len(sims), sims))

    taus = np.linspace(0.02, 0.98, n_frames)

    # ---- damage-state coordinate, computed once (model-independent) ----
    # extent = fraction of elements already damaged; phase from the max.
    print("computing the damage-state coordinate for every frame ...")
    extent = np.full((len(sims), n_frames), np.nan)
    phase = np.empty((len(sims), n_frames), dtype=object)
    for si, sid in enumerate(sims):
        idxs = groups[sid]
        for ti, tau in enumerate(taus):
            D = seq[idxs[min(int(len(idxs) * tau), len(idxs) - 1)]].x[:, -2].numpy()
            extent[si, ti] = float((D > 0.05).mean())
            mx = float(D.max())
            phase[si, ti] = ("pristine" if mx < 0.05
                             else "initiation" if mx < 0.95 else "propagation")
    print("   extent range %.4f - %.4f;  phase counts %s"
          % (np.nanmin(extent), np.nanmax(extent),
             {p: int((phase == p).sum()) for p in PHASES}))

    results = {}
    for run, lab, col in ARMS:
        model, stats, _ = load(run)
        # per category -> array (n_sims, n_frames)
        geo = {c: np.full((len(sims), n_frames), np.nan) for c in GEO}
        dmg = {c: np.full((len(sims), n_frames), np.nan) for c in DMG}
        for si, sid in enumerate(sims):
            idxs = groups[sid]
            gcats = geometric_categories(seq[idxs[0]])
            for ti, tau in enumerate(taus):
                d = seq[idxs[min(int(len(idxs) * tau), len(idxs) - 1)]]
                A, k = received_attention(model, stats, d)
                for c in GEO:
                    geo[c][si, ti] = enrichment(A, k, gcats[c])
                dcats = damage_categories(d)
                for c in DMG:
                    dmg[c][si, ti] = enrichment(A, k, dcats[c])
            print("   %-24s sim %3d (%d/%d)" % (lab, sid, si + 1, len(sims)))
        results[run] = dict(
            label=lab, colour=col, taus=taus.tolist(), sims=sims,
            raw_geo={c: geo[c].tolist() for c in GEO},
            raw_dmg={c: dmg[c].tolist() for c in DMG},
            geo={c: dict(mean=np.nanmean(geo[c], 0).tolist(),
                         se=(np.nanstd(geo[c], 0, ddof=1)
                             / np.sqrt(len(sims))).tolist()) for c in GEO},
            dmg={c: dict(mean=np.nanmean(dmg[c], 0).tolist(),
                         se=(np.nanstd(dmg[c], 0, ddof=1)
                             / np.sqrt(len(sims))).tolist()) for c in DMG})

    os.makedirs(out, exist_ok=True)
    json.dump(dict(meta=dict(data=DATA, sims=sims, n_frames=n_frames,
                             note="enrichment of RECEIVED attention, "
                                  "degree-normalised; 1.0 = uniform"),
                   arms=results),
              open(os.path.join(out, "attention_by_category.json"), "w"),
              indent=2)

    # ---------- rebin onto the damage-state coordinate ----------
    edges = np.linspace(np.nanmin(extent), np.nanmax(extent), N_EXTENT_BINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    which = np.clip(np.digitize(extent, edges) - 1, 0, N_EXTENT_BINS - 1)

    def rebin(arr):
        """Mean per extent bin, with a BETWEEN-SIMULATION standard error."""
        mean = np.full(N_EXTENT_BINS, np.nan)
        se = np.full(N_EXTENT_BINS, np.nan)
        for b in range(N_EXTENT_BINS):
            per_sim = []
            for si in range(arr.shape[0]):
                v = arr[si][which[si] == b]
                v = v[np.isfinite(v)]
                if v.size:
                    per_sim.append(v.mean())
            if len(per_sim) >= 2:
                mean[b] = float(np.mean(per_sim))
                se[b] = float(np.std(per_sim, ddof=1) / np.sqrt(len(per_sim)))
            elif per_sim:
                mean[b] = float(per_sim[0])
        return mean, se

    def by_phase(arr):
        """Mean per physically defined phase, between-simulation s.e."""
        out = {}
        for ph in PHASES:
            per_sim = []
            for si in range(arr.shape[0]):
                v = arr[si][phase[si] == ph]
                v = v[np.isfinite(v)]
                if v.size:
                    per_sim.append(v.mean())
            out[ph] = dict(
                mean=float(np.mean(per_sim)) if per_sim else float("nan"),
                se=float(np.std(per_sim, ddof=1) / np.sqrt(len(per_sim)))
                if len(per_sim) >= 2 else float("nan"),
                n_sims=len(per_sim))
        return out

    for run, lab, col in ARMS:
        S = results[run]
        S["extent_centres"] = centres.tolist()
        S["binned"] = {}
        S["phase"] = {}
        for key, group, raw in (("geo", GEO, "raw_geo"), ("dmg", DMG, "raw_dmg")):
            for c in group:
                a = np.asarray(S[raw][c], dtype=float)
                m, e = rebin(a)
                S["binned"][c] = dict(mean=m.tolist(), se=e.tolist())
                S["phase"][c] = by_phase(a)

    import csv as _csv
    with open(os.path.join(out, "attention_by_category.csv"), "w",
              newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["# x axis is DAMAGE EXTENT (fraction of elements with "
                    "D>0.05), not time"])
        w.writerow(["arm", "damage_extent"] + GEO + DMG
                   + [c + "_se" for c in GEO + DMG])
        for run, lab, col in ARMS:
            S = results[run]
            for b in range(N_EXTENT_BINS):
                row = [lab, "%.4f" % centres[b]]
                for c in GEO + DMG:
                    row.append("%.4f" % S["binned"][c]["mean"][b])
                for c in GEO + DMG:
                    row.append("%.4f" % S["binned"][c]["se"][b])
                w.writerow(row)
    with open(os.path.join(out, "attention_by_phase.csv"), "w",
              newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["arm", "phase"] + GEO + DMG + [c + "_se" for c in GEO + DMG])
        for run, lab, col in ARMS:
            S = results[run]
            for ph in PHASES:
                row = [lab, ph]
                row += ["%.4f" % S["phase"][c][ph]["mean"] for c in GEO + DMG]
                row += ["%.4f" % S["phase"][c][ph]["se"] for c in GEO + DMG]
                w.writerow(row)
    print("wrote %s/attention_by_category.csv and attention_by_phase.csv" % out)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 200,
                         "savefig.bbox": "tight"})
    n = len(ARMS)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 8.0), sharex=True)
    axes = np.atleast_2d(axes)
    for j, (run, lab, col) in enumerate(ARMS):
        S = results[run]
        for row, (group, cols, key) in enumerate(
                [(GEO, GEO_COL, "geo"), (DMG, DMG_COL, "dmg")]):
            ax = axes[row, j]
            for c in group:
                m = np.array(S["binned"][c]["mean"])
                e = np.array(S["binned"][c]["se"])
                ax.plot(centres, m, "o-", color=cols[c], ms=3.4, lw=1.7,
                        label=c)
                ax.fill_between(centres, m - e, m + e, color=cols[c],
                                alpha=0.18, lw=0)
            ax.axhline(1.0, color="#444444", ls="--", lw=1.0)
            ax.set_title("%s%s" % (lab, "" if row else ""), loc="left",
                         fontsize=9.5)
            if j == 0:
                ax.set_ylabel("attention enrichment\n(%s)"
                              % ("geometry" if row == 0 else "damage state"))
            if row == 1:
                ax.set_xlabel("damage extent\n(fraction of elements with "
                              "$D>0.05$)")
            if j == n - 1:
                ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Where the local attention goes, as damage develops. "
                 "Enrichment of received attention against a degree-matched "
                 "uniform baseline (dashed 1.0). The abscissa is the "
                 "specimen's own DAMAGE EXTENT, not clock time, so simulations "
                 "are aligned by physical state; bands are between-simulation "
                 "standard errors over %d test simulations." % len(sims),
                 x=0.005, ha="left", fontsize=9, color="#666666")
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(out, "fig_attention_by_category." + e))
    plt.close(fig)
    print("\nwrote %s/fig_attention_by_category.png" % out)

    for run, lab, col in ARMS:
        S = results[run]
        print("\n" + "=" * 78)
        print("%s   attention enrichment by DAMAGE PHASE (1.0 = uniform)" % lab)
        print("=" * 78)
        print("%-14s" % "" + "".join("%18s" % p for p in PHASES))
        print("-" * 78)
        for c in GEO + DMG:
            print("%-14s" % c + "".join(
                "  %7.3f +- %-6.3f" % (S["phase"][c][p]["mean"],
                                       S["phase"][c][p]["se"])
                for p in PHASES))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=12)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    main(a.sims, a.frames, a.out)
