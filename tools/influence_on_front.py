#!/usr/bin/env python3
"""Influence of the damage regions on each node's prediction, against a
distance-matched null.

The plate is partitioned by damage state into four regions:
    tip      D < 0.05 and FI > 0.9   (about to initiate)
    process  0.05 < D < 0.95         (initiated, still evolving)
    failed   D >= 0.95
    intact   D < 0.05 and FI < 0.5   (far field)

The influence I(i, j) of node j on output node i is the norm of the gradient of
the prediction at i with respect to the input features of j, and the share of
region R is share(i, R) = sum_{j in R, j != i} I(i, j) / sum_{j != i} I(i, j).

Influence decays with graph distance, so each share is compared with a null that
keeps every node's influence and hop distance from i and permutes only the region
labels within each hop shell. Choosing k_h of the n_h nodes of shell h gives the
exact expectation and variance

    E[share_null] = sum_h (k_h / n_h) * S_h ,   Var_h = k_h (n_h - k_h) / (n_h - 1) * sigma_h^2 ,

where S_h is the total influence of shell h, and the enrichment is

    enrichment(i, R) = share(i, R) / E[share_null(i, R)] .

An enrichment of 1 means the model reads region R only as much as its distance
predicts; above 1, it preferentially draws information from R.

This module is a library: it holds the model loader, the hop-distance shells,
the region definitions and the exact-null enrichment, and is imported by
front_enrichment_testset.py, neff_testset.py, global_attention_by_state.py,
attention_by_category.py and paper_analysis/neff_perplexity_testset.py.
"""
import glob, os, re, sys, types
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code
ROOT = os.environ.get("GTP_ROOT", REPO)                              # data/ and results/
os.chdir(ROOT)
sys.path.insert(0, REPO)
from src.gtp.models.transformer import FCLGA_GraphTransformer
from src.gtp.config.config import Config

# (run, label, colour, FI R2 of the released model)
MODELS = [("ablation_no_attention", "attentionless",          "#999999", 0.502),
          ("ablation_local_only",   "local only",             "#E69F00", 0.576),
          ("ablation_global_only",  "global only",            "#009E73", 0.699),
          ("ablation_reference",    "local-global (gate)",    "#0072B2", 0.728)]
D_LO, D_HI = 0.05, 0.95


def read_cfg(p):
    cfg = {}
    for line in open(p):
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*?)\s*$", line)
        if m:
            v = m.group(2).strip().strip("'\"")
            if v in ("true", "false"):
                v = (v == "true")
            else:
                try:
                    v = float(v) if ("." in v or "e" in v.lower()) else int(v)
                except ValueError:
                    pass
            cfg[m.group(1)] = v
    return cfg


def weights(run):
    """The released weights of a run: results/<run>/best_models/model_<run>_*.pt."""
    found = glob.glob("results/%s/best_models/model_%s_*.pt" % (run, run))
    if len(found) != 1:
        raise SystemExit("expected one weights file results/%s/best_models/model_%s_*.pt, "
                         "found %d" % (run, run, len(found)))
    return found[0]


def load(run):
    c = read_cfg("results/%s/config.yaml" % run)
    ns = types.SimpleNamespace(
        num_layers=c["num_layers"], hidden_dim=c["hidden_dim"],
        attention_freq=c["attention_freq"], dropout=c["dropout_rate"],
        two_stage=c.get("two_stage_enabled", True), rate_cap=0,
        alpha_global=c.get("alpha_global", 0.2),
        local_attention_enabled=c.get("local_attention_enabled", True),
        global_attention_enabled=c.get("global_attention_enabled", True))
    m = FCLGA_GraphTransformer(15, 4, c["hidden_dim"], 1,
                               Config.from_args(ns).to_objectview()).eval()
    sd = torch.load(weights(run), map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    m.load_state_dict(sd, strict=True)
    st = torch.load("results/%s/normalization_stats.pt" % run,
                    map_location="cpu", weights_only=False)
    return m, st, c


def predict(model, stats, d):
    out = model(d, stats[0], stats[1], stats[2], stats[3])
    return (out[0] if isinstance(out, tuple) else out).squeeze(-1)


def hop_distances(edge_index, n_nodes, sources):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path
    r = edge_index[0].numpy(); s = edge_index[1].numpy()
    ii = np.concatenate([r, s]); jj = np.concatenate([s, r])
    A = coo_matrix((np.ones(len(ii), dtype=np.int8), (ii, jj)),
                   shape=(n_nodes, n_nodes)).tocsr()
    D = shortest_path(A, method="D", unweighted=True, directed=False,
                      indices=np.asarray(sources))
    D[~np.isfinite(D)] = -1
    return D.astype(np.int16)


def shell_index(d_row):
    """Group node indices by hop distance, excluding self and unreachable."""
    shells = {}
    for h in np.unique(d_row):
        if h <= 0:
            continue
        shells[int(h)] = np.where(d_row == h)[0]
    return shells


def enrichment_for_node(infl, d_row, regions, shells):
    """Influence share per region, against the EXACT distance-matched null.

    For each hop shell h, the null draws k_h of the n_h nodes uniformly at
    random, so it expects a fraction k_h/n_h of that shell's influence. Summing
    over shells gives the null expectation in closed form; the variance is that
    of a simple random sample sum without replacement. No shuffling needed, and
    the result carries no Monte-Carlo error.
    """
    ok = d_row > 0
    tot = float(infl[ok].sum())
    if tot <= 0:
        return None
    out = {}
    for name, mask in regions.items():
        obs = float(infl[ok & mask].sum())
        exp = 0.0
        var = 0.0
        for h, idx in shells.items():
            n_h = len(idx)
            k_h = int(mask[idx].sum())
            if k_h == 0 or n_h < 2:
                continue
            v = infl[idx].astype(np.float64)
            S_h = v.sum()
            exp += (k_h / n_h) * S_h
            if k_h < n_h:
                sig2 = v.var()
                var += k_h * (n_h - k_h) / (n_h - 1.0) * sig2
        if exp <= 0:
            continue
        sd = np.sqrt(var)
        out[name] = dict(obs=obs / tot, null=exp / tot,
                         enrichment=obs / exp,
                         z=float((obs - exp) / sd) if sd > 0 else float("nan"))
    return out


REGIONS = ["tip", "process", "failed", "intact"]


def build_regions(D, FI):
    return {"tip":     (D < D_LO) & (FI > 0.9),
            "process": (D > D_LO) & (D < D_HI),
            "failed":  (D >= D_HI),
            "intact":  (D < D_LO) & (FI < 0.5)}
