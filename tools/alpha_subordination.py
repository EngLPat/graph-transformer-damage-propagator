#!/usr/bin/env python3
"""Share of the processor update carried by the global-attention branch.

At each layer with a global block the processor computes

    x_mid  = x_pre + alpha * A(x_pre)      (global residual)
    x_next = ProcessorLayer(x_mid)         (local, edge-conditioned update)

For each alpha_global model this reports the global share

    s = g / (g + l),   g = ||alpha * A(x_pre)||_F,   l = ||x_next - x_mid||_F

averaged over the global blocks. s near 0 means the layer is dominated by
message passing on the mesh, s near 1 by the dense global operator. Runs on
CPU on a single frame of a small mesh.

Usage:  python3 tools/alpha_subordination.py
"""
import json, os, re, sys
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code
ROOT = os.environ.get("GTP_ROOT", REPO)                              # data/ and results/
os.chdir(ROOT)
sys.path.insert(0, REPO)

from src.gtp.models.transformer import FCLGA_GraphTransformer
from src.gtp.config.config import Config
from src.gtp.evaluation.evaluator import Evaluator  # noqa: F401  (arch introspection)
from tools.influence_on_front import weights

DATA = ("data/processed/extrap_onehole_top/"
        "extrap_onehole_top_dmgstate_dataset_uniform_dt0.01.pt")

# (run dir, alpha)
ARMS = [
    ("ablation_local_only", 0.0),   # no global branch at all
    ("alpha_0p05", 0.05),
    ("alpha_0p1", 0.1),
    ("ablation_reference", 0.2),
    ("alpha_0p4", 0.4),
    ("alpha_0p8", 0.8),
]


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
                    v = float(v) if "." in v or "e" in v.lower() else int(v)
                except ValueError:
                    pass
            cfg[m.group(1)] = v
    return cfg


def build(run):
    """Rebuild the model from the run's own config.yaml, not from DEFAULTS."""
    c = read_cfg(os.path.join("results", run, "config.yaml"))
    import types
    ns = types.SimpleNamespace(
        num_layers=c["num_layers"], hidden_dim=c["hidden_dim"],
        attention_freq=c["attention_freq"], dropout=c["dropout_rate"],
        two_stage=c.get("two_stage_enabled", True), rate_cap=0,
        alpha_global=c.get("alpha_global", 0.2),
        local_attention_enabled=c.get("local_attention_enabled", True),
        global_attention_enabled=c.get("global_attention_enabled", True),
    )
    cfg = Config.from_args(ns).to_objectview()
    return FCLGA_GraphTransformer(15, 4, c["hidden_dim"], 1, cfg).eval(), c


def measure(run, sample):
    model, c = build(run)
    sd = torch.load(weights(run), map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(sd, strict=True)

    rec = {"g": [], "l": []}
    if not model.global_attention_enabled:
        return {"share": 0.0, "n": 0, "note": "no global branch"}

    # Hook the shared global-attention module: its OUTPUT is A(x_pre).
    def gh(_m, _i, out):
        rec["g"].append(float(torch.linalg.norm(model.alpha_global * out)))
    h1 = model.global_attention.register_forward_hook(gh)

    # Hook every processor layer: local update = ||out_nodes - in_nodes||.
    hs = [h1]
    def mk(idx):
        def ph(_m, inp, out):
            x_in = inp[0]
            x_out = out[0] if isinstance(out, tuple) else out
            rec["l"].append((idx, float(torch.linalg.norm(x_out - x_in))))
        return ph
    for i, L in enumerate(model.processor):
        hs.append(L.register_forward_hook(mk(i)))

    stats = torch.load(os.path.join("results", run, "normalization_stats.pt"),
                       map_location="cpu", weights_only=False)
    mvx, svx, mve, sve = stats[0], stats[1], stats[2], stats[3]
    with torch.no_grad():
        model(sample, mvx, svx, mve, sve)
    for h in hs:
        h.remove()

    # Global blocks fire at layers i % attention_freq == 0, BEFORE that layer.
    freq = c["attention_freq"]
    gl = [v for i, v in rec["l"] if i % freq == 0]
    n = min(len(rec["g"]), len(gl))
    if n == 0:
        return None
    shares = [rec["g"][k] / (rec["g"][k] + gl[k]) for k in range(n)]
    return {"share": sum(shares) / n, "per_block": shares, "n": n}


if __name__ == "__main__":
    print("loading a small real mesh:", DATA)
    ds = torch.load(DATA, map_location="cpu", weights_only=False)
    seq = ds["data"] if isinstance(ds, dict) and "data" in ds else ds
    sample = seq[len(seq) // 2]
    print("sample: %d nodes, %d edges\n" % (sample.x.shape[0],
                                            sample.edge_index.shape[1]))
    rows = []
    print("%-8s %-30s %6s  %s" % ("alpha", "run", "share", "per global block"))
    for run, a in ARMS:
        try:
            r = measure(run, sample)
        except Exception as e:
            print("%-8s %-30s  FAILED: %s" % (a, run, e)); continue
        if r is None:
            print("%-8s %-30s  no data" % (a, run)); continue
        pb = " ".join("%.3f" % s for s in r.get("per_block", []))
        print("%-8s %-30s %6.3f  %s" % (a, run, r["share"], pb))
        rows.append({"alpha": a, "run": run, "share": r["share"],
                     "per_block": r.get("per_block", [])})
    os.makedirs("results/figures", exist_ok=True)
    json.dump(rows, open("results/figures/alpha_subordination.json", "w"),
              indent=2)
    print("\nwrote results/figures/alpha_subordination.json")
