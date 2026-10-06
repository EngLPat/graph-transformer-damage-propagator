"""Counterfactual Y_T sweep on one post-initiation state of EACH of the 60 test
simulations (5 steps after the first FI > 1): the Y_T model's predicted FI rate at
nine values of Y_T in [51, 69] MPa, normalised per state by the prediction at
Y_T = 60 MPa. Produces the sweep numbers of the material-variability study.

Usage:  GTP_ROOT=<folder with data/ and results/> python tools/paper_analysis/sweep_many.py <out_dir>
<out_dir> must hold dt001_yt/test_results/rollout_arrays.npz (from dt001.sh; it
lists the test simulations) and receives sweep_many.json.
"""
import sys, os, json, glob
from pathlib import Path
import numpy as np, torch

CODE = str(Path(__file__).resolve().parents[2])        # this repository
sys.path.insert(0, CODE)
ROOT = os.environ.get("GTP_ROOT", CODE)                # data/ and results/
S = sys.argv[1]

from src.gtp.config import Config
from src.gtp.data.loader import load_dataset
from src.gtp.evaluation.evaluator import Evaluator

YT_COL = 13
DS = ("data/processed/laminate_yt_variability/"
      "laminate_yt_variability_dmgstate_ytfeat_"
      "dataset_uniform_dt0.01.pt")
RUN = "results/yt_variability"


def load_model(ckpt_path, stats_path, sample, device):
    """Rebuild the model as trained: the run's config.yaml first (forward-only
    flags such as attention_freq cannot be inferred from the weights), then any
    config embedded in the checkpoint; frozen, in eval mode."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = Config()
    run_cfg = Path(ckpt_path).parent.parent / "config.yaml"
    if run_cfg.exists():
        import yaml
        with open(run_cfg) as fp:
            for k, v in (yaml.safe_load(fp) or {}).items():
                if v is not None:
                    setattr(config, k, v)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
        for k, v in ckpt["config"].items():
            if v is not None:
                setattr(config, k, v)
    model = Evaluator(config=config, device=device).load_model(
        Path(ckpt_path), sample.x.shape[1], sample.edge_attr.shape[1])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    stats = [s.to(device) for s in torch.load(stats_path, weights_only=False)]
    return model, stats


def predict(model, stats, sample, yt, device):
    s = sample.clone().to(device)
    if yt is not None:
        s.x[:, YT_COL] = float(yt)
    s.previous_damage = s.x[:, 14:15].clone()
    with torch.no_grad():
        y = model(s, stats[0], stats[1], stats[2], stats[3])
    y = y[0] if isinstance(y, (tuple, list)) else y
    return y.reshape(-1).float().cpu().numpy()


test_sims=sorted(set(np.asarray(np.load(S+"/dt001_yt/test_results/rollout_arrays.npz",allow_pickle=True)["sim_ids_per_frame"]).astype(int).tolist()))
ds=load_dataset(os.path.join(ROOT,DS),verbose=False)
ck=glob.glob(os.path.join(ROOT,RUN,"best_models","model_*.pt"))[0]
model,stats=load_model(ck,os.path.join(ROOT,RUN,"normalization_stats.pt"),ds[0],"cuda:0")
by_sim={}
for i,s in enumerate(ds):
    sid=int(s.sample_idx)
    if sid in test_sims: by_sim.setdefault(sid,[]).append(i)
sweep=np.linspace(51.029,68.999,9)
rows=[]
for sid in test_sims:
    idx=by_sim[sid]
    dmg=[k for k in idx if float(ds[k].x[:,15].max())>1.0]
    if not dmg: continue
    k=idx[min(idx.index(dmg[0])+5,len(idx)-1)]
    obs=np.array([float(predict(model,stats,ds[k],float(q),"cuda:0").mean()) for q in sweep])
    ref=float(predict(model,stats,ds[k],60.0,"cuda:0").mean())
    rows.append((obs/ref).tolist())
R=np.array(rows)
print(f"{len(R)} states; mean relative rate at 51 / 69 MPa: {R[:,0].mean():.3f} / {R[:,-1].mean():.3f}; share of states decreasing: {(R[:,-1]<R[:,0]).mean():.2f}")
json.dump(dict(sweep=sweep.tolist(),rel=R.tolist(),mean=R.mean(0).tolist()),open(S+"/sweep_many.json","w"))
