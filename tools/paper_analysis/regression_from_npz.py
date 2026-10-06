"""Rebuild the laminate regression hexbin raster from rollout_arrays.npz, replicating
tools/export_fields_for_tikz.py cmd_regression (70 bins, mincnt 10, linear
colour cap = P99 of kept bin counts, Reds, transparent, 3.2 in at 350 dpi).
The npz holds the same pooled (gt, pred) FI samples as the per-frame VTKs."""
import sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
npz, out_png = sys.argv[1], sys.argv[2]
d = np.load(npz, allow_pickle=True)
def flat(a):
    if a.dtype == object:
        return np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in a])
    return a.astype(np.float64).ravel()
gt = flat(d["ground_truth_damage"])
pr = flat(d["predicted_damage"])
n = gt.size
r2 = 1 - ((pr-gt)**2).sum()/((gt-gt.mean())**2).sum()
lo = min(gt.min(), pr.min()); hi = max(gt.max(), pr.max())
H, xe, ye = np.histogram2d(gt, pr, bins=70, range=[[lo, hi], [lo, hi]])
c = H[H > 0]; keep = c[c >= 10]
vmax = float(np.percentile(keep, 99.0))
fig = plt.figure(figsize=(3.2, 3.2), dpi=350); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_axis_off()
ax.hexbin(gt, pr, gridsize=70, cmap="Reds", norm=plt.Normalize(vmin=0, vmax=vmax),
          extent=[0, hi, 0, hi], mincnt=10, linewidths=0.05)
fig.savefig(out_png, transparent=True, dpi=350); plt.close(fig)
print(json.dumps({"N": int(n), "R2": round(float(r2), 4), "lo": float(lo), "hi": float(hi), "P99_cap": vmax}))
