#!/usr/bin/env python
"""Iso-view 3D raster of the full 8-ply laminate (the 4 modelled plies mirrored
about the mid-plane symmetry) coloured by FI or damage, at chosen pseudo-times.
Each ply is a layer of its element quads at an (exaggerated) through-thickness z.
Output: transparent PNGs with only the coloured laminate (no axes, no text),
to be embedded via \\addplot graphics in the TikZ field figure, as for the 2D
per-ply panels.

Example
-------
python tools/render_laminate_3d.py --run results/<run> \
  --sim sim_013 --field fi --which pred \
  --times 0,0.25,0.5,0.75,1.0 --out <figure dir>/data
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_fields_for_tikz import read_vtu_full, _hex_footprint_quads, _frame_files, _vtu_time


def _boundary(qsub):
    """Boundary edges of a ply's quad set = edges in exactly one quad. Returns
    [((a,b), elem_local_idx), ...] so each side wall can be coloured by its
    cross-section element's field."""
    from collections import defaultdict
    cnt = defaultdict(int)
    info = {}
    for ei, q in enumerate(qsub):
        pts = [tuple(np.round(p, 5)) for p in np.asarray(q)[:, :2]]
        for i in range(4):
            a, b = pts[i], pts[(i + 1) % 4]
            key = (a, b) if a <= b else (b, a)
            cnt[key] += 1
            info[key] = ((a, b), ei)
    return [info[k] for k, c in cnt.items() if c == 1]


def _render(verts2d, ply_id, plies, values, vmin, vmax, cmap, extent,
            out_png, dpi, elev, azim, z_frac, mirror=True):
    x0, x1, y0, y1 = extent
    span = max(x1 - x0, y1 - y0)
    H = span * z_frac                    # exaggerated total laminate height
    cell = H / 8.0
    t_slab = cell                        # plies TOUCH (cohesive interfaces are
                                         # zero-thickness) -> solid laminate block
    cm = plt.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # full 8-ply stack: modelled plies 0..3 then their mirror 3..0 (symmetry);
    # field duplicated onto the mirrored ply. --no-mirror (single ply) -> 1 layer.
    layer_order = list(plies) + list(plies[::-1]) if mirror else list(plies)
    polys, facecolors = [], []
    for zi, p in enumerate(layer_order):
        m = np.where(ply_id == p)[0]
        qsub = [verts2d[i] for i in m]
        cols = cm(norm(values[m]))
        z_bot = zi * cell
        z_top = z_bot + t_slab
        # top face of the slab (colored by field)
        for k, q in enumerate(qsub):
            polys.append([(vx, vy, z_top) for vx, vy in q])
            facecolors.append(cols[k])
        # side walls (extruded boundary) give the slab visible thickness,
        # coloured by the cross-section element's field
        for (a, b), ei in _boundary(qsub):
            polys.append([(a[0], a[1], z_bot), (b[0], b[1], z_bot),
                          (b[0], b[1], z_top), (a[0], a[1], z_top)])
            facecolors.append(cols[ei])

    fig = plt.figure(figsize=(4.2, 3.0), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    pc = Poly3DCollection(polys, facecolors=facecolors,
                          edgecolors=(0, 0, 0, 0.12), linewidths=0.05, shade=True)
    ax.add_collection3d(pc)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(0, 8 * cell)
    try:
        ax.set_box_aspect((x1 - x0, y1 - y0, 8 * cell))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.savefig(out_png, transparent=True, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--field", default="fi", choices=["fi", "damage"])
    ap.add_argument("--which", default="pred", choices=["gt", "pred"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--times", default=None, help="comma pseudo-times, e.g. 0,0.25,0.5,0.75,1.0")
    ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--elev", type=float, default=22)
    ap.add_argument("--azim", type=float, default=-58)
    ap.add_argument("--z-frac", type=float, default=0.35)
    ap.add_argument("--cmap", default=None,
                    help="colormap override (default viridis/fi, magma/damage; "
                         "use plasma for the damage ANIMATION so it reads on black)")
    ap.add_argument("--norm-column", action="store_true",
                    help="normalise each frame to its own max (matches the per-ply "
                         "FI/FI^max_t paper colorbar for the integrated figure row)")
    ap.add_argument("--no-mirror", action="store_true",
                    help="do not mirror plies about the mid-plane (use for a single ply)")
    ap.add_argument("--animate", action="store_true",
                    help="render every --frame-stride-th frame and assemble a "
                         "transparent GIF + black-bg MP4 of the 3D iso evolution")
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--light", action="store_true",
                    help="paper-style animation: mp4 composited on WHITE (default black)")
    args = ap.parse_args()

    files = _frame_files(args.run, args.sim)
    if args.times:
        wanted = [float(t) for t in args.times.split(",")]
        allt = [_vtu_time(f) for f in files]
        idx = [int(np.argmin([abs((t if t is not None else 1e9) - w) for t in allt]))
               for w in wanted]
    else:
        idx = np.unique(np.linspace(0, len(files) - 1, args.n_frames).round().astype(int)).tolist()

    pts, conn, starts, cen, f0, _ = read_vtu_full(files[0])
    verts2d = _hex_footprint_quads(pts, conn, starts, cen)
    ply_id = f0.get("ply_id", np.zeros(len(cen))).astype(int)
    plies = sorted(np.unique(ply_id).tolist())
    extent = (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
    cmap = args.cmap or ("magma" if args.field == "damage" else "viridis")   # match per-ply paper figs
    vmax = args.vmax if args.vmax is not None else (1.0 if args.field == "damage" else 3.0)
    key = f"{args.field}_{args.which}"
    os.makedirs(args.out, exist_ok=True)

    if args.animate:
        import tempfile
        from render_field_gif import _assemble_gif, _assemble_mp4
        from PIL import Image
        tmp = tempfile.mkdtemp(prefix="lam3d_")
        pngs = []
        anim_idx = list(range(0, len(files), args.frame_stride))
        for k, fi_ in enumerate(anim_idx):
            _, _, _, _, fields, t = read_vtu_full(files[fi_])
            p = os.path.join(tmp, f"frame_{k:04d}.png")
            _render(verts2d, ply_id, plies, fields[key], 0.0, vmax, cmap, extent,
                    p, args.dpi, args.elev, args.azim, args.z_frac, not args.no_mirror)
            pngs.append(p)
            if k % 10 == 0:
                print(f"  frame {k+1}/{len(anim_idx)}  t={t:.2f}")
        base = os.path.join(args.out, f"{args.sim}_{args.field}3d_{args.which}")
        _assemble_gif(pngs, base + ".gif", args.fps)
        with Image.open(pngs[0]) as im0:
            size = im0.size
        _assemble_mp4(tmp, "frame_%04d.png", base + ".mp4", args.fps, size,
                      bg=("white" if getattr(args, "light", False) else "black"))
        print(f"GIF/MP4 -> {base}.gif / .mp4  ({len(pngs)} frames)")
        return

    for k, fi_ in enumerate(idx):
        _, _, _, _, fields, t = read_vtu_full(files[fi_])
        # per-frame normalisation (column norm) to match the per-ply paper figure's
        # FI/FI^max_t colorbar, so the integrated row shares one 0..1 colorbar.
        vmx = float(fields[key].max()) or 1.0 if args.norm_column else vmax
        out = os.path.join(args.out, f"{args.sim}_{args.field}3d_f{k:02d}_{args.which}.png")
        _render(verts2d, ply_id, plies, fields[key], 0.0, vmx, cmap, extent,
                out, args.dpi, args.elev, args.azim, args.z_frac, not args.no_mirror)
        print(f"  t={t:.2f} -> {out} (vmax={vmx:.2f})")


if __name__ == "__main__":
    main()
