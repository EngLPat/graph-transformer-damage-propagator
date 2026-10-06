#!/usr/bin/env python
"""Render the GLOBAL ATTENTION field (received attention per node) as tikz-native
element-footprint panels, matching the FI/damage field figures
(tools/export_fields_for_tikz.py fields-mesh).

Input: the per-frame npz files dumped by --attention-analysis
       (<out>/attention_analysis/attn_field_frame_*.npz), each containing
       received_attn / received_max / entropy / long_range_ratio (per node),
       plus mesh_pos, damage, frame_idx.
Geometry (element quads, ply_id, ply_angle) is read from the run's VTU frames,
exactly as the field exporter does, and the per-node attention scalar is mapped
onto the same quads by element index.

Output (into --out):
  data/<sim>_attn_f??_ply?.png     transparent element-footprint panels
  fig_attention_<sim>.tex          \\input-able pgfplots figure (npl rows x
                                   n_frames cols, shared colorbar, ply braces)

Each column is normalised to its own max, so the SPATIAL pattern (where the
model attends) is visible even when the absolute attention is diffuse; the
colour therefore reads as "received attention / column max".

Example
-------
python tools/render_attention_field.py \
  --attn-dir <test dir>/test_results/attention_analysis \
  --run results/<run> --sim sim_013 \
  --out <figure dir>
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_fields_for_tikz import (
    read_vtu_full, _hex_footprint_quads, _frame_files, _render_mesh_panel_png,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attn-dir", required=True, help="dir with attn_field_frame_*.npz")
    ap.add_argument("--run", required=True, help="results/<run> with test_results/vtk")
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True, help="output dir (tex + data/)")
    ap.add_argument("--metric", default="received_attn",
                    choices=["received_attn", "received_max", "entropy",
                             "long_range_ratio"])
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--panel-width", type=float, default=0.15,
                    help="per-panel width as a fraction of \\linewidth")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--edge-alpha", type=float, default=0.12)
    ap.add_argument("--edge-lw", type=float, default=0.25)
    ap.add_argument("--clip-sink", action="store_true",
                    help="cap the per-frame colour scale at --clip-pct so an "
                         "attention-sink node does not flatten the rest of the "
                         "field (mainly for the received_* metrics)")
    ap.add_argument("--clip-pct", type=float, default=99.0,
                    help="percentile cap used when --clip-sink is set (default 99)")
    args = ap.parse_args()

    # ── npz frames, ordered by frame_idx (= time order) ──
    npz_files = sorted(glob.glob(os.path.join(args.attn_dir, "attn_field_frame_*.npz")))
    if not npz_files:
        sys.exit(f"No attn_field_frame_*.npz in {args.attn_dir}")
    frames = []
    for f in npz_files:
        d = np.load(f)
        frames.append((int(d["frame_idx"]), d[args.metric], d.get("mesh_pos")))
    frames.sort(key=lambda t: t[0])
    n = len(frames)

    # ── geometry from the first VTU (constant mesh) ──
    files = _frame_files(args.run, args.sim)
    pts, conn, starts, cen, f0, _ = read_vtu_full(files[0])
    verts = _hex_footprint_quads(pts, conn, starts, cen)
    ply_id = f0.get("ply_id", np.zeros(len(cen))).astype(int)
    ply_ang = f0.get("ply_angle_deg", np.zeros(len(cen)))
    plies = sorted(np.unique(ply_id).tolist())
    npl = len(plies)
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    extent = (x0, x1, y0, y1)

    # alignment sanity: attention is per graph-node == per element-centroid
    if len(frames[0][1]) != len(cen):
        sys.exit(f"attention length {len(frames[0][1])} != #elements {len(cen)} "
                 "(node/element ordering mismatch)")
    mp = frames[0][2]
    if mp is not None and mp.shape == cen.shape:
        d = float(np.abs(mp - cen).max())
        if d > 1e-3:
            print(f"  WARNING: npz mesh_pos differs from VTU centroids (max {d:.2e})")

    os.makedirs(args.out, exist_ok=True)
    datadir = os.path.join(args.out, "data")
    os.makedirs(datadir, exist_ok=True)

    # ── render one panel per (frame, ply); per-column normalisation ──
    png_names, times = {}, []
    for k, (fidx, vals, _mp) in enumerate(frames):
        vals = np.asarray(vals, dtype=float)
        if args.clip_sink:
            cap = float(np.nanpercentile(vals, args.clip_pct))
            if cap > 0:
                vals = np.minimum(vals, cap)   # sink saturates; structure underneath shows
        vmax = float(np.nanmax(vals)) or 1.0
        valsn = vals / vmax
        times.append(k / (n - 1) if n > 1 else 0.0)  # approx pseudo-time (even sampling)
        for p in plies:
            m = ply_id == p
            png = f"{args.sim}_attn_f{k:02d}_ply{p}.png"
            _render_mesh_panel_png(
                [verts[i] for i in np.where(m)[0]], valsn[m], 0.0, 1.0, extent,
                os.path.join(datadir, png), dpi=args.dpi,
                edge_alpha=args.edge_alpha, edge_lw=args.edge_lw, cmap=args.cmap)
            png_names[(k, p)] = png

    # ── TeX: npl rows x n cols + shared colorbar + ply braces ──
    cb_label = r"received attention\,/\,col.\ max"
    lines = [
        "% Auto-generated by tools/render_attention_field.py — do not edit.",
        f"% sim={args.sim} metric={args.metric}; {npl} plies x {n} pseudo-times.",
        "% Global received-attention on the element footprint (same recipe as",
        "% the FI/damage field figures); each column normalised to its own max.",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        f"  group style={{group size={n} by {npl}, horizontal sep=1.0mm, "
        "vertical sep=1.0mm},",
        f"  width={args.panel_width}\\linewidth, scale only axis, axis equal image,",
        r"  enlargelimits=false, xtick=\empty, ytick=\empty,",
        f"  colormap/{args.cmap}, point meta min=0, point meta max=1,",
        r"  title style={font=\scriptsize},",
        r"]",
    ]
    for p_i, p in enumerate(plies):
        for k in range(n):
            opts = []
            if p_i == 0:
                opts.append(f"title={{$t\\approx{times[k]:.2f}$}}")
            lines.append(f"\\nextgroupplot[{', '.join(opts)}]")
            png = png_names[(k, p)]
            lines.append(
                f"\\addplot graphics[xmin={x0:.4g}, xmax={x1:.4g}, "
                f"ymin={y0:.4g}, ymax={y1:.4g}] "
                f"{{figures_tikz/data/{png}}};")
    lines.append(r"\end{groupplot}")

    # shared colorbar: standalone hidden axis spanning the full grid height
    last_col, bottom_row = n, npl
    lines += [
        f"\\path let \\p1=(group c{last_col}r1.north east),"
        f" \\p2=(group c{last_col}r{bottom_row}.south east) in"
        r" \pgfextra{\xdef\GridH{\the\dimexpr\y1-\y2\relax}}"
        r" coordinate (cbmid) at ($(\x1,\y1)!0.5!(\x1,\y2)$);",
        r"\begin{axis}[",
        r"  hide axis, scale only axis, width=0pt, height=\GridH,",
        f"  colormap/{args.cmap}, point meta min=0, point meta max=1,",
        r"  at={(cbmid)}, anchor=west, xshift=4mm,",
        r"  colorbar right, colorbar style={",
        f"    ylabel={{{cb_label}}}, ylabel style={{font=\\scriptsize}},",
        r"    width=3mm, tick label style={font=\scriptsize}},",
        r"]",
        r"\addplot[draw=none] coordinates {(0,0) (1,1)};",
        r"\end{axis}",
    ]
    # ply braces left of each row
    for p_i, p in enumerate(plies):
        ang = float(ply_ang[ply_id == p][0]) if (ply_id == p).any() else 0.0
        r = p_i + 1
        lines.append(
            "\\draw[decorate, decoration={brace, amplitude=4pt}] "
            f"($(group c1r{r}.south west)+(-6mm,0)$) -- "
            f"($(group c1r{r}.north west)+(-6mm,0)$) "
            "node[midway, rotate=90, anchor=south, outer sep=3pt, "
            "font=\\scriptsize\\bfseries] "
            f"{{Ply {p_i + 1} ($ {ang:.0f}^\\circ$)}};")
    lines.append(r"\end{tikzpicture}")

    out = os.path.join(args.out, f"fig_attention_{args.sim}.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"attention figure: {n} pseudo-times x {npl} plies -> {out}")
    print(f"  panels in {datadir}")


if __name__ == "__main__":
    main()
