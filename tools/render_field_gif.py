#!/usr/bin/env python
"""Render a high-resolution, transparent-background, white-text animation of a
rollout field (FI or damage) for a single simulation, reusing the SAME element-
footprint PolyCollection recipe as the paper field figures (export_fields_for_tikz).

Outputs (into --out):
  <sim>_<field>_<which>.gif   transparent background, white text (for slides)
  <sim>_<field>_<which>.mp4   H.264, flattened on black (no alpha in mp4),
                              white text -> looks seamless on a dark slide.

Examples
--------
# Laminate (4 plies), predicted FI rollout:
python tools/render_field_gif.py \
  --run results/<run> --sim sim_013 \
  --field fi --which pred --out figures_anim/laminate_fi

# Laminate predicted damage:
python tools/render_field_gif.py \
  --run results/<run> --sim sim_013 \
  --field damage --which pred --out figures_anim/laminate_dmg

# Ground-truth vs predicted side by side:
python tools/render_field_gif.py --run ... --sim sim_013 --field fi --which both ...
"""
import argparse
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_fields_for_tikz import read_vtu_full, _hex_footprint_quads, _frame_files

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from PIL import Image

WHITE = "#FFFFFF"


def _silhouette_segments(quads, decimals=5):
    """Outer boundary of a set of element quads = edges that belong to exactly
    one quad (the plate's outer square + the hole rim). Returns line segments;
    interior (shared) edges are dropped, so there is no element grid."""
    from collections import defaultdict
    cnt = defaultdict(int)
    seg = {}
    for q in quads:
        pts = [tuple(np.round(p, decimals)) for p in np.asarray(q)[:, :2]]
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            key = (a, b) if a <= b else (b, a)
            cnt[key] += 1
            seg[key] = (a, b)
    return [seg[k] for k, c in cnt.items() if c == 1]


# Global style: prefer Arial Nova (falls back to Arial -> DejaVu if not
# installed), and make ALL text white so nothing (incl. colorbar ticks) renders
# black on the transparent/black background.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "cmr10", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm",       # Computer Modern (matches the paper)
    "axes.unicode_minus": False,
    "text.color": WHITE,
    "axes.labelcolor": WHITE,
    "axes.edgecolor": WHITE,
    "xtick.color": WHITE,
    "ytick.color": WHITE,
})

# "damagepop": near-binary damage colormap. The big colour jump is in [0, 0.12]
# (black -> red), so intact (0) reads as background and any damage pops red;
# 0.12 -> 1 is a gentle red -> orange, so the *pattern* dominates over the
# magnitude (0.1 and 1.0 look similar, 0 looks completely different).
_DAMAGEPOP = LinearSegmentedColormap.from_list("damagepop", [
    (0.00, "#000000"),   # intact: black
    (0.04, "#5a0000"),   # immediate dark-red as soon as damage starts
    (0.12, "#d61a00"),   # bright red by ~0.12
    (1.00, "#ff7a1a"),   # gentle warm shift to orange across the rest
])
try:
    matplotlib.colormaps.register(_DAMAGEPOP, force=True)
except Exception:
    pass


def _draw_frame(verts, ply_id, plies, ply_label, values_by_ply, vmin, vmax,
                cmap, extent, title, out_png, dpi, which_cols, col_titles,
                cbar_label, sil_by_ply):
    """One animation frame: a row of ply panels (optionally GT/Pred rows),
    shared colorbar, transparent background, all text white."""
    x0, x1, y0, y1 = extent
    npl = len(plies)
    nrows = len(which_cols)
    panel_w = 2.4
    panel_h = panel_w * (y1 - y0) / (x1 - x0)
    panel_gap = 0.30   # horizontal spacing between plates (so plies read as separate)
    row_gap = 0.30     # vertical spacing between Ground-truth and Predicted rows
    left_m = 0.42      # left margin (room for the rotated row labels)
    # Content spans the panels (+ inter-panel gaps) + a gap + colorbar; enforce a
    # minimum figure width so the centred suptitle is not clipped on narrow
    # (single-ply) figures, and centre the panel/colorbar block within that width.
    panels_w = npl * panel_w + (npl - 1) * panel_gap
    # trailing 0.55 = right margin for the colorbar TICK LABELS (0..1 / 0..3);
    # too small and they fall outside the figure and are clipped (not rendered).
    content_w = left_m + panels_w + 0.18 + 0.16 + 0.85
    fig_w = max(content_w, 4.6)
    x_off = (fig_w - content_w) / 2.0
    fig_h = nrows * panel_h + (nrows - 1) * row_gap + 0.9  # +headroom for title
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    norm = Normalize(vmin=vmin, vmax=vmax)
    for r, col in enumerate(which_cols):
        for c, p in enumerate(plies):
            ax = fig.add_axes([
                (x_off + left_m + c * (panel_w + panel_gap)) / fig_w,
                1.0 - (0.55 + (r + 1) * panel_h + r * row_gap) / fig_h,
                panel_w / fig_w,
                panel_h / fig_h,
            ])
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            ax.set_axis_off()
            mask = ply_id == p
            pc = PolyCollection(
                [verts[i] for i in np.where(mask)[0]],
                array=values_by_ply[col][mask],
                cmap=cmap, norm=norm,
                # thin black element edges: visible on the coloured field,
                # invisible on the black background
                edgecolors=(0, 0, 0, 0.35), linewidths=0.1, alpha=1.0,
            )
            ax.add_collection(pc)
            # Optional plate silhouette (outer square + hole rim), thin and
            # translucent
            if sil_by_ply is not None:
                ax.add_collection(LineCollection(
                    sil_by_ply[p], colors=(1, 1, 1, 0.5), linewidths=0.4))
            if r == 0:
                ax.set_title(ply_label[p], color=WHITE, fontsize=15, fontweight="bold", pad=4)
            if c == 0 and nrows > 1:
                ax.text(-0.06, 0.5, col_titles[col], transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        color=WHITE, fontsize=16, fontweight="bold")

    # Shared colorbar (white label + ticks)
    cax = fig.add_axes([(x_off + left_m + panels_w + 0.18) / fig_w, 0.12,
                        0.16 / fig_w, 0.70])
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(cbar_label, color=WHITE, fontsize=15, fontweight="bold")
    cb.ax.yaxis.set_tick_params(color=WHITE, labelcolor=WHITE, labelsize=13)
    cb.outline.set_edgecolor(WHITE)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=WHITE)

    fig.suptitle(title, color=WHITE, fontsize=19, fontweight="bold", y=0.985)
    fig.savefig(out_png, transparent=True, dpi=dpi)
    plt.close(fig)


def _assemble_gif(png_paths, out_gif, fps):
    """Transparent animated GIF via Pillow. Uses ONE GLOBAL palette derived from
    a montage of sampled frames, so a fixed field value keeps the SAME colour in
    every frame (a per-frame ADAPTIVE palette made the colours drift/fluctuate).
    Index 255 is reserved transparent."""
    rgba = [Image.open(p).convert("RGBA") for p in png_paths]
    w, h = rgba[0].size
    sample = rgba[:: max(1, len(rgba) // 20)]            # ~20 frames for the palette
    # Seed the palette with white, greys and black (a full-width band) so that the
    # white text and grid get their own palette slot; otherwise median-cut maps
    # them to the nearest field colour.
    swatch = Image.new("RGB", (w, h), (0, 0, 0))
    band = max(1, h // 5)
    for j, c in enumerate([(255, 255, 255), (200, 200, 200), (120, 120, 120), (40, 40, 40)]):
        swatch.paste(c, (0, j * band, w, (j + 1) * band))
    montage = Image.new("RGB", (w, h * (len(sample) + 1)), (0, 0, 0))
    montage.paste(swatch, (0, 0))
    for i, im in enumerate(sample):
        montage.paste(im.convert("RGB"), (0, (i + 1) * h))
    global_pal = montage.quantize(colors=255, method=Image.MEDIANCUT)

    frames = []
    for im in rgba:
        alpha = im.getchannel("A")
        p = im.convert("RGB").quantize(palette=global_pal, dither=Image.Dither.NONE)
        mask = alpha.point(lambda a: 255 if a <= 128 else 0)
        p.paste(255, mask=mask)                          # index 255 = transparent
        frames.append(p)
    frames[0].save(
        out_gif, save_all=True, append_images=frames[1:],
        duration=int(1000 / fps), loop=0,
        transparency=255, disposal=2, optimize=False,
    )


def _assemble_mp4(png_dir, pattern, out_mp4, fps, size, bg="black"):
    """H.264 mp4 (no alpha). Transparent PNGs are explicitly composited over a
    solid canvas (matplotlib writes white RGB under alpha=0, so we must overlay
    on a real colour rather than let yuv420p drop the alpha). bg='black' for
    dark slides (white text); bg='white' for paper-style (--light, black text)."""
    import subprocess
    w, h = size
    w += w % 2; h += h % 2  # libx264 needs even dimensions
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(png_dir, pattern),
        "-filter_complex",
        f"color={bg}:s={w}x{h}:r={fps}[bg];"
        f"[bg][0:v]overlay=shortest=1:format=auto,format=yuv420p",
        "-c:v", "libx264", "-crf", "16",
        out_mp4,
    ]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="results/<run> dir with test_results/vtk")
    ap.add_argument("--sim", required=True, help="e.g. sim_013")
    ap.add_argument("--field", required=True, choices=["fi", "damage"])
    ap.add_argument("--which", default="pred", choices=["pred", "gt", "both"])
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--vmax", type=float, default=None,
                    help="colour scale max (default: 3.0 for fi, 1.0 for damage)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--cmap", default=None,
                    help="colormap override (default: viridis/fi, plasma/damage). "
                         "'damagepop' = near-binary black->red.")
    ap.add_argument("--silhouette", action="store_true",
                    help="draw the white plate outline (outer square + hole). "
                         "Off by default.")
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument("--light", action="store_true",
                    help="paper-style output: black text, mp4 composited on a "
                         "WHITE canvas (default: white text, black canvas).")
    args = ap.parse_args()

    if args.light:
        global WHITE
        WHITE = "#000000"  # rebind the module text colour used in _draw_frame
        plt.rcParams.update({
            "text.color": WHITE, "axes.labelcolor": WHITE, "axes.edgecolor": WHITE,
            "xtick.color": WHITE, "ytick.color": WHITE,
        })

    files = _frame_files(args.run, args.sim)[:: args.frame_stride]
    os.makedirs(args.out, exist_ok=True)

    cmap = args.cmap or ("plasma" if args.field == "damage" else "viridis")
    vmax = args.vmax if args.vmax is not None else (1.0 if args.field == "damage" else 3.0)
    cols = {"pred": ["pred"], "gt": ["gt"], "both": ["gt", "pred"]}[args.which]
    col_titles = {"gt": "Ground truth", "pred": "Predicted"}

    # Geometry + ply layout from the first frame (constant across frames)
    pts, conn, starts, cen, f0, _ = read_vtu_full(files[0])
    verts = _hex_footprint_quads(pts, conn, starts, cen)
    ply_id = f0.get("ply_id", np.zeros(len(cen))).astype(int)
    ply_ang = f0.get("ply_angle_deg", np.zeros(len(cen)))
    plies = sorted(np.unique(ply_id).tolist())
    ply_label = {p: f"Ply {i+1} (${int(round(float(ply_ang[ply_id==p][0])))}^\\circ$)"
                 for i, p in enumerate(plies)}
    # Per-ply plate silhouette (outer square + hole), computed ONCE (mesh is
    # constant across frames). Only when --silhouette is requested.
    sil_by_ply = ({p: _silhouette_segments([verts[i] for i in np.where(ply_id == p)[0]])
                   for p in plies} if args.silhouette else None)
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    extent = (x0, x1, y0, y1)
    # matrix-cracking failure index FI_mc and damage variable D
    fname = r"$\mathrm{FI}_{mc}$" if args.field == "fi" else r"$D$"
    cbar_label = fname

    tmp = tempfile.mkdtemp(prefix="anim_")
    png_paths = []
    for k, fpath in enumerate(files):
        _, _, _, _, fields, t = read_vtu_full(fpath)
        vals = {"gt": fields[f"{args.field}_gt"], "pred": fields[f"{args.field}_pred"]}
        title = f"{fname},  $t = {t:.2f}$"   # no em-dash/underscore (cmr10 lacks them)
        out_png = os.path.join(tmp, f"frame_{k:04d}.png")
        _draw_frame(verts, ply_id, plies, ply_label, vals, 0.0, vmax,
                    cmap, extent, title, out_png, args.dpi, cols, col_titles,
                    cbar_label, sil_by_ply)
        png_paths.append(out_png)
        if k % 20 == 0:
            print(f"  frame {k+1}/{len(files)}  t={t:.2f}")

    base = os.path.join(args.out, f"{args.sim}_{args.field}_{args.which}")
    _assemble_gif(png_paths, base + ".gif", args.fps)
    print(f"GIF  -> {base}.gif  ({len(png_paths)} frames, transparent, {args.dpi} dpi)")
    if not args.no_mp4:
        with Image.open(png_paths[0]) as im0:
            size = im0.size
        bg = "white" if args.light else "black"
        _assemble_mp4(tmp, "frame_%04d.png", base + ".mp4", args.fps, size, bg=bg)
        print(f"MP4  -> {base}.mp4  (H.264, composited on {bg})")


if __name__ == "__main__":
    main()
