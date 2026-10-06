#!/usr/bin/env python
"""Export test-run data as CSV + pgfplots-ready TeX for the paper's figures.

Three subcommands, all reading from a results directory produced by `--mode test`:

  fields      Per-ply (x, y, gt, pred) tables for FI / damage at selected frames,
              read from the VTU files in <run>/test_results/vtk/<sim>/.
              Also emits a ready-to-\\input groupplot TeX panel (GT row / Pred row).

  loss        (epoch, train, val) table parsed from the execution_log_train_*.txt
              files of one or more runs (later logs win on duplicated epochs).

  regression  Binned 2-D histogram of (gt, pred) pooled over frames, plus R2/RMSE/N,
              for the previous-paper style correlation plot.

Examples (gtp conda env):
  python tools/export_fields_for_tikz.py fields \
      --run results/<run> --sim sim_003 \
      --field fi --n-frames 5 --out <figure dir>
  python tools/export_fields_for_tikz.py loss \
      --run results/<run> --name attention \
      --out <figure dir>
  python tools/export_fields_for_tikz.py regression \
      --run results/<run> --sim all --field fi \
      --out <figure dir>
"""

import argparse
import base64
import glob
import json
import os
import re
import sys
import zlib
import xml.etree.ElementTree as ET

import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# Minimal VTU (inline-binary, zlib-compressed) reader, without meshio
# ──────────────────────────────────────────────────────────────────────────

_VTK_DTYPES = {
    "Float32": np.float32, "Float64": np.float64,
    "Int8": np.int8, "Int16": np.int16, "Int32": np.int32, "Int64": np.int64,
    "UInt8": np.uint8, "UInt16": np.uint16, "UInt32": np.uint32,
    "UInt64": np.uint64,
}


def _decode_inline_binary(text, dtype, header_dtype=np.uint32):
    """Decode a VTK inline base64 DataArray compressed with zlib."""
    raw = "".join(text.split())
    hsize = np.dtype(header_dtype).itemsize
    # First header word: number of blocks
    head = base64.b64decode(raw[: 4 * ((hsize + 2) // 3)] + "==")
    nblocks = int(np.frombuffer(head[:hsize], header_dtype)[0])
    header_len = hsize * (3 + nblocks)
    b64_header_len = 4 * ((header_len + 2) // 3)
    header = np.frombuffer(
        base64.b64decode(raw[:b64_header_len])[:header_len], header_dtype
    )
    comp_sizes = header[3:]
    data = base64.b64decode(raw[b64_header_len:])
    out, off = [], 0
    for cs in comp_sizes:
        out.append(zlib.decompress(data[off:off + int(cs)]))
        off += int(cs)
    return np.frombuffer(b"".join(out), dtype=dtype)


def read_vtu_full(path):
    """Return (points[N,3], conn, starts, centroids[M,3], {name: arr[M]}, t)
    from one frame VTU: geometry and per-cell fields."""
    root = ET.parse(path).getroot()
    compressed = "compressor" in root.attrib
    piece = root.find(".//Piece")

    def read_da(da):
        dtype = _VTK_DTYPES[da.attrib["Type"] if "Type" in da.attrib
                            else da.attrib["type"]]
        if da.attrib.get("format") == "ascii":
            return np.fromstring(da.text, dtype=dtype, sep=" ")
        if not compressed:
            raw = base64.b64decode("".join(da.text.split()))
            n = np.frombuffer(raw[:4], np.uint32)[0]
            return np.frombuffer(raw[4:4 + n], dtype=dtype)
        return _decode_inline_binary(da.text, dtype)

    pts_da = piece.find("Points/DataArray")
    ncomp = int(pts_da.attrib.get("NumberOfComponents", 3))
    points = read_da(pts_da).reshape(-1, ncomp)

    conn = read_da(piece.find("Cells/DataArray[@Name='connectivity']")).astype(np.int64)
    offsets = read_da(piece.find("Cells/DataArray[@Name='offsets']"))
    starts = np.concatenate([[0], offsets[:-1]]).astype(np.int64)
    counts = (offsets - starts).astype(np.float64)
    sums = np.add.reduceat(points[conn], starts, axis=0)
    centroids = sums / counts[:, None]

    fields = {}
    for da in piece.findall("CellData/DataArray"):
        fields[da.attrib["Name"]] = read_da(da)

    time_s = None
    tda = root.find(".//FieldData/DataArray[@Name='time_s']")
    if tda is not None:
        time_s = float(read_da(tda)[0])
    return points, conn, starts, centroids, fields, time_s


def read_vtu_celldata(path):
    """Return (centroids[M,3], {name: array[M]}, time_s) from one frame VTU.

    Fields are exported per element (CellData); plotting positions are the
    cell centroids, recovered from connectivity/offsets.
    """
    _, _, _, centroids, fields, time_s = read_vtu_full(path)
    return centroids, fields, time_s


def _hex_footprint_quads(points, conn, starts, centroids):
    """Project each hex cell to its in-plane (XY) footprint quad.

    For each cell, the 4 corners below the centroid z (bottom face) are
    ordered counter-clockwise around the footprint centroid. This is the
    same element footprint the testing script's hex-quad renderer draws
    (src/gtp/visualization/plotting.py:_build_element_hex_mesh).

    Returns verts [M, 4, 2].
    """
    M = len(starts)
    verts = np.empty((M, 4, 2))
    ends = np.concatenate([starts[1:], [len(conn)]])
    for i in range(M):
        corners = points[conn[starts[i]:ends[i]]]
        zc = centroids[i, 2]
        bottom = corners[corners[:, 2] < zc]
        if len(bottom) != 4:  # degenerate: fall back to 4 lowest-z corners
            bottom = corners[np.argsort(corners[:, 2])[:4]]
        xy = bottom[:, :2]
        c = xy.mean(axis=0)
        order = np.argsort(np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0]))
        verts[i] = xy[order]
    return verts


# ──────────────────────────────────────────────────────────────────────────
# fields
# ──────────────────────────────────────────────────────────────────────────

def _frame_files(run, sim):
    files = sorted(glob.glob(os.path.join(
        run, "test_results", "vtk", sim, f"{sim}_frame_*.vtu")))
    if not files:
        sys.exit(f"No VTU frames under {run}/test_results/vtk/{sim}")
    return files


def cmd_fields(args):
    files = _frame_files(args.run, args.sim)
    if args.frames:
        idx = [int(i) for i in args.frames.split(",")]
    else:
        idx = np.unique(np.linspace(0, len(files) - 1, args.n_frames)
                        .round().astype(int)).tolist()
    os.makedirs(args.out, exist_ok=True)
    datadir = os.path.join(args.out, "data")
    os.makedirs(datadir, exist_ok=True)

    gt_key, pr_key = f"{args.field}_gt", f"{args.field}_pred"
    meta = {"sim": args.sim, "field": args.field, "frames": idx,
            "times": [], "plies": {}, "vmin": None, "vmax": None}
    vmin, vmax = np.inf, -np.inf
    csv_names = {}

    for fi_ in idx:
        points, fields, t = read_vtu_celldata(files[fi_])
        if gt_key not in fields:
            sys.exit(f"Field {gt_key} not in {files[fi_]} "
                     f"(have: {sorted(fields)})")
        meta["times"].append(t)
        ply_id = fields.get("ply_id", np.zeros(len(points)))
        for p in np.unique(ply_id).astype(int):
            m = ply_id == p
            ang = float(np.unique(fields.get(
                "ply_angle_deg", np.zeros(len(points)))[m])[0])
            meta["plies"][str(p)] = ang
            arr = np.column_stack([points[m, 0], points[m, 1],
                                   fields[gt_key][m], fields[pr_key][m]])
            vmin = min(vmin, arr[:, 2:].min())
            vmax = max(vmax, arr[:, 2:].max())
            name = f"{args.sim}_{args.field}_f{fi_:04d}_ply{p}.csv"
            np.savetxt(os.path.join(datadir, name), arr,
                       header="x,y,gt,pred", comments="", delimiter=",",
                       fmt="%.6g")
            csv_names[(fi_, p)] = name
    meta["vmin"], meta["vmax"] = float(vmin), float(vmax)
    with open(os.path.join(datadir,
              f"{args.sim}_{args.field}_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    _emit_fields_tex(args, idx, meta, csv_names)
    print(f"fields: {len(idx)} frames x {len(meta['plies'])} plies -> {datadir}")


def _render_panel_png(csv_path, col, vmin, vmax, out_png, dpi=300):
    """Rasterise one field panel (square element markers, viridis, no axes).

    The PNG carries only the data; axes, titles, and the colorbar stay vector
    (pgfplots), which keeps pdflatex memory low (raw scatter panels exceed
    pdflatex's main memory at ~15k marks per figure).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.genfromtxt(csv_path, delimiter=",", names=True)
    x, y, v = arr["x"], arr["y"], arr[col]
    # element size ~ median nearest-neighbour distance between centroids
    # (robust on meshes mixing structured blocks and rings around the hole)
    from scipy.spatial import cKDTree
    dists, _ = cKDTree(np.column_stack([x, y])).query(
        np.column_stack([x, y]), k=2)
    elem = float(np.median(dists[:, 1]))
    x0, x1 = x.min() - elem / 2, x.max() + elem / 2
    y0, y1 = y.min() - elem / 2, y.max() + elem / 2
    w_in = 2.4
    h_in = w_in * (y1 - y0) / (x1 - x0)
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_axis_off()
    # marker edge in points; 1.35x oversize closes gaps where the mesh mixes
    # element aspect ratios (structured transition bands, rings at the hole)
    s_pts = 1.35 * elem * w_in * 72.0 / (x1 - x0)
    ax.scatter(x, y, c=v, s=s_pts ** 2, marker="s", cmap="viridis",
               vmin=vmin, vmax=vmax, linewidths=0)
    fig.savefig(out_png, transparent=True, dpi=dpi)
    plt.close(fig)
    return x0, x1, y0, y1


def _emit_fields_tex(args, idx, meta, csv_names):
    """Groupplot: columns = frames, rows = (GT, Pred), for one ply.

    Panels are PNG rasters placed with \\addplot graphics inside vector axes
    (default). With --vector, emits pure pgfplots scatter panels instead
    (needs lualatex: pdflatex runs out of memory).
    """
    ply = args.ply
    label = {"fi": r"$\mathrm{FI}$", "damage": r"$D$"}.get(
        args.field, args.field)
    n = len(idx)
    datadir = os.path.join(args.out, "data")
    lines = [
        "% Auto-generated by tools/export_fields_for_tikz.py — do not edit by hand.",
        f"% sim={args.sim} field={args.field} ply={ply} "
        f"(angle {meta['plies'].get(str(ply), '?')} deg)",
        "% Panels are rasterised (PNG) inside vector pgfplots axes; the colorbar",
        "% uses pgfplots' viridis, which matches matplotlib's.",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        f"  group style={{group size={n} by 2, horizontal sep=2mm, "
        "vertical sep=2mm},",
        r"  width=0.21\linewidth, scale only axis, axis equal image,",
        r"  enlargelimits=false, xtick=\empty, ytick=\empty,",
        f"  colormap/viridis, point meta min={meta['vmin']:.4g}, "
        f"point meta max={meta['vmax']:.4g},",
        r"]",
    ]
    for row in ("gt", "pred"):
        for k, fi_ in enumerate(idx):
            t = meta["times"][k]
            opts = []
            if row == "gt":
                opts.append(f"title={{$t={t:.2f}$}}")
            if k == 0:
                side = "Abaqus" if row == "gt" else "GT rollout"
                opts.append(f"ylabel={{{side}}}")
            if row == "pred" and k == n - 1:
                opts.append(
                    f"colorbar, colorbar style={{title={label}, "
                    "width=2.5mm, height=2.6cm, ytick distance=1, "
                    "tick label style={font=\\scriptsize}}")
            lines.append(f"\\nextgroupplot[{', '.join(opts)}]")
            csv_name = csv_names[(fi_, ply)]
            csv_path = os.path.join(datadir, csv_name)
            if args.vector:
                lines.append(
                    f"\\addplot[scatter, only marks, mark=square*, "
                    f"mark size=\\FieldMarkSize, scatter src=explicit] "
                    f"table[col sep=comma, x=x, y=y, meta={row}] "
                    f"{{figures_tikz/data/{csv_name}}};")
            else:
                png_name = csv_name.replace(".csv", f"_{row}.png")
                x0, x1, y0, y1 = _render_panel_png(
                    csv_path, row, meta["vmin"], meta["vmax"],
                    os.path.join(datadir, png_name))
                lines.append(
                    f"\\addplot graphics[xmin={x0:.4g}, xmax={x1:.4g}, "
                    f"ymin={y0:.4g}, ymax={y1:.4g}] "
                    f"{{figures_tikz/data/{png_name}}};")
    lines += [r"\end{groupplot}", r"\end{tikzpicture}"]
    out = os.path.join(args.out,
                       f"fig_fields_{args.field}_{args.sim}_ply{ply}.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"fields: TeX panel -> {out}")


# ──────────────────────────────────────────────────────────────────────────
# fields-mesh: element quads (same mesh as the testing script's GIFs)
# ──────────────────────────────────────────────────────────────────────────

def _render_mesh_panel_png(verts, values, vmin, vmax, extent, out_png,
                           dpi=400, edge_alpha=0.15, edge_lw=0.3,
                           cmap="viridis"):
    """Rasterise one ply panel using the SAME PolyCollection recipe as the
    testing script (src/gtp/visualization/plotting.py:_plot_field_on_hex_quads:
    edgecolors=(0,0,0,edge_alpha), linewidths=edge_lw)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize

    x0, x1, y0, y1 = extent
    w_in = 2.6
    h_in = w_in * (y1 - y0) / (x1 - x0)
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_axis_off()
    pc = PolyCollection(
        verts,
        array=np.asarray(values),
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        edgecolors=(0, 0, 0, edge_alpha),
        linewidths=edge_lw,
        alpha=1.0,
    )
    ax.add_collection(pc)
    fig.savefig(out_png, transparent=True, dpi=dpi)
    plt.close(fig)


def _vtu_time(path):
    """Read just the time_s FieldData value from a frame VTU (cheap)."""
    root = ET.parse(path).getroot()
    compressed = "compressor" in root.attrib
    tda = root.find(".//FieldData/DataArray[@Name='time_s']")
    if tda is None:
        return None
    dtype = _VTK_DTYPES[tda.attrib.get("Type", tda.attrib.get("type"))]
    if tda.attrib.get("format") == "ascii":
        return float(np.fromstring(tda.text, dtype=dtype, sep=" ")[0])
    if not compressed:
        raw = base64.b64decode("".join(tda.text.split()))
        n = np.frombuffer(raw[:4], np.uint32)[0]
        return float(np.frombuffer(raw[4:4 + n], dtype=dtype)[0])
    return float(_decode_inline_binary(tda.text, dtype)[0])


# matplotlib's magma colormap as a pgfplots definition (for the colorbar);
# D = 1 maps to the light-yellow top end, as requested.
_MAGMA_PGF = (
    "colormap={magma}{rgb255=(0,0,4) rgb255=(28,16,68) rgb255=(79,18,123) "
    "rgb255=(129,37,129) rgb255=(181,54,122) rgb255=(229,80,100) "
    "rgb255=(251,135,97) rgb255=(254,194,135) rgb255=(252,253,191)}"
)


def cmd_fields_mesh(args):
    """All-plies element-mesh panels: per ply a (Ground Truth / Predicted)
    row-pair x n_frames pseudo-times, one shared colorbar. Panels drawn on
    the true element footprint quads parsed from the testing script's VTUs.

    Colour normalisation (--norm):
      fixed   one physical scale for everything (FI 0..3, D 0..1)
      column  per pseudo-time scale, shared between Ground Truth and
              Predicted and across plies; the column max is printed in the
              title and the colorbar shows the normalised field (0..1).
              Fixes the 'early frames look empty' problem when FI ~ 0.3
              would be invisible against a 0..3 scale.
    """
    files = _frame_files(args.run, args.sim)
    if args.times:
        wanted = [float(t) for t in args.times.split(",")]
        all_times = [_vtu_time(f) for f in files]
        idx = sorted(set(
            int(np.argmin([abs((t if t is not None else 1e9) - w)
                           for t in all_times])) for w in wanted))
    elif args.frames:
        idx = [int(i) for i in args.frames.split(",")]
    else:
        idx = np.unique(np.linspace(0, len(files) - 1, args.n_frames)
                        .round().astype(int)).tolist()
    os.makedirs(args.out, exist_ok=True)
    datadir = os.path.join(args.out, "data")
    os.makedirs(datadir, exist_ok=True)

    gt_key, pr_key = f"{args.field}_gt", f"{args.field}_pred"
    cmap = "magma" if args.field == "damage" else "viridis"

    # Geometry from the first frame (mesh constant across frames)
    points, conn, starts, centroids, fields0, _ = read_vtu_full(files[0])
    if gt_key not in fields0:
        sys.exit(f"Field {gt_key} not in {files[0]} (have: {sorted(fields0)})")
    verts_all = _hex_footprint_quads(points, conn, starts, centroids)
    ply_id = fields0.get("ply_id", np.zeros(len(centroids))).astype(int)
    ply_ang = fields0.get("ply_angle_deg", np.zeros(len(centroids)))
    plies = sorted(np.unique(ply_id).tolist())
    x0, x1 = points[:, 0].min(), points[:, 0].max()
    y0, y1 = points[:, 1].min(), points[:, 1].max()
    extent = (x0, x1, y0, y1)

    # Physical scales (fixed mode); column mode rescales per pseudo-time
    vmin = 0.0
    vmax_fixed = 3.0 if args.field == "fi" else 1.0

    explicit_scales = ([float(s) for s in args.scales.split(",")]
                       if getattr(args, "scales", None) else None)
    times, png_names, col_max, col_raw = [], {}, [], []
    for k_i, fi_ in enumerate(idx):
        _, _, _, _, fields, t = read_vtu_full(files[fi_])
        times.append(t)
        raw_max = max(float(fields[gt_key].max()),
                      float(fields[pr_key].max()), 1e-6)
        if explicit_scales is not None:
            # user-chosen per-column scale; values above it saturate at the
            # top colour (Normalize clips in rendering)
            cmax = explicit_scales[min(k_i, len(explicit_scales) - 1)]
        elif args.norm == "column":
            cmax = raw_max
        elif args.norm == "snap":
            # snap to the physical thresholds: pre-initiation columns get
            # the initiation scale (FI <= 1), post-initiation the cap
            cmax = 1.0 if raw_max <= 1.0 + 1e-6 else vmax_fixed
        else:
            cmax = vmax_fixed
        col_max.append(cmax)
        col_raw.append(raw_max)
        for p in plies:
            m = ply_id == p
            for kind, key in (("gt", gt_key), ("pred", pr_key)):
                png = f"{args.sim}_{args.field}mesh_f{fi_:04d}_ply{p}_{kind}.png"
                _render_mesh_panel_png(
                    verts_all[m], fields[key][m], vmin, cmax, extent,
                    os.path.join(datadir, png), dpi=args.dpi,
                    edge_alpha=args.edge_alpha, edge_lw=args.edge_lw,
                    cmap=cmap)
                png_names[(fi_, p, kind)] = png

    # ── TeX: 2*n_plies rows x n_frames cols + shared colorbar ──
    label = {"fi": r"$\mathrm{FI}$", "damage": r"$D$"}.get(args.field, args.field)
    n = len(idx)
    npl = len(plies)
    if args.norm in ("column", "snap") or explicit_scales is not None:
        cb_min, cb_max = 0.0, 1.0
        cb_label = (r"$\mathrm{FI}/\mathrm{FI}^{\max}_{t}$"
                    if args.field == "fi" else r"$D/D^{\max}_{t}$")
    else:
        cb_min, cb_max = vmin, vmax_fixed
        cb_label = label
    cmap_tex = (f"colormap/{cmap}" if cmap == "viridis"
                else _MAGMA_PGF)
    lines = [
        "% Auto-generated by tools/export_fields_for_tikz.py fields-mesh — do not edit.",
        f"% sim={args.sim} field={args.field} ALL {npl} plies; true element-footprint",
        "% quads parsed from the testing script's VTUs, rendered with the same",
        f"% PolyCollection recipe as the test GIFs ({cmap}, edge alpha "
        f"{args.edge_alpha}, lw {args.edge_lw}).",
        f"% Rows: per ply, Ground Truth (top) / Predicted (bottom). norm={args.norm}.",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        f"  group style={{group size={n} by {2 * npl}, horizontal sep=1.0mm, "
        "vertical sep=1.0mm},",
        r"  width=0.175\linewidth, scale only axis, axis equal image,",
        r"  enlargelimits=false, xtick=\empty, ytick=\empty,",
        f"  {cmap_tex}, point meta min={cb_min:.4g}, point meta max={cb_max:.4g},",
        r"  ylabel style={font=\scriptsize}, title style={font=\scriptsize},",
        r"]",
    ]
    for p_i, p in enumerate(plies):
        for row in ("gt", "pred"):
            for k, fi_ in enumerate(idx):
                opts = []
                if p_i == 0 and row == "gt":
                    # short title only; the per-column max goes to the
                    # bottom xlabel so that titles cannot collide
                    opts.append(f"title={{$t={times[k]:.2f}$}}")
                if ((args.norm in ("column", "snap")
                        or explicit_scales is not None) and row == "pred"
                        and p_i == npl - 1):
                    # label: the actual column max; for snap mode the
                    # (possibly larger) colour scale is stated separately.
                    # Compact two-line form + tiny font so neighbouring
                    # columns can never collide.
                    if ((args.norm == "snap" or explicit_scales is not None)
                            and abs(col_max[k] - col_raw[k]) > 5e-3):
                        xlab = (f"$\\max {col_raw[k]:.2f}$\\\\"
                                f"(sc.\\,{col_max[k]:.0f})")
                    else:
                        xlab = f"$\\max {col_raw[k]:.2f}$"
                    opts.append(
                        f"xlabel={{{xlab}}}, "
                        "xlabel style={font=\\tiny, align=center}")
                if k == 0:
                    side = "Ground truth" if row == "gt" else "Predicted"
                    opts.append(f"ylabel={{{side}}}")
                # Single shared colorbar attached to the last-column gt panel
                # of the middle ply. It must (a) span the FULL 2*npl-row grid
                # and (b) be centred on the grid. row pitch ~= panel height +
                # vsep ~= 1.5 cm here; the colorbar's own panel centre sits
                # (npl%2 - 0.5) rows off the grid centre, so yshift corrects it.
                if (row == "gt" and p_i == npl // 2 and k == n - 1):
                    pass  # colorbar handled by the standalone axis below
                lines.append(f"\\nextgroupplot[{', '.join(opts)}]")
                png = png_names[(fi_, p, row)]
                lines.append(
                    f"\\addplot graphics[xmin={x0:.4g}, xmax={x1:.4g}, "
                    f"ymin={y0:.4g}, ymax={y1:.4g}] "
                    f"{{figures_tikz/data/{png}}};")
    lines.append(r"\end{groupplot}")

    # ── Shared colorbar: a standalone hidden axis to the right of the grid,
    # spanning the FULL grid height and vertically centred. Its height is read
    # from the actual cell node coordinates (reliable after \end{groupplot}),
    # so it is correct for any ply count without cm guesses.
    last_col = n
    bottom_row = 2 * npl
    lines += [
        f"\\path let \\p1=(group c{last_col}r1.north east),"
        f" \\p2=(group c{last_col}r{bottom_row}.south east) in"
        r" \pgfextra{\xdef\GridH{\the\dimexpr\y1-\y2\relax}}"
        r" coordinate (cbmid) at ($(\x1,\y1)!0.5!(\x1,\y2)$);",
        r"\begin{axis}[",
        r"  hide axis, scale only axis, width=0pt, height=\GridH,",
        f"  {cmap_tex}, point meta min={cb_min:.4g}, point meta max={cb_max:.4g},",
        r"  at={(cbmid)}, anchor=west, xshift=4mm,",
        r"  colorbar right, colorbar style={",
        f"    ylabel={{{cb_label}}}, ylabel style={{font=\\scriptsize}},",
        r"    width=3mm, tick label style={font=\scriptsize}},",
        r"]",
        r"\addplot[draw=none] coordinates {(0,0) (1,1)};",
        r"\end{axis}",
    ]

    # Brace + ply label left of each Ground-truth/Predicted row pair.
    # Must follow \end{groupplot}: the named nodes (group c<i>r<j>) only
    # exist once the group is finalised. Requires
    # \usetikzlibrary{decorations.pathreplacing} in the preamble.
    for p_i, p in enumerate(plies):
        ang = float(ply_ang[ply_id == p][0]) if (ply_id == p).any() else 0.0
        r1, r2 = 2 * p_i + 1, 2 * p_i + 2
        lines.append(
            "\\draw[decorate, decoration={brace, amplitude=4pt}] "
            f"($(group c1r{r2}.south west)+(-8.5mm,0)$) -- "
            f"($(group c1r{r1}.north west)+(-8.5mm,0)$) "
            "node[midway, rotate=90, anchor=south, outer sep=3pt, "
            "font=\\scriptsize\\bfseries] "
            f"{{Ply {p_i + 1} ($ {ang:.0f}^\\circ$)}};")
    lines.append(r"\end{tikzpicture}")
    out = os.path.join(args.out,
                       f"fig_fields_{args.field}_{args.sim}_allplies.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"fields-mesh: {n} frames x {npl} plies x 2 -> {datadir}")
    print(f"fields-mesh: TeX panel -> {out}")


# ──────────────────────────────────────────────────────────────────────────
# loss
# ──────────────────────────────────────────────────────────────────────────

_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+):\s+Train=([\d.eE+-]+)\s+Val=([\d.eE+-]+)")


def cmd_loss(args):
    logs = []
    for run in args.run.split(","):  # fresh + resume dirs, chronological order
        logs += sorted(glob.glob(os.path.join(run,
                                              "execution_log_train_*.txt")))
    if not logs:
        sys.exit(f"No execution_log_train_*.txt under {args.run}")
    hist = {}
    for log in logs:  # chronological: later logs overwrite duplicated epochs
        with open(log, errors="ignore") as f:
            for line in f:
                m = _EPOCH_RE.search(line)
                if m:
                    hist[int(m.group(1))] = (float(m.group(2)),
                                             float(m.group(3)))
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    out = os.path.join(args.out, "data", f"loss_{args.name}.csv")
    with open(out, "w") as f:
        f.write("epoch,train,val\n")
        for e in sorted(hist):
            f.write(f"{e},{hist[e][0]:.6g},{hist[e][1]:.6g}\n")
    print(f"loss: {len(hist)} epochs ({min(hist)}..{max(hist)}) -> {out}")


# ──────────────────────────────────────────────────────────────────────────
# regression
# ──────────────────────────────────────────────────────────────────────────

def cmd_regression(args):
    vtk_root = os.path.join(args.run, "test_results", "vtk")
    sims = (sorted(os.listdir(vtk_root)) if args.sim == "all"
            else [args.sim])
    gt_key, pr_key = f"{args.field}_gt", f"{args.field}_pred"
    gts, prs = [], []
    for sim in sims:
        for fpath in _frame_files(args.run, sim)[::args.frame_stride]:
            _, fields, _ = read_vtu_celldata(fpath)
            gts.append(fields[gt_key])
            prs.append(fields[pr_key])
    gt = np.concatenate(gts)
    pr = np.concatenate(prs)
    n = gt.size
    ss_res = float(((pr - gt) ** 2).sum())
    ss_tot = float(((gt - gt.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(ss_res / n))

    lo = min(gt.min(), pr.min())
    hi = max(gt.max(), pr.max())
    H, xe, ye = np.histogram2d(gt, pr, bins=args.bins,
                               range=[[lo, hi], [lo, hi]])
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    ii, jj = np.nonzero(H)

    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    out = os.path.join(args.out, "data",
                       f"regression_{args.field}_{args.name}.csv")
    with open(out, "w") as f:
        f.write("gt,pred,count\n")
        for i, j in zip(ii, jj):
            f.write(f"{xc[i]:.6g},{yc[j]:.6g},{int(H[i, j])}\n")
    # ── true-hexagon raster (matplotlib hexbin), previous-paper look ──
    # The PNG carries only the hex bins; axes, diagonal, annotation box and
    # colorbar stay vector (pgfplots). Bins with < mincnt samples are not
    # drawn (kills the pale <10-count speckle), and the linear colour scale
    # is capped at the vmax-pct percentile of the REMAINING bin counts so a
    # uniform near-cap halo cannot form.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.colors import LogNorm

    counts_all = H[ii, jj]
    keep = counts_all[counts_all >= args.mincnt]
    hex_vmax = float(np.percentile(keep, args.vmax_pct)) if keep.size else 1.0
    hex_max = float(keep.max()) if keep.size else 1.0

    # Optionally exclude pairs whose GROUND TRUTH sits at the FI cap: the
    # vertical band at gt=cap is an artifact of clamping in the subroutine,
    # not a model error. R2/RMSE above stay computed on ALL points.
    if getattr(args, "drop_cap", False):
        keep_m = gt < (hi - 1e-6)
        gt_plot, pr_plot = gt[keep_m], pr[keep_m]
        print(f"regression: hexbin excludes {int((~keep_m).sum()):,} "
              f"GT-at-cap points (gt == {hi:g})")
    else:
        gt_plot, pr_plot = gt, pr

    def _hex_png(norm, suffix):
        fig = plt.figure(figsize=(3.2, 3.2), dpi=args.png_dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_axis_off()
        ax.hexbin(gt_plot, pr_plot, gridsize=args.bins, cmap="Reds", norm=norm,
                  extent=[0, hi, 0, hi], mincnt=args.mincnt, linewidths=0.05)
        p = os.path.join(args.out, "data",
                         f"regression_{args.field}_{args.name}_hex{suffix}.png")
        fig.savefig(p, transparent=True, dpi=args.png_dpi)
        plt.close(fig)
        return p

    # Log colour scale (LogNorm(vmin=1)): the counts span ~5 decades, which a
    # linear map cannot show
    hex_png = _hex_png(LogNorm(vmin=max(1, args.mincnt), vmax=hex_max), "")
    hex_png_lin = _hex_png(plt.Normalize(vmin=0, vmax=hex_vmax), "_linear")

    meta = {"r2": r2, "rmse": rmse, "n": int(n), "lo": float(lo),
            "hi": float(hi), "bin_width": float(xe[1] - xe[0]),
            "sims": len(sims), "mincnt": args.mincnt,
            "hex_vmax": hex_vmax, "hex_max": hex_max,
            "vmax_pct": args.vmax_pct}
    with open(out.replace(".csv", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"regression: N={n:,} R2={r2:.4f} RMSE={rmse:.4g} "
          f"({len(ii)} non-empty bins) -> {out}")
    print(f"regression: hexbin rasters (mincnt={args.mincnt}, "
          f"log vmax={hex_max:.0f}; linear vmax=P{args.vmax_pct}="
          f"{hex_vmax:.0f}) -> {hex_png} / {hex_png_lin}")


# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fields")
    p.add_argument("--run", required=True)
    p.add_argument("--sim", required=True)
    p.add_argument("--field", default="fi", choices=["fi", "damage",
                                                     "fi_rate"])
    p.add_argument("--frames", default=None,
                   help="comma-separated frame indices (default: evenly spaced)")
    p.add_argument("--n-frames", type=int, default=5)
    p.add_argument("--ply", type=int, default=0,
                   help="ply index for the emitted TeX panel (CSVs cover all plies)")
    p.add_argument("--vector", action="store_true",
                   help="pure pgfplots scatter panels (needs lualatex) instead "
                        "of PNG rasters inside vector axes")
    p.add_argument("--out", default="results/figures_tikz")
    p.set_defaults(func=cmd_fields)

    p = sub.add_parser("fields-mesh",
                       help="all-plies panels on TRUE element footprint quads "
                            "(same mesh as the testing GIFs)")
    p.add_argument("--run", required=True)
    p.add_argument("--sim", required=True)
    p.add_argument("--field", default="fi", choices=["fi", "damage"])
    p.add_argument("--frames", default=None,
                   help="comma-separated frame indices (default: evenly spaced)")
    p.add_argument("--times", default=None,
                   help="comma-separated pseudo-times (e.g. 0,0.15,0.3,0.5,1); "
                        "mapped to the nearest stored frames, wins over --frames")
    p.add_argument("--n-frames", type=int, default=5)
    p.add_argument("--norm", default="fixed",
                   choices=["fixed", "column", "snap"],
                   help="colour scale: fixed physical (FI 0-3, D 0-1); "
                        "column = per-pseudo-time max shared by GT/Pred; "
                        "snap = per-column but snapped to the physical "
                        "thresholds (1 pre-initiation, cap after)")
    p.add_argument("--scales", default=None,
                   help="explicit comma-separated per-column colour scales "
                        "(e.g. 1,1,3,3,3); overrides --norm; values above "
                        "a column's scale saturate at the top colour")
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--edge-alpha", type=float, default=0.15,
                   help="element edge alpha (testing-script default 0.15)")
    p.add_argument("--edge-lw", type=float, default=0.3,
                   help="element edge linewidth (testing-script default 0.3)")
    p.add_argument("--out", default="results/figures_tikz")
    p.set_defaults(func=cmd_fields_mesh)

    p = sub.add_parser("loss")
    p.add_argument("--run", required=True)
    p.add_argument("--name", required=True,
                   help="output tag, e.g. attention / standard")
    p.add_argument("--out", default="results/figures_tikz")
    p.set_defaults(func=cmd_loss)

    p = sub.add_parser("regression")
    p.add_argument("--run", required=True)
    p.add_argument("--sim", default="all")
    p.add_argument("--field", default="fi", choices=["fi", "damage",
                                                     "fi_rate"])
    p.add_argument("--name", default="model3")
    p.add_argument("--bins", type=int, default=70)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--mincnt", type=int, default=10,
                   help="hide hex bins with fewer samples (default 10)")
    p.add_argument("--drop-cap", action="store_true",
                   help="exclude GT-at-cap points (e.g. FI==3) from the "
                        "hexbin raster (a clamping artefact, not a model error)")
    p.add_argument("--vmax-pct", type=float, default=99.0,
                   help="linear colorbar cap percentile of kept bins")
    p.add_argument("--png-dpi", type=int, default=350)
    p.add_argument("--out", default="results/figures_tikz")
    p.set_defaults(func=cmd_regression)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
