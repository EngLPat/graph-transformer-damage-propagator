#!/usr/bin/env python
"""Sanity-check single-ply .inp decks (and .sta/.odb results) for a study.

Deck checks (per .inp in data/raw/<study>/geometry/):
  1. element type C3D8R present; NO COH3D8 *Element block (single ply = no interfaces)
  2. through-thickness extent == expected ply thickness (default 0.125 mm)
  3. field output uses "number interval=100" (not "frequency=" on *Output, field)
  4. at least two *Boundary blocks (clamped edge + displaced edge)
  5. prescribed displacement magnitude inside [0.35, 0.7] mm
  6. *Elastic material card and *Depvar (USDFLD state variables) present
  7. plate x/y extents inside [28, 40] mm; in-plane element size ~ target (median edge)

Result checks (per plate dir in data/raw/<study>/simulations/, if present):
  8. .sta contains "THE ANALYSIS HAS COMPLETED SUCCESSFULLY"
  9. .odb exists and is non-trivial (> 1 MB)

Also renders a mesh plot of the first N decks (top-face element footprints) to
data/raw/<study>/deck_check_mesh.png so mesh quality can be eyeballed.

Usage:
  python tools/check_singleply_decks.py --study singleply \
      --thickness 0.125 --elem-size 1.5 --plot 4
"""
import argparse
import glob
import os
import re
import sys

import numpy as np


def parse_inp(path):
    nodes = {}
    elems_c3d8r, elems_coh = [], []
    boundaries = []
    field_output_lines, disp_values = [], []
    has_elastic = has_depvar = False
    section = None
    cur_eltype = None
    with open(path, errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            low = s.lower()
            if s.startswith("**"):
                continue
            if s.startswith("*"):
                if low.startswith("*node"):
                    section = "node"
                elif low.startswith("*element"):
                    section = "element"
                    m = re.search(r"type=(\w+)", s, re.I)
                    cur_eltype = m.group(1).upper() if m else "?"
                elif low.startswith("*boundary"):
                    section = "boundary"
                    boundaries.append([])
                elif low.startswith("*output, field") or low.startswith("*output,field"):
                    section = None
                    field_output_lines.append(s)
                elif low.startswith("*elastic"):
                    has_elastic = True
                    section = None
                elif low.startswith("*depvar"):
                    has_depvar = True
                    section = None
                else:
                    section = None
                continue
            if section == "node":
                p = s.split(",")
                if len(p) >= 4:
                    try:
                        nodes[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
                    except ValueError:
                        pass
            elif section == "element":
                p = s.split(",")
                try:
                    ids = [int(x) for x in p[1:] if x.strip()]
                except ValueError:
                    continue
                if cur_eltype == "C3D8R":
                    elems_c3d8r.append(ids)
                elif cur_eltype.startswith("COH"):
                    elems_coh.append(ids)
            elif section == "boundary":
                boundaries[-1].append(s)
                p = [x.strip() for x in s.split(",")]
                # "<nset>, <dof1>, <dof2>, <value>": displacement BC rows carry a value
                if len(p) >= 4:
                    try:
                        v = float(p[3])
                        if abs(v) > 1e-9:
                            disp_values.append(abs(v))
                    except ValueError:
                        pass
    return dict(nodes=nodes, c3d8r=elems_c3d8r, coh=elems_coh,
                boundaries=boundaries, field=field_output_lines,
                disp=disp_values, elastic=has_elastic, depvar=has_depvar)


def check_deck(path, thickness, elem_size, tol=0.15):
    d = parse_inp(path)
    errs, warns = [], []
    if not d["c3d8r"]:
        errs.append("no C3D8R elements")
    if d["coh"]:
        errs.append(f"{len(d['coh'])} COH3D8 elements present (single ply must have none)")
    xyz = np.array(list(d["nodes"].values()))
    if len(xyz):
        tz = xyz[:, 2].max() - xyz[:, 2].min()
        if abs(tz - thickness) > 1e-3:
            errs.append(f"thickness {tz:.4f} != expected {thickness}")
        lx, ly = np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1])
        if not (28 - 1e-3 <= lx <= 40 + 1e-3) or not (28 - 1e-3 <= ly <= 40 + 1e-3):
            errs.append(f"plate extents {lx:.1f}x{ly:.1f} outside [28,40] mm")
        # median in-plane edge length of the first 200 elements
        idset = d["nodes"]
        edges = []
        for el in d["c3d8r"][:200]:
            for a, b in zip(el[:4], el[1:4] + el[:1]):
                if a in idset and b in idset:
                    pa, pb = idset[a], idset[b]
                    edges.append(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
        if edges:
            med = float(np.median(edges))
            if abs(med - elem_size) / elem_size > tol:
                warns.append(f"median in-plane edge {med:.3f} vs target {elem_size}")
    if len(d["boundaries"]) < 2:
        errs.append(f"only {len(d['boundaries'])} *Boundary blocks")
    if d["disp"]:
        u = max(d["disp"])
        if not (0.35 - 1e-6 <= u <= 0.7 + 1e-6):
            errs.append(f"displacement {u} outside [0.35, 0.7] mm")
    else:
        errs.append("no displacement value found in *Boundary")
    ok_field = any("number interval=100" in ln for ln in d["field"])
    bad_field = any("frequency" in ln.lower() for ln in d["field"])
    if not ok_field:
        errs.append("field output missing 'number interval=100'")
    if bad_field:
        errs.append("field output uses 'frequency=' (use number interval instead)")
    if not d["elastic"]:
        errs.append("no *Elastic card")
    if not d["depvar"]:
        errs.append("no *Depvar (USDFLD state variables)")
    return errs, warns, d


def plot_meshes(paths, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, paths):
        d = parse_inp(p)
        xyz = d["nodes"]
        zmax = max(v[2] for v in xyz.values())
        quads = []
        for el in d["c3d8r"]:
            pts = [xyz[i] for i in el if i in xyz]
            top = [pt[:2] for pt in pts if abs(pt[2] - zmax) < 1e-6]
            if len(top) == 4:
                quads.append(top)
        ax.add_collection(PolyCollection(quads, facecolor="0.9", edgecolor="k", lw=0.3))
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(os.path.basename(p), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"  mesh plot -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--thickness", type=float, default=0.125)
    ap.add_argument("--elem-size", type=float, default=0.9)
    ap.add_argument("--plot", type=int, default=4,
                    help="render the mesh of the first N decks")
    args = ap.parse_args()

    geo = os.path.join("data", "raw", args.study, "geometry")
    inps = sorted(glob.glob(os.path.join(geo, "plate_*.inp")))
    if not inps:
        sys.exit(f"no decks found under {geo}")
    print(f"Checking {len(inps)} decks in {geo} ...")
    n_bad = 0
    for p in inps:
        errs, warns, _ = check_deck(p, args.thickness, args.elem_size)
        if errs:
            n_bad += 1
            print(f"  FAIL {os.path.basename(p)}: " + "; ".join(errs))
        for w in warns:
            print(f"  warn {os.path.basename(p)}: {w}")
    print(f"Decks: {len(inps) - n_bad}/{len(inps)} OK")

    if args.plot:
        plot_meshes(inps[: args.plot],
                    os.path.join("data", "raw", args.study, "deck_check_mesh.png"))

    sim = os.path.join("data", "raw", args.study, "simulations")
    stas = sorted(glob.glob(os.path.join(sim, "*", "*.sta")))
    if stas:
        good = 0
        for s in stas:
            ok = "SUCCESSFULLY" in open(s, errors="ignore").read()
            odb = s.replace(".sta", ".odb")
            big = os.path.exists(odb) and os.path.getsize(odb) > 1e6
            good += ok and big
            if not (ok and big):
                print(f"  FAIL result {os.path.dirname(s)} (sta ok={ok}, odb ok={big})")
        print(f"Results: {good}/{len(stas)} simulations completed successfully")

    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
