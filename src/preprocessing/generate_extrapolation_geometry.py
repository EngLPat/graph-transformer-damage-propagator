"""Extrapolation-geometry generator: no-hole, two-hole and stitched plates.

Inference-only test cases for the trained laminate model (8-ply symmetric
[45/-45/0/90]_s laminate, 0.375 mm plies, z-symmetry and direct displacement
boundary conditions, 0.9 mm uniform mesh) on geometries outside the training
distribution. The datasets built from these cases are evaluated with
`python -m src.gtp --mode test` and the normalisation statistics of the
trained model.

Plate size, layup, mesh and boundary conditions match the training corpus
(the laminate dataset):

  nohole    34 x 34 mm plate, no hole, u1 = 0.5 mm: does the model predict
            (near-)zero damage growth without a stress concentrator?

  twohole   34 x 34 mm plate, two r = 3.0 mm holes at (5,-5) and (5,5),
            u1 = 0.5 mm: two stress concentrations and the ligament
            between them.

  stitched  68 x 34 mm plate made of two training-like plates in series
            (bc | plate | plate), one r = 3.0 mm hole per half at (-17,0)
            and (+17,0), u1 = 1.0 mm, so each half sees the nominal strain
            of the training plates.

CASES also holds a diagonal two-hole plate and the single-hole constituents
of both two-hole plates.

The 2D mesh comes from Abaqus CAE (tools/abaqus_mesh_multi.py): plate
minus N holes, full-span partition lines through every hole centre so
each hole-adjacent face has the single-quarter-arc topology that the
STRUCTURED quad mesher accepts (same as the "+" partition of the
single-hole generator).  Extrusion to C3D8R plies + COH3D8 interfaces,
boundary sets and the .inp writer are reused unchanged from
generate_cohesive_laminate_uniform.py.

Metadata written to data/processed/<study>/plate_geometry_data.pt is the
standard 6-param tensor (first hole or zeros) plus a "holes" key, a
per-sample list of (cx, cy, r), which build_cohesive_dataset.py uses
for the is_hole feature when present (no holes -> is_hole = 0 everywhere).

Usage
-----
    # generate all three cases (inp + mesh preview plots)
    python -m src.preprocessing.generate_extrapolation_geometry --case all

    # single case, custom study name / displacement
    python -m src.preprocessing.generate_extrapolation_geometry \\
        --case stitched --displacement 0.7 --study extrap_stitched_u07

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import torch

from .generate_geometry import PROJECT_ROOT, _format_set_list  # noqa: F401
from .generate_cohesive_laminate_uniform import (
    ELEM_SIZE_DEFAULT,
    extrude_to_3d_cohesive,
    identify_boundary_sets_cohesive,
    write_inp_cohesive,
)

_ABAQUS_MESH_SCRIPT = PROJECT_ROOT / "tools" / "abaqus_mesh_multi.py"

# Layup of the laminate training study (0.375 mm plies, 4-ply half of
# the symmetric 8-ply laminate; the z-symmetry BC provides the other half).
LAYUP_T375 = [(45.0, 0.375), (-45.0, 0.375), (0.0, 0.375), (90.0, 0.375)]

# ═══════════════════════════════════════════════════════════════════════════
# CASE PRESETS
# ═══════════════════════════════════════════════════════════════════════════
# Single-plate reference dims (mid-range of the training LHS box:
# half sizes 14-20 mm, hole r 2-5 mm, hole offset +/-5 mm, disp 0.35-0.7 mm).
_HALF = 17.0
_R    = 3.0
_DISP = 0.5

CASES = {
    "nohole": {
        "plate_half_x": _HALF,
        "plate_half_y": _HALF,
        "holes":        [],
        "xcuts":        [],
        "ycuts":        [],
        "displacement": _DISP,
        "note": "34x34 mm plate, no hole",
    },
    "twohole": {
        "plate_half_x": _HALF,
        "plate_half_y": _HALF,
        "holes":        [(5.0, -5.0, _R), (5.0, 5.0, _R)],
        # cuts through both hole centres + midline separating the two arcs
        "xcuts":        [5.0],
        "ycuts":        [-5.0, 0.0, 5.0],
        "displacement": _DISP,
        "note": "34x34 mm plate, two r=3 holes at (5,-5) and (5,5)",
    },
    # ── Isolated constituents of "twohole" ────────────────────────────────
    # Each keeps the SAME partition cuts as twohole, so the meshes are
    # node-identical apart from the missing hole. That makes superposition a
    # node-for-node sum with no co-registration step, and lets the attention
    # fields be compared tile-for-tile against the two-hole case.
    # Purpose: test whether the model's two-hole response is (a) the
    # superposition of these two single-hole responses, or (b) something that
    # accounts for the ligament interaction between them.
    "onehole_top": {
        "plate_half_x": _HALF,
        "plate_half_y": _HALF,
        "holes":        [(5.0, 5.0, _R)],
        "xcuts":        [5.0],
        "ycuts":        [-5.0, 0.0, 5.0],
        "displacement": _DISP,
        "note": "34x34 mm plate, ONE r=3 hole at (5,5) — upper hole of twohole",
    },
    "onehole_bot": {
        "plate_half_x": _HALF,
        "plate_half_y": _HALF,
        "holes":        [(5.0, -5.0, _R)],
        "xcuts":        [5.0],
        "ycuts":        [-5.0, 0.0, 5.0],
        "displacement": _DISP,
        "note": "34x34 mm plate, ONE r=3 hole at (5,-5) — lower hole of twohole",
    },
    # ── Isolated constituents of "twohole_diag" ───────────────────────────
    # The DIAGONAL pair is the cleaner superposition test. In "twohole" both
    # holes sit at x = +5, so each hole's own vertical damage band lies on top
    # of the other's and superposition is nearly indistinguishable from
    # interaction. Here the holes are at different x, so their bands should not
    # overlap: under superposition the two-hole field is two separate bands,
    # and anything connecting them is genuine interaction.
    # Same partition cuts as twohole_diag so the three meshes are as close as
    # the mesher allows.
    "onehole_diag_a": {
        "plate_half_x": _HALF,
        "plate_half_y": _HALF,
        "holes":        [(5.0, 5.0, _R)],
        "xcuts":        [-5.0, 0.0, 5.0],
        "ycuts":        [-5.0, 0.0, 5.0],
        "displacement": _DISP,
        "note": "34x34 mm plate, ONE r=3 hole at (5,5) — upper hole of twohole_diag",
    },
    "onehole_diag_b": {
        "plate_half_x": _HALF,
        "plate_half_y": _HALF,
        "holes":        [(-5.0, -5.0, _R)],
        "xcuts":        [-5.0, 0.0, 5.0],
        "ycuts":        [-5.0, 0.0, 5.0],
        "displacement": _DISP,
        "note": "34x34 mm plate, ONE r=3 hole at (-5,-5) — lower hole of twohole_diag",
    },
    "twohole_diag": {
        "plate_half_x": _HALF,
        "plate_half_y": _HALF,
        "holes":        [(-5.0, -5.0, _R), (5.0, 5.0, _R)],
        # cuts through both hole centres + midlines separating the arcs
        "xcuts":        [-5.0, 0.0, 5.0],
        "ycuts":        [-5.0, 0.0, 5.0],
        "displacement": _DISP,
        "note": "34x34 mm plate, two r=3 holes at (-5,-5) and (5,5) (diagonal)",
    },
    "stitched": {
        "plate_half_x": 2.0 * _HALF,
        "plate_half_y": _HALF,
        "holes":        [(-_HALF, 0.0, _R), (_HALF, 0.0, _R)],
        "xcuts":        [-_HALF, 0.0, _HALF],
        "ycuts":        [0.0],
        # series assembly: 2x length -> 2x displacement = same nominal strain
        "displacement": 2.0 * _DISP,
        "note": "68x34 mm double plate (bc|plate|plate), one r=3 hole per half",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 2D MESH VIA ABAQUS CAE
# ═══════════════════════════════════════════════════════════════════════════

def generate_multi_hole_mesh_2d(plate_half_x, plate_half_y, holes,
                                xcuts, ycuts, elem_size):
    """Boundary-fitted 2D quad mesh of a plate with 0..N holes.

    Dispatches to tools/abaqus_mesh_multi.py (Abaqus CAE structured quad
    mesher).  Returns (nodes_2d, elements_2d) in the same format as
    generate_plus_partition_mesh_2d: 1-indexed contiguous IDs, CCW quads.
    """
    if not _ABAQUS_MESH_SCRIPT.exists():
        raise RuntimeError(
            "Cannot find Abaqus mesh script at %s" % _ABAQUS_MESH_SCRIPT)

    with tempfile.TemporaryDirectory(prefix="abq_meshx_") as tmp:
        params_path = Path(tmp) / "mesh_params.json"
        params_path.write_text(json.dumps({
            "plate_half_x": plate_half_x,
            "plate_half_y": plate_half_y,
            "elem_size":    elem_size,
            "holes":        [list(h) for h in holes],
            "xcuts":        list(xcuts),
            "ycuts":        list(ycuts),
        }))
        cmd = [
            "abaqus", "cae", "noGUI=" + str(_ABAQUS_MESH_SCRIPT), "--",
            str(params_path), tmp,
        ]
        result = subprocess.run(cmd, cwd=tmp,
                                capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                "Abaqus mesh script failed (rc=%d):\nstdout:\n%s\nstderr:\n%s"
                % (result.returncode, result.stdout, result.stderr))

        nodes_path = Path(tmp) / "multi_nodes.txt"
        elems_path = Path(tmp) / "multi_elements.txt"
        if not (nodes_path.exists() and elems_path.exists()):
            raise RuntimeError(
                "Abaqus did not produce mesh files.\nstdout:\n" + result.stdout)
        if "fallback=True" in result.stdout:
            print("  WARNING: structured meshing fell back to FREE quad; "
                  "inspect the mesh preview before running the simulation.")

        nodes_2d = []
        with open(nodes_path) as fp:
            for ln in fp:
                p = [s.strip() for s in ln.split(",")]
                if len(p) >= 3:
                    nodes_2d.append((int(p[0]), float(p[1]), float(p[2])))

        elements_2d = []
        with open(elems_path) as fp:
            for ln in fp:
                p = [s.strip() for s in ln.split(",")]
                if len(p) >= 5:
                    elements_2d.append((int(p[0]),
                                        int(p[1]), int(p[2]),
                                        int(p[3]), int(p[4])))

    return nodes_2d, elements_2d


# ═══════════════════════════════════════════════════════════════════════════
# MESH PREVIEW PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_mesh_preview(nodes_2d, elements_2d, case_cfg, out_path, title):
    """Save a 2D mesh preview: element edges, holes, BC annotations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    coords = {nid: (x, y) for nid, x, y in nodes_2d}
    segs = []
    for _, n1, n2, n3, n4 in elements_2d:
        quad = [coords[n1], coords[n2], coords[n3], coords[n4], coords[n1]]
        for a, b in zip(quad[:-1], quad[1:]):
            segs.append([a, b])

    hx = case_cfg["plate_half_x"]
    hy = case_cfg["plate_half_y"]

    fig, ax = plt.subplots(figsize=(max(7.0, 7.0 * hx / hy), 7.0))
    ax.add_collection(LineCollection(segs, colors="0.55", linewidths=0.3))

    for (cx, cy, r) in case_cfg["holes"]:
        circ = plt.Circle((cx, cy), r, fill=False, color="crimson",
                          linestyle="--", linewidth=1.2)
        ax.add_patch(circ)
        ax.annotate(f"r={r:g}", (cx, cy), ha="center", va="center",
                    fontsize=8, color="crimson")

    # BC annotations: left edge fixed (u1=0), right edge displaced (u1=disp)
    ax.plot([-hx, -hx], [-hy, hy], color="royalblue", linewidth=3,
            label="fixed edge (u1=0, BL corner pinned)")
    ax.plot([hx, hx], [-hy, hy], color="darkorange", linewidth=3,
            label=f"displaced edge (u1={case_cfg['displacement']:g} mm, direct BC)")
    n_arrows = 7
    for k in range(n_arrows):
        ya = -hy + (2 * hy) * (k + 0.5) / n_arrows
        ax.annotate("", xy=(hx + 0.12 * hy, ya), xytext=(hx, ya),
                    arrowprops=dict(arrowstyle="->", color="darkorange"))

    ax.set_xlim(-hx - 0.18 * hy, hx + 0.18 * hy)
    ax.set_ylim(-hy * 1.12, hy * 1.12)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8,
              ncol=2, frameon=False)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"  Mesh preview: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL API
# ═══════════════════════════════════════════════════════════════════════════

def generate_extrapolation_case(case_name, case_cfg, study,
                                layup=None,
                                elem_size=ELEM_SIZE_DEFAULT,
                                material_model="usdfld",
                                direct_bc=True,
                                sym_bc=True,
                                preview=True):
    """Generate one extrapolation .inp + metadata + mesh preview."""
    if layup is None:
        layup = LAYUP_T375

    hx = case_cfg["plate_half_x"]
    hy = case_cfg["plate_half_y"]
    x_min, x_max = -hx, hx
    y_min, y_max = -hy, hy
    disp = case_cfg["displacement"]
    holes = case_cfg["holes"]

    out_inp_dir  = PROJECT_ROOT / "data" / "raw" / study / "geometry"
    out_meta_dir = PROJECT_ROOT / "data" / "processed" / study
    out_inp_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{case_name}] meshing via Abaqus CAE "
          f"({2*hx:g} x {2*hy:g} mm, {len(holes)} hole(s), "
          f"elem {elem_size:g} mm) ...")
    nodes_2d, elements_2d = generate_multi_hole_mesh_2d(
        hx, hy, holes, case_cfg["xcuts"], case_cfg["ycuts"], elem_size,
    )
    print(f"  2D mesh: {len(nodes_2d)} nodes, {len(elements_2d)} quads")

    if preview:
        plot_mesh_preview(
            nodes_2d, elements_2d, case_cfg,
            out_meta_dir / "plots" / f"mesh_preview_{case_name}.png",
            title=(f"{case_name}: {case_cfg['note']}  |  "
                   f"layup {'/'.join(f'{a:g}' for a, _ in layup)} "
                   f"@ {layup[0][1]:g} mm, elem {elem_size:g} mm, "
                   f"symBC+directBC"),
        )

    nodes_3d, elements_3d, coh_elements, ply_elem_ranges, coh_elem_ranges = \
        extrude_to_3d_cohesive(nodes_2d, elements_2d, layup)

    left_nodes, bl_nodes, bl_elem, right_elems = \
        identify_boundary_sets_cohesive(nodes_3d, elements_3d,
                                        x_min, x_max, y_min)

    sample_name = "plate_001"
    filepath = out_inp_dir / f"{sample_name}.inp"
    write_inp_cohesive(
        filepath=filepath,
        nodes_3d=nodes_3d,
        elements_3d=elements_3d,
        coh_elements=coh_elements,
        job_name=sample_name,
        layup=layup,
        left_nodes=left_nodes,
        bl_nodes=bl_nodes,
        bl_elem=bl_elem,
        right_elems=right_elems,
        ply_elem_ranges=ply_elem_ranges,
        coh_elem_ranges=coh_elem_ranges,
        displacement=disp,
        plate_x_max=x_max,
        plate_y_min=y_min,
        plate_y_max=y_max,
        material_model=material_model,
        direct_bc=direct_bc,
        sym_bc=sym_bc,
    )
    print(f"  Written: {filepath}  "
          f"({len(elements_3d)} C3D8R + {len(coh_elements)} COH3D8)")

    # Metadata: standard 6-param tensor (first hole or zeros for no-hole)
    # + "holes" key consumed by build_cohesive_dataset.build_node_features.
    if holes:
        h0 = holes[0]
    else:
        h0 = (0.0, 0.0, 0.0)
    param_tensor = torch.tensor(
        [[h0[2], h0[0], h0[1], disp, hx, hy]], dtype=torch.float32,
    )
    metadata = {
        "params":       param_tensor,
        "param_names":  ["hole_radius", "hole_center_x", "hole_center_y",
                         "displacement", "plate_half_x", "plate_half_y"],
        "sample_names": [sample_name],
        "layup":        layup,
        "num_samples":  1,
        "model_type":   "cohesive_laminate",
        "holes":        [[(float(cx), float(cy), float(r))
                          for (cx, cy, r) in holes]],
        "extrapolation_case": case_name,
        "case_note":    case_cfg["note"],
    }
    meta_path = out_meta_dir / "plate_geometry_data.pt"
    torch.save(metadata, meta_path)
    print(f"  Metadata: {meta_path}")
    return filepath


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate extrapolation test geometries (no-hole / "
                    "two-hole / stitched double plate) matching the laminate "
                    "training recipe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--case", type=str, default="all",
                        choices=list(CASES.keys()) + ["all"],
                        help="Which case to generate (default: all).")
    parser.add_argument("--study", type=str, default=None,
                        help="Study name override (single case only). "
                             "Default: extrap_<case>.")
    parser.add_argument("--layup", type=str, default=None,
                        help="Comma-separated fibre angles "
                             "(default: 45,-45,0,90).")
    parser.add_argument("--ply-thickness", type=float, default=0.375,
                        help="Ply thickness in mm (default: 0.375).")
    parser.add_argument("--elem-size", type=float, default=ELEM_SIZE_DEFAULT,
                        help=f"In-plane element edge length in mm "
                             f"(default: {ELEM_SIZE_DEFAULT}, matches the "
                             f"training mesh).")
    parser.add_argument("--displacement", type=float, default=None,
                        help="Override the case's applied u1 (mm).")
    parser.add_argument("--material-model", type=str, default="usdfld",
                        choices=["usdfld", "elastic"])
    parser.add_argument("--direct-bc", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Direct right-edge displacement BC "
                             "(default on, as in the training corpus).")
    parser.add_argument("--sym-bc", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="z-symmetry BC on the top face "
                             "(default on, as in the training corpus).")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip the mesh preview plot.")
    args = parser.parse_args()

    if args.layup is not None:
        angles = [float(a) for a in args.layup.split(",")]
        layup = [(a, round(args.ply_thickness, 6)) for a in angles]
    else:
        layup = [(a, round(args.ply_thickness, 6)) for a, _ in LAYUP_T375]

    case_names = list(CASES.keys()) if args.case == "all" else [args.case]
    if args.study and len(case_names) > 1:
        raise SystemExit("--study can only be used with a single --case.")

    for case_name in case_names:
        cfg = dict(CASES[case_name])
        if args.displacement is not None:
            cfg["displacement"] = args.displacement
        study = args.study or f"extrap_{case_name}"
        print("=" * 70)
        print(f"CASE {case_name}  ->  study '{study}'")
        print(f"  {cfg['note']}")
        print("=" * 70)
        generate_extrapolation_case(
            case_name, cfg, study,
            layup=layup,
            elem_size=args.elem_size,
            material_model=args.material_model,
            direct_bc=args.direct_bc,
            sym_bc=args.sym_bc,
            preview=not args.no_preview,
        )


if __name__ == "__main__":
    main()
