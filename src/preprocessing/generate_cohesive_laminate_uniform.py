"""Parametric deck generator for cohesive laminates with a uniform mesh.

Each sample is a rectangular plate with an off-centre circular hole, loaded by
a displacement on its right edge. Abaqus CAE (tools/abaqus_mesh_plus.py)
meshes the 2D plate with structured quads, using a \"+\" partition through the
hole centre and one global seed. The 2D mesh is extruded into C3D8R plies with
COH3D8 cohesive interfaces between them, and one Abaqus .inp deck is written
per sample, together with the metadata of the parametric study.

Usage
-----
    # One plate with explicit geometry, 0.9 mm global seed
    python -m src.preprocessing.generate_cohesive_laminate_uniform \\
        --layup "45,-45,0,90" --ply-thickness 0.125 \\
        --elem-size 0.9 \\
        --plate-half-x 16.8241 --plate-half-y 19.4678 \\
        --hole-radius 4.5667 --hole-cx 1.7493 --hole-cy 0.4163 \\
        --displacement 0.4936 \\
        --study cp_plyres_uniform_p1

Output
------
    data/raw/<study>/geometry/plate_001.inp …
    data/processed/<study>/plate_geometry_data.pt   (metadata tensor)

Material (IM7/8552)
-------------------
    Ply (C3D8R) : elastic engineering constants (from generate_geometry.py)
    Interface   : Camanho et al. (2003) traction-separation law
                  K = 1e6 N/mm³,  τ_n=80 MPa, τ_s=τ_t=100 MPa
                  GIc=0.277 N/mm, GIIc=GIIIc=0.788 N/mm, η=1.45

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# ── Shared constants and the parameter sampler ──────────────────────────────
from .generate_geometry import (
    PROJECT_ROOT,
    PARAM_RANGES,
    PLATE_HALF_SIZE,
    INNER_SQUARE_FACTOR,
    GEOMETRY_MARGIN,
    N_SAMPLES,
    RANDOM_SEED,
    # Material constants
    MAT_E1, MAT_E2, MAT_E3,
    MAT_NU12, MAT_NU13, MAT_NU23,
    MAT_G12, MAT_G13, MAT_G23,
    # Step parameters
    STEP_INITIAL_INC, STEP_TOTAL_TIME, STEP_MIN_INC,
    STEP_MAX_INC, STEP_MAX_NUM_INC,
    # Functions
    sample_parameters,
    _format_set_list,
)

# ═══════════════════════════════════════════════════════════════════════════
# COHESIVE INTERFACE PROPERTIES  (IM7/8552, Camanho et al. 2003)
# Ref: Camanho, P.P., Dávila, C.G., de Moura, M.F. (2003). Numerical
#      simulation of mixed-mode progressive delamination in composite
#      materials. J. Composite Materials, 37(16), 1415–1438.
# ═══════════════════════════════════════════════════════════════════════════
COH_K       = 1.0e6    # Penalty stiffness (N/mm³ = MPa/mm), all three modes
COH_T_N     = 80.0     # Mode-I interlaminar tensile strength (MPa)
COH_T_S     = 100.0    # Mode-II interlaminar shear strength (MPa)
COH_T_T     = 100.0    # Mode-III interlaminar shear strength (MPa)
COH_GIC     = 0.277    # Mode-I   fracture energy (N/mm = kJ/m²)
COH_GIIC    = 0.788    # Mode-II  fracture energy
COH_GIIIC   = 0.788    # Mode-III fracture energy (assumed = Mode-II)
COH_BK_ETA  = 1.45     # Benzeggagh–Kenane exponent for IM7/8552
COH_VISC    = 1.0e-4   # Viscous regularisation, applied via *Section Controls
                       # (cohesive elements with TRACTION SEPARATION ignore
                       #  *Damage Stabilization, so *Section Controls is used).

# Default layup of the cohesive laminate (override via CLI)
# QI [45/-45/0/90] converges reliably; cross-ply [0/90/0/90] does not because
# both 90° plies fail simultaneously once transverse FI > 1.
LAYUP_DEFAULT = [(45.0, 0.125), (-45.0, 0.125),
                 (0.0, 0.125), (90.0, 0.125)]


# Default target element edge length (mm); the matrix process-zone length of
# the ply material is L_pz ≈ 1.82 mm.
ELEM_SIZE_DEFAULT = 0.9


_ABAQUS_MESH_SCRIPT = (PROJECT_ROOT / "tools" / "abaqus_mesh_plus.py")


def generate_plus_partition_mesh_2d(x_min, x_max, y_min, y_max,
                                    hole_r, hole_cx, hole_cy,
                                    elem_size):
    """Boundary-fitted 2D mesh with a single \"+\" partition through hole.

    Implementation: dispatch to Abaqus CAE's own meshing engine (see
    tools/abaqus_mesh_plus.py).  The Abaqus script creates the plate +
    hole geometry, adds the \"+\" partition, applies structured quad mesh
    controls with one global seed (= ``elem_size``), generates the mesh,
    and writes the resulting 2D nodes / elements to plain text files.

    The result is a Cartesian far field with a smooth bend around the hole.

    Returns
    -------
    nodes_2d    : list of (id, x, y)
    elements_2d : list of (id, n1, n2, n3, n4)
        Node IDs are 1..N contiguous, element IDs are 1..M contiguous, all
        quads in CCW winding (matches what extrude_to_3d_cohesive expects).
    """
    import subprocess
    import tempfile

    if not _ABAQUS_MESH_SCRIPT.exists():
        raise RuntimeError(
            "Cannot find Abaqus mesh script at %s" % _ABAQUS_MESH_SCRIPT)

    plate_half_x = 0.5 * (x_max - x_min)
    plate_half_y = 0.5 * (y_max - y_min)
    # The Abaqus script assumes the plate is centred on the origin in its
    # own coordinate system; our caller may pass off-centre x_min, x_max
    # if it's already shifted, so check.
    centred = (abs(x_max + x_min) < 1e-9) and (abs(y_max + y_min) < 1e-9)
    if not centred:
        raise RuntimeError(
            "Abaqus mesh script expects plate centred on origin; "
            "got x_min=%.6f x_max=%.6f y_min=%.6f y_max=%.6f"
            % (x_min, x_max, y_min, y_max))

    with tempfile.TemporaryDirectory(prefix="abq_mesh_") as tmp:
        cmd = [
            "abaqus", "cae", "noGUI=" + str(_ABAQUS_MESH_SCRIPT), "--",
            ("%.10f" % plate_half_x),
            ("%.10f" % plate_half_y),
            ("%.10f" % hole_r),
            ("%.10f" % hole_cx),
            ("%.10f" % hole_cy),
            ("%.10f" % elem_size),
            tmp,
        ]
        # cwd=tmp keeps Abaqus's .rec/.rpy/.jnl scratch files in the
        # temp dir so they get cleaned up automatically.
        result = subprocess.run(cmd, cwd=tmp,
                                capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                "Abaqus mesh script failed (rc=%d):\nstdout:\n%s\nstderr:\n%s"
                % (result.returncode, result.stdout, result.stderr))

        nodes_path = Path(tmp) / "plus_nodes.txt"
        elems_path = Path(tmp) / "plus_elements.txt"
        if not (nodes_path.exists() and elems_path.exists()):
            raise RuntimeError(
                "Abaqus did not produce mesh files.\nstdout:\n" + result.stdout)

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
# 3-D EXTRUSION WITH DUPLICATED INTERFACE NODES
# ═══════════════════════════════════════════════════════════════════════════

def extrude_to_3d_cohesive(nodes_2d, elements_2d, layup):
    """Extrude the 2-D mesh into C3D8R plies with duplicated interface nodes.

    Every ply-ply interface has two independent node layers at the same
    z-coordinate:
        • layer (2k+1): top face of ply k
        • layer (2k+2): bottom face of ply k+1
    COH3D8 elements bridge these two layers and can open (delaminate).

    Parameters
    ----------
    nodes_2d : list of (id, x, y)
        In-plane mesh nodes, 1-indexed.
    elements_2d : list of (id, n1, n2, n3, n4)
        2-D quad elements, 1-indexed.
    layup : list of (angle_deg, thickness_mm)
        One entry per ply, ordered bottom→top.

    Returns
    -------
    nodes_3d : list of (id, x, y, z)
        2*n_plies node layers, total 2*n_plies*n2d nodes.
    elements_3d : list of (id, n1..n8)
        C3D8R elements only, n_plies*n_elems_2d total.
    coh_elements : list of (id, n1..n8)
        COH3D8 elements at each interface, (n_plies-1)*n_elems_2d total.
    ply_elem_ranges : list of (first_eid, last_eid)
        1-based inclusive C3D8R ID range for each ply.
    coh_elem_ranges : list of (first_eid, last_eid)
        1-based inclusive COH3D8 ID range for each interface.
    """
    n2d       = len(nodes_2d)
    n_elems   = len(elements_2d)
    n_plies   = len(layup)

    # z-coordinates at each ply boundary (n_plies + 1 physical z-levels)
    z_levels = [0.0]
    for _, t in layup:
        z_levels.append(z_levels[-1] + t)

    # ── Node layout ──────────────────────────────────────────────────────
    # Layer 2k   = bottom of ply k  (offset = 2k * n2d)
    # Layer 2k+1 = top    of ply k  (offset = (2k+1) * n2d)
    # Adjacent interface layers 2k+1 and 2k+2 share z but have different IDs.
    nodes_3d = []
    for ply_idx in range(n_plies):
        z_bot = z_levels[ply_idx]
        z_top = z_levels[ply_idx + 1]
        for layer, z in ((2 * ply_idx, z_bot), (2 * ply_idx + 1, z_top)):
            off = layer * n2d
            for nid, x, y in nodes_2d:
                nodes_3d.append((nid + off, x, y, z))

    # ── C3D8R elements ───────────────────────────────────────────────────
    elements_3d   = []
    ply_elem_ranges = []
    for ply_idx in range(n_plies):
        bot_off = (2 * ply_idx) * n2d
        top_off = (2 * ply_idx + 1) * n2d
        e_start = ply_idx * n_elems
        for eid, n1, n2, n3, n4 in elements_2d:
            elements_3d.append((
                e_start + eid,
                n1 + bot_off, n2 + bot_off, n3 + bot_off, n4 + bot_off,
                n1 + top_off, n2 + top_off, n3 + top_off, n4 + top_off,
            ))
        ply_elem_ranges.append((e_start + 1, e_start + n_elems))

    # ── COH3D8 elements at each interface ────────────────────────────────
    # Interface between ply k and ply k+1:
    #   bottom face of COH3D8 = top of ply k   (layer 2k+1)
    #   top    face of COH3D8 = bottom of ply k+1 (layer 2k+2)
    coh_elements  = []
    coh_elem_ranges = []
    coh_id_start  = n_plies * n_elems + 1
    for iface in range(n_plies - 1):
        bot_off = (2 * iface + 1) * n2d   # top of ply iface
        top_off = (2 * iface + 2) * n2d   # bottom of ply iface+1
        e_start = coh_id_start + iface * n_elems
        for eid, n1, n2, n3, n4 in elements_2d:
            coh_elements.append((
                e_start + (eid - 1),
                # bottom face (top surface of lower ply)
                n1 + bot_off, n2 + bot_off, n3 + bot_off, n4 + bot_off,
                # top face (bottom surface of upper ply)
                n1 + top_off, n2 + top_off, n3 + top_off, n4 + top_off,
            ))
        last_coh_id = e_start + n_elems - 1
        coh_elem_ranges.append((e_start, last_coh_id))

    return nodes_3d, elements_3d, coh_elements, ply_elem_ranges, coh_elem_ranges


# ═══════════════════════════════════════════════════════════════════════════
# BOUNDARY SET IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def identify_boundary_sets_cohesive(nodes_3d, elements_3d, x_min, x_max,
                                    y_min, tol=1e-6):
    """Return node/element sets for cohesive-laminate BCs.

    Uses kinematic coupling on the right edge so that all duplicate
    interface nodes are properly constrained.

    Returns
    -------
    left_nodes  : all node IDs where x ≈ x_min  (fix DOF 1)
    bl_nodes    : subset at y ≈ y_min (fix DOFs 1,2,3)
    bl_elem     : one element touching the BL corner
    right_elems_by_face : dict {face_label: [eid, ...]} for elements whose
        named face lies on x ≈ x_max. face_label is one of "S1".."S6".
        Abaqus's structured-quad mesher produces different C3D8R node
        orderings per partition quadrant, so the right-edge face is not
        always S4 and all six faces are checked.
    """
    coords = {nid: (x, y, z) for nid, x, y, z in nodes_3d}

    left_nodes = sorted(
        [nid for nid, (x, _, _) in coords.items() if abs(x - x_min) < tol],
        key=lambda n: (coords[n][2], coords[n][1]),
    )
    bl_nodes = sorted(
        [nid for nid in left_nodes if abs(coords[nid][1] - y_min) < tol],
        key=lambda n: coords[n][2],
    )

    bl_set = set(bl_nodes)
    bl_elem = next(
        eid for eid, n1, n2, n3, n4, n5, n6, n7, n8 in elements_3d
        if bl_set & {n1, n2, n3, n4, n5, n6, n7, n8}
    )

    # Abaqus C3D8R face conventions (0-indexed local nodes):
    #   S1: 0,1,2,3  S2: 4,5,6,7  S3: 0,1,5,4
    #   S4: 1,2,6,5  S5: 2,3,7,6  S6: 3,0,4,7
    FACE_LOCAL = {
        "S1": (0, 1, 2, 3), "S2": (4, 5, 6, 7),
        "S3": (0, 1, 5, 4), "S4": (1, 2, 6, 5),
        "S5": (2, 3, 7, 6), "S6": (3, 0, 4, 7),
    }
    right_elems_by_face: dict[str, list[int]] = {f: [] for f in FACE_LOCAL}
    for eid, n1, n2, n3, n4, n5, n6, n7, n8 in elements_3d:
        local = (n1, n2, n3, n4, n5, n6, n7, n8)
        for face, idxs in FACE_LOCAL.items():
            if all(abs(coords[local[i]][0] - x_max) < tol for i in idxs):
                right_elems_by_face[face].append(eid)
                break  # one face per element
    # drop empty buckets, sort for deterministic output
    right_elems_by_face = {f: sorted(v) for f, v in right_elems_by_face.items() if v}

    return left_nodes, bl_nodes, bl_elem, right_elems_by_face


# ═══════════════════════════════════════════════════════════════════════════
# .INP WRITER
# ═══════════════════════════════════════════════════════════════════════════

def write_inp_cohesive(filepath, nodes_3d, elements_3d, coh_elements,
                       job_name, layup,
                       left_nodes, bl_nodes, bl_elem, right_elems,
                       ply_elem_ranges, coh_elem_ranges,
                       displacement,
                       plate_x_max, plate_y_min, plate_y_max,
                       material_model="usdfld",
                       direct_bc=False,
                       sym_bc=False,
                       yt=None):
    """Write an ABAQUS .inp for a multi-ply laminate with cohesive interfaces.

    The right edge is loaded through a kinematic coupling (or a direct
    displacement), and COH3D8 elements at each ply interface model
    delamination.

    Parameters
    ----------
    material_model : str
        ``"usdfld"``: custom USDFLD3D subroutine (CDM LaRC05, field-dependent
        stiffness).  Requires ``user=usdfld3d.f90`` at run time.  No XFEM.
        ``"elastic"``: purely elastic plies (no ply damage).  Useful for
        isolating the cohesive delamination response.

    Output variables
    ----------------
    usdfld  → C3D8R : SDV1-10, FV1-4, S
              COH3D8: SDEG, QUADSCRT, STATUS
    elastic → C3D8R : S
              COH3D8: SDEG, QUADSCRT, STATUS
    """
    total_t = sum(t for _, t in layup)
    num_nodes = len(nodes_3d)
    num_c3d8r = len(elements_3d)

    ply_mat_name = ("CDM_LARC05_3D" if material_model == "usdfld"
                    else "PLY_ELASTIC")

    with open(filepath, "w") as f:
        # ── Heading ──────────────────────────────────────────────────────
        f.write("*Heading\n")
        f.write(f"** Job name: {job_name} Model name: Model-1\n")
        f.write("** Generated by: generate_cohesive_laminate.py\n")
        f.write(f"** Layup: {' / '.join(f'{a}deg x {t}mm' for a, t in layup)}\n")
        if material_model == "usdfld":
            f.write("** Plies: USDFLD CDM LaRC05 (intralaminar damage)\n")
        else:
            f.write("** Plies: elastic (no intralaminar damage)\n")
        f.write("** Interfaces: COH3D8 traction-separation (delamination)\n")
        f.write("*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
        f.write("**\n** PARTS\n**\n")

        # ── Part ─────────────────────────────────────────────────────────
        f.write("*Part, name=Plate\n")

        # Nodes
        f.write("*Node\n")
        for nid, x, y, z in nodes_3d:
            f.write(f"      {nid},{x:16.8g},{y:16.8g},{z:16.8g}\n")

        # C3D8R elements (plies)
        f.write("*Element, type=C3D8R\n")
        for row in elements_3d:
            eid = row[0]
            nids = ", ".join(str(n) for n in row[1:])
            f.write(f"   {eid}, {nids}\n")

        # COH3D8 elements (interfaces)
        if coh_elements:
            f.write("*Element, type=COH3D8\n")
            for row in coh_elements:
                eid = row[0]
                nids = ", ".join(str(n) for n in row[1:])
                f.write(f"   {eid}, {nids}\n")

        # All-node / all-C3D8R sets
        f.write("*Nset, nset=Set-11, generate\n")
        f.write(f"     1,  {num_nodes},      1\n")
        f.write("*Elset, elset=Set-11, generate\n")
        f.write(f"     1,  {num_c3d8r},      1\n")

        # All-ply element set (C3D8R only)
        f.write("*Elset, elset=ALL_PLY, generate\n")
        f.write(f"     1,  {ply_elem_ranges[-1][1]},      1\n")

        # Per-ply element sets, orientations, and solid sections
        for ply_idx, (angle_deg, ply_t) in enumerate(layup):
            first_e, last_e = ply_elem_ranges[ply_idx]
            f.write(f"*Elset, elset=PLY-{ply_idx}, generate\n")
            f.write(f"     {first_e},  {last_e},      1\n")
            f.write(f"*Orientation, name=Ori-{ply_idx}\n")
            f.write("          1.,           0.,           0.,"
                    "           0.,           1.,           0.\n")
            f.write(f"3, {angle_deg:g}.\n")
            f.write(f"** Section: PLY-{ply_idx}  ({angle_deg}deg, {ply_t}mm)\n")
            _ply_controls = (
                "controls=PLY_HG, " if ply_mat_name == "CDM_LARC05_3D" else ""
            )
            f.write(
                f"*Solid Section, elset=PLY-{ply_idx}, "
                f"orientation=Ori-{ply_idx}, "
                f"{_ply_controls}"
                f"material={ply_mat_name}\n"
            )
            f.write(",\n")

        # Cohesive sections, one per interface
        for iface, (first_e, last_e) in enumerate(coh_elem_ranges):
            f.write(f"** Interface {iface} (between ply {iface} and ply {iface+1})\n")
            f.write(f"*Elset, elset=INTERFACE-{iface}, generate\n")
            f.write(f"     {first_e},  {last_e},      1\n")
            f.write(
                f"*Cohesive Section, elset=INTERFACE-{iface}, "
                "material=COHESIVE_IM7_8552, "
                "response=TRACTION SEPARATION, "
                "stack direction=3, controls=COH_VISC\n"
            )
            f.write("1.0,\n")  # constitutive thickness = 1.0 for traction-separation

        # All-cohesive elset (inside Part so Plate-1.ALL_IFACE resolves)
        if coh_elem_ranges:
            all_coh_first = coh_elem_ranges[0][0]
            all_coh_last = coh_elem_ranges[-1][1]
            f.write("*Elset, elset=ALL_IFACE, generate\n")
            f.write(f"     {all_coh_first},  {all_coh_last},      1\n")

        f.write("*End Part\n")

        # ── Assembly ─────────────────────────────────────────────────────
        f.write("**  \n**\n** ASSEMBLY\n**\n")
        f.write("*Assembly, name=Assembly\n")
        f.write("**  \n")
        f.write("*Instance, name=Plate-1, part=Plate\n")
        f.write("*End Instance\n")
        f.write("**  \n")

        # Reference node for kinematic coupling (right edge)
        ref_x = plate_x_max
        ref_y = (plate_y_min + plate_y_max) / 2.0
        ref_z = total_t / 2.0
        f.write("*Node\n")
        f.write(f"      1,{ref_x:16.8g},{ref_y:16.8g},{ref_z:16.8g}\n")

        # Bottom-left corner set
        f.write("*Nset, nset=Set-6, instance=Plate-1\n")
        f.write(_format_set_list(bl_nodes) + "\n")
        f.write("*Elset, elset=Set-6, instance=Plate-1\n")
        f.write(f" {bl_elem},\n")

        # Left edge nodes
        f.write("*Nset, nset=Set-7, instance=Plate-1\n")
        f.write(_format_set_list(left_nodes) + "\n")

        # Reference node sets (for coupling + BC)
        f.write("*Nset, nset=Set-9\n")
        f.write(" 1,\n")
        f.write("*Nset, nset=m_Set-8\n")
        f.write(" 1,\n")

        # Right-edge surface for kinematic coupling. The structured-quad
        # mesher does not guarantee that S4 is the right face (the "+"
        # partition quadrants have different node orderings), so one *Elset
        # and one *Surface entry are written per face; right_elems is a dict
        # {face: [eids]}.
        if isinstance(right_elems, dict):
            face_buckets = right_elems
        else:
            # a flat list of element ids is taken as face S4
            face_buckets = {"S4": list(right_elems)}

        if direct_bc:
            # ── Direct-BC variant (opt-in): no *Coupling. Apply u1 displacement
            # directly to right-edge nodes via a *Nset. Recover the node IDs
            # from the right-edge element faces (face-local node mapping in
            # Abaqus C3D8R: S2={1,2,5,6}, S3={2,3,6,7}, S4={3,4,7,8} etc.).
            FACE_LOCAL_NODES = {
                "S1": (0, 1, 2, 3),  # bottom (-z)
                "S2": (4, 5, 6, 7),  # top    (+z)
                "S3": (0, 1, 5, 4),  # face y=ymin
                "S4": (1, 2, 6, 5),  # face x=xmax  (standard "right" for canonical)
                "S5": (2, 3, 7, 6),  # face y=ymax
                "S6": (3, 0, 4, 7),  # face x=xmin
            }
            # elements_3d rows are (eid, n1..n8)
            elem_by_id = {row[0]: row[1:] for row in elements_3d}
            right_node_set = set()
            for face, eids in face_buckets.items():
                local = FACE_LOCAL_NODES.get(face)
                if local is None:
                    continue
                for eid in eids:
                    conn = elem_by_id[eid]
                    for li in local:
                        right_node_set.add(conn[li])
            right_nodes_sorted = sorted(right_node_set)
            f.write("** Direct-BC variant: nodes on right edge for u1 displacement\n")
            f.write("*Nset, nset=Set-RightDirect, instance=Plate-1\n")
            f.write(_format_set_list(right_nodes_sorted) + "\n")
        else:
            for face, eids in face_buckets.items():
                if not eids:
                    continue
                f.write(f"*Elset, elset=_s_Surf-2_{face}, internal, instance=Plate-1\n")
                f.write(_format_set_list(eids) + "\n")
            f.write("*Surface, type=ELEMENT, name=s_Surf-2\n")
            for face, eids in face_buckets.items():
                if eids:
                    f.write(f"_s_Surf-2_{face}, {face}\n")

            # Kinematic coupling: ref node → right surface.  Restricted to DOF 1
            # ("Poisson-relaxed tie") so the right-edge nodes can contract freely
            # in y and z under load; a rigid (all-DOF) coupling produces spurious
            # damage at the loaded corners.
            f.write("** Constraint: Constraint-2\n")
            f.write(
                "*Coupling, constraint name=Constraint-2, "
                "ref node=m_Set-8, surface=s_Surf-2\n"
            )
            f.write("*Kinematic\n")
            f.write("1, 1\n")

        # No XFEM enrichment: USDFLD handles damage via field variables
        f.write("** No XFEM — using USDFLD CDM for ply damage\n")

        # Optional symmetric BC on the top face (z = total_t): the model is the
        # bottom half of a symmetric [layup]_s laminate, so a 4-ply layup
        # represents an 8-ply [a,b,c,d]_s at half the mesh cost.
        if sym_bc:
            top_z = sum(t for _, t in layup)
            top_ids = sorted(nid for nid, x, y, z in nodes_3d
                             if abs(z - top_z) < 1e-4)
            f.write("** symBC: top-face (z=t) nodes for z-reflection plane\n")
            f.write("*Nset, nset=TopFaceZ, instance=Plate-1\n")
            f.write(_format_set_list(top_ids) + "\n")

        f.write("*End Assembly\n")

        # ── Property table (USDFLD only, before the material) ───────────
        if material_model == "usdfld":
            f.write("** \n")
            f.write("** ---------------------------------------------------------------\n")
            f.write("** Property table type declaration (global, for USDFLD subroutine)\n")
            f.write("** Property order (plyTools.f90 IPROP_* constants, 25 values for 3D):\n")
            f.write("**  1:E11  2:E22  3:nu12  4:G12  5:Xt  6:Xc  7:Yt  8:Yc\n")
            f.write("**  9:Sl  10:a0  11:nL  12:nT  13:ILSS  14:Zt  15:G1cMat  16:G2cMat\n")
            f.write("** 17:G1cFibT  18:G1cFibK  19:GExpMM  20:E11comp  21:E33  22:nu13  "
                    "23:nu23  24:G13  25:G23\n")
            f.write("** ---------------------------------------------------------------\n")
            f.write("*PROPERTY TABLE TYPE, NAME=LARC05PROPERTIES, PROPERTIES=25\n")

        # ── Materials ────────────────────────────────────────────────────
        f.write("** \n** MATERIALS\n** \n")

        if material_model == "usdfld":
            # CDM LaRC05 via USDFLD3D subroutine (same as usdfld_cdm_test.inp)
            f.write("*Material, name=CDM_LARC05_3D\n")
            f.write("** Activate USDFLD subroutine (computes FV1-4 from LaRC05)\n")
            f.write("*User Defined Field\n")
            f.write("** 10 SDVs: SDV1-5 = failure indices, SDV6-10 = damage variables\n")
            f.write("*Depvar\n")
            f.write("     10,\n")
            f.write("1, FI_1_Matrix,   Matrix failure index\n")
            f.write("2, FI_2_Split,    Matrix splitting failure index\n")
            f.write("3, FI_3_FibreTen, Fibre tension failure index\n")
            f.write("4, FI_4_Kink,     Fibre kinking failure index\n")
            f.write("5, FI_MAX,        Maximum local failure index\n")
            f.write("6, DMG_1,         Damage for matrix failure\n")
            f.write("7, DMG_2,         Damage for matrix splitting\n")
            f.write("8, DMG_3,         Damage for tensile fibre failure\n")
            f.write("9, DMG_4,         Damage for fibre kinking\n")
            f.write("10, DMG_COMBINED, Combined health variable\n")
            f.write("**\n")
            f.write("** Field-dependent elastic stiffness (4 field dependencies)\n")
            f.write("**   FV1 = transverse matrix damage  -> degrades E22, G12, nu12\n")
            f.write("**   FV2 = matrix mode-2 damage      -> degrades E22, G12, G23\n")
            f.write("**   FV3 = through-thickness damage   -> degrades E33, G13, G23\n")
            f.write("**   FV4 = fibre damage (combined)    -> degrades E11\n")
            f.write("** Residual stiffness = 0.001 * undamaged (avoids singularity)\n")
            f.write("*Elastic, type=ENGINEERING CONSTANTS, dependencies=4\n")
            f.write("** E11,       E22,     E33,    nu12,   nu13,   nu23,   G12,     G13,\n")
            f.write("** G23,  FV1,  FV2,  FV3,  FV4\n")
            # ── CDM_LARC05_3D elastic + LaRC05 property table constants ──
            # Locally scoped so the module-level MAT_* (used by the elastic-
            # only ply path and by other generators) is not disturbed. These
            # are the IM7/8552 values consumed by the USDFLD subroutine.
            E1, E2, E3 = 161000.0, 11400.0, 11400.0
            NU12, NU13, NU23 = 0.32, 0.32, 0.43
            G12, G13, G23 = 5170.0, 5170.0, 3980.0
            RES = 0.001  # residual stiffness factor for degraded rows
            e1r, e2r, e3r = E1 * RES, E2 * RES, E3 * RES
            g12r, g13r, g23r = G12 * RES, G13 * RES, G23 * RES
            # LaRC05 strengths / fracture params (IPROP_* order for 3D)
            XT, XC = 2806.0, 1200.1      # E11 tensile / compressive strength
            # Transverse tensile strength: the constant varied by the
            # material-variability study (nominal 60 MPa when yt is None); it
            # appears in one line of the deck, the *Property Table row below.
            YT, YC = (60.0 if yt is None else float(yt)), 199.8
            SL, A0 = 90.0, 53.0          # long. shear / kinking-plane angle
            NL, NT = 0.082, -1.0         # longitudinal / transverse friction
            ILSS, ZT = 50.0, 40.0        # interlaminar / through-thickness
            G1C_MAT, G2C_MAT = 0.293, 0.631
            G1C_FIB_T, G1C_FIB_K = 112.7, 100.0
            G_EXP_MM = 1.0
            f.write("** --- Undamaged (FV1=0, FV2=0, FV3=0, FV4=0) ---\n")
            f.write(f"{E1}, {E2}, {E3}, {NU12}, {NU13}, {NU23}, {G12}, {G13},\n")
            f.write(f"{G23}, 0., 0., 0., 0.\n")
            f.write("** --- FV1=1: transverse matrix cracking (E22, G12 -> 0) ---\n")
            f.write(f"{E1}, {e2r}, {E3}, {RES}, {NU13}, {RES}, {g12r}, {G13},\n")
            f.write(f"{G23}, 1., 0., 0., 0.\n")
            f.write("** --- FV2=1: matrix mode 2 (E22, G12, G23 -> 0) ---\n")
            f.write(f"{E1}, {e2r}, {E3}, {RES}, {NU13}, {RES}, {g12r}, {G13},\n")
            f.write(f"{g23r}, 0., 1., 0., 0.\n")
            f.write("** --- FV3=1: through-thickness damage (E33, G13, G23 -> 0) ---\n")
            f.write(f"{E1}, {E2}, {e3r}, {NU12}, {RES}, {RES}, {G12}, {g13r},\n")
            f.write(f"{g23r}, 0., 0., 1., 0.\n")
            f.write("** --- FV4=1: fibre damage (E11 -> 0) ---\n")
            f.write(f"{e1r}, {E2}, {E3}, {RES}, {RES}, {NU23}, {g12r}, {g13r},\n")
            f.write(f"{G23}, 0., 0., 0., 1.\n")
            f.write("**\n")
            f.write("** LaRC05 material constants for IM7/8552\n")
            f.write("** Refs: Camanho 2007, Pinho 2006, Laffan 2012, Camanho 2003\n")
            f.write("*Property Table, type=LARC05PROPERTIES\n")
            f.write(f"{E1}, {E2}, {NU12}, {G12}, {XT}, {XC}, {YT}, {YC},\n")
            f.write(f"{SL}, {A0}, {NL}, {NT}, {ILSS}, {ZT}, {G1C_MAT}, {G2C_MAT},\n")
            f.write(f"{G1C_FIB_T}, {G1C_FIB_K}, {G_EXP_MM}, {E1}, {E2}, {NU12}, {NU23}, {G13},\n")
            f.write(f"{G23}\n")
        else:
            # Purely elastic plies (no damage)
            f.write(f"*Material, name={ply_mat_name}\n")
            f.write("*Elastic, type=ENGINEERING CONSTANTS\n")
            f.write(
                f"{MAT_E1},{MAT_E2},{MAT_E3},"
                f" {MAT_NU12}, {MAT_NU13}, {MAT_NU23},"
                f"{MAT_G12},{MAT_G13}\n"
            )
            f.write(f"{MAT_G23},\n")

        # Cohesive interface material (same for all interfaces)
        f.write("** \n")
        f.write("*Material, name=COHESIVE_IM7_8552\n")
        f.write("** Penalty stiffness (N/mm^3):  K_nn, K_ss, K_tt\n")
        f.write("** Ref: Camanho et al. (2003), J. Composite Materials\n")
        f.write("*Elastic, type=TRACTION\n")
        f.write(f"{COH_K:g}, {COH_K:g}, {COH_K:g}\n")
        f.write("** Damage initiation: quadratic stress criterion\n")
        f.write("** tau_n0 (Mode I), tau_s0 (Mode II), tau_t0 (Mode III)  [MPa]\n")
        f.write("*Damage Initiation, criterion=QUADS\n")
        f.write(f"{COH_T_N:g}, {COH_T_S:g}, {COH_T_T:g}\n")
        f.write("** Damage evolution: BK mixed-mode fracture energy\n")
        f.write("** GIc, GIIc, GIIIc [N/mm = kJ/m^2],  eta (BK exponent)\n")
        f.write("*Damage Evolution, type=ENERGY, mixed mode behavior=BK,"
                f" power={COH_BK_ETA:g}\n")
        f.write(f"{COH_GIC:g}, {COH_GIIC:g}, {COH_GIIIC:g}\n")
        f.write("** Viscous regularisation: use *Section Controls (cohesive elements\n")
        f.write("** with TRACTION SEPARATION response ignore *Damage Stabilization).\n")

        # ── Step ─────────────────────────────────────────────────────────
        f.write("** ----------------------------------------------------------------\n")
        f.write("** \n** STEP: Step-1\n** \n")
        f.write("*Section Controls, name=PLY_HG, hourglass=ENHANCED\n")
        f.write(f"*Section Controls, name=COH_VISC, viscosity={COH_VISC:g}\n")
        f.write(f"*Step, name=Step-1, nlgeom=NO, inc={STEP_MAX_NUM_INC}\n")
        f.write("*Static\n")
        f.write(
            f"{STEP_INITIAL_INC}, {STEP_TOTAL_TIME}, "
            f"{STEP_MIN_INC}, {STEP_MAX_INC}\n"
        )
        f.write("** \n")
        f.write("*Controls, parameters=time incrementation\n")
        f.write("12, 15, , , , , , 20, , , \n")

        # Boundary conditions
        f.write("** \n** BOUNDARY CONDITIONS\n** \n")
        f.write("** Name: BOTTOMLEFT Type: Displacement/Rotation\n")
        f.write("*Boundary\n")
        f.write("Set-6, 1, 1\n")
        f.write("Set-6, 2, 2\n")
        f.write("Set-6, 3, 3\n")
        f.write("** Name: LEFT Type: Displacement/Rotation\n")
        f.write("*Boundary\n")
        f.write("Set-7, 1, 1\n")
        if sym_bc:
            f.write("** Name: TOP_FACE_SYMMETRY  (z=t reflection plane "
                    f"— simulates [layup]_s)\n")
            f.write("*Boundary\n")
            f.write("TopFaceZ, 3, 3\n")
        f.write("** Name: RIGHT Type: Displacement/Rotation\n")
        f.write("*Boundary\n")
        if direct_bc:
            f.write(f"Set-RightDirect, 1, 1, {displacement}\n")
        else:
            f.write(f"Set-9, 1, 1, {displacement}\n")

        # Output requests
        f.write("** \n** OUTPUT REQUESTS\n** \n")
        f.write("*Restart, write, frequency=0\n")
        f.write("** \n")
        f.write("*Output, field, number interval=100\n")

        # Ply output
        if material_model == "usdfld":
            f.write("** SDV1-5: failure indices, SDV6-10: damage variables\n")
            f.write("** FV1-4: field variables (stiffness reduction drivers)\n")
            f.write("*Element Output, elset=Plate-1.ALL_PLY, "
                    "position=INTEGRATION POINTS, directions=YES\n")
            f.write("SDV1, SDV2, SDV3, SDV4, SDV5, SDV6, SDV7, SDV8, SDV9, SDV10,\n")
            f.write("FV1, FV2, FV3, FV4, S\n")
        else:
            f.write("*Element Output, elset=Plate-1.ALL_PLY, "
                    "position=INTEGRATION POINTS, directions=YES\n")
            f.write("S\n")

        # Cohesive interface output
        if coh_elem_ranges:
            f.write("** \n** Interface damage (COH3D8) — delamination\n")
            f.write("*Element Output, elset=Plate-1.ALL_IFACE\n")
            f.write("SDEG, QUADSCRT, STATUS\n")

        f.write("** \n")
        f.write("*Node Output\n")
        f.write("U, RF\n")
        f.write("** \n")
        f.write("*Output, history, variable=PRESELECT, number interval=100\n")
        f.write("*End Step\n")


# ═══════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL API
# ═══════════════════════════════════════════════════════════════════════════

def generate_single_plate_cohesive(params, output_dir, sample_name,
                                   layup=None,
                                   material_model="usdfld",
                                   elem_size=ELEM_SIZE_DEFAULT,
                                   direct_bc=False,
                                   sym_bc=False):
    """Generate one \"+\"-partition cohesive-laminate .inp from a parameter dict.

    The 2D in-plane mesh is generated by Abaqus CAE (see
    tools/abaqus_mesh_plus.py): plate + circular hole, \"+\" partition
    through hole centre, structured-quad meshing with a single global
    seed.  Result is extruded into the C3D8R / COH3D8 laminate by the
    existing 3D pipeline.  No annular ring, no transition square, no
    other partitions.
    """
    if layup is None:
        layup = params.get("layup", LAYUP_DEFAULT)

    hx   = params["plate_half_x"]
    hy   = params["plate_half_y"]
    x_min, x_max = -hx, hx
    y_min, y_max = -hy, hy
    disp = params["displacement"]

    nodes_2d, elements_2d = generate_plus_partition_mesh_2d(
        x_min, x_max, y_min, y_max,
        params["hole_radius"],
        params["hole_center_x"],
        params["hole_center_y"],
        elem_size,
    )

    # 3-D extrusion with cohesive interface nodes
    nodes_3d, elements_3d, coh_elements, ply_elem_ranges, coh_elem_ranges = \
        extrude_to_3d_cohesive(nodes_2d, elements_2d, layup)

    # Boundary sets
    left_nodes, bl_nodes, bl_elem, right_elems = identify_boundary_sets_cohesive(
        nodes_3d, elements_3d, x_min, x_max, y_min,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{sample_name}.inp"

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
        yt=params.get("yt"),
    )

    n_c3d8r = len(elements_3d)
    n_coh   = len(coh_elements)
    return filepath, n_c3d8r, n_coh


def generate_dataset_cohesive(n_samples=N_SAMPLES, seed=RANDOM_SEED,
                               layup=None, study=None,
                               material_model="usdfld",
                               elem_size=ELEM_SIZE_DEFAULT,
                               direct_bc=False,
                               sym_bc=False,
                               yt_range=None):
    """Generate a full parametric dataset of cohesive-laminate .inp files.

    ``yt_range``: ``None`` (default) writes the nominal transverse tensile
    strength of 60 MPa into every deck. Given ``(lo, hi)``, each sample draws
    its own YT for the material-variability study: sample *i* takes stratum
    *i* of ``n_samples`` equal bins with a jitter inside it, and the
    assignment is then shuffled. The draw uses its own generator, seeded from
    ``seed + 1``, so the geometry sampling is unchanged.
    """
    if layup is None:
        layup = LAYUP_DEFAULT

    # Match run_pipeline.py layout: data/raw/<study>/geometry/
    if study:
        out_inp_dir  = PROJECT_ROOT / "data" / "raw" / study / "geometry"
        out_meta_dir = PROJECT_ROOT / "data" / "processed" / study
    else:
        out_inp_dir  = PROJECT_ROOT / "data" / "raw" / "geometry"
        out_meta_dir = PROJECT_ROOT / "data" / "processed"

    layup_str  = " / ".join(f"{a}°×{t}mm" for a, t in layup)
    print("=" * 70)
    print("COHESIVE LAMINATE DATASET GENERATION")
    print("=" * 70)
    print(f"  Samples  : {n_samples}")
    print(f"  Layup    : [{layup_str}]  total={sum(t for _,t in layup):.4g} mm")
    print(f"  Plies    : {len(layup)}  →  {len(layup)-1} cohesive interface(s)")
    print("  Material : IM7/8552 (elastic plies + cohesive interfaces)")
    print(f"  Output   : {out_inp_dir}")
    print("=" * 70)

    all_params = sample_parameters(
        n_samples=n_samples,
        param_ranges=PARAM_RANGES,
        plate_half_size=PLATE_HALF_SIZE,
        inner_factor=INNER_SQUARE_FACTOR,
        margin=GEOMETRY_MARGIN,
        seed=seed,
    )
    for p in all_params:
        p["layup"] = layup

    if yt_range is not None:
        import numpy as _np
        lo, hi = float(yt_range[0]), float(yt_range[1])
        _rng = _np.random.default_rng(seed + 1)
        edges = _np.linspace(lo, hi, n_samples + 1)
        draws = edges[:-1] + _rng.random(n_samples) * (edges[1:] - edges[:-1])
        _rng.shuffle(draws)
        for p, v in zip(all_params, draws):
            p["yt"] = float(v)
        print(f"  YT varied : {lo:g} to {hi:g} MPa, stratified over "
              f"{n_samples} samples (nominal 60)")

    sample_names = [f"plate_{i+1:03d}" for i in range(n_samples)]
    param_tensor = torch.tensor(
        [[p["hole_radius"], p["hole_center_x"], p["hole_center_y"],
          p["displacement"], p["plate_half_x"], p["plate_half_y"]]
         for p in all_params],
        dtype=torch.float32,
    )

    for i, (p, name) in enumerate(zip(all_params, sample_names)):
        fp, n_c, n_coh = generate_single_plate_cohesive(
            p, out_inp_dir, name, layup=layup,
            material_model=material_model,
            elem_size=elem_size,
            direct_bc=direct_bc,
            sym_bc=sym_bc,
        )
        if (i + 1) % max(1, n_samples // 10) == 0 or i == n_samples - 1:
            print(f"  [{i+1:4d}/{n_samples}]  {fp.name}"
                  f"  ({n_c} C3D8R + {n_coh} COH3D8)")

    metadata = {
        "params":       param_tensor,
        "param_names":  ["hole_radius", "hole_center_x", "hole_center_y",
                         "displacement", "plate_half_x", "plate_half_y"],
        "sample_names": sample_names,
        "layup":        layup,
        "num_samples":  n_samples,
        "model_type":   "cohesive_laminate",
    }
    # Only present when the study actually varied it, so existing metadata
    # files keep exactly the keys they have today.
    if yt_range is not None:
        metadata["yt"] = torch.tensor([p["yt"] for p in all_params],
                                      dtype=torch.float32)
        metadata["yt_range"] = [float(yt_range[0]), float(yt_range[1])]
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_meta_dir / "plate_geometry_data.pt"
    torch.save(metadata, meta_path)
    print(f"\n  Metadata saved: {meta_path}")
    print("=" * 70)
    return metadata


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate cohesive-laminate .inp decks with a uniform "
                    "structured mesh, meshed by Abaqus CAE with a '+' "
                    "partition through the hole.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (one plate with explicit geometry, 1 mm seed):\n"
            "  python -m src.preprocessing.generate_cohesive_laminate_uniform \\\n"
            "      --layup '45,-45,0,90' --ply-thickness 0.125 \\\n"
            "      --elem-size 1.0 \\\n"
            "      --plate-half-x 16.8241 --plate-half-y 19.4678 \\\n"
            "      --hole-radius 4.5667 --hole-cx 1.7493 --hole-cy 0.4163 \\\n"
            "      --displacement 0.4936 \\\n"
            "      --study cp_plyres_uniform_p1\n"
        ),
    )
    parser.add_argument("--n", type=int, default=1,
                        help="Number of samples (default: 1).")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed (used only when geometry overrides absent).")
    parser.add_argument("--layup", type=str, default=None,
                        help="Comma-separated fibre angles, e.g. '45,-45,0,90'.")
    parser.add_argument("--ply-thickness", type=float, default=None,
                        help="Thickness per ply (mm). Default: 0.125.")
    parser.add_argument("--study", type=str, default=None,
                        help="Study name; creates data/raw/<study>/geometry/ "
                             "and data/processed/<study>/.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output directory for .inp files.")
    # Uniform-mesh control
    parser.add_argument("--elem-size", type=float, default=ELEM_SIZE_DEFAULT,
                        help=f"In-plane element edge length, mm "
                             f"(default: {ELEM_SIZE_DEFAULT}).")
    # Optional: a direct *Boundary on the right-edge nodes instead of the
    # kinematic coupling
    parser.add_argument("--direct-bc", action="store_true", default=False,
                        help="Apply the u1 displacement directly to the "
                             "right-edge nodes via *Boundary, without *Coupling. "
                             "Default: off (kinematic coupling).")
    # Optional z-symmetry on the top face (z = total_t): the 4-ply model is
    # then the bottom half of a symmetric [layup]_s laminate
    parser.add_argument("--sym-bc", action="store_true", default=False,
                        help="Add a z-symmetry BC on the top face (the model "
                             "is the bottom half of [layup]_s). Default: off.")
    # Material-variability study: vary the transverse tensile strength;
    # omitted, every deck carries the nominal 60 MPa
    parser.add_argument("--yt-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Vary the transverse tensile strength YT over "
                             "[LO, HI] MPa, one stratified draw per sample. "
                             "Default: off (nominal 60 MPa everywhere).")
    # Explicit geometry of a single plate (skips the LHS sampler)
    parser.add_argument("--plate-half-x", type=float, default=None)
    parser.add_argument("--plate-half-y", type=float, default=None)
    parser.add_argument("--hole-radius", type=float, default=None)
    parser.add_argument("--hole-cx", type=float, default=None)
    parser.add_argument("--hole-cy", type=float, default=None)
    parser.add_argument("--displacement", type=float, default=None)
    parser.add_argument("--material-model", type=str, default="usdfld",
                        choices=["usdfld", "elastic"])
    args = parser.parse_args()

    # Build layup
    if args.layup is not None:
        angles = [float(a) for a in args.layup.split(",")]
        t_ply  = args.ply_thickness if args.ply_thickness else 0.125
        layup  = [(a, round(t_ply, 6)) for a in angles]
    else:
        layup = LAYUP_DEFAULT

    # If all six geometry overrides are given, skip the LHS sampler entirely.
    geom_vals = [args.plate_half_x, args.plate_half_y, args.hole_radius,
                 args.hole_cx, args.hole_cy, args.displacement]
    explicit_geom = all(v is not None for v in geom_vals)

    if explicit_geom:
        params_list = [{
            "plate_half_x":   args.plate_half_x,
            "plate_half_y":   args.plate_half_y,
            "hole_radius":    args.hole_radius,
            "hole_center_x":  args.hole_cx,
            "hole_center_y":  args.hole_cy,
            "displacement":   args.displacement,
            "layup":          layup,
        }]
        if args.study:
            out_dir = PROJECT_ROOT / "data" / "raw" / args.study / "geometry"
        else:
            out_dir = Path(args.out_dir) if args.out_dir else (
                PROJECT_ROOT / "data" / "raw" / "geometry_uniform")
        for i, p in enumerate(params_list):
            name = f"plate_{i+1:03d}"
            fp, n_c, n_coh = generate_single_plate_cohesive(
                p, out_dir, name, layup=layup,
                material_model=args.material_model,
                elem_size=args.elem_size,
                direct_bc=args.direct_bc,
                sym_bc=args.sym_bc,
            )
            print(f"  Written: {fp}  ({n_c} C3D8R + {n_coh} COH3D8)")
            print(f"    elem_size={args.elem_size} mm  (uniform grid, "
                  f"smooth hole boundary)")
    elif args.out_dir:
        # Sampled single-plate mode (test, no explicit geometry)
        params_list = sample_parameters(
            n_samples=args.n,
            param_ranges=PARAM_RANGES,
            plate_half_size=PLATE_HALF_SIZE,
            inner_factor=INNER_SQUARE_FACTOR,
            margin=GEOMETRY_MARGIN,
            seed=args.seed,
        )
        out = Path(args.out_dir)
        for i, p in enumerate(params_list):
            p["layup"] = layup
            name = f"plate_{i+1:03d}"
            fp, n_c, n_coh = generate_single_plate_cohesive(
                p, out, name, layup=layup,
                material_model=args.material_model,
                elem_size=args.elem_size,
                direct_bc=args.direct_bc,
                sym_bc=args.sym_bc,
            )
            print(f"  Written: {fp}  ({n_c} C3D8R + {n_coh} COH3D8)")
    else:
        generate_dataset_cohesive(
            n_samples=args.n,
            seed=args.seed,
            layup=layup,
            study=args.study,
            material_model=args.material_model,
            elem_size=args.elem_size,
            direct_bc=args.direct_bc,
            sym_bc=args.sym_bc,
            yt_range=args.yt_range,
        )


if __name__ == "__main__":
    main()
