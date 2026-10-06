"""Build a PyG dataset from cohesive-laminate simulations.

Stages
------
1. Parse each .inp file (geometry and ply topology).
2. Extract the element fields of each .odb with the Abaqus Python worker
   (_abaqus_extract_cohesive_worker.py). The graph uses the matrix failure
   index SDV_FI_1_Matrix and the matrix damage SDV_DMG_1 of the ply elements.
3. Build one graph per pair of consecutive frames and write the .pt dataset.
4. Optionally, write diagnostic plots.

Graph
-----
Nodes are the ply elements (C3D8R). Cohesive elements (COH3D8) are not nodes;
their centroids only enter the z range that normalises layer_norm. Edges join
face-sharing elements of the same ply and, through the thickness, each element
to the nearest element (in plane) of the next ply. Edge features are the
centroid offset and distance (dx, dy, dz, |d|).

Node features
-------------
  0-2   x, y, z       element centroid (mm)
  3-4   cos, sin      of the ply angle
  5-6   cos, sin      of the ply angle (equal to 3-4)
  7     layer_norm    (z - z_min) / (z_max - z_min)
  8     is_hole       a corner node lies on a hole edge
  9     is_fixed      a corner node lies on the fixed boundary
 10     is_disp       a corner node lies on the displaced boundary
 11     disp_amt      applied displacement (mm) if is_disp, else 0
 12     node_type     0 for every node
 13     Y_T           transverse tensile strength (--include-yt-feature)
 next   D_now         current matrix damage (--include-damage-feature)
 last   FI_now        current matrix failure index

Targets
-------
  data.y        failure-index rate dFI/dt (FI capped at FI_CAP, rate >= 0)
  data.y_dmg    matrix-damage rate dD/dt (rate >= 0)
  data.dmg_now  current matrix damage D
data.y_dmg and data.dmg_now are present when SDV_DMG_1 was extracted.

Usage
-----
    # Dataset from existing ODB files
    python -m src.preprocessing.build_cohesive_dataset \\
        --study <study> --include-damage-feature --resample-dt 0.01 --plot

    # Dry run: parse the .inp files only, no ODB extraction
    python -m src.preprocessing.build_cohesive_dataset \\
        --study <study> --dry-run --plot

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "results_cohesive"

WORKER_SCRIPT = Path(__file__).parent / "_abaqus_extract_cohesive_worker.py"
ABAQUS_CMD = "abaqus"

# Cache version: bump this whenever the Abaqus worker output format changes
# (e.g. new fields added, field names changed, data layout changed).
# Stale NPZ caches with a missing or older version are automatically re-extracted.
CACHE_VERSION = 3  # v3: iface_fi (QUADSCRT)

# Failure-index cap.  FI > 1 means the failure criterion is exceeded;
# values beyond ~3 are physically meaningless (fully degraded element
# continuing to accumulate numerical FI).  Clamping here bounds both the
# input feature and the derived rate, so the model focuses on the 0→3
# transition instead of chasing unbounded values.
FI_CAP = 3.0

# ═══════════════════════════════════════════════════════════════════════════
# INP PARSING
# ═══════════════════════════════════════════════════════════════════════════

def _parse_nodes(lines, start):
    nodes = {}
    for line in lines[start:]:
        line = line.strip()
        if line.startswith("*"):
            break
        parts = line.split(",")
        if len(parts) >= 4:
            nid = int(parts[0])
            nodes[nid] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return nodes


def _parse_elements(lines, start, elem_type):
    elements = {}
    for line in lines[start:]:
        line = line.strip()
        if line.startswith("*"):
            break
        parts = [int(x) for x in line.split(",")]
        if len(parts) >= 2:
            eid = parts[0]
            elements[eid] = (elem_type, parts[1:])
    return elements


def _parse_elset_generate(line_after):
    """Parse '*Elset, generate' → (first, last, step)."""
    parts = [int(x.strip()) for x in line_after.split(",") if x.strip()]
    if len(parts) == 3:
        return range(parts[0], parts[1] + 1, parts[2])
    return []


def parse_cohesive_inp(inp_path):
    """Parse a cohesive laminate .inp file.

    Returns
    -------
    dict with:
        nodes       : {nid: (x, y, z)}
        ply_elems   : {eid: [n1..n8]}   C3D8R
        iface_elems : {eid: [n1..n8]}   COH3D8
        ply_angles  : {ely_idx: angle_deg}   PLY-k → angle
        n_plies     : int
        n_ifaces    : int
        ply_ranges  : [(first, last), ...]  C3D8R ID ranges per ply
        iface_ranges: [(first, last), ...]  COH3D8 ID ranges per interface
        hole_radius : float (from comments or Set-9 reference node proximity)
        displacement: float
        plate_x_max : float
    """
    lines = Path(inp_path).read_text().splitlines()
    nodes = {}
    ply_elems = {}
    iface_elems = {}
    ply_angles = {}       # ply_idx → angle_deg
    ply_elset_ranges = []
    iface_elset_ranges = []
    displacement = 0.0
    ref_node_x = None
    fixed_nids = set()    # Set-7: left edge fixed nodes
    disp_eids  = set()    # _s_Surf-2_S4: right edge displaced elements

    i = 0
    current_elem_type = None
    in_assembly = False
    while i < len(lines):
        line = lines[i].strip()
        low = line.lower()

        if low.startswith("*assembly"):
            in_assembly = True
        elif low.startswith("*end assembly"):
            in_assembly = False

        # Only parse mesh nodes from the Part section; skip Assembly ref nodes
        # (the Assembly *Node section has a single ref node whose ID can collide
        # with real mesh node IDs, corrupting element centroids).
        if low.startswith("*node") and not low.startswith("*node output") \
                and not in_assembly:
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("*"):
                l = lines[i].strip()
                if l:
                    parts = l.split(",")
                    if len(parts) >= 4:
                        nid = int(parts[0])
                        nodes[nid] = (float(parts[1]), float(parts[2]),
                                      float(parts[3]))
                i += 1
            continue

        if low.startswith("*element,"):
            m = re.search(r"type\s*=\s*(\w+)", low)
            current_elem_type = m.group(1).upper() if m else "UNKNOWN"
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("*"):
                l = lines[i].strip()
                if l:
                    parts = [int(x) for x in l.split(",")]
                    eid = parts[0]
                    if current_elem_type == "C3D8R":
                        ply_elems[eid] = parts[1:]
                    elif current_elem_type == "COH3D8":
                        iface_elems[eid] = parts[1:]
                i += 1
            continue

        # Per-ply element sets (PLY-k, generate)
        if low.startswith("*elset") and "elset=ply-" in low:
            m = re.search(r"elset=ply-(\d+)", low)
            if m:
                ply_idx = int(m.group(1))
                if "generate" in low and i + 1 < len(lines):
                    r = _parse_elset_generate(lines[i + 1])
                    if r:
                        ply_elset_ranges.append((ply_idx, r.start, r.stop - 1))

        # Per-interface element sets (INTERFACE-k, generate)
        if low.startswith("*elset") and "elset=interface-" in low:
            m = re.search(r"elset=interface-(\d+)", low)
            if m:
                iface_idx = int(m.group(1))
                if "generate" in low and i + 1 < len(lines):
                    r = _parse_elset_generate(lines[i + 1])
                    if r:
                        iface_elset_ranges.append((iface_idx, r.start, r.stop - 1))

        # Fiber angles from *Orientation
        if low.startswith("*orientation"):
            m = re.search(r"name=ori-(\d+)", low)
            if m:
                ply_idx = int(m.group(1))
                # The rotation angle is on the next-next line: "3, angle."
                for j in range(i + 1, min(i + 4, len(lines))):
                    l2 = lines[j].strip()
                    if l2.startswith("3,"):
                        angle_str = l2.split(",")[1].strip().rstrip(".")
                        ply_angles[ply_idx] = float(angle_str)
                        break

        # Displacement from *Boundary Set-9
        if low.startswith("*boundary") and i + 1 < len(lines):
            next_l = lines[i + 1].strip()
            if next_l.lower().startswith("set-9"):
                parts = next_l.split(",")
                if len(parts) >= 4:
                    try:
                        displacement = float(parts[3])
                    except ValueError:
                        pass

        # Set-7: left edge fixed nodes (DOF 1 = x-displacement fixed)
        if low.startswith("*nset") and "nset=set-7" in low:
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("*"):
                for part in lines[i].strip().split(","):
                    part = part.strip()
                    if part:
                        try:
                            fixed_nids.add(int(part))
                        except ValueError:
                            pass
                i += 1
            continue

        # _s_Surf-2_S4: right edge C3D8R elements (displaced surface)
        if low.startswith("*elset") and "_s_surf-2_s4" in low:
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("*"):
                for part in lines[i].strip().split(","):
                    part = part.strip()
                    if part:
                        try:
                            disp_eids.add(int(part))
                        except ValueError:
                            pass
                i += 1
            continue

        # Reference node x (= plate_x_max) from assembly *Node
        if "ref_x" not in locals() and low.startswith("*node") and \
                i > 0 and "assembly" in " ".join(lines[max(0,i-20):i]).lower():
            for j in range(i + 1, min(i + 4, len(lines))):
                l2 = lines[j].strip()
                if l2 and not l2.startswith("*"):
                    parts = l2.split(",")
                    if len(parts) >= 2 and parts[0].strip() == "1":
                        try:
                            ref_node_x = float(parts[1])
                        except ValueError:
                            pass
                        break

        i += 1

    # Build range lists
    ply_elset_ranges.sort(key=lambda x: x[0])
    iface_elset_ranges.sort(key=lambda x: x[0])

    ply_ranges = [(r[1], r[2]) for r in ply_elset_ranges]
    iface_ranges = [(r[1], r[2]) for r in iface_elset_ranges]

    if not nodes:
        raise ValueError(f"{inp_path}: no nodes found")
    if ref_node_x is None:
        raise ValueError(f"{inp_path}: no reference node in the assembly; "
                         "cannot locate the displaced edge")

    # Estimate hole radius from nearest node to origin
    hole_r = min(math.sqrt(x**2 + y**2)
                 for nid, (x, y, z) in nodes.items())

    plate_x_max = ref_node_x

    return {
        "nodes":        nodes,
        "ply_elems":    ply_elems,
        "iface_elems":  iface_elems,
        "ply_angles":   ply_angles,
        "n_plies":      len(ply_ranges),
        "n_ifaces":     len(iface_ranges),
        "ply_ranges":   ply_ranges,
        "iface_ranges": iface_ranges,
        "hole_radius":  hole_r,
        "displacement": displacement,
        "plate_x_max":  plate_x_max,
        "fixed_nids":   fixed_nids,
        "disp_eids":    disp_eids,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CENTROID AND FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def _centroid(nids, nodes):
    coords = np.array([nodes[n] for n in nids if n in nodes])
    return coords.mean(axis=0) if len(coords) else np.zeros(3)


def build_node_features(parsed, sample_params):
    """Build the static feature matrix and the element-to-node-index map.

    Only ply (C3D8R) elements become graph nodes. Interface (COH3D8)
    elements are not nodes, but their centroids still enter the z range
    that normalises layer_norm.

    Parameters
    ----------
    parsed : dict from parse_cohesive_inp
    sample_params : dict with hole/displacement geometry params

    Returns
    -------
    feats : np.ndarray [N, 13], or [N, 14] with Y_T
    ply_eid_to_idx : dict {eid: node_index}
    centroids : np.ndarray [N, 3]
    """
    nodes    = parsed["nodes"]
    ply_e    = parsed["ply_elems"]
    iface_e  = parsed["iface_elems"]
    angles   = parsed["ply_angles"]     # ply_idx → deg
    ply_ranges   = parsed["ply_ranges"]    # [(first, last), ...]
    hole_radius  = sample_params["hole_radius"]
    hole_cx      = sample_params["hole_center_x"]
    hole_cy      = sample_params["hole_center_y"]
    displacement = sample_params["displacement"]
    # Per-sample transverse tensile strength, broadcast to every node of the
    # sample when the caller puts it in sample_params (--include-yt-feature).
    # It is the last static column, so the head of x (z at 2, is_hole /
    # is_fixed / is_disp at 8/9/10) and the dynamic columns at the end
    # (damage-now at -2, FI-now at -1) keep their positions.
    yt = sample_params.get("yt")
    # Multi-hole support (extrapolation studies): sample_params may carry a
    # "holes" list of (cx, cy, r).  Falls back to the single-hole params;
    # hole_radius <= 0 means "no hole" (is_hole = 0 everywhere).
    holes = sample_params.get("holes")
    if holes is None:
        holes = ([(float(hole_cx), float(hole_cy), float(hole_radius))]
                 if float(hole_radius) > 0 else [])
    hole_tols = [max(0.3, r * 0.10) for (_, _, r) in holes]
    fixed_nids   = parsed["fixed_nids"]
    plate_x_max  = parsed["plate_x_max"]
    disp_tol     = 0.05  # tolerance for right-edge node detection (mm)

    # Map each ply element to its ply index
    eid_to_ply = {}
    for pi, (first, last) in enumerate(ply_ranges):
        for eid in range(first, last + 1):
            eid_to_ply[eid] = pi

    rows = []
    ply_eid_to_idx   = {}
    centroids_list   = []

    # ── Pass 1: compute all centroids (ply then interface, sorted) ────────
    ply_cens   = {}   # eid → np.array [3]
    iface_cens = {}   # eid → np.array [3]
    for eid in sorted(ply_e.keys()):
        ply_cens[eid] = _centroid(ply_e[eid], nodes)
    for eid in sorted(iface_e.keys()):
        iface_cens[eid] = _centroid(iface_e[eid], nodes)

    # Global z range from ply centroids → used to normalize layer_norm for
    # all nodes (ply and interface alike), so they each get a unique value
    # that reflects their true through-thickness position.
    all_z = ([c[2] for c in ply_cens.values()] +
             [c[2] for c in iface_cens.values()])
    z_min = min(all_z) if all_z else 0.0
    z_max = max(all_z) if all_z else 1.0
    z_range = z_max - z_min if z_max > z_min else 1.0

    # ── Ply nodes ─────────────────────────────────────────────────────────
    for eid in sorted(ply_e.keys()):
        pi = eid_to_ply.get(eid, 0)
        angle_deg = angles.get(pi, 0.0)
        theta = math.radians(angle_deg)
        ca, sa = math.cos(theta), math.sin(theta)

        cen = ply_cens[eid]
        centroids_list.append(cen)

        layer_norm = (cen[2] - z_min) / z_range

        # Node-based boundary/hole detection (checks actual corner nodes)
        nids = ply_e[eid]
        is_hole = 0.0
        is_fixed = 0.0
        is_disp = 0.0
        for nid in nids:
            if nid not in nodes:
                continue
            nx, ny, _ = nodes[nid]
            # Hole adjacency: node within hole_tol of any hole circumference
            for (hcx, hcy, hr), htol in zip(holes, hole_tols):
                d_hole = math.sqrt((nx - hcx)**2 + (ny - hcy)**2)
                if abs(d_hole - hr) < htol:
                    is_hole = 1.0
                    break
            # Fixed boundary (left edge, x ≈ x_min)
            if nid in fixed_nids:
                is_fixed = 1.0
            # Displaced boundary: node on right edge (x ≈ plate_x_max)
            if abs(nx - plate_x_max) < disp_tol:
                is_disp = 1.0

        disp_amt = displacement if is_disp else 0.0
        row = [
            cen[0], cen[1], cen[2],
            ca, sa,
            ca, sa,
            layer_norm,
            is_hole, is_fixed, is_disp, disp_amt,
            0.0,  # node_type = 0 (ply)
        ]
        if yt is not None:
            row.append(float(yt))
        rows.append(row)
        ply_eid_to_idx[eid] = len(rows) - 1

    feats = np.array(rows, dtype=np.float32)
    centroids = np.array(centroids_list, dtype=np.float32)
    return feats, ply_eid_to_idx, centroids


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def _hex_faces(nids):
    """Return the 6 faces of a C3D8R element as frozensets of 4 node IDs."""
    n = nids
    return [
        frozenset([n[0], n[1], n[2], n[3]]),  # bottom
        frozenset([n[4], n[5], n[6], n[7]]),  # top
        frozenset([n[0], n[1], n[5], n[4]]),  # front
        frozenset([n[1], n[2], n[6], n[5]]),  # right
        frozenset([n[2], n[3], n[7], n[6]]),  # back
        frozenset([n[3], n[0], n[4], n[7]]),  # left
    ]


def build_edges(parsed, ply_eid_to_idx, centroids, ply_ranges,
                through_thickness_tol=0.05):
    """Build edge_index for the ply graph.

    In-plane edges: face-sharing within each ply.
    Through-thickness: direct ply ↔ ply between adjacent plies
    (nearest in-plane centroid match); interface elements are not nodes.

    Returns
    -------
    edge_index : torch.LongTensor [2, E]
    """
    edges = []
    ply_e   = parsed["ply_elems"]

    # ── In-plane edges within each ply ──────────────────────────────────
    for pi, (first, last) in enumerate(ply_ranges):
        face_to_elems = defaultdict(list)
        for eid in range(first, last + 1):
            if eid not in ply_e:
                continue
            for face in _hex_faces(ply_e[eid]):
                face_to_elems[face].append(eid)
        for face, eids in face_to_elems.items():
            if len(eids) == 2:
                a, b = ply_eid_to_idx[eids[0]], ply_eid_to_idx[eids[1]]
                edges += [[a, b], [b, a]]

    # ── Direct ply ↔ ply through-thickness edges ────────────────────────
    # Connect each element in ply pi to its nearest in-plane neighbour
    # in ply (pi+1), for all adjacent ply pairs.
    ply_eid_to_ply_idx = {}
    for pi, (first, last) in enumerate(ply_ranges):
        for eid in range(first, last + 1):
            ply_eid_to_ply_idx[eid] = pi

    ply_centroids_2d = {eid: centroids[idx, :2]
                        for eid, idx in ply_eid_to_idx.items()}

    # Group ply elements by ply index
    ply_by_idx = defaultdict(list)
    for eid, pi in ply_eid_to_ply_idx.items():
        if eid in ply_centroids_2d:
            ply_by_idx[pi].append((eid, ply_centroids_2d[eid]))

    n_plies = parsed["n_plies"]
    for pi in range(n_plies - 1):
        elems_lo = ply_by_idx[pi]
        elems_hi = ply_by_idx[pi + 1]
        if not elems_lo or not elems_hi:
            continue
        # For each element in lower ply, find nearest in upper ply
        hi_cens = np.array([c for _, c in elems_hi])
        for eid_lo, cen_lo in elems_lo:
            dists = np.linalg.norm(hi_cens - cen_lo, axis=1)
            best_j = int(np.argmin(dists))
            if dists[best_j] < through_thickness_tol * 10:
                eid_hi = elems_hi[best_j][0]
                node_lo = ply_eid_to_idx[eid_lo]
                node_hi = ply_eid_to_idx[eid_hi]
                edges += [[node_lo, node_hi], [node_hi, node_lo]]

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # Deduplicate
    ei = torch.unique(ei, dim=1)
    return ei


# ═══════════════════════════════════════════════════════════════════════════
# ODB EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_odb(odb_path, npz_path, n_ply_elems):
    """Run the Abaqus worker on one ODB and write its fields to npz_path."""
    cmd = [
        ABAQUS_CMD, "python",
        str(WORKER_SCRIPT),
        str(odb_path),
        str(npz_path),
        str(n_ply_elems),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise RuntimeError(
            f"Abaqus worker failed for {odb_path}:\n{result.stderr}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# DATASET BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_sample(inp_path, odb_path, npz_cache, sample_params, dry_run=False,
                  sample_idx=0, sim_id=None,
                  include_damage_feature=False):
    """Build a list of PyG Data objects (one per timestep pair) for one sample.

    Parameters
    ----------
    sim_id : str or None
        Unique identifier for this simulation (e.g. 'plate_001').
        If None, falls back to ``inp_path.stem``.
    """
    if sim_id is None:
        sim_id = inp_path.stem
    parsed = parse_cohesive_inp(inp_path)
    feats, ply_eid_to_idx, centroids = build_node_features(parsed, sample_params)

    n_ply_elems = len(ply_eid_to_idx)

    edge_index = build_edges(
        parsed, ply_eid_to_idx, centroids, parsed["ply_ranges"],
    )

    if dry_run:
        # Return minimal Data without damage (for plotting geometry only)
        x = torch.from_numpy(feats)
        data = Data(x=x, edge_index=edge_index)
        data.n_ply_nodes   = n_ply_elems
        data.n_iface_nodes = 0             # interface elements are not graph nodes
        data.displacement  = parsed["displacement"]
        data.sample_idx    = sample_idx
        data.sim_id        = sim_id
        return [data]

    # Extract from ODB if not cached or cache is stale.
    # Always pass the total ply element count (from parsed .inp) to the worker.
    if npz_cache.exists():
        _cached = np.load(npz_cache)
        _ver = int(_cached["_cache_version"]) if "_cache_version" in _cached else 0
        if _ver < CACHE_VERSION:
            print(f"    Cache version {_ver} < {CACHE_VERSION}, re-extracting: "
                  f"{npz_cache.name}")
            npz_cache.unlink()
    if not npz_cache.exists():
        extract_odb(odb_path, npz_cache, len(parsed["ply_elems"]))

    raw = np.load(npz_cache)
    ply_labels   = raw["ply_labels"]
    ply_damage   = raw["ply_damage"]    # [T, P]  SDV_FI_1_Matrix
    frame_times  = raw["frame_times"]   # [T]

    # SDV_DMG_1 (matrix damage): present from cache version 2 on
    has_matrix_dmg = "ply_matrix_dmg" in raw
    ply_matrix_dmg = raw["ply_matrix_dmg"] if has_matrix_dmg else None

    if ply_damage.size == 0:
        raise ValueError(
            f"ply_damage array is empty for {odb_path}: the worker "
            f"found no ply elements. Delete cached NPZ and re-run."
        )
    if float(ply_damage.max()) == 0.0:
        raise ValueError(
            f"Ply FI is all zero in {odb_path}: SDV_FI_1_Matrix is zero across "
            f"all frames. Simulation may not have reached failure."
        )

    # Build label → node_index maps from what we have in the graph
    ply_label_to_node   = {lab: ply_eid_to_idx[lab]
                            for lab in ply_labels if lab in ply_eid_to_idx}

    N = feats.shape[0]
    T = len(frame_times)
    data_list = []

    # Precompute edge attributes (4-dim: dx, dy, dz, dist)
    pos = torch.from_numpy(centroids)
    u_i = pos[edge_index[0]]
    u_j = pos[edge_index[1]]
    u_ij = u_i - u_j
    dist = torch.norm(u_ij, p=2, dim=1, keepdim=True)
    edge_attr = torch.cat([u_ij, dist], dim=-1).float()

    feat_static = torch.from_numpy(feats)

    for t in range(T - 1):
        # FI state (primary target channel)
        fi_curr = np.zeros(N, dtype=np.float32)
        fi_next = np.zeros(N, dtype=np.float32)

        for k, lab in enumerate(ply_labels):
            if lab in ply_label_to_node:
                ni = ply_label_to_node[lab]
                fi_curr[ni] = ply_damage[t, k]
                fi_next[ni] = ply_damage[t + 1, k]

        t_now  = float(frame_times[t])
        t_next = float(frame_times[t + 1])
        dt = t_next - t_now
        if dt <= 0:
            continue

        fi_now_t  = torch.from_numpy(fi_curr).unsqueeze(1)   # [N, 1]
        fi_next_t = torch.from_numpy(fi_next).unsqueeze(1)

        # Clamp FI at FI_CAP: values beyond this are physically meaningless
        # (fully degraded element). Clamping before rate computation ensures
        # dFI/dt → 0 once an element reaches the cap.
        fi_now_t  = fi_now_t.clamp(max=FI_CAP)
        fi_next_t = fi_next_t.clamp(max=FI_CAP)

        rate = (fi_next_t - fi_now_t) / dt                   # [N, 1]
        rate = rate.clamp(min=0.0)  # non-negative FI rates only (monotonic loading)

        # Two-stage quantities; computed here (before x) because the current
        # matrix damage optionally enters the feature vector. Only present if
        # the NPZ carries SDV_DMG_1.
        mdmg_now_t = mdmg_next_t = None
        if has_matrix_dmg:
            mdmg_curr = np.zeros(N, dtype=np.float32)
            mdmg_next = np.zeros(N, dtype=np.float32)
            for k, lab in enumerate(ply_labels):
                if lab in ply_label_to_node:
                    ni = ply_label_to_node[lab]
                    mdmg_curr[ni] = ply_matrix_dmg[t, k]
                    mdmg_next[ni] = ply_matrix_dmg[t + 1, k]
            mdmg_now_t  = torch.from_numpy(mdmg_curr).unsqueeze(1)
            mdmg_next_t = torch.from_numpy(mdmg_next).unsqueeze(1)

        # Features: static + FI_now; with include_damage_feature,
        # static + D_now + FI_now.
        # D is inserted BEFORE FI: the model reads x[:, -1] as the FI state
        # (two-stage gate, optional FI log transform), so FI must stay last.
        if include_damage_feature:
            if mdmg_now_t is None:
                raise RuntimeError(
                    f"--include-damage-feature requires SDV_DMG_1 in "
                    f"{npz_cache.name} (matrix damage missing)."
                )
            x = torch.cat([feat_static, mdmg_now_t, fi_now_t], dim=1)
        else:
            x = torch.cat([feat_static, fi_now_t], dim=1)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=rate,
            mesh_pos=pos,
            delta_t=torch.tensor([dt], dtype=torch.float),
            current_time=torch.tensor([t_now], dtype=torch.float),
            next_time=torch.tensor([t_next], dtype=torch.float),
            sample_idx=torch.tensor([sample_idx], dtype=torch.long),
        )
        data.n_ply_nodes   = n_ply_elems
        data.n_iface_nodes = 0             # interface elements are not graph nodes
        data.displacement  = parsed["displacement"]
        data.sim_id        = sim_id
        # Read by the model's loss masking: False keeps the ply-only FI-rate mask.
        data.iface_trainable = False

        # Two-stage target: matrix damage rate (tensors hoisted above x)
        if has_matrix_dmg:
            dmg_rate = ((mdmg_next_t - mdmg_now_t) / dt).clamp(min=0.0)
            data.y_dmg   = dmg_rate      # [N, 1] matrix damage rate
            data.dmg_now = mdmg_now_t     # [N, 1] current matrix damage

        data_list.append(data)

    return data_list


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_dataset(study_dir, all_data_list, parsed_list, out_dir,
                 sample_params_list=None):
    """Generate diagnostic plots for the cohesive laminate dataset."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Graph geometry: node positions coloured by type ───────────────
    if parsed_list:
        parsed = parsed_list[0]
        sp0 = sample_params_list[0] if sample_params_list else {}
        feats, ply_eid_to_idx, centroids = build_node_features(parsed, sp0)
        node_type = feats[:, -1]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, view, zi in zip(axes, ["Top view (z=mid)", "Side view (y=0)"],
                                 [None, 1]):
            ply_mask   = node_type == 0
            if zi is None:
                ax.scatter(centroids[ply_mask, 0],   centroids[ply_mask, 1],
                           s=4, c="steelblue", label="Ply C3D8R", alpha=0.6)
                ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
            else:
                ax.scatter(centroids[ply_mask, 0],   centroids[ply_mask, 2],
                           s=4, c="steelblue", label="Ply C3D8R", alpha=0.6)
                ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
            ax.set_title(view); ax.legend(markerscale=3)
            ax.set_aspect("equal")
        plt.suptitle("Node positions, sample 0", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_dir / "01_node_positions.png", dpi=150)
        plt.close()
        print(f"  Saved: {out_dir / '01_node_positions.png'}")

    # ── 2. Node feature distributions ───────────────────────────────────
    if all_data_list and all_data_list[0]:
        data0 = all_data_list[0][0]
        x = data0.x.numpy()
        feat_names = ["x", "y", "z",
                      "cos_angle_1", "sin_angle_1",
                      "cos_angle_2", "sin_angle_2",
                      "layer_norm", "is_hole", "is_fixed",
                      "is_disp", "disp_amt", "node_type"]
        fig, axes = plt.subplots(4, 4, figsize=(16, 13))
        for k, (ax, name) in enumerate(zip(axes.flat, feat_names)):
            ax.hist(x[:, k], bins=40, color="steelblue", edgecolor="none",
                    alpha=0.8)
            ax.set_title(name, fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in axes.flat[len(feat_names):]:
            ax.axis("off")
        plt.suptitle("Node feature distributions, sample 0, t=0", fontsize=11)
        plt.tight_layout()
        plt.savefig(out_dir / "02_feature_distributions.png", dpi=150)
        plt.close()
        print(f"  Saved: {out_dir / '02_feature_distributions.png'}")

    # ── 3. Damage evolution over time (per sample) ───────────────────────
    if all_data_list and len(all_data_list[0]) > 1:
        n_cols = 1
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5), squeeze=False)
        colors = cm.viridis(np.linspace(0, 1, min(len(all_data_list), 5)))
        for si, (data_list, color) in enumerate(
                zip(all_data_list[:5], colors)):
            times = [d.current_time.item() for d in data_list]
            # Current FI/damage is the last column of x (appended during build)
            ply_max = [d.x[:d.n_ply_nodes, -1].max().item() for d in data_list]
            axes[0, 0].plot(times, ply_max, color=color, label=f"s{si}")
        axes[0, 0].set_title("Ply FI (SDV_FI_1_Matrix), max over elements")
        for ax in axes.flat:
            ax.set_xlabel("Load time"); ax.set_ylabel("Value [-]")
            ax.legend(fontsize=7)
        plt.suptitle("FI/damage evolution, first 5 samples", fontsize=11)
        plt.tight_layout()
        plt.savefig(out_dir / "03_damage_evolution.png", dpi=150)
        plt.close()
        print(f"  Saved: {out_dir / '03_damage_evolution.png'}")

    # ── 4. Spatial damage map at last timestep (sample 0) ───────────────
    if all_data_list and all_data_list[0]:
        data_last = all_data_list[0][-1]
        if hasattr(data_last, "y") and data_last.y is not None and parsed_list:
            sp0 = sample_params_list[0] if sample_params_list else {}
            feats, ply_eid_to_idx, centroids = build_node_features(parsed_list[0], sp0)
            # Current FI/damage is last column of x
            y = data_last.x[:, -1].numpy()
            node_type = feats[:, -1]

            DAMAGE_THRESH = 1e-4  # nodes below this are shown grey (undamaged)
            plot_items = [
                (node_type == 0, "Ply FI (SDV_FI_1_Matrix)"),
            ]
            fig, axes = plt.subplots(1, len(plot_items),
                                     figsize=(7 * len(plot_items), 6),
                                     squeeze=False)
            for ax, (mask, title) in zip(axes.flat, plot_items):
                if not mask.any():
                    continue
                vals = y[mask]
                pos_m = centroids[mask]
                # Background: undamaged nodes in light grey
                undmg = vals < DAMAGE_THRESH
                ax.scatter(pos_m[undmg, 0], pos_m[undmg, 1],
                           c="lightgrey", s=6, alpha=0.5, zorder=1)
                # Foreground: damaged nodes with adaptive colorbar
                dmg = ~undmg
                if dmg.any():
                    vmin = vals[dmg].min()
                    sc = ax.scatter(
                        pos_m[dmg, 0], pos_m[dmg, 1],
                        c=vals[dmg], cmap="hot_r", vmin=vmin, vmax=1,
                        s=8, alpha=0.95, zorder=2,
                    )
                    plt.colorbar(sc, ax=ax)
                    ax.set_title(f"{title}\n{dmg.sum()} nodes > {DAMAGE_THRESH:.0e}", fontsize=9)
                else:
                    ax.set_title(f"{title}\n(all zero at last frame)", fontsize=9)
                ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
                ax.set_aspect("equal")
            plt.suptitle("Damage spatial map, sample 0, last frame", fontsize=11)
            plt.tight_layout()
            plt.savefig(out_dir / "04_spatial_damage_map.png", dpi=150)
            plt.close()
            print(f"  Saved: {out_dir / '04_spatial_damage_map.png'}")

    # ── 5. Dataset statistics ─────────────────────────────────────────────
    if all_data_list:
        n_samples  = len(all_data_list)
        n_timesteps = [len(dl) for dl in all_data_list]
        max_ply_dmg   = []
        for dl in all_data_list:
            if dl and hasattr(dl[-1], "y") and dl[-1].y is not None:
                last = dl[-1]
                n_p = last.n_ply_nodes
                max_ply_dmg.append(last.x[:n_p, -1].max().item())

        n_cols = 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
        axes[0].hist(n_timesteps, bins=20, color="steelblue", edgecolor="k")
        axes[0].set_title("Timesteps per sample"); axes[0].set_xlabel("N timesteps")
        if max_ply_dmg:
            axes[1].hist(max_ply_dmg, bins=20, color="tomato", edgecolor="k")
            axes[1].set_title("Max ply FI at end-of-sim")
            axes[1].set_xlabel("FI max [-]")
        plt.suptitle(f"Dataset statistics, {n_samples} samples", fontsize=11)
        plt.tight_layout()
        plt.savefig(out_dir / "05_dataset_statistics.png", dpi=150)
        plt.close()
        print(f"  Saved: {out_dir / '05_dataset_statistics.png'}")

    # ── 6. Per-ply damage profiles ────────────────────────────────────────
    if parsed_list and all_data_list and all_data_list[0]:
        parsed = parsed_list[0]
        n_plies = parsed["n_plies"]
        angles  = parsed["ply_angles"]
        sp0 = sample_params_list[0] if sample_params_list else {}
        feats, ply_eid_to_idx, centroids = build_node_features(parsed, sp0)
        node_type = feats[:, -1]

        fig, axes = plt.subplots(1, n_plies, figsize=(5 * n_plies, 5),
                                 squeeze=False)
        data_last = all_data_list[0][-1] if all_data_list[0] else None
        DAMAGE_THRESH = 1e-4
        if data_last and hasattr(data_last, "x") and data_last.x is not None:
            y = data_last.x[:, -1].numpy()  # current damage = last col of x
            # Identify ply nodes per ply using ply_ranges (not layer_norm formula,
            # which is z-based and can't be inverted back to ply index)
            ply_eid_to_idx_plot = {}
            for _pi, (_first, _last) in enumerate(parsed["ply_ranges"]):
                for _eid in range(_first, _last + 1):
                    ply_eid_to_idx_plot[_eid] = _pi
            # Build per-ply node mask from node positions in ply_eid_to_idx
            for pi in range(n_plies):
                node_indices_pi = [idx for eid, idx in
                                   ply_eid_to_idx.items()
                                   if ply_eid_to_idx_plot.get(eid) == pi]
                mask = np.zeros(len(node_type), dtype=bool)
                if node_indices_pi:
                    mask[node_indices_pi] = True
                ax = axes[0, pi]
                if mask.sum() > 0:
                    vals = y[mask]
                    pos_m = centroids[mask]
                    undmg = vals < DAMAGE_THRESH
                    ax.scatter(pos_m[undmg, 0], pos_m[undmg, 1],
                               c="lightgrey", s=8, alpha=0.5, zorder=1)
                    dmg = ~undmg
                    if dmg.any():
                        vmin = vals[dmg].min()
                        sc = ax.scatter(
                            pos_m[dmg, 0], pos_m[dmg, 1],
                            c=vals[dmg], cmap="hot_r", vmin=vmin, vmax=1,
                            s=10, alpha=0.95, zorder=2,
                        )
                        plt.colorbar(sc, ax=ax)
                angle = angles.get(pi, "?")
                ax.set_title(f"Ply {pi} ({angle}°)", fontsize=9)
                ax.set_xlabel("x"); ax.set_ylabel("y")
                ax.set_aspect("equal")
        plt.suptitle("SDV_DMG_1 per ply, sample 0, last frame", fontsize=11)
        plt.tight_layout()
        plt.savefig(out_dir / "06_per_ply_damage.png", dpi=150)
        plt.close()
        print(f"  Saved: {out_dir / '06_per_ply_damage.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# FAIL-FAST VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

_REQUIRED_ATTRS = ['sample_idx', 'sim_id', 'mesh_pos', 'n_ply_nodes',
                   'current_time', 'next_time', 'delta_t', 'y']


def _validate_dataset(flat_list):
    """Validate every Data object has required metadata. Raise on failure."""
    from collections import defaultdict

    print("\n  Validating dataset integrity ...")
    errors = []

    # 1. Required attributes must be present and non-None
    for i, d in enumerate(flat_list):
        for attr in _REQUIRED_ATTRS:
            val = getattr(d, attr, None)
            if val is None:
                errors.append(
                    f"sample {i}: missing required attribute '{attr}'"
                )

    # 2. sample_idx must be set and consistent across timesteps
    sim_groups = defaultdict(list)
    for i, d in enumerate(flat_list):
        sid = getattr(d, 'sample_idx', None)
        if sid is not None:
            sim_groups[int(sid.item())].append(i)

    if len(sim_groups) == 0:
        errors.append("No valid sample_idx found in any Data object")

    # 3. Per-simulation: mesh_pos must be consistent within a sim,
    #    and different sims must not silently reuse identical geometry
    #    without acknowledgement
    sim_node_counts = {}
    for sid, indices in sim_groups.items():
        ref_pos = flat_list[indices[0]].mesh_pos
        sim_node_counts[sid] = ref_pos.shape[0]
        for idx in indices[1:]:
            other_pos = flat_list[idx].mesh_pos
            if not torch.equal(ref_pos, other_pos):
                errors.append(
                    f"sim {sid}: mesh_pos changed between timesteps "
                    f"(index {indices[0]} vs {idx})"
                )
                break

    # 4. sim_id must be a non-empty string
    for i, d in enumerate(flat_list):
        sim_id = getattr(d, 'sim_id', None)
        if sim_id is not None and (not isinstance(sim_id, str) or len(sim_id) == 0):
            errors.append(f"sample {i}: sim_id must be a non-empty string, got {sim_id!r}")

    # 5. delta_t must be positive
    for i, d in enumerate(flat_list):
        dt = getattr(d, 'delta_t', None)
        if dt is not None and dt.item() <= 0:
            errors.append(f"sample {i}: delta_t <= 0 ({dt.item():.6e})")

    if errors:
        msg = (f"DATASET VALIDATION FAILED: {len(errors)} error(s):\n"
               + "\n".join(f"  - {e}" for e in errors[:20]))
        if len(errors) > 20:
            msg += f"\n  ... and {len(errors) - 20} more"
        raise ValueError(msg)

    # Summary
    n_sims = len(sim_groups)
    n_transitions = len(flat_list)
    node_counts = sorted(set(sim_node_counts.values()))
    print(f"  PASSED: {n_sims} simulation(s), {n_transitions} transitions")
    print(f"  Node counts: {node_counts}")
    sim_ids = set(getattr(d, 'sim_id', None) for d in flat_list)
    print(f"  sim_ids: {sorted(s for s in sim_ids if s is not None)[:10]}"
          f"{'...' if len(sim_ids) > 10 else ''}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def _build_one_worker(task):
    """Top-level (picklable) worker: build ONE simulation and write its result to
    a temp .pt, returning the PATH (not the tensors). Passing torch tensors back
    through the Pool would share them via file descriptors and hit the OS open-fd
    limit ('Too many open files'); writing to disk and returning a path avoids it.
    Returns (sample_idx, name, tmp_path, n_nodes, n_steps, error_or_None)."""
    i, inp_path, odb_path, npz_cache, sample_params, flags, tmpdir = task
    name = inp_path.stem
    try:
        data_list = build_sample(
            inp_path, odb_path, npz_cache, sample_params,
            dry_run=flags["dry_run"], sample_idx=i, sim_id=name,
            include_damage_feature=flags["include_damage_feature"],
        )
        parsed = parse_cohesive_inp(inp_path)
        tmp_path = os.path.join(tmpdir, f"_build_{i:05d}.pt")
        torch.save({"data_list": data_list, "parsed": parsed,
                    "sample_params": sample_params}, tmp_path)
        n_nodes = data_list[0].x.shape[0] if data_list else 0
        return (i, name, tmp_path, n_nodes, len(data_list), None)
    except Exception as e:  # noqa: BLE001
        return (i, name, None, 0, 0, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Build cohesive laminate GNN dataset (SDV_DMG_1 + SDEG)."
    )
    parser.add_argument("--study", required=True,
                        help="Study name; matches data/raw/<study>/geometry/ "
                             "and data/raw/<study>/simulations/ layout.")
    parser.add_argument("--out", default=None,
                        help="Output .pt filename (default: <study>_dataset.pt).")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse .inp only, skip ODB extraction.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for quick tests).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker processes for per-sim construction "
                             "(default 8; set 1 for sequential).")
    parser.add_argument("--resample-dt", type=float, default=0.02,
                        help="Resample to uniform dt after building "
                             "(default: 0.02). Set to 0 to disable.")
    parser.add_argument("--no-resample", action="store_true",
                        help="Disable automatic uniform-dt resampling.")
    parser.add_argument("--include-yt-feature", action="store_true",
                        default=False,
                        help="Broadcast the per-sample transverse tensile "
                             "strength YT to every node of its sample, as the "
                             "last static column (x gains one dim; the damage "
                             "and failure-index columns stay last). Requires a "
                             "study generated with --yt-range.")
    parser.add_argument("--include-damage-feature", action="store_true",
                        default=False,
                        help="Append the current matrix damage D_now as an "
                             "input feature (x becomes 15-dim: [static, "
                             "D_now, FI_now]; FI stays last). Output name "
                             "gets a _dmgstate suffix unless --out is given. "
                             "Requires SDV_DMG_1 in the NPZ caches.")
    args = parser.parse_args()

    study = args.study
    # Match run_pipeline.py layout: data/raw/<study>/geometry/ and simulations/
    geom_dir = RAW_DIR / study / "geometry"
    sim_dir  = RAW_DIR / study / "simulations"
    # Distinct default name for the 15-feature variant so the canonical
    # 14-feature datasets are never clobbered.
    _default_name = (f"{study}_dmgstate_dataset.pt"
                     if args.include_damage_feature
                     else f"{study}_dataset.pt")
    out_path_name = args.out or _default_name
    out_dir  = PROCESSED_DIR / study
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_path_name
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    # Load per-sample geometry params (hole center/radius, displacement, plate dims)
    meta_path = PROCESSED_DIR / study / "plate_geometry_data.pt"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Geometry metadata not found: {meta_path}\n"
            f"Run generate_cohesive_laminate_uniform.py first."
        )
    meta = torch.load(meta_path, weights_only=False)
    param_names = meta["param_names"]
    params_arr  = meta["params"]  # list of lists or array, shape [N, n_params]
    # Optional multi-hole metadata (extrapolation studies): per-sample list
    # of (cx, cy, r).  Absent on single-hole studies.
    holes_meta  = meta.get("holes", None)
    # Per-sample transverse tensile strength, written by the generator's
    # --yt-range option and required by --include-yt-feature
    yt_meta = meta.get("yt", None)
    if args.include_yt_feature and yt_meta is None:
        raise SystemExit(
            f"--include-yt-feature was given but {meta_path} carries no 'yt' "
            f"tensor.\nThat study was generated without --yt-range, so there is "
            f"no per-sample strength to broadcast."
        )

    # Find all .inp files. Accept both naming conventions:
    #   plate_XXX.inp                 (generator --study mode)
    #   cohesive_lam_plate_XXX.inp    (generator --out-dir mode)
    inp_files = sorted(
        set(geom_dir.glob("plate_[0-9]*.inp"))
        | set(geom_dir.glob("cohesive_lam_plate_[0-9]*.inp"))
    )
    if args.max_samples:
        inp_files = inp_files[:args.max_samples]

    print("=" * 65)
    print("COHESIVE LAMINATE DATASET BUILD")
    print("=" * 65)
    print(f"  Study        : {study}")
    print(f"  Geometry dir : {geom_dir}")
    print(f"  Simulation dir: {sim_dir}")
    print(f"  Samples      : {len(inp_files)}")
    print("  Interfaces   : EXCLUDED (ply-only)")
    # Width is 13 static + FI_now, plus D_now and plus YT when each is asked for.
    # Reported rather than hardcoded: an earlier version always printed "15-dim"
    # and silently under-reported the YT build by one column.
    _xdim = 14 + int(args.include_damage_feature) + int(args.include_yt_feature)
    print(f"  D-now feature: {'ON' if args.include_damage_feature else 'off'}")
    print(f"  YT feature   : {'ON (broadcast per sample)' if args.include_yt_feature else 'off'}")
    print(f"  Node features: {_xdim}-dim x")
    print(f"  Dry run      : {args.dry_run}")
    print("=" * 65)

    all_data_list      = []
    parsed_list        = []
    sample_params_list = []
    failed = []

    # Build the per-sim task list (each is independent -> embarrassingly parallel).
    # Workers write results to temp .pt files on the (big) processed disk and
    # return paths, so no torch tensors cross the Pool (avoids the open-fd limit).
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp(prefix="_build_tmp_", dir=str(PROCESSED_DIR / study))
    flags = dict(dry_run=args.dry_run,
                 include_damage_feature=args.include_damage_feature)
    tasks = []
    for i, inp_path in enumerate(inp_files):
        sample_name = inp_path.stem
        odb_path  = sim_dir / sample_name / f"{sample_name}.odb"
        npz_cache = INTERIM_DIR / f"{study}_{sample_name}.npz"
        sample_params = dict(zip(param_names, params_arr[i]))
        # --include-yt-feature only: hand this sample's transverse tensile
        # strength to build_node_features, which broadcasts it to every node.
        # The key is absent otherwise, so the feature width is unchanged.
        if args.include_yt_feature:
            sample_params["yt"] = float(yt_meta[i])
        if holes_meta is not None:
            sample_params["holes"] = holes_meta[i]
        tasks.append((i, inp_path, odb_path, npz_cache, sample_params, flags, tmpdir))

    nworkers = max(1, min(args.workers, len(tasks)))
    paths = {}    # sample_idx -> (tmp_path, sample_params)
    print(f"  Building {len(tasks)} sims with {nworkers} worker(s) "
          f"(first line appears after one sim finishes, ~25s) ...", flush=True)
    t0 = time.time()
    if nworkers > 1:
        import multiprocessing as _mp
        with _mp.get_context("fork").Pool(nworkers) as pool:
            for done, r in enumerate(pool.imap_unordered(_build_one_worker, tasks), 1):
                i, name, tmp_path, n_nodes, n_steps, err = r
                if err:
                    print(f"  [{done:3d}/{len(tasks)}]  {name}  FAILED: {err}", flush=True)
                    failed.append(name)
                else:
                    paths[i] = (tmp_path, tasks[i][4])
                    print(f"  [{done:3d}/{len(tasks)}]  {name}  nodes={n_nodes}  "
                          f"steps={n_steps}", flush=True)
    else:
        for task in tasks:
            i, name, tmp_path, n_nodes, n_steps, err = _build_one_worker(task)
            if err:
                print(f"  [{i+1:3d}/{len(tasks)}]  {name}  FAILED: {err}", flush=True)
                failed.append(name)
            else:
                paths[i] = (tmp_path, task[4])
                print(f"  [{i+1:3d}/{len(tasks)}]  {name}  nodes={n_nodes}  "
                      f"steps={n_steps}", flush=True)
    print(f"  Build (construction): {time.time()-t0:.0f}s; loading results ...", flush=True)

    # Load each sim's temp file in order, then clean up the temp dir.
    for i in sorted(paths):
        tmp_path, sp = paths[i]
        blob = torch.load(tmp_path, weights_only=False)
        all_data_list.append(blob["data_list"])
        parsed_list.append(blob["parsed"])
        sample_params_list.append(sp)
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Flatten into a single list of Data objects
    flat_list = [d for dl in all_data_list for d in dl]
    print(f"\n  Total Data objects: {len(flat_list)}")
    print(f"  Failed samples   : {len(failed)}")

    # ══════════════════════════════════════════════════════════════════
    # FAIL-FAST VALIDATION: abort on missing/inconsistent metadata
    # ══════════════════════════════════════════════════════════════════
    if flat_list:
        _validate_dataset(flat_list)

    if flat_list:
        torch.save(flat_list, out_path)
        print(f"  Saved raw dataset: {out_path}")

        # Resample to uniform dt (on by default, dt=0.02)
        do_resample = (not args.no_resample) and args.resample_dt > 0
        if do_resample:
            from src.preprocessing.resample_uniform_dt import resample_dataset  # noqa
            print(f"\n  Resampling to uniform dt={args.resample_dt} ...")
            resample_dataset(input_path=out_path, dt=args.resample_dt,
                             dmg_in_features=args.include_damage_feature)

    if args.plot:
        print("\n  Generating plots...")
        plot_dir = out_dir / "plots"
        plot_dataset(geom_dir, all_data_list, parsed_list, plot_dir,
                     sample_params_list)

    if failed:
        print(f"\n  Failed: {failed}")

    print("=" * 65)


if __name__ == "__main__":
    main()
