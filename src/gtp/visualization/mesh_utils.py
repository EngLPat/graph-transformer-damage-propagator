"""
Mesh utilities for loading hex element connectivity from ABAQUS .inp files
and extracting quad faces for visualization.

The simulation mesh consists of C3D8R (8-node reduced-integration brick)
elements.  For 2D visualization we extract the bottom quad faces (nodes
0-3 of each hex); for 3D visualization we extract all exterior faces.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# ── .inp file parsing ─────────────────────────────────────────────────────

def parse_inp_nodes_and_elements(filepath: str) -> Dict:
    """
    Parse an ABAQUS .inp file for node positions and element connectivity.

    Args:
        filepath: Path to the .inp file

    Returns:
        Dictionary with:
          'nodes':    {node_id: (x, y, z)}, 1-based ABAQUS IDs
          'elements': {elem_id: [n1, n2, ..., n8]}, 1-based ABAQUS IDs
    """
    filepath = Path(filepath)
    nodes = {}
    elements = {}
    section = None

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("**"):
                continue
            if stripped.startswith("*"):
                if stripped.startswith("*Node") and "Output" not in stripped:
                    section = "nodes"
                elif stripped.startswith("*Element"):
                    section = "elements"
                else:
                    section = None
                continue
            if section == "nodes":
                parts = stripped.split(",")
                nid = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                z = float(parts[3]) if len(parts) > 3 else 0.0
                nodes[nid] = (x, y, z)
            elif section == "elements":
                parts = [p.strip() for p in stripped.split(",") if p.strip()]
                try:
                    eid = int(parts[0])
                    enodes = [int(p) for p in parts[1:]]
                    elements[eid] = enodes
                except ValueError:
                    section = None

    return {"nodes": nodes, "elements": elements}


def _find_inp_for_positions(
    pos_3d: np.ndarray,
    geometry_dir: str = None,
) -> Optional[str]:
    """
    Find the .inp file whose node positions best match *pos_3d*.

    Matching is by node count first, then by a quick centroid comparison.

    Args:
        pos_3d: [N, 3] mesh positions from the Data object
        geometry_dir: Directory containing plate_*.inp files.
                      Defaults to data/raw/geometry/ relative to project root.

    Returns:
        Path to the matching .inp file, or None.
    """
    if geometry_dir is None:
        # Try to resolve relative to this file's location
        project_root = Path(__file__).resolve().parents[3]  # src/gtp/visualization -> project root
        geometry_dir = project_root / "data" / "raw" / "geometry"

    geometry_dir = Path(geometry_dir)
    if not geometry_dir.exists():
        return None

    inp_files = sorted(geometry_dir.glob("plate_*.inp"))
    if not inp_files:
        return None

    target_n = pos_3d.shape[0]
    target_centroid = pos_3d.mean(axis=0)

    best_file = None
    best_dist = float("inf")

    for inp_file in inp_files:
        data = parse_inp_nodes_and_elements(str(inp_file))
        n_nodes = len(data["nodes"])
        if n_nodes != target_n:
            continue
        # Centroid match
        coords = np.array(list(data["nodes"].values()))
        centroid = coords.mean(axis=0)
        dist = np.linalg.norm(centroid - target_centroid)
        if dist < best_dist:
            best_dist = dist
            best_file = str(inp_file)

    return best_file


# ── Quad face extraction ──────────────────────────────────────────────────

# ABAQUS C3D8R node ordering (0-based local index):
#   Bottom face: 0, 1, 2, 3   (Z_min)
#   Top face:    4, 5, 6, 7   (Z_max)
#   Front:  0, 1, 5, 4
#   Back:   3, 2, 6, 7
#   Left:   0, 3, 7, 4
#   Right:  1, 2, 6, 5

_HEX_BOTTOM = [0, 1, 2, 3]
_HEX_TOP = [4, 5, 6, 7]
_HEX_ALL_FACES = [
    [0, 1, 2, 3],  # bottom
    [4, 5, 6, 7],  # top
    [0, 1, 5, 4],  # front
    [3, 2, 6, 7],  # back
    [0, 3, 7, 4],  # left
    [1, 2, 6, 5],  # right
]


def extract_quad_faces(
    inp_path: str,
    face: str = "bottom",
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Extract quad face connectivity from hex elements.

    Args:
        inp_path: Path to .inp file
        face: Which face(s) to extract:
              'bottom': Z_min face (nodes 0-3), for 2D XY plots
              'top':    Z_max face (nodes 4-7)
              'all':    all 6 faces (for 3D rendering)

    Returns:
        Tuple of:
          positions: [N, 3] node positions (0-based index)
          quads:     [Q, 4] quad connectivity (0-based node indices)
          id_map:    {abaqus_id: 0-based_index}
    """
    data = parse_inp_nodes_and_elements(inp_path)
    nodes = data["nodes"]
    elements = data["elements"]

    # Build 0-based index mapping
    sorted_ids = sorted(nodes.keys())
    id_map = {nid: idx for idx, nid in enumerate(sorted_ids)}
    positions = np.array([nodes[nid] for nid in sorted_ids], dtype=np.float32)

    # Select face template(s)
    if face == "bottom":
        templates = [_HEX_BOTTOM]
    elif face == "top":
        templates = [_HEX_TOP]
    elif face == "all":
        templates = _HEX_ALL_FACES
    else:
        raise ValueError(f"Unknown face type: {face!r}")

    quads = []
    for enodes in elements.values():
        # enodes is list of ABAQUS node IDs (1-based)
        idx = [id_map[nid] for nid in enodes]
        for tmpl in templates:
            quads.append([idx[t] for t in tmpl])

    quads = np.array(quads, dtype=np.int64)
    return positions, quads, id_map


def load_quad_faces_for_data(
    pos_3d: np.ndarray,
    geometry_dir: str = None,
    face: str = "bottom",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    High-level function: given mesh positions, find the .inp file and
    extract quad faces.

    Args:
        pos_3d: [N, 3] node positions from the Data object
        geometry_dir: Directory with plate_*.inp files (auto-detected)
        face: 'bottom', 'top', or 'all'

    Returns:
        Tuple of (positions [N,3], quads [Q,4]) or None if not found.
    """
    inp_path = _find_inp_for_positions(pos_3d, geometry_dir)
    if inp_path is None:
        return None

    positions, quads, id_map = extract_quad_faces(inp_path, face=face)

    # Verify that positions match (within tolerance)
    if positions.shape[0] != pos_3d.shape[0]:
        return None

    return positions, quads


def compute_reindex_map(
    inp_positions: np.ndarray,
    data_positions: np.ndarray,
    tol: float = 1e-3,
) -> Optional[np.ndarray]:
    """
    Compute mapping from .inp node indices to Data node indices.

    The .inp file and the PyG Data object may have different node orderings.
    This function finds the correspondence by matching coordinates.

    Args:
        inp_positions: [N, 3] from .inp file (sorted by ABAQUS ID)
        data_positions: [N, 3] from Data object

    Returns:
        reindex: [N] array where reindex[inp_idx] = data_idx,
                 or None if matching fails.
    """
    from scipy.spatial import cKDTree

    if inp_positions.shape != data_positions.shape:
        return None

    tree = cKDTree(data_positions)
    dists, indices = tree.query(inp_positions)

    if np.any(dists > tol):
        return None

    return indices


def get_visualization_quads(
    pos_3d: np.ndarray,
    geometry_dir: str = None,
    face: str = "bottom",
) -> Optional[np.ndarray]:
    """
    Get quad face connectivity ready for matplotlib PolyCollection.

    This is the main entry point for visualization code.  It returns
    quad faces with indices that correspond to *pos_3d* (the Data
    object's node ordering).

    Args:
        pos_3d: [N, 3] node positions from the Data object
        geometry_dir: Directory with .inp files
        face: 'bottom', 'top', or 'all'

    Returns:
        quads: [Q, 4] connectivity (0-based, in Data ordering),
               or None if .inp file not found.
    """
    result = load_quad_faces_for_data(pos_3d, geometry_dir, face=face)
    if result is None:
        return None

    inp_positions, quads = result

    # Check if node orderings match
    if np.allclose(inp_positions, pos_3d, atol=1e-3):
        return quads

    # Need to re-index
    reindex = compute_reindex_map(inp_positions, pos_3d)
    if reindex is None:
        return None

    # Re-map quad indices from inp ordering to data ordering
    return reindex[quads]


# Used by plotting._build_element_hex_mesh (test-time field plots and animations).
def parse_inp_elements_and_plies(filepath):
    """Parse .inp for nodes, elements, ply elsets, and fiber angles.

    Returns
    -------
    dict with keys:
        nodes       : dict {nid: (x,y,z)}
        elements    : dict {eid: [n1,...,n8]}
        ply_elsets  : list of (ply_name, elem_ids, fiber_angle_deg)
    """
    filepath = Path(filepath)
    nodes = {}
    elements = {}
    ply_elsets = []      # [(name, set(eids), angle)]
    orientations = {}    # {name: angle_deg}

    with open(filepath, "r") as f:
        section = None
        current_elset_name = None
        current_elset_ids = []
        current_elset_generate = False
        node_section_seen = False

        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("**"):
                continue

            if stripped.startswith("*"):
                # Flush any pending elset
                if section == "elset" and current_elset_name:
                    if current_elset_name.startswith("PLY-"):
                        if current_elset_generate and len(current_elset_ids) == 3:
                            start, end, step = current_elset_ids
                            ids = list(range(start, end + 1, step))
                        else:
                            ids = current_elset_ids
                        ply_elsets.append((current_elset_name, ids))
                    current_elset_name = None
                    current_elset_ids = []
                    current_elset_generate = False

                if stripped.startswith("*Node") and "Output" not in stripped:
                    if not node_section_seen:
                        section = "nodes"
                        node_section_seen = True
                    else:
                        section = None
                elif stripped.startswith("*Element, type="):
                    section = "elements"
                elif stripped.startswith("*Elset"):
                    # Parse elset header
                    parts_kv = {}
                    for part in stripped.split(","):
                        part = part.strip()
                        if "=" in part:
                            k, v = part.split("=", 1)
                            parts_kv[k.strip().lower()] = v.strip()
                    name = parts_kv.get("elset", "")
                    if name.startswith("PLY-"):
                        section = "elset"
                        current_elset_name = name
                        current_elset_ids = []
                        current_elset_generate = "generate" in stripped.lower()
                    else:
                        section = None
                elif stripped.startswith("*Orientation"):
                    # e.g. *Orientation, name=Ori-0
                    parts_kv = {}
                    for part in stripped.split(","):
                        part = part.strip()
                        if "=" in part:
                            k, v = part.split("=", 1)
                            parts_kv[k.strip().lower()] = v.strip()
                    ori_name = parts_kv.get("name", "")
                    section = "orientation"
                    _ori_line_count = 0
                    current_elset_name = ori_name  # reuse for temp storage
                elif stripped.startswith("*Solid Section"):
                    # Link elset to orientation
                    parts_kv = {}
                    for part in stripped.split(","):
                        part = part.strip()
                        if "=" in part:
                            k, v = part.split("=", 1)
                            parts_kv[k.strip().lower()] = v.strip()
                    elset_name = parts_kv.get("elset", "")
                    ori_name = parts_kv.get("orientation", "")
                    if elset_name.startswith("PLY-") and ori_name in orientations:
                        # Find the matching ply elset and attach the angle
                        for ii, item in enumerate(ply_elsets):
                            pn = item[0]
                            pids = item[1]
                            if pn == elset_name:
                                ply_elsets[ii] = (pn, pids, orientations[ori_name])
                                break
                    section = None
                else:
                    section = None
                continue

            # Data lines
            if section == "nodes":
                parts = stripped.split(",")
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3]) if len(parts) > 3 else 0.0
                nodes[nid] = (x, y, z)

            elif section == "elements":
                parts = [p.strip() for p in stripped.split(",") if p.strip()]
                eid = int(parts[0])
                enodes = [int(p) for p in parts[1:]]
                elements[eid] = enodes

            elif section == "elset":
                parts = [int(p.strip()) for p in stripped.split(",") if p.strip()]
                current_elset_ids.extend(parts)

            elif section == "orientation":
                _ori_line_count += 1
                if _ori_line_count == 1:
                    # Line 1: direction cosines (skip)
                    pass
                elif _ori_line_count == 2:
                    # Line 2: "3, angle"
                    parts = stripped.split(",")
                    if len(parts) >= 2:
                        try:
                            axis = int(parts[0].strip())
                            if axis == 3:
                                angle = float(parts[1].strip())
                                orientations[current_elset_name] = angle
                        except (ValueError, IndexError):
                            pass
                    section = None

    # Flush final elset
    if section == "elset" and current_elset_name and current_elset_name.startswith("PLY-"):
        if current_elset_generate and len(current_elset_ids) == 3:
            start, end, step = current_elset_ids
            ids = list(range(start, end + 1, step))
        else:
            ids = current_elset_ids
        ply_elsets.append((current_elset_name, ids))

    # Build final ply info: ensure all have angles
    result_plies = []
    for item in ply_elsets:
        if len(item) == 3:
            name, eids, angle = item
        else:
            name, eids = item
            angle = 0.0  # fallback
        result_plies.append((name, eids, angle))

    # Fallback for single-ply meshes with no PLY-* elsets:
    # assign all elements to one ply using the first orientation found
    if not result_plies and elements:
        all_eids = sorted(elements.keys())
        angle = list(orientations.values())[0] if orientations else 90.0
        result_plies = [("PLY-0", all_eids, angle)]

    return {
        "nodes": nodes,
        "elements": elements,
        "ply_elsets": result_plies,
    }
