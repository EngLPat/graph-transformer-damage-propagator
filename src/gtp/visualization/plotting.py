"""
Shared plotting utilities for GTP visualization.

Provides reusable plotting functions for training and testing.
Uses actual quad element connectivity from the C3D8R hex mesh when
available, falling back to Delaunay triangulation otherwise.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import SymLogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay

from .mesh_utils import get_visualization_quads


def safe_to_numpy(tensor_or_array) -> np.ndarray:
    """
    Safely convert tensor/array to 1D numpy array.
    
    Ensures contiguous memory and 1D shape for matplotlib compatibility.
    
    Args:
        tensor_or_array: PyTorch tensor, numpy array, or list
        
    Returns:
        1D numpy array
    """
    if torch.is_tensor(tensor_or_array):
        arr = tensor_or_array.contiguous().cpu().detach().numpy()
    elif isinstance(tensor_or_array, np.ndarray):
        arr = np.ascontiguousarray(tensor_or_array)
    else:
        arr = np.array(tensor_or_array)
    
    return np.ravel(arr)


def create_2d_projection(pos_3d: np.ndarray) -> Tuple[np.ndarray, Delaunay]:
    """
    Create 2D projection and Delaunay triangulation from 3D positions.
    
    DEPRECATED: Use create_mesh_projection() for quad-aware rendering.
    Kept for backward compatibility.
    
    Args:
        pos_3d: Node positions [N, 3] as numpy array
        
    Returns:
        Tuple of (pos_2d [N, 2], triangulation)
    """
    if torch.is_tensor(pos_3d):
        pos_3d = pos_3d.cpu().numpy()
    
    pos_2d = pos_3d[:, :2]
    tri = Delaunay(pos_2d)
    
    return pos_2d, tri


def create_mesh_projection(pos_3d_or_data, face: str = "bottom",
                           ply: int = None) -> dict:
    """
    Create 2D projection with quad element connectivity.
    
    Preferred path: read ``data.cells`` directly from the PyG Data object
    (already stored during preprocessing, so no file I/O).
    
    Fallback chain:
        1. data.cells  (instant)
        2. .inp file parsing via mesh_utils  (slow, disk)
        3. Delaunay triangulation  (always works, wrong topology)
    
    Args:
        pos_3d_or_data: Either a PyG Data object (preferred) or [N,3]
                        numpy/tensor of node positions.
        face: 'bottom' (default), 'top', or 'all'
        
    Returns:
        Dictionary with:
          'pos_2d': [N, 2] XY coordinates
          'quads':  [Q, 4] quad connectivity or None
          'tri':    Delaunay triangulation (fallback) or None
          'mode':   'quad' or 'tri'
    """
    # Accept either Data object or raw positions
    data_obj = None
    if hasattr(pos_3d_or_data, 'mesh_pos'):
        data_obj = pos_3d_or_data
        pos_3d = data_obj.mesh_pos
    else:
        pos_3d = pos_3d_or_data
    
    if torch.is_tensor(pos_3d):
        pos_3d = pos_3d.cpu().numpy()
    
    pos_2d = pos_3d[:, :2]
    quads = None
    
    # --- Path 1: use data.cells directly (best) ---
    if data_obj is not None and hasattr(data_obj, 'cells') and data_obj.cells is not None:
        cells = data_obj.cells
        if torch.is_tensor(cells):
            cells = cells.cpu().numpy()
        
        # C3D8R hex face templates (0-based local node indices)
        _HEX_FACES = {
            'bottom': [[0, 1, 2, 3]],
            'top':    [[4, 5, 6, 7]],
            'all':    [[0,1,2,3], [4,5,6,7], [0,1,5,4],
                       [3,2,6,7], [0,3,7,4], [1,2,6,5]],
        }
        templates = _HEX_FACES.get(face, _HEX_FACES['bottom'])
        face_list = []
        for elem in cells:
            for tmpl in templates:
                face_list.append(elem[tmpl])
        quads = np.array(face_list, dtype=np.int64)
    
    # --- Path 2: .inp file fallback ---
    if quads is None:
        quads = get_visualization_quads(pos_3d, face=face)

    if quads is not None:
        return {
            'pos_2d': pos_2d,
            'quads': quads,
            'tri': None,
            'mode': 'quad',
        }

    # --- Path 2.5: element-level cohesive data → hex footprint quads ---
    # (nodes are element centroids; Delaunay would draw wrong triangles).
    # One ply at a time: the plies overlap in the XY projection.
    if data_obj is not None and _is_element_level_data(data_obj):
        try:
            hexm = _build_element_hex_mesh(data_obj)
        except Exception:
            hexm = None
        if hexm and hexm.get('plies'):
            plies = hexm['plies']
            k = 0 if ply is None else int(ply)
            k = max(0, min(k, len(plies) - 1))
            pm = plies[k]
            verts = pm['pos_2d'][pm['quads']]          # [Q, 4, 2]
            elem_idx = pm['elem_idx']                   # index into data arrays
            return {
                'pos_2d': pos_2d,                       # all element centroids
                'contour_pos': pos_2d[elem_idx],        # this ply's centroids
                'verts': verts,
                'elem_idx': elem_idx,
                'n_plies': len(plies),
                'ply': k,
                'ply_name': pm.get('name', f'ply{k}'),
                'quads': None,
                'tri': None,
                'mode': 'elem_quad',
            }

    # --- Path 3: Delaunay fallback ---
    tri = Delaunay(pos_2d)
    return {
        'pos_2d': pos_2d,
        'quads': None,
        'tri': tri,
        'mode': 'tri',
    }


# ── Element-level (integration-point-as-nodes) hex mesh support ──────────

def _is_element_level_data(data) -> bool:
    """Detect element-level dataset by checking for multiple z-layers
    and absence of cells."""
    if hasattr(data, 'cells') and data.cells is not None:
        return False
    pos = data.mesh_pos if hasattr(data, 'mesh_pos') else None
    if pos is None:
        return False
    if torch.is_tensor(pos):
        pos = pos.cpu().numpy()
    z_unique = np.unique(np.round(pos[:, 2], 4))
    # Cohesive laminates carry 2N-1 z-levels (N plies + N-1 interfaces);
    # ceiling set to cover realistic laminates up to ~16 plies.
    return len(z_unique) >= 1 and len(z_unique) <= 32


def _build_element_hex_mesh(data):
    """Build per-ply (and per-interface, for cohesive data) quad mesh.

    Parses the .inp file **for this specific simulation**, matches elements
    to dataset indices via centroid matching, and returns per-layer mesh
    info for quad plotting.

    Resolution order for finding the correct .inp:
      1. ``data.sim_id`` → look for ``<sim_id>.inp`` in geometry dirs
      2. Centroid-matching: compare .inp element centroids against
         ``data.mesh_pos`` to find the correct geometry file
      3. Raise if no match (never silently reuse the wrong geometry)

    Returns
    -------
    dict with:
        plies : list of dicts, each with:
            'name'     : str, ply name
            'z'        : float, ply centroid z
            'mask'     : [N_total] bool mask into dataset elements
            'pos_2d'   : [M, 2] mesh node XY positions
            'quads'    : [Q, 4] quad connectivity (indices into pos_2d)
            'elem_idx' : [Q] index into dataset elements for each quad
        interfaces : list of dicts (same schema, empty for non-cohesive).
                     For cohesive graphs, ply nodes occupy indices
                     [0, n_ply) and interfaces [n_ply, n_ply+n_iface);
                     interface elem_idx values reflect that offset.
    """
    from pathlib import Path

    pos = data.mesh_pos
    if torch.is_tensor(pos):
        pos = pos.cpu().numpy()

    from .mesh_utils import parse_inp_elements_and_plies

    is_cohesive = (hasattr(data, 'n_ply_nodes')
                   and getattr(data, 'n_ply_nodes', None) is not None)

    # For cohesive graphs, ply nodes are first (0..n_ply-1), interfaces
    # after (n_ply..n_ply+n_iface-1). Match against the ply portion only.
    if is_cohesive:
        n_ply_target = int(data.n_ply_nodes)
    else:
        n_ply_target = pos.shape[0]
    pos_ply = pos[:n_ply_target]

    # ── Find the .inp file for THIS simulation ──
    project_root = Path(__file__).resolve().parents[3]
    geom_dirs = sorted(project_root.glob("data/raw/*/geometry"))
    inp_path = None

    # --- Helpers: expected layup + z-layers from data.x for validation ---
    # data.x columns (per build_cohesive_dataset.build_node_features):
    #   0,1,2 = centroid xyz; 3,4 = cos/sin of lower ply angle; 7 = layer_norm
    def _expected_layup_from_data(_data, _n_ply_target):
        import math
        xnp = _data.x.cpu().numpy() if torch.is_tensor(_data.x) else _data.x
        ply_rows = xnp[:_n_ply_target]
        # z is always taken from mesh_pos, which does not depend on the
        # node-feature layout
        zs = pos[:_n_ply_target, 2]
        # angle columns: 3,4 in the standard 14/15-dim layout; 0,1 in a
        # layout without the three position columns (11/12-dim).
        ang_off = 3 if xnp.shape[1] >= 14 else 0
        # unique z layers in stack order
        z_unique = []
        for z in zs:
            z_round = float(np.round(z, 4))
            if not z_unique or abs(z_round - z_unique[-1]) > 1e-3:
                if z_round not in z_unique:
                    z_unique.append(z_round)
        z_unique = sorted(set(round(z, 4) for z in z_unique))
        angles = []
        for zl in z_unique:
            mask = np.abs(zs - zl) < 1e-3
            if not mask.any():
                continue
            ca = float(np.mean(ply_rows[mask, ang_off]))
            sa = float(np.mean(ply_rows[mask, ang_off + 1]))
            ang = math.degrees(math.atan2(sa, ca))
            angles.append(ang)
        return z_unique, angles

    def _angles_match(a_list, b_list, tol=5.0):
        if len(a_list) != len(b_list):
            return False
        for a, b in zip(a_list, b_list):
            da = ((a - b) % 180.0)
            da = min(da, 180.0 - da)
            if da > tol:
                return False
        return True

    expected_z, expected_angles = None, None
    try:
        expected_z, expected_angles = _expected_layup_from_data(data, n_ply_target)
    except Exception:
        pass

    def _validate_candidate(candidate_path):
        """Check both ply count, ply angles, and ply z-levels match data."""
        try:
            if is_cohesive:
                from src.preprocessing.build_cohesive_dataset import (
                    parse_cohesive_inp,
                )
                pc = parse_cohesive_inp(candidate_path)
                if len(pc["ply_elems"]) != n_ply_target:
                    return False
                inp_angles = [pc["ply_angles"].get(i, 0.0)
                              for i in range(len(pc["ply_ranges"]))]
                # Per-ply mid-z (using element-centroid z of each ply range)
                inp_zs = []
                for (first, last) in pc["ply_ranges"]:
                    eids = [eid for eid in pc["ply_elems"]
                            if first <= eid <= last]
                    if not eids:
                        continue
                    cz = float(np.mean([
                        np.mean([pc["nodes"][n][2]
                                 for n in pc["ply_elems"][eid]])
                        for eid in eids[:50]
                    ]))
                    inp_zs.append(round(cz, 4))
            else:
                parsed_check = parse_inp_elements_and_plies(str(candidate_path))
                if len(parsed_check["elements"]) != n_ply_target:
                    return False
                # Non-cohesive: trust ply_elsets order; angle check via ply_elsets if available
                ply_elsets_check = parsed_check.get("ply_elsets", [])
                inp_angles = [a for (_, _, a) in ply_elsets_check]
                inp_zs = []  # skip z-check for non-cohesive
        except Exception:
            return False
        if expected_angles is not None and not _angles_match(expected_angles, inp_angles):
            return False
        if expected_z is not None and inp_zs:
            if len(expected_z) != len(inp_zs):
                return False
            if max(abs(a - b) for a, b in zip(expected_z, inp_zs)) > 1e-2:
                return False
        return True

    # --- Strategy 1: use sim_id if available, validated by layup + z-layers ---
    sim_id = getattr(data, 'sim_id', None)
    if sim_id is not None and isinstance(sim_id, str) and len(sim_id) > 0:
        for gdir in geom_dirs:
            candidate = gdir / f"{sim_id}.inp"
            if candidate.exists() and _validate_candidate(candidate):
                inp_path = candidate
                break

    # --- Strategy 2: match by per-element centroids against mesh_pos,
    #     gated by layup + z-layer validation to avoid cross-study mismatches ---
    if inp_path is None:
        best_match = None
        best_dist = float('inf')

        for gdir in geom_dirs:
            candidates = sorted(
                set(gdir.glob("plate_[0-9]*.inp"))
                | set(gdir.glob("cohesive_lam_plate_[0-9]*.inp"))
            )
            for candidate in candidates:
                try:
                    # Layup and z check first: rejects decks of other studies
                    if not _validate_candidate(candidate):
                        continue

                    inp_centroids = None
                    if is_cohesive:
                        from src.preprocessing.build_cohesive_dataset import (
                            parse_cohesive_inp,
                        )
                        pc = parse_cohesive_inp(candidate)
                        ply_elems = pc["ply_elems"]
                        nodes_c = pc["nodes"]
                        sorted_ply_eids = sorted(ply_elems.keys())
                        inp_centroids = np.array([
                            np.mean([nodes_c[n]
                                     for n in ply_elems[eid][:4]], axis=0)
                            for eid in sorted_ply_eids
                        ])
                    else:
                        parsed_check = parse_inp_elements_and_plies(
                            str(candidate))
                        nd = parsed_check["nodes"]
                        ed = parsed_check["elements"]
                        s_eids = sorted(ed.keys())
                        inp_centroids = np.array([
                            np.mean([nd[n] for n in ed[eid][:4]], axis=0)
                            for eid in s_eids
                        ])

                    if inp_centroids is None:
                        continue

                    dist = float(np.abs(
                        inp_centroids[:, :2] - pos_ply[:, :2]
                    ).mean())
                    if dist < best_dist:
                        best_dist = dist
                        best_match = candidate
                except Exception:
                    continue

        if best_match is not None and best_dist < 0.01:
            inp_path = best_match
            print(f"  _build_element_hex_mesh: matched {inp_path.name} "
                  f"(mean_abs_dist={best_dist:.6f})")

    if inp_path is None:
        raise RuntimeError(
            f"_build_element_hex_mesh: could not find .inp file for simulation "
            f"(sim_id={sim_id!r}, n_ply_target={n_ply_target}, "
            f"centroid_mean=({pos_ply[:, 0].mean():.2f}, {pos_ply[:, 1].mean():.2f})). "
            f"Searched: {[str(g) for g in geom_dirs]}. "
            f"Set data.sim_id to the .inp filename stem to enable direct lookup."
        )

    # ── Parse .inp with the parser for cohesive or non-cohesive decks ──
    iface_elems = {}
    iface_elsets = []
    if is_cohesive:
        from src.preprocessing.build_cohesive_dataset import parse_cohesive_inp
        parsed_coh = parse_cohesive_inp(inp_path)
        nodes_dict = parsed_coh["nodes"]
        ply_elems = parsed_coh["ply_elems"]       # {eid: [n1..n8]}
        iface_elems = parsed_coh["iface_elems"]    # {eid: [n1..n8]}
        ply_angles = parsed_coh["ply_angles"]      # {ply_idx: angle_deg}
        ply_ranges = parsed_coh["ply_ranges"]
        iface_ranges = parsed_coh["iface_ranges"]

        elements_dict = ply_elems  # ply-only
        sorted_eids = sorted(elements_dict.keys())
        sorted_iface_eids = sorted(iface_elems.keys())

        ply_elsets = []
        for pi, (first, last) in enumerate(ply_ranges):
            angle = ply_angles.get(pi, 0.0)
            eids = [eid for eid in sorted_eids if first <= eid <= last]
            ply_elsets.append((f"PLY-{pi}", eids, angle))

        for ii, (first, last) in enumerate(iface_ranges):
            eids = [eid for eid in sorted_iface_eids if first <= eid <= last]
            iface_elsets.append((f"IFACE-{ii}", eids))
    else:
        parsed = parse_inp_elements_and_plies(str(inp_path))
        nodes_dict = parsed["nodes"]
        elements_dict = parsed["elements"]
        ply_elsets = parsed["ply_elsets"]
        sorted_eids = sorted(elements_dict.keys())
        sorted_iface_eids = []

    # ── Build mapping: sorted element IDs → dataset indices ──
    n_elem_inp = len(sorted_eids)

    if n_elem_inp != n_ply_target:
        raise RuntimeError(
            f"_build_element_hex_mesh: ply element count mismatch between .inp "
            f"({n_elem_inp}) and dataset ({n_ply_target}) for {inp_path.name}. "
            f"The .inp file does not match this simulation's data."
        )

    # ── Build 0-based mesh node positions ──
    sorted_nids = sorted(nodes_dict.keys())
    nid_to_idx = {nid: j for j, nid in enumerate(sorted_nids)}
    mesh_node_pos = np.array([nodes_dict[nid] for nid in sorted_nids])
    pos_2d_mesh = mesh_node_pos[:, :2]

    # ── Build per-ply quad meshes using ply_elsets from .inp ──
    eid_to_didx = {eid: i for i, eid in enumerate(sorted_eids)}
    z_vals = np.round(pos[:, 2], 4)

    plies = []
    for ply_name, ply_eids, angle in ply_elsets:
        quads = []
        quad_didx = []
        for eid in ply_eids:
            if eid not in elements_dict:
                continue
            enodes = elements_dict[eid]
            face = [nid_to_idx[n] for n in enodes[:4]]
            quads.append(face)
            quad_didx.append(eid_to_didx.get(eid, -1))

        if not quads:
            continue

        quad_didx_arr = np.array(quad_didx, dtype=np.int64)
        ply_z = float(np.median(pos[quad_didx_arr, 2]))
        mask = np.abs(z_vals - round(ply_z, 4)) < 1e-3

        plies.append({
            'name': ply_name,
            'angle': angle,
            'z': ply_z,
            'mask': mask,
            'pos_2d': pos_2d_mesh,
            'quads': np.array(quads, dtype=np.int64),
            'elem_idx': quad_didx_arr,
        })

    # ── Per-interface quads (cohesive WITH-iface datasets only) ──
    # Gate on the dataset actually containing interface rows. `n_ply_nodes`
    # is set on the Data object for BOTH ply-only and with-iface cohesive
    # builds, so `is_cohesive` alone is insufficient. A ply-only dataset has
    # pos.shape[0] == n_ply_target; only with-iface builds extend pos beyond.
    # Rendering interfaces on a ply-only dataset would index both data.mesh_pos
    # and downstream values_per_element past their end.
    interfaces = []
    has_iface_rows = pos.shape[0] > n_ply_target
    if is_cohesive and iface_elsets and has_iface_rows:
        iface_eid_to_didx = {
            eid: n_ply_target + i for i, eid in enumerate(sorted_iface_eids)
        }
        for iface_name, iface_eids in iface_elsets:
            quads = []
            quad_didx = []
            for eid in iface_eids:
                if eid not in iface_elems:
                    continue
                enodes = iface_elems[eid]
                # COH3D8 bottom face = nodes[:4] (zero-thickness so top==bottom)
                face = [nid_to_idx[n] for n in enodes[:4]]
                quads.append(face)
                quad_didx.append(iface_eid_to_didx.get(eid, -1))

            if not quads:
                continue

            quad_didx_arr = np.array(quad_didx, dtype=np.int64)
            iface_z = float(np.median(pos[quad_didx_arr, 2]))
            mask = np.abs(z_vals - round(iface_z, 4)) < 1e-3

            interfaces.append({
                'name': iface_name,
                'angle': None,
                'z': iface_z,
                'mask': mask,
                'pos_2d': pos_2d_mesh,
                'quads': np.array(quads, dtype=np.int64),
                'elem_idx': quad_didx_arr,
            })

    if not plies and not interfaces:
        return None

    return {'plies': plies, 'interfaces': interfaces}


def _plot_field_on_hex_quads(ax, ply_mesh, values_per_element, cmap='viridis',
                              vmin=None, vmax=None, norm=None,
                              edgecolor='k', edge_alpha=0.15, edge_linewidth=0.3):
    """Plot a scalar field on hex element quads (one value per element).

    Parameters
    ----------
    ax : matplotlib Axes
    ply_mesh : dict from _build_element_hex_mesh with 'pos_2d', 'quads', 'elem_idx'
    values_per_element : [N_total] array, indexed by elem_idx
    """
    pos_2d = ply_mesh['pos_2d']
    quads = ply_mesh['quads']
    elem_idx = ply_mesh['elem_idx']

    # Build polygon vertices: [Q, 4, 2]
    verts = pos_2d[quads]

    # One value per element (no averaging needed)
    elem_values = values_per_element[elem_idx]

    if norm is None and vmin is not None:
        norm = Normalize(vmin=vmin, vmax=vmax)

    cmap_obj = plt.get_cmap(cmap)

    pc = PolyCollection(
        verts,
        array=elem_values,
        cmap=cmap_obj,
        norm=norm,
        edgecolors=(0, 0, 0, edge_alpha),
        linewidths=edge_linewidth,
        alpha=1.0,
    )
    ax.add_collection(pc)
    ax.set_xlim(pos_2d[:, 0].min(), pos_2d[:, 0].max())
    ax.set_ylim(pos_2d[:, 1].min(), pos_2d[:, 1].max())
    return pc


def plot_element_damage_frame_perply(
    data, hex_mesh, pred_damage, true_damage, pred_rate, true_rate,
    timestep, save_path, current_time=None, rate_range=None,
    field_name: str = "Damage",
):
    """Plot per-ply damage/rate frames for element-level data.

    Layout: n_plies columns × 2 rows (true on top, pred on bottom).
    Each column shows one ply's damage field on the actual hex mesh.
    """
    plies = hex_mesh.get('plies', [])
    interfaces = hex_mesh.get('interfaces', [])
    layers = sorted(plies + interfaces, key=lambda l: l['z'])
    n_layers = len(layers)
    has_gt = true_damage is not None and true_rate is not None

    pred_damage = safe_to_numpy(pred_damage)
    pred_rate = safe_to_numpy(pred_rate)

    time_str = f"t={current_time:.4f}s" if current_time is not None else f"Step {timestep}"

    # Rate normalisation
    if rate_range is not None:
        rate_vmin, rate_vmax = rate_range
    else:
        rate_vmax = max(float(np.max(np.abs(pred_rate))), 1.0)
        if has_gt:
            true_damage = safe_to_numpy(true_damage)
            true_rate = safe_to_numpy(true_rate)
            rate_vmax = max(rate_vmax, float(np.max(np.abs(true_rate))), 1.0)
        rate_vmin = 0.0
    rate_norm = _make_rate_norm(rate_vmin, rate_vmax)

    if has_gt:
        if not isinstance(true_damage, np.ndarray):
            true_damage = safe_to_numpy(true_damage)
        if not isinstance(true_rate, np.ndarray):
            true_rate = safe_to_numpy(true_rate)

    # ── Layout: n_layers cols × 2 rows (true, pred) showing damage only ──
    n_rows = 2 if has_gt else 1
    fig, axes = plt.subplots(n_rows, n_layers, figsize=(5 * n_layers, 5 * n_rows),
                              squeeze=False)

    # Auto-scale vmax for FI (can exceed 1.0)
    dmg_vmax = 1.0
    dmg_vmax = max(dmg_vmax, float(np.max(pred_damage)))
    if has_gt:
        dmg_vmax = max(dmg_vmax, float(np.max(true_damage)))
    field_label = field_name

    for li, layer in enumerate(layers):
        angle = layer.get('angle', None)
        if angle is not None:
            col_title = f"{layer['name']} ({angle:.0f}°)\nz={layer['z']:.3f}"
        else:
            col_title = f"{layer['name']}\nz={layer['z']:.3f}"

        # True
        if has_gt:
            ax = axes[0, li]
            ax.set_aspect('equal')
            ax.set_axis_off()
            mp = _plot_field_on_hex_quads(ax, layer, true_damage,
                                           cmap='viridis', vmin=0.0, vmax=dmg_vmax)
            ax.set_title(f"True {field_label}\n{col_title}\n{time_str}",
                         fontsize=10, fontweight='bold')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(mp, cax=cax).set_label(f'{field_label} [-]')

        # Pred
        ax = axes[n_rows - 1, li]
        ax.set_aspect('equal')
        ax.set_axis_off()
        mp = _plot_field_on_hex_quads(ax, layer, pred_damage,
                                       cmap='viridis', vmin=0.0, vmax=dmg_vmax)
        title_prefix = f"Pred {field_label}" if has_gt else field_label
        ax.set_title(f"{title_prefix}\n{col_title}\n{time_str}",
                     fontsize=10, fontweight='bold')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(mp, cax=cax).set_label(f'{field_label} [-]')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_element_field_perply(
    hex_mesh, pred_field, true_field,
    save_path, field_label="FI", cmap='viridis',
    current_time=None, timestep=0,
    use_rate_norm=False, vmin=0.0, vmax=None,
    iface_vmax=1.0,
):
    """General per-ply hex-element frame renderer.

    Renders one row per truth type (GT on top, Pred below), one column
    per ply.  Works for any scalar field: FI, FI rate, damage, damage
    rate.

    Parameters
    ----------
    hex_mesh : dict from _build_element_hex_mesh
    pred_field, true_field : [N] arrays (true_field may be None)
    field_label : str shown in titles and colorbar
    cmap : matplotlib colormap name
    use_rate_norm : if True, use symlog normalisation suitable for rates
    vmin, vmax : explicit colour limits (auto-scaled if vmax is None)
    """
    plies = hex_mesh.get('plies', [])
    interfaces = hex_mesh.get('interfaces', [])
    # Through-thickness order: interleaves plies + interfaces by z.
    layers = sorted(plies + interfaces, key=lambda l: l['z'])
    n_layers = len(layers)
    has_gt = true_field is not None

    pred_field = safe_to_numpy(pred_field)
    if has_gt:
        true_field = safe_to_numpy(true_field)

    time_str = f"t={current_time:.4f}s" if current_time is not None else f"Step {timestep}"

    if vmax is None:
        vmax = max(1.0, float(np.max(np.abs(pred_field))))
        if has_gt:
            vmax = max(vmax, float(np.max(np.abs(true_field))))

    # Ply colormap uses [vmin, vmax]; interface layers use [vmin, iface_vmax].
    # Rationale: iface fields (SDEG, QUADSCRT) are physically bounded by 1,
    # while ply FI is bounded by FI_CAP = 3. Sharing the same scale washes
    # out the iface variation. Rate-normalised fields use the same range
    # for both (rates aren't bounded the same way).
    if use_rate_norm:
        norm = _make_rate_norm(vmin, vmax)
        iface_norm = norm
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        iface_norm = Normalize(vmin=vmin, vmax=iface_vmax)

    n_rows = 2 if has_gt else 1
    fig, axes = plt.subplots(n_rows, n_layers,
                              figsize=(5 * n_layers, 5 * n_rows),
                              squeeze=False)

    for li, layer in enumerate(layers):
        angle = layer.get('angle', None)
        is_iface = (angle is None)
        layer_norm = iface_norm if is_iface else norm
        layer_clabel = field_label + (" (iface)" if is_iface else "")
        if angle is not None:
            col_title = f"{layer['name']} ({angle:.0f}°)\nz={layer['z']:.3f}"
        else:
            col_title = f"{layer['name']}\nz={layer['z']:.3f}"

        if has_gt:
            ax = axes[0, li]
            ax.set_aspect('equal'); ax.set_axis_off()
            mp = _plot_field_on_hex_quads(ax, layer, true_field,
                                           cmap=cmap, norm=layer_norm)
            ax.set_title(f"True {field_label}\n{col_title}\n{time_str}",
                         fontsize=10, fontweight='bold')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(mp, cax=cax).set_label(layer_clabel)

        ax = axes[n_rows - 1, li]
        ax.set_aspect('equal'); ax.set_axis_off()
        mp = _plot_field_on_hex_quads(ax, layer, pred_field,
                                       cmap=cmap, norm=layer_norm)
        title_prefix = f"Pred {field_label}" if has_gt else field_label
        ax.set_title(f"{title_prefix}\n{col_title}\n{time_str}",
                     fontsize=10, fontweight='bold')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(mp, cax=cax).set_label(layer_clabel)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _make_rate_norm(
    vmin: float,
    vmax: float,
    use_symlog: bool = True,
) -> Normalize:
    """
    Create a colour normalisation suitable for damage rate fields.
    
    Rates span many orders of magnitude (0 to 60,000+).  A symmetric
    log scale with a linear region near zero makes both progressive
    (rate < 100) and catastrophic (rate > 100) nodes visible.
    
    Args:
        vmin: Minimum value
        vmax: Maximum value
        use_symlog: If True, use SymLogNorm; else linear Normalize
        
    Returns:
        matplotlib Normalize instance
    """
    if not use_symlog or vmax <= 1.0:
        return Normalize(vmin=vmin, vmax=vmax)
    
    # linthresh: values in [-linthresh, linthresh] are linear
    # Good default: ~1% of max, but at least 0.01
    linthresh = max(0.01, vmax * 0.01)
    return SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)


def _plot_field_on_mesh(
    ax,
    mesh: dict,
    values: np.ndarray,
    cmap: str = 'viridis',
    vmin: float = None,
    vmax: float = None,
    norm=None,
    edgecolor: str = 'k',
    edge_alpha: float = 0.1,
    edge_linewidth: float = 0.3,
):
    """
    Plot a scalar field on the mesh using quads or triangles.
    
    Args:
        ax: Matplotlib Axes
        mesh: Dictionary from create_mesh_projection()
        values: [N] scalar values per node
        cmap: Colormap name
        vmin, vmax: Value range for colour mapping
        norm: Optional matplotlib Normalize (overrides vmin/vmax)
        edgecolor: Edge line colour
        edge_alpha: Edge line transparency
        edge_linewidth: Edge line width
    
    Returns:
        The matplotlib collection for colorbar creation.
    """
    pos_2d = mesh['pos_2d']
    
    if mesh['mode'] == 'quad' and mesh['quads'] is not None:
        quads = mesh['quads']
        # Build polygon vertices: [Q, 4, 2]
        verts = pos_2d[quads]
        
        # Per-element values: average of 4 corner nodes
        elem_values = values[quads].mean(axis=1)
        
        if norm is None and vmin is not None:
            norm = Normalize(vmin=vmin, vmax=vmax)
        
        cmap_obj = plt.get_cmap(cmap)
        
        # Create PolyCollection
        pc = PolyCollection(
            verts,
            array=elem_values,
            cmap=cmap_obj,
            norm=norm,
            edgecolors=edgecolor,
            linewidths=edge_linewidth,
            alpha=1.0,
        )
        pc.set_edgecolor((0, 0, 0, edge_alpha))
        ax.add_collection(pc)
        
        # Set axis limits
        ax.set_xlim(pos_2d[:, 0].min(), pos_2d[:, 0].max())
        ax.set_ylim(pos_2d[:, 1].min(), pos_2d[:, 1].max())
        
        return pc
    
    elif mesh.get('mode') == 'elem_quad':
        # Element-level cohesive data: one value per element, drawn on the
        # element footprint quads of the selected ply (same recipe as
        # _plot_field_on_hex_quads / the test GIFs).
        verts = mesh['verts']
        vals = np.asarray(values)
        # accept full-length arrays (index by elem_idx) or arrays already
        # restricted to this ply (len == n quads)
        elem_values = (vals if len(vals) == len(mesh['elem_idx'])
                       else vals[mesh['elem_idx']])
        if norm is None and vmin is not None:
            norm = Normalize(vmin=vmin, vmax=vmax)
        pc = PolyCollection(
            verts,
            array=elem_values,
            cmap=plt.get_cmap(cmap),
            norm=norm,
            edgecolors=(0, 0, 0, edge_alpha),
            linewidths=edge_linewidth,
            alpha=1.0,
        )
        ax.add_collection(pc)
        ax.set_xlim(verts[..., 0].min(), verts[..., 0].max())
        ax.set_ylim(verts[..., 1].min(), verts[..., 1].max())
        return pc

    elif mesh.get('mode') == 'scatter' or mesh.get('tri') is None:
        # Element-level: centroids as points, no mesh connectivity
        if norm is None and vmin is not None:
            norm = Normalize(vmin=vmin, vmax=vmax)
        sc = ax.scatter(
            pos_2d[:, 0], pos_2d[:, 1],
            c=values, cmap=cmap, norm=norm,
            s=12, edgecolors='none', marker='s',
        )
        ax.set_xlim(pos_2d[:, 0].min(), pos_2d[:, 0].max())
        ax.set_ylim(pos_2d[:, 1].min(), pos_2d[:, 1].max())
        return sc

    else:
        # Fallback: Delaunay triangulation
        tri = mesh['tri']
        # matplotlib 3.9+ forbids passing norm together with vmin/vmax
        if norm is None and vmin is not None:
            norm = Normalize(vmin=vmin, vmax=vmax)
        mesh_plot = ax.tripcolor(
            pos_2d[:, 0], pos_2d[:, 1], tri.simplices,
            values, shading='flat', cmap=cmap,
            norm=norm,
        )
        ax.triplot(
            pos_2d[:, 0], pos_2d[:, 1], tri.simplices,
            'k-', alpha=edge_alpha, linewidth=edge_linewidth,
        )
        return mesh_plot


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[Union[str, Path]] = None,
    test_loss: Optional[float] = None,
    title: str = "Training Progress",
    show: bool = False
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the figure
        test_loss: Optional final test loss to mark
        title: Plot title
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = list(range(1, len(train_losses) + 1))
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    if test_loss is not None:
        ax.axhline(y=test_loss, color='g', linestyle='--', 
                   label=f'Test Loss: {test_loss:.6f}', linewidth=2)
        ax.scatter([len(epochs)], [test_loss], color='g', s=100, zorder=5)
    
    # Mark best validation loss
    best_idx = np.argmin(val_losses)
    best_val = val_losses[best_idx]
    ax.scatter([best_idx + 1], [best_val], color='orange', s=150, 
               marker='*', label=f'Best Val: {best_val:.6f}', zorder=5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Log scale if losses span multiple orders of magnitude
    if len(train_losses) > 0 and max(train_losses) / (min(train_losses) + 1e-10) > 100:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # Also save as PDF for publication quality
        pdf_path = save_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_loss_components(
    loss_components: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Loss Component Breakdown",
    show: bool = False
) -> None:
    """
    Plot individual loss components over training epochs.
    
    Args:
        loss_components: Dictionary with keys 'rate', 'dmg_rate', 'total'
        save_path: Path to save the figure
        title: Plot title
        show: Whether to display the plot
    """
    # Determine layout from which components have signal
    dmg_rate = loss_components.get('dmg_rate', [])
    has_dmg_rate = dmg_rate and any(v > 0 for v in dmg_rate)

    n_extra = int(has_dmg_rate)
    n_cols = 2 + (1 if n_extra > 0 else 0)
    fig, axes = plt.subplots(2, n_cols, figsize=(7 * n_cols, 10))

    epochs = list(range(1, len(loss_components.get('total', [])) + 1))

    if not epochs:
        plt.close()
        return

    # Total loss
    ax = axes[0, 0]
    total = loss_components.get('total', [])
    if total:
        ax.plot(epochs, total, 'b-', linewidth=2, label='Total Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Training Loss', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        if max(total) / (min(total) + 1e-10) > 100:
            ax.set_yscale('log')

    # FI rate loss (Huber log-space)
    ax = axes[0, 1]
    rate = loss_components.get('rate', [])
    if rate:
        ax.plot(epochs, rate, 'g-', linewidth=2, label='FI Rate Loss (Huber)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('FI Rate Prediction Loss', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        if max(rate) / (min(rate) + 1e-10) > 100:
            ax.set_yscale('log')

    # Damage rate loss (two-stage)
    if has_dmg_rate:
        col = 2 if n_cols > 2 else 1
        ax = axes[1, col] if n_cols > 2 else axes[1, 1]
        ax.plot(epochs, dmg_rate, 'c-', linewidth=2, label='Damage Rate Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Damage Rate Loss (FI>1 masked)', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)

    # Hide unused subplots
    for r in range(2):
        for c in range(n_cols):
            if not axes[r, c].has_data():
                axes[r, c].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_damage_comparison(
    data,
    pred_damage: np.ndarray,
    true_damage: np.ndarray,
    pred_rate: np.ndarray,
    true_rate: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    timestep: Optional[int] = None,
    current_time: Optional[float] = None,
    show: bool = False,
    rate_range: Optional[Tuple[float, float]] = None,
    field_name: str = "Damage",
) -> None:
    """
    Create comprehensive 2x3 damage comparison plot.
    
    Uses actual quad element connectivity from C3D8R hex mesh when
    available, falling back to Delaunay triangulation otherwise.
    Rate fields use symmetric-log colour scale for visibility across
    the full range (progressive 0-100, catastrophic 100-60000).
    
    Layout:
        Row 1: [True Damage] [True Rate] [Error Map]
        Row 2: [Pred Damage] [Pred Rate] [Rate Comparison]
    
    Args:
        data: Graph data with mesh_pos attribute
        pred_damage: Predicted accumulated damage [N]
        true_damage: Ground truth accumulated damage [N]
        pred_rate: Predicted damage rate [N]
        true_rate: Ground truth damage rate [N]
        save_path: Path to save figure
        timestep: Current timestep index
        current_time: Current simulation time
        show: Whether to display the plot
        rate_range: Shared (vmin, vmax) for rate plots.
                    Default None = auto (symlog from data).
    """
    # Get mesh projection (quad or tri)
    mesh = create_mesh_projection(data)
    
    # Convert to numpy
    pred_damage = safe_to_numpy(pred_damage)
    true_damage = safe_to_numpy(true_damage)
    pred_rate = safe_to_numpy(pred_rate)
    true_rate = safe_to_numpy(true_rate)
    
    # Compute errors
    damage_error = pred_damage - true_damage
    rate_error = pred_rate - true_rate
    
    # Time info
    if current_time is not None:
        time_str = f"t={current_time:.4f}s"
    elif timestep is not None:
        time_str = f"Step {timestep}"
    else:
        time_str = ""
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Shared rate range, auto-scaled with symlog
    if rate_range is not None:
        rate_vmin, rate_vmax = rate_range
    else:
        rate_vmin = 0.0
        rate_vmax = max(
            float(np.max(np.abs(true_rate))),
            float(np.max(np.abs(pred_rate))),
            1.0,  # minimum vmax to avoid degenerate scale
        )
    
    rate_norm = _make_rate_norm(rate_vmin, rate_vmax)
    
    # Auto-scaled damage/FI range (FI may exceed 1)
    dmg_vmax = max(1.0, float(np.max(pred_damage)), float(np.max(true_damage)))
    plots_data = [
        # (values, title, cmap, row, col, vmin, vmax, norm, is_rate)
        (true_damage, f'True {field_name}\n{time_str}', 'viridis', 0, 0, 0.0, dmg_vmax, None, False),
        (true_rate, f'True {field_name} Rate\n{time_str}', 'plasma', 0, 1, rate_vmin, rate_vmax, rate_norm, True),
        (damage_error, f'{field_name} Error (Pred - True)', 'RdBu_r', 0, 2, None, None, None, False),
        (pred_damage, f'Predicted {field_name}\n{time_str}', 'viridis', 1, 0, 0.0, dmg_vmax, None, False),
        (pred_rate, f'Predicted {field_name} Rate\n{time_str}', 'plasma', 1, 1, rate_vmin, rate_vmax, rate_norm, True),
        (rate_error, 'Rate Error (Pred - True)', 'RdBu_r', 1, 2, None, None, None, False),
    ]
    
    for values, title, cmap, row, col, vmin, vmax, norm, is_rate in plots_data:
        ax = axes[row, col]
        ax.set_aspect('equal')
        ax.set_axis_off()
        
        # Symmetric colorbar for error plots
        if 'Error' in title:
            max_abs = max(abs(values.min()), abs(values.max()), 1e-10)
            vmin, vmax = -max_abs, max_abs
            norm = None
        
        mesh_plot = _plot_field_on_mesh(
            ax, mesh, values, cmap=cmap,
            vmin=vmin, vmax=vmax, norm=norm,
        )
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
        
        if is_rate:
            cbar.set_label('Rate [1/s]')
        elif 'Error' in title:
            cbar.set_label('Error')
        else:
            cbar.set_label(f'{field_name} [-]')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_damage_evolution_frame(
    data,
    pred_damage: np.ndarray,
    true_damage: Optional[np.ndarray],
    pred_rate: np.ndarray,
    true_rate: Optional[np.ndarray],
    timestep: int,
    save_path: Union[str, Path],
    current_time: Optional[float] = None,
    rate_range: Optional[Tuple[float, float]] = None,
    field_name: str = "Damage",
) -> None:
    """
    Plot a single frame for damage evolution visualization.
    
    Uses actual quad element connectivity from C3D8R hex mesh when
    available.  Rate fields use symmetric-log colour scale.
    
    Args:
        data: Graph data with mesh_pos
        pred_damage: Predicted accumulated damage
        true_damage: Ground truth accumulated damage (None for extended rollout)
        pred_rate: Predicted damage rate
        true_rate: Ground truth damage rate (None for extended rollout)
        timestep: Current timestep index
        save_path: Path to save figure
        current_time: Current simulation time
        rate_range: Shared (vmin, vmax) for rate plots.
                    Default None = auto (symlog from data).
    """
    # Get mesh projection (quad or tri)
    mesh = create_mesh_projection(data)
    
    # Convert to numpy
    pred_damage = safe_to_numpy(pred_damage)
    pred_rate = safe_to_numpy(pred_rate)
    
    # Check if we have ground truth
    has_ground_truth = true_damage is not None and true_rate is not None
    
    if has_ground_truth:
        true_damage = safe_to_numpy(true_damage)
        true_rate = safe_to_numpy(true_rate)
    
    # Time info
    if current_time is not None:
        time_str = f"t={current_time:.4f}s"
    else:
        time_str = f"Step {timestep}"
    
    # Rate normalisation, auto-scaled with symlog
    if rate_range is not None:
        rate_vmin, rate_vmax = rate_range
    else:
        if has_ground_truth:
            rate_vmax = max(
                float(np.max(np.abs(true_rate))),
                float(np.max(np.abs(pred_rate))),
                1.0,
            )
        else:
            rate_vmax = max(float(np.max(np.abs(pred_rate))), 1.0)
        rate_vmin = 0.0
    
    rate_norm = _make_rate_norm(rate_vmin, rate_vmax)
    
    is_fi = field_name.upper() == "FI"

    if has_ground_truth:
        # 2×3 layout:
        #   Row 1: True Field  |  True Rate   |  True State / Error
        #   Row 2: Pred Field  |  Pred Rate   |  Pred State / Error
        fig, axes = plt.subplots(2, 3, figsize=(21, 12))

        # Helper to plot a panel
        def _panel(ax, values, title, cmap, vmin, vmax, norm, label, is_state=False):
            ax.set_aspect('equal')
            ax.set_axis_off()
            mp = _plot_field_on_mesh(ax, mesh, values, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
            ax.set_title(title, fontsize=12, fontweight='bold')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(mp, cax=cax, orientation='vertical')
            if is_state:
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels(['Active', 'Failed'])
            else:
                cbar.set_label(label)

        # Auto-scaled damage/FI range (FI may exceed 1)
        dmg_vmax = max(1.0, float(np.max(true_damage)), float(np.max(pred_damage)))

        # Row 1: Ground truth
        _panel(axes[0, 0], true_damage,  f'True {field_name}\n{time_str}',  'viridis',  0.0, dmg_vmax, None, f'{field_name} [-]')
        _panel(axes[0, 1], true_rate,    f'True {field_name} Rate\n{time_str}',    'plasma',   rate_vmin, rate_vmax, rate_norm, 'Rate [1/s]')

        # Row 2: Predictions
        _panel(axes[1, 0], pred_damage,  f'Pred {field_name}\n{time_str}',  'viridis',  0.0, dmg_vmax, None, f'{field_name} [-]')
        _panel(axes[1, 1], pred_rate,    f'Pred {field_name} Rate\n{time_str}',    'plasma',   rate_vmin, rate_vmax, rate_norm, 'Rate [1/s]')

        if is_fi:
            # For FI: col 3 shows absolute error (failure_threshold is meaningless for FI)
            abs_err = np.abs(pred_damage - true_damage)
            max_err = max(float(abs_err.max()), 1e-6)
            _panel(axes[0, 2], abs_err, f'{field_name} |Error|\n{time_str}', 'Reds', 0.0, max_err, None, '|Error|')
            rate_err = np.abs(pred_rate - true_rate)
            max_rerr = max(float(rate_err.max()), 1e-6)
            _panel(axes[1, 2], rate_err, f'Rate |Error|\n{time_str}', 'Reds', 0.0, max_rerr, None, '|Error|')
        else:
            # For damage [0,1]: col 3 shows failure state
            failure_threshold = 0.95
            true_state = (true_damage > failure_threshold).astype(float)
            pred_state = (pred_damage > failure_threshold).astype(float)
            _panel(axes[0, 2], true_state,   f'True State\n{time_str}',   'RdYlGn_r', 0.0, 1.0, None, '', is_state=True)
            _panel(axes[1, 2], pred_state,   f'Pred State\n{time_str}',   'RdYlGn_r', 0.0, 1.0, None, '', is_state=True)
    
    else:
        # Extended rollout: predictions only (no ground truth)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Predicted damage/FI
        ax = axes[0]
        ax.set_aspect('equal')
        ax.set_axis_off()
        dmg_vmax = max(1.0, float(np.max(pred_damage)))
        mesh_plot = _plot_field_on_mesh(
            ax, mesh, pred_damage, cmap='viridis', vmin=0.0, vmax=dmg_vmax,
        )
        ax.set_title(f'Predicted {field_name}\n{time_str}', fontsize=12, fontweight='bold')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
        cbar.set_label(f'{field_name} [-]')

        # Predicted rate
        ax = axes[1]
        ax.set_aspect('equal')
        ax.set_axis_off()
        mesh_plot = _plot_field_on_mesh(
            ax, mesh, pred_rate, cmap='plasma', norm=rate_norm,
        )
        ax.set_title(f'Predicted {field_name} Rate\n{time_str}', fontsize=12, fontweight='bold')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
        cbar.set_label('Rate [1/s]')

        if is_fi:
            # For FI: show FI threshold exceedance
            ax = axes[2]
            ax.set_aspect('equal')
            ax.set_axis_off()
            fi_exceed = np.clip(pred_damage, 0, None)  # already non-negative
            mesh_plot = _plot_field_on_mesh(
                ax, mesh, fi_exceed, cmap='hot_r', vmin=0.0, vmax=max(3.0, dmg_vmax),
            )
            ax.set_title(f'FI Magnitude\n{time_str}', fontsize=12, fontweight='bold')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
            cbar.set_label('FI [-]')
        else:
            # For damage: show failure state
            failure_threshold = 0.95
            pred_state = (pred_damage > failure_threshold).astype(float)
            ax = axes[2]
            ax.set_aspect('equal')
            ax.set_axis_off()
            mesh_plot = _plot_field_on_mesh(
                ax, mesh, pred_state, cmap='RdYlGn_r', vmin=0.0, vmax=1.0,
            )
            ax.set_title(f'Predicted State\n{time_str}', fontsize=12, fontweight='bold')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Active', 'Failed'])

        fig.suptitle(f'Extended Rollout: {field_name} (No Ground Truth)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_hyperparameters(
    config,
    output_path: Union[str, Path],
    additional_info: Optional[Dict] = None
) -> None:
    """
    Save hyperparameters to JSON file.
    
    Args:
        config: Configuration object with hyperparameters
        output_path: Path to save the file
        additional_info: Additional information to include
    """
    output_path = Path(output_path)
    
    # Extract hyperparameters from config
    hyperparams = {
        'model': {
            'num_layers': config.num_layers,
            'hidden_dim': config.hidden_dim,
            'dropout_rate': config.dropout_rate,
            'attention_freq': config.attention_freq,
        },
        'training': {
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'optimizer': config.optimizer,
            'scheduler': config.scheduler,
            'patience': config.patience,
        },
        'data': {
            'train_ratio': getattr(config, 'train_ratio', 0.7),
            'val_ratio': getattr(config, 'val_ratio', 0.15),
            'test_ratio': getattr(config, 'test_ratio', 0.15),
        },
    }
    
    if additional_info:
        hyperparams['additional'] = additional_info
    
    with open(output_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)


def load_hyperparameters(path: Union[str, Path]) -> Dict:
    """
    Load hyperparameters from JSON file.
    
    Args:
        path: Path to hyperparameters file
        
    Returns:
        Dictionary of hyperparameters
    """
    path = Path(path)
    with open(path, 'r') as f:
        return json.load(f)


def plot_global_attention_analysis(
    mesh_pos: np.ndarray,
    entropy: np.ndarray,
    max_weight: np.ndarray,
    attn_reach: np.ndarray,
    long_range_ratio: np.ndarray,
    damage: np.ndarray,
    received_attn: np.ndarray = None,
    received_max: np.ndarray = None,
    received_entropy: np.ndarray = None,
    received_reach: np.ndarray = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Global Attention Analysis",
    data_template=None,
    show: bool = False,
    ply: int = None,
) -> None:
    """
    Plot global-attention statistics on the mesh.

    For element-level cohesive data, ``ply`` selects which ply's TRUE hex
    footprint quads are drawn (the plies overlap in the XY projection).

    If received_* arrays are provided, produces 2×4 = 8 panels:
      Top row, outgoing (query side): entropy, max weight, reach, long-range ratio
      Bottom row, incoming (key side): total received, max received, entropy, reach

    Otherwise falls back to the original 2×2 outgoing-only layout.

    The damage field is shown as contour lines for spatial reference.
    """
    mesh = create_mesh_projection(
        data_template if data_template is not None else mesh_pos, ply=ply
    )
    pos_2d = mesh['pos_2d']
    dmg = safe_to_numpy(damage)
    # element-level: damage contours restricted to the selected ply
    contour_pos = mesh.get('contour_pos', pos_2d)
    contour_dmg = dmg[mesh['elem_idx']] if 'elem_idx' in mesh else dmg

    has_received = received_attn is not None

    if has_received:
        fig, axes = plt.subplots(2, 4, figsize=(32, 14))
        fields = [
            # Top row: outgoing (query side)
            (entropy,          'viridis',  'Query Entropy',                 None,  None),
            (max_weight,       'hot',      'Query Max Weight',              None,  None),
            (attn_reach,       'plasma',   'Query Reach (exp entropy)',     None,  None),
            (long_range_ratio, 'coolwarm', 'Query Long-Range Ratio',       None,  None),
            # Bottom row: incoming (key side)
            (received_attn,    'YlGnBu',  'Received Attention (col sum)',  None,  None),
            (received_max,     'hot',      'Received Max (from one node)', None,  None),
            (received_entropy, 'viridis',  'Received Entropy',             None,  None),
            (received_reach,   'plasma',   'Received Reach (exp entropy)', None,  None),
        ]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fields = [
            (entropy,          'viridis',  'Attention Entropy',             None,  None),
            (max_weight,       'hot',      'Max Attention Weight',          None,  None),
            (attn_reach,       'plasma',   'Effective Reach (exp entropy)', None,  None),
            (long_range_ratio, 'coolwarm', 'Long-Range Attention Ratio',   None,  None),
        ]

    for ax, (field, cmap, subtitle, vmin, vmax) in zip(axes.flat, fields):
        ax.set_aspect('equal')
        ax.set_axis_off()

        f = safe_to_numpy(field)
        if vmin is None:
            vmin = float(f.min())
        if vmax is None:
            vmax = max(float(f.max()), vmin + 1e-6)

        mesh_plot = _plot_field_on_mesh(ax, mesh, f, cmap=cmap,
                                         vmin=vmin, vmax=vmax)

        # Overlay damage contour lines for spatial reference
        try:
            ax.tricontour(contour_pos[:, 0], contour_pos[:, 1], contour_dmg,
                          levels=[0.3, 0.6, 0.9],
                          colors='white', linewidths=0.8, linestyles='--')
        except Exception:
            pass

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
        ax.set_title(subtitle, fontsize=10, fontweight='bold')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

