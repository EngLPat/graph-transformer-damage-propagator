"""Per-simulation VTU/PVD exporter for Paraview / Blender.

Reads the .inp geometry of the simulation being tested, builds a proper 3-D
hex (+ quad-for-iface) UnstructuredGrid via pyvista, and writes one .vtu per
rollout frame plus a .pvd time-series index. Per-cell scalars attached:

    Static (same every frame):
        ply_id        : ply index (0..n_plies-1); NaN on iface cells
        node_type     : 0 = ply (hex), 1 = interface (quad)
        z_centroid    : through-thickness coordinate

    Time-varying (one value per cell per frame):
        fi_gt         : ground-truth FI at this cell at this time
        fi_pred       : predicted FI
        fi_rate_gt    : ground-truth FI rate dFI/dt
        fi_rate_pred  : predicted FI rate
        damage_gt     : ground-truth damage variable  (only if two-stage)
        damage_pred   : predicted damage variable     (only if two-stage)

The function is called from `_run_testing` inside a try/except so a failure
here never breaks the rest of the test pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _safe_get_array(d: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    arr = d.get(key)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.size == 0:
        return None
    return arr


def _expected_layup_from_data(data_template, n_ply_target: int) -> Optional[List[float]]:
    """Derive the layup angles (per ply, bottom→top) from the dataset's
    static features. Each ply has a unique z layer; the cos/sin encoding
    in data.x[:, 3:5] gives the ply angle modulo 180°.

    Returns a list of angles in degrees per ply (bottom→top), or None
    if the encoding can't be parsed.
    """
    try:
        import math
        x = data_template.x
        if hasattr(x, "cpu"):
            x = x.cpu().numpy()
        x = x[:n_ply_target]
        # z is always taken from mesh_pos, which does not depend on the
        # node-feature layout
        mp = data_template.mesh_pos
        if hasattr(mp, "cpu"):
            mp = mp.cpu().numpy()
        z = np.asarray(mp)[:n_ply_target, 2]
        # angle columns: 3,4 standard layout; 0,1 when positions dropped
        ang_off = 3 if x.shape[1] >= 14 else 0
        # Group rows by unique z layers
        z_uniq = np.unique(np.round(z, 4))
        if len(z_uniq) < 2:
            return None
        angles = []
        for zv in sorted(z_uniq):
            mask = np.abs(z - zv) < 1e-3
            if not mask.any():
                continue
            ca = float(np.mean(x[mask, ang_off]))
            sa = float(np.mean(x[mask, ang_off + 1]))
            angles.append(math.degrees(math.atan2(sa, ca)))
        return angles
    except Exception:
        return None


def _angles_match(a: Sequence[float], b: Sequence[float], tol: float = 5.0) -> bool:
    """Compare two layups in order (bottom→top), modulo 180°."""
    if len(a) != len(b):
        return False
    for ai, bi in zip(a, b):
        d = ((ai - bi + 90) % 180) - 90
        if abs(d) > tol:
            return False
    return True


def _locate_inp_for_sim(data_template, n_ply_target: int) -> Optional[Path]:
    """Find the .inp file that matches this simulation.

    Disambiguates between studies by also checking the ply-layup angles.
    Returns None on failure rather than raising.
    """
    from src.preprocessing.build_cohesive_dataset import parse_cohesive_inp

    project_root = Path(__file__).resolve().parents[3]
    geom_dirs = sorted(project_root.glob("data/raw/*/geometry"))
    if not geom_dirs:
        return None

    expected_angles = _expected_layup_from_data(data_template, n_ply_target)

    # Expected ply mid-z values from the dataset
    expected_z = None
    try:
        x_arr = data_template.x
        if hasattr(x_arr, "cpu"):
            x_arr = x_arr.cpu().numpy()
        expected_z = sorted(np.unique(np.round(x_arr[:n_ply_target, 2], 4)).tolist())
    except Exception:
        pass

    def _candidate_matches(cand: Path) -> Optional[Dict]:
        """Return parsed dict if cand passes all match checks, else None."""
        try:
            pc = parse_cohesive_inp(cand)
        except Exception:
            return None
        if len(pc["ply_elems"]) != n_ply_target:
            return None
        if expected_angles is not None:
            ply_angles = pc.get("ply_angles", {})
            cand_angles = [ply_angles.get(pi, 0.0)
                           for pi in sorted(ply_angles.keys())]
            if not _angles_match(cand_angles, expected_angles):
                return None
        # Z-layer (ply thickness) match
        if expected_z is not None:
            try:
                nodes_c = pc["nodes"]
                ply_elems = pc["ply_elems"]
                cand_z = []
                for first, last in pc["ply_ranges"]:
                    eids = [eid for eid in range(first, last + 1) if eid in ply_elems]
                    if not eids:
                        continue
                    zs = [np.mean([nodes_c[n][2] for n in ply_elems[e]])
                          for e in eids[:3]]
                    cand_z.append(float(np.mean(zs)))
                cand_z_sorted = sorted(cand_z)
                if len(cand_z_sorted) == len(expected_z):
                    diffs = [abs(a - b) for a, b in zip(cand_z_sorted, expected_z)]
                    if max(diffs) > 1e-3:
                        return None
            except Exception:
                pass
        return pc

    # Strategy 1: sim_id filename match (now layup-aware)
    sim_id = getattr(data_template, "sim_id", None)
    if isinstance(sim_id, str) and sim_id:
        for gdir in geom_dirs:
            cand = gdir / f"{sim_id}.inp"
            if cand.exists() and _candidate_matches(cand) is not None:
                return cand

    # Strategy 2: centroid match (layup-aware)
    pos = data_template.mesh_pos
    if hasattr(pos, "cpu"):
        pos = pos.cpu().numpy()
    pos_ply = pos[:n_ply_target]

    best = None
    best_dist = float("inf")
    for gdir in geom_dirs:
        cands = sorted(
            set(gdir.glob("plate_[0-9]*.inp"))
            | set(gdir.glob("cohesive_lam_plate_[0-9]*.inp"))
        )
        for cand in cands:
            pc = _candidate_matches(cand)
            if pc is None:
                continue
            ply_elems = pc["ply_elems"]
            nodes_c = pc["nodes"]
            sorted_eids = sorted(ply_elems.keys())
            cens = np.array([
                np.mean([nodes_c[n] for n in ply_elems[eid][:4]], axis=0)
                for eid in sorted_eids
            ])
            dist = float(np.abs(cens[:, :2] - pos_ply[:, :2]).mean())
            if dist < best_dist:
                best_dist = dist
                best = cand

    if best is not None and best_dist < 0.01:
        return best
    return None


def _build_unstructured_grid(parsed_coh: Dict, n_ply_target: int,
                              include_iface: bool):
    """Build a pyvista UnstructuredGrid (hexes for plies, quads for iface).

    Cell ordering matches the dataset row ordering produced by
    `build_node_features` (plies first, sorted by eid; then iface, sorted by
    eid). This means per-cell scalars can be passed by simple row-slicing.
    """
    import pyvista as pv

    nodes_dict = parsed_coh["nodes"]
    ply_elems  = parsed_coh["ply_elems"]
    iface_elems = parsed_coh["iface_elems"]
    ply_angles = parsed_coh["ply_angles"]
    ply_ranges = parsed_coh["ply_ranges"]

    sorted_nids = sorted(nodes_dict.keys())
    nid_to_idx = {nid: j for j, nid in enumerate(sorted_nids)}
    points = np.array([nodes_dict[nid] for nid in sorted_nids],
                      dtype=np.float64)

    # ── Map ply eid → ply index (for ply_id scalar) ──
    eid_to_ply = {}
    for pi, (first, last) in enumerate(ply_ranges):
        for eid in range(first, last + 1):
            eid_to_ply[eid] = pi

    cells_flat: List[int] = []
    cell_types: List[int] = []
    ply_id_per_cell: List[float] = []
    node_type_per_cell: List[int] = []
    z_centroid_per_cell: List[float] = []
    angle_per_cell: List[float] = []

    sorted_ply_eids = sorted(ply_elems.keys())
    for eid in sorted_ply_eids:
        n8 = ply_elems[eid]
        cells_flat.extend([8] + [nid_to_idx[n] for n in n8])
        cell_types.append(pv.CellType.HEXAHEDRON)
        ply_idx = eid_to_ply.get(eid, -1)
        ply_id_per_cell.append(float(ply_idx))
        node_type_per_cell.append(0)
        z_centroid_per_cell.append(
            float(np.mean([nodes_dict[n][2] for n in n8])))
        angle_per_cell.append(float(ply_angles.get(ply_idx, 0.0)))

    sorted_iface_eids: List[int] = []
    if include_iface and iface_elems:
        sorted_iface_eids = sorted(iface_elems.keys())
        for eid in sorted_iface_eids:
            n8 = iface_elems[eid]
            # COH3D8: render the bottom face as a quad (zero thickness)
            cells_flat.extend([4] + [nid_to_idx[n] for n in n8[:4]])
            cell_types.append(pv.CellType.QUAD)
            ply_id_per_cell.append(float("nan"))
            node_type_per_cell.append(1)
            z_centroid_per_cell.append(
                float(np.mean([nodes_dict[n][2] for n in n8[:4]])))
            angle_per_cell.append(float("nan"))

    grid = pv.UnstructuredGrid(
        np.array(cells_flat, dtype=np.int64),
        np.array(cell_types, dtype=np.uint8),
        points,
    )
    grid.cell_data["ply_id"]     = np.array(ply_id_per_cell, dtype=np.float32)
    grid.cell_data["node_type"]  = np.array(node_type_per_cell, dtype=np.int8)
    grid.cell_data["z_centroid"] = np.array(z_centroid_per_cell, dtype=np.float32)
    grid.cell_data["ply_angle_deg"] = np.array(angle_per_cell, dtype=np.float32)

    n_cells_total = len(cell_types)
    return grid, n_cells_total, len(sorted_ply_eids), len(sorted_iface_eids)


def export_sim_to_vtu(
    sim_id: int,
    frame_indices: Sequence[int],
    rollout_results: Dict[str, Any],
    data_template,
    output_dir: Path,
    is_two_stage: bool = False,
    times: Optional[Sequence[float]] = None,
) -> Optional[Path]:
    """Write per-frame VTU + a .pvd index for one simulation.

    Returns the path to the .pvd file, or None on failure.
    """
    import pyvista as pv  # noqa: F401 (fail fast if missing)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset row layout
    n_ply = int(getattr(data_template, "n_ply_nodes", 0)) or 0
    n_iface = int(getattr(data_template, "n_iface_nodes", 0)) or 0
    if n_ply == 0:
        # Non-cohesive fallback: treat all nodes as ply
        pos = data_template.mesh_pos
        n_ply = (pos.shape[0] if hasattr(pos, "shape") else len(pos))
    include_iface = n_iface > 0

    inp_path = _locate_inp_for_sim(data_template, n_ply_target=n_ply)
    if inp_path is None:
        print(f"  [vtk_export] skipped sim {sim_id}: could not locate .inp file")
        return None

    from src.preprocessing.build_cohesive_dataset import parse_cohesive_inp
    parsed = parse_cohesive_inp(inp_path)
    grid, n_cells, n_ply_cells, n_iface_cells = _build_unstructured_grid(
        parsed, n_ply_target=n_ply, include_iface=include_iface,
    )

    # Rollout arrays, shape [N_frames_total, N_nodes]
    pred_fi    = _safe_get_array(rollout_results, "predicted_damage")
    gt_fi      = _safe_get_array(rollout_results, "ground_truth_damage")
    pred_rate  = _safe_get_array(rollout_results, "predicted_rates")
    gt_rate    = _safe_get_array(rollout_results, "ground_truth_rates")
    pred_dmg   = (_safe_get_array(rollout_results, "predicted_matrix_damage")
                  if is_two_stage else None)
    gt_dmg     = (_safe_get_array(rollout_results, "ground_truth_matrix_damage")
                  if is_two_stage else None)

    all_times = _safe_get_array(rollout_results, "times")

    pvd_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    n_written = 0
    for local_i, frame_idx in enumerate(frame_indices):
        t = (float(all_times[frame_idx])
             if all_times is not None and frame_idx < len(all_times)
             else local_i * 0.01)
        per_frame = grid.copy(deep=False)

        def _attach(name: str, arr: Optional[np.ndarray]):
            if arr is None:
                return
            if frame_idx >= len(arr):
                return
            v = np.asarray(arr[frame_idx]).reshape(-1)
            # Map rows → cells: ply cells share order with ply rows;
            # iface cells share order with iface rows (which are appended
            # after the ply rows in the dataset).
            out = np.full(n_cells, np.nan, dtype=np.float32)
            n_take_ply = min(len(v), n_ply_cells)
            out[:n_take_ply] = v[:n_take_ply]
            if include_iface and len(v) > n_ply:
                n_take_if = min(len(v) - n_ply, n_iface_cells)
                out[n_ply_cells:n_ply_cells + n_take_if] = v[n_ply:n_ply + n_take_if]
            per_frame.cell_data[name] = out

        _attach("fi_pred",       pred_fi)
        _attach("fi_gt",         gt_fi)
        _attach("fi_rate_pred",  pred_rate)
        _attach("fi_rate_gt",    gt_rate)
        if is_two_stage:
            _attach("damage_pred", pred_dmg)
            _attach("damage_gt",   gt_dmg)

        per_frame.field_data["time_s"] = [t]
        vtu_name = f"sim_{sim_id:03d}_frame_{local_i:04d}.vtu"
        per_frame.save(output_dir / vtu_name)
        pvd_lines.append(
            f'    <DataSet timestep="{t:.6f}" group="" part="0" file="{vtu_name}"/>')
        n_written += 1

    pvd_lines.extend(['  </Collection>', '</VTKFile>', ''])
    pvd_path = output_dir / f"sim_{sim_id:03d}.pvd"
    pvd_path.write_text("\n".join(pvd_lines))

    print(f"  [vtk_export] sim {sim_id}: {n_written} frames "
          f"({n_ply_cells} hex plies + {n_iface_cells} quad iface) "
          f"→ {pvd_path}")
    return pvd_path
