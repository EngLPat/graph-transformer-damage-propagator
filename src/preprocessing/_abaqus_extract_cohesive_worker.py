# -*- coding: utf-8 -*-
"""Abaqus worker: extract SDV_FI_1_Matrix + SDV_DMG_1 (ply) and SDEG (interface).

Called by build_cohesive_dataset.py via subprocess::

    abaqus python src/preprocessing/_abaqus_extract_cohesive_worker.py \\
        <odb_path> <out_npz> <n_ply_elems>

Extracts three fields at integration-point / centroidal level:
  - SDV_FI_1_Matrix : matrix cracking failure index (unbounded, >=0), C3D8R ply elements
  - SDV_DMG_1       : matrix damage variable [0,1], C3D8R ply elements
  - SDEG            : scalar damage variable [0,1], COH3D8 interface elements

The USDFLD subroutine (usdfld3d.f90) writes named SDVs (SDV_DMG_1, SDV_FI_1_Matrix, ...)
rather than the generic SDV1/SDV2/... numbering, so field names must match exactly.

Element ordering is preserved from the ODB, aligned by element label.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""
from __future__ import print_function

import sys
import traceback

import numpy as np
import odbAccess

PLY_FI_FIELD   = "SDV_FI_1_Matrix"  # matrix failure index from USDFLD subroutine
PLY_DMG_FIELD  = "SDV_DMG_1"        # matrix damage variable from USDFLD subroutine
IFACE_FIELD    = "SDEG"             # cohesive damage from COH3D8 elements
IFACE_FI_FIELD = "QUADSCRT"         # cohesive QUADS criterion (non-negative interface FI)

# Cache version: must match CACHE_VERSION in build_cohesive_dataset.py.
# Bump both together when the NPZ format changes.
# v2: added ply_matrix_dmg (SDV_DMG_1)
# v3: added iface_fi (QUADSCRT) for interface FI-rate target
CACHE_VERSION = 3


def extract_cohesive_odb(odb_path, n_ply_elems):
    """Extract SDV_FI_1_Matrix, SDV_DMG_1, and SDEG from a cohesive laminate ODB.

    Parameters
    ----------
    odb_path : str
    n_ply_elems : int
        Unused; kept for API compatibility. Element split is done by field
        availability (SDV_FI_1_Matrix for plies, SDEG for interfaces).

    Returns
    -------
    dict with keys:
        ply_labels       int64   [P]
        ply_damage       float32 [T, P]  -- SDV_FI_1_Matrix (unclamped FI)
        ply_matrix_dmg   float32 [T, P]  -- SDV_DMG_1 (matrix damage [0,1])
        iface_labels     int64   [I]
        iface_damage     float32 [T, I]  -- SDEG
        frame_times      float64 [T]
    """
    odb = odbAccess.openOdb(str(odb_path), readOnly=True)
    try:
        step = odb.steps[list(odb.steps.keys())[-1]]
        frames = step.frames
        n_frames = len(frames)

        # ── First pass: collect element labels from first frame that has the fields
        ply_labels_set = set()
        iface_labels_set = set()

        for frame in frames:
            fo = frame.fieldOutputs
            if PLY_FI_FIELD in fo:
                for val in fo[PLY_FI_FIELD].values:
                    eid = getattr(val, "elementLabel", None)
                    if eid is not None:
                        ply_labels_set.add(eid)
            if IFACE_FIELD in fo:
                for val in fo[IFACE_FIELD].values:
                    eid = getattr(val, "elementLabel", None)
                    if eid is not None:
                        iface_labels_set.add(eid)
            if ply_labels_set or iface_labels_set:
                break

        ply_labels   = np.array(sorted(ply_labels_set),   dtype=np.int64)
        iface_labels = np.array(sorted(iface_labels_set), dtype=np.int64)
        ply_idx   = {lab: i for i, lab in enumerate(ply_labels)}
        iface_idx = {lab: i for i, lab in enumerate(iface_labels)}

        # Hard validation: required fields must be found
        last_fo = frames[-1].fieldOutputs
        available = sorted(last_fo.keys())
        if len(ply_labels) == 0:
            raise RuntimeError(
                "Field '{}' not found in ODB '{}'.\n"
                "Available field outputs in last frame:\n  {}\n"
                "Check PLY_FI_FIELD constant in this worker.".format(
                    PLY_FI_FIELD, odb_path, "\n  ".join(available))
            )
        if len(iface_labels) == 0 and IFACE_FIELD in last_fo:
            raise RuntimeError(
                "Field '{}' not found in ODB '{}'.\n"
                "Available field outputs in last frame:\n  {}\n"
                "Check IFACE_FIELD constant in this worker.".format(
                    IFACE_FIELD, odb_path, "\n  ".join(available))
            )

        has_dmg      = PLY_DMG_FIELD in last_fo
        has_iface_fi = IFACE_FI_FIELD in last_fo
        if not has_dmg:
            print("  WARNING: '{}' not found; ply_matrix_dmg will be zeros".format(
                PLY_DMG_FIELD))
        if len(iface_labels) > 0 and not has_iface_fi:
            print("  WARNING: '{}' not found on interfaces; iface_fi will be zeros. "
                  "Regenerate .inp with QUADSCRT in Element Output for a v2 dataset."
                  .format(IFACE_FI_FIELD))

        P = len(ply_labels)
        I = len(iface_labels)
        print("  Found {} ply elements ({}+{}), {} interface elements ({}+{})".format(
            P, PLY_FI_FIELD,
            PLY_DMG_FIELD if has_dmg else "MISSING",
            I, IFACE_FIELD,
            IFACE_FI_FIELD if has_iface_fi else "MISSING"))

        ply_damage     = np.zeros((n_frames, P), dtype=np.float32)  # FI
        ply_matrix_dmg = np.zeros((n_frames, P), dtype=np.float32)  # damage
        iface_damage   = np.zeros((n_frames, I), dtype=np.float32)  # SDEG
        iface_fi       = np.zeros((n_frames, I), dtype=np.float32)  # QUADSCRT
        frame_times    = np.zeros(n_frames, dtype=np.float64)

        # ── Second pass: extract frame by frame ──────────────────────────
        for t, frame in enumerate(frames):
            frame_times[t] = frame.frameValue
            fo = frame.fieldOutputs

            if PLY_FI_FIELD in fo and P > 0:
                for val in fo[PLY_FI_FIELD].values:
                    eid = getattr(val, "elementLabel", None)
                    if eid in ply_idx:
                        raw = val.data
                        ply_damage[t, ply_idx[eid]] = (
                            float(raw[0]) if hasattr(raw, "__len__") else float(raw)
                        )

            if has_dmg and PLY_DMG_FIELD in fo and P > 0:
                for val in fo[PLY_DMG_FIELD].values:
                    eid = getattr(val, "elementLabel", None)
                    if eid in ply_idx:
                        raw = val.data
                        ply_matrix_dmg[t, ply_idx[eid]] = (
                            float(raw[0]) if hasattr(raw, "__len__") else float(raw)
                        )

            if IFACE_FIELD in fo and I > 0:
                for val in fo[IFACE_FIELD].values:
                    eid = getattr(val, "elementLabel", None)
                    if eid in iface_idx:
                        raw = val.data
                        iface_damage[t, iface_idx[eid]] = (
                            float(raw[0]) if hasattr(raw, "__len__") else float(raw)
                        )

            if has_iface_fi and IFACE_FI_FIELD in fo and I > 0:
                for val in fo[IFACE_FI_FIELD].values:
                    eid = getattr(val, "elementLabel", None)
                    if eid in iface_idx:
                        raw = val.data
                        iface_fi[t, iface_idx[eid]] = (
                            float(raw[0]) if hasattr(raw, "__len__") else float(raw)
                        )

        ply_damage     = np.clip(ply_damage,     0.0, None)  # FI is non-negative but unbounded above 1
        ply_matrix_dmg = np.clip(ply_matrix_dmg, 0.0, 1.0)   # damage ∈ [0,1]
        iface_damage   = np.clip(iface_damage,   0.0, 1.0)
        iface_fi       = np.clip(iface_fi,       0.0, None)  # QUADSCRT >= 0 (can exceed 1)

        return {
            "ply_labels":     ply_labels,
            "ply_damage":     ply_damage,       # SDV_FI_1_Matrix
            "ply_matrix_dmg": ply_matrix_dmg,   # SDV_DMG_1
            "iface_labels":   iface_labels,
            "iface_damage":   iface_damage,     # SDEG
            "iface_fi":       iface_fi,         # QUADSCRT (v3)
            "frame_times":    frame_times,
            "_cache_version": np.array(CACHE_VERSION, dtype=np.int64),
        }
    finally:
        odb.close()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: abaqus python _abaqus_extract_cohesive_worker.py "
              "<odb_path> <out_npz> <n_ply_elems>")
        sys.exit(1)

    odb_path    = sys.argv[1]
    out_npz     = sys.argv[2]
    n_ply_elems = int(sys.argv[3])

    try:
        result = extract_cohesive_odb(odb_path, n_ply_elems)
        np.savez_compressed(out_npz, **result)
        print("  Saved: {}".format(out_npz))
        print("  Frames: {}".format(len(result["frame_times"])))
        P = result["ply_damage"].shape[1]
        I = result["iface_damage"].shape[1]
        print("  Ply FI max: {:.4f}".format(
            result["ply_damage"].max() if P > 0 else 0.0))
        print("  Ply DMG max: {:.4f}".format(
            result["ply_matrix_dmg"].max() if P > 0 else 0.0))
        print("  Iface damage max: {:.4f}".format(
            result["iface_damage"].max() if I > 0 else 0.0))
        print("  Iface FI max: {:.4f}".format(
            result["iface_fi"].max() if I > 0 else 0.0))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
