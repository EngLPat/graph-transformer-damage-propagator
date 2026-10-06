# -*- coding: utf-8 -*-
"""Abaqus CAE script: 2D structured-quad mesh on a plate with 0..N holes.

Generalization of tools/abaqus_mesh_plus.py (single hole, "+" partition)
for the extrapolation study: no-hole plates, two-hole plates, and long
"stitched" double plates.  Geometry and partition lines are passed via a
JSON file because the hole list is variable-length.

Topology
    Rectangular plate centred on the origin, minus N circular holes.
    Partition lines are FULL-SPAN horizontal (y=const) and vertical
    (x=const) lines, chosen by the caller so that every face containing
    a hole arc has exactly one quarter-arc biting one of its corners,
    the same 5-sided face topology the single-hole "+" partition
    produces, which Abaqus's STRUCTURED quad mesher accepts.

Run as
    abaqus cae noGUI=tools/abaqus_mesh_multi.py -- <params_json> <out_dir>

params_json
    {
      "plate_half_x": 17.0,
      "plate_half_y": 17.0,
      "elem_size":    0.9,
      "holes":  [[cx, cy, r], ...],     # may be empty
      "xcuts":  [x1, x2, ...],          # vertical partition lines
      "ycuts":  [y1, y2, ...]           # horizontal partition lines
    }

Output
    <out_dir>/multi_nodes.txt     "<id>, <x>, <y>"                one per line
    <out_dir>/multi_elements.txt  "<id>, <n1>, <n2>, <n3>, <n4>"  CCW quads

Intended caller: src/preprocessing/generate_extrapolation_geometry.py.
"""
from __future__ import print_function

import json
import os
import sys

from abaqus import mdb
from abaqusConstants import (
    DEFORMABLE_BODY, FREE, QUAD, SIDE1, STANDARD_EXPLICIT,
    STRUCTURED, TWO_D_PLANAR,
)


def parse_args():
    # Abaqus consumes its own leading flags; our 2 user-args land at the END
    # of sys.argv (same trick as abaqus_mesh_plus.py).
    if len(sys.argv) < 3:
        print("Usage: abaqus cae noGUI=abaqus_mesh_multi.py -- "
              "params_json out_dir")
        sys.exit(2)
    return sys.argv[-2], sys.argv[-1]


def main():
    params_path, out_dir = parse_args()
    with open(params_path) as fp:
        p = json.load(fp)

    hx        = float(p["plate_half_x"])
    hy        = float(p["plate_half_y"])
    elem_size = float(p["elem_size"])
    holes     = [(float(h[0]), float(h[1]), float(h[2]))
                 for h in p.get("holes", [])]
    xcuts     = [float(v) for v in p.get("xcuts", [])]
    ycuts     = [float(v) for v in p.get("ycuts", [])]

    # Every hole must be fully inside the plate.
    for (cx, cy, r) in holes:
        if cx + r > hx or cx - r < -hx or cy + r > hy or cy - r < -hy:
            print("ERROR: hole (%.3f, %.3f, r=%.3f) extends outside plate."
                  % (cx, cy, r))
            sys.exit(3)

    sh = 4.0 * max(hx, hy)

    mdb.Model(name="MeshOnly", modelType=STANDARD_EXPLICIT)
    m = mdb.models["MeshOnly"]

    # ---- Part sketch: outer rectangle minus circular holes ----------------
    s = m.ConstrainedSketch(name="plate", sheetSize=sh)
    s.rectangle(point1=(-hx, -hy), point2=(hx, hy))
    for (cx, cy, r) in holes:
        s.CircleByCenterPerimeter(center=(cx, cy), point1=(cx + r, cy))
    part = m.Part(name="Plate", dimensionality=TWO_D_PLANAR,
                  type=DEFORMABLE_BODY)
    part.BaseShell(sketch=s)
    del m.sketches["plate"]

    # ---- Partition: full-span lines at each xcut / ycut --------------------
    if xcuts or ycuts:
        face = part.faces[0]
        trans = part.MakeSketchTransform(sketchPlane=face,
                                         sketchPlaneSide=SIDE1,
                                         origin=(0.0, 0.0, 0.0))
        s2 = m.ConstrainedSketch(name="cuts", sheetSize=sh, transform=trans)
        for yc in ycuts:
            s2.Line(point1=(-hx, yc), point2=(hx, yc))
        for xc in xcuts:
            s2.Line(point1=(xc, -hy), point2=(xc, hy))
        part.PartitionFaceBySketch(faces=part.faces, sketch=s2)
        del m.sketches["cuts"]

    # ---- Mesh controls: structured quad on every sub-face ------------------
    # Fallback to FREE quad if the structured mesher rejects the topology
    # (should not happen with well-chosen cuts, but fail soft with a warning
    # so the caller still gets a mesh to inspect).
    used_fallback = False
    try:
        part.setMeshControls(regions=part.faces, elemShape=QUAD,
                             technique=STRUCTURED)
        part.seedPart(size=elem_size, deviationFactor=0.1, minSizeFactor=0.1)
        part.generateMesh()
    except Exception as e:
        print("WARNING: structured meshing failed (%s); retrying FREE quad."
              % e)
        used_fallback = True
    if used_fallback or len(part.elements) == 0:
        if not used_fallback:
            print("WARNING: structured meshing produced 0 elements; "
                  "retrying FREE quad.")
        part.deleteMesh()
        part.setMeshControls(regions=part.faces, elemShape=QUAD,
                             technique=FREE)
        part.seedPart(size=elem_size, deviationFactor=0.1, minSizeFactor=0.1)
        part.generateMesh()

    # ---- Dump nodes and elements to plain text -----------------------------
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    nodes    = part.nodes
    elements = part.elements
    label_for_index = [n.label for n in nodes]

    n_quad = 0
    n_other = 0
    with open(os.path.join(out_dir, "multi_nodes.txt"), "w") as fp:
        for n in nodes:
            xy = n.coordinates
            fp.write("%d, %.10f, %.10f\n" % (n.label, xy[0], xy[1]))
    with open(os.path.join(out_dir, "multi_elements.txt"), "w") as fp:
        for e in elements:
            conn = e.connectivity
            if len(conn) == 4:
                fp.write("%d, %d, %d, %d, %d\n" % (
                    e.label,
                    label_for_index[conn[0]],
                    label_for_index[conn[1]],
                    label_for_index[conn[2]],
                    label_for_index[conn[3]],
                ))
                n_quad += 1
            else:
                n_other += 1

    print("DONE  nodes=%d  quads=%d  non-quads=%d  fallback=%s  out=%s"
          % (len(nodes), n_quad, n_other, used_fallback, out_dir))


if __name__ == "__main__":
    main()
