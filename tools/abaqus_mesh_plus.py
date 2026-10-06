# -*- coding: utf-8 -*-
"""Abaqus CAE script: generate a 2D structured-quad mesh on a plate-with-hole.

Topology
    Rectangular plate with one circular hole + a "+" partition through the
    hole centre.  No other partitions.  Each of the 4 resulting sub-faces is
    meshed with Abaqus's STRUCTURED quad-dominated algorithm at a single
    global seed size.

Run as
    abaqus cae noGUI=tools/abaqus_mesh_plus.py -- \
        <plate_half_x> <plate_half_y> <hole_r> <hole_cx> <hole_cy> \
        <elem_size> <out_dir>

Output
    <out_dir>/plus_nodes.txt     "<id>, <x>, <y>"           one per line
    <out_dir>/plus_elements.txt  "<id>, <n1>, <n2>, <n3>, <n4>"  CCW

Intended caller: src/preprocessing/generate_cohesive_laminate_uniform.py,
which shells out, reads the two files back, and feeds the 2D mesh into the
existing 3D extrusion / cohesive / .inp writer pipeline.
"""
from __future__ import print_function

import os
import sys

from abaqus import mdb
from abaqusConstants import (
    DEFORMABLE_BODY, QUAD, SIDE1, STANDARD_EXPLICIT,
    STRUCTURED, TWO_D_PLANAR,
)


def parse_args():
    # Abaqus consumes the leading flags ('-cae', '-noGUI=...', '-tmpdir', etc.)
    # and our 7 user-args land at the END of sys.argv.  No reliable separator,
    # so take the last 7 unconditionally.
    if len(sys.argv) < 8:
        print("Usage: abaqus cae noGUI=abaqus_mesh_plus.py -- "
              "plate_half_x plate_half_y hole_r hole_cx hole_cy "
              "elem_size out_dir")
        sys.exit(2)
    a = sys.argv[-7:]
    return (float(a[0]), float(a[1]), float(a[2]),
            float(a[3]), float(a[4]), float(a[5]), a[6])


def main():
    (plate_half_x, plate_half_y, hole_r,
     hole_cx, hole_cy, elem_size, out_dir) = parse_args()

    # Plate must fully contain the hole.
    if hole_cx + hole_r > plate_half_x or hole_cx - hole_r < -plate_half_x \
       or hole_cy + hole_r > plate_half_y or hole_cy - hole_r < -plate_half_y:
        print("ERROR: hole extends outside plate.")
        sys.exit(3)

    # Sheet size large enough for sketcher
    sh = 4.0 * max(plate_half_x, plate_half_y)

    mdb.Model(name="MeshOnly", modelType=STANDARD_EXPLICIT)
    m = mdb.models["MeshOnly"]

    # ---- Part sketch: outer rectangle minus circular hole -----------------
    s = m.ConstrainedSketch(name="plate", sheetSize=sh)
    s.rectangle(point1=(-plate_half_x, -plate_half_y),
                point2=( plate_half_x,  plate_half_y))
    s.CircleByCenterPerimeter(center=(hole_cx, hole_cy),
                              point1=(hole_cx + hole_r, hole_cy))
    part = m.Part(name="Plate", dimensionality=TWO_D_PLANAR,
                  type=DEFORMABLE_BODY)
    part.BaseShell(sketch=s)
    del m.sketches["plate"]

    # ---- "+" partition: one horizontal + one vertical line through hole --
    face = part.faces[0]
    trans = part.MakeSketchTransform(sketchPlane=face,
                                     sketchPlaneSide=SIDE1,
                                     origin=(0.0, 0.0, 0.0))
    s2 = m.ConstrainedSketch(name="plus", sheetSize=sh, transform=trans)
    s2.Line(point1=(-plate_half_x, hole_cy),
            point2=( plate_half_x, hole_cy))
    s2.Line(point1=(hole_cx, -plate_half_y),
            point2=(hole_cx,  plate_half_y))
    part.PartitionFaceBySketch(faces=part.faces, sketch=s2)
    del m.sketches["plus"]

    # ---- Mesh controls: structured quad on every sub-face -----------------
    part.setMeshControls(regions=part.faces, elemShape=QUAD,
                         technique=STRUCTURED)

    # ---- Global seed + generate mesh -------------------------------------
    part.seedPart(size=elem_size, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()

    # ---- Dump nodes and elements to plain text ---------------------------
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    nodes    = part.nodes
    elements = part.elements

    # Map node sequence-index → label (Abaqus connectivity returns indices)
    label_for_index = [n.label for n in nodes]

    n_quad = 0
    n_other = 0
    with open(os.path.join(out_dir, "plus_nodes.txt"), "w") as fp:
        for n in nodes:
            xy = n.coordinates
            fp.write("%d, %.10f, %.10f\n" % (n.label, xy[0], xy[1]))
    with open(os.path.join(out_dir, "plus_elements.txt"), "w") as fp:
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

    print("DONE  nodes=%d  quads=%d  non-quads=%d  out=%s"
          % (len(nodes), n_quad, n_other, out_dir))


if __name__ == "__main__":
    main()
