#!/usr/bin/env python3
"""Regenerate dataset/{train,test} as clean binary STL meshes using VTK.

Each mesh is watertight and free of degenerate triangles; every written file
is validated (binary layout, zero-area triangles, closed 2-manifold) before
the script exits.
"""

import math
import struct
import sys
from collections import Counter
from pathlib import Path

from vtk import (
    vtkCellArray,
    vtkCleanPolyData,
    vtkConeSource,
    vtkCubeSource,
    vtkCylinderSource,
    vtkPoints,
    vtkPolyData,
    vtkSTLWriter,
    vtkSphereSource,
    vtkTransform,
    vtkTransformFilter,
    vtkTriangleFilter,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


def as_polydata(src) -> vtkPolyData:
    """Run `src`, weld points, drop degenerate cells, and triangulate."""
    src.Update()
    clean = vtkCleanPolyData()
    clean.SetInputData(src.GetOutput())
    clean.Update()
    tri = vtkTriangleFilter()
    tri.SetInputData(clean.GetOutput())
    tri.Update()
    return tri.GetOutput()


def sphere(radius: float) -> vtkPolyData:
    f = vtkSphereSource()
    f.SetRadius(radius)
    f.SetThetaResolution(48)
    f.SetPhiResolution(48)
    return as_polydata(f)


def ellipsoid(rx: float, ry: float, rz: float) -> vtkPolyData:
    s = vtkSphereSource()
    s.SetRadius(1.0)
    s.SetThetaResolution(48)
    s.SetPhiResolution(48)
    t = vtkTransform()
    t.Scale(rx, ry, rz)
    tf = vtkTransformFilter()
    tf.SetInputConnection(s.GetOutputPort())
    tf.SetTransform(t)
    return as_polydata(tf)


def box(lx: float, ly: float, lz: float) -> vtkPolyData:
    c = vtkCubeSource()
    c.SetXLength(lx)
    c.SetYLength(ly)
    c.SetZLength(lz)
    return as_polydata(c)


def cylinder(radius: float, height: float, res: int = 48) -> vtkPolyData:
    c = vtkCylinderSource()
    c.SetRadius(radius)
    c.SetHeight(height)
    c.SetResolution(res)
    c.SetCapping(True)
    return as_polydata(c)


def cone(radius: float, height: float, res: int = 48) -> vtkPolyData:
    c = vtkConeSource()
    c.SetRadius(radius)
    c.SetHeight(height)
    c.SetResolution(res)
    c.SetCapping(True)
    return as_polydata(c)


def capsule(radius: float, height: float, res: int = 32) -> vtkPolyData:
    c = vtkCylinderSource()
    c.SetRadius(radius)
    c.SetHeight(height)
    c.SetResolution(res)
    c.SetCapping(True)
    c.SetCapsuleCap(True)
    return as_polydata(c)


def mesh_from_points_cells(points, cells) -> vtkPolyData:
    pts = vtkPoints()
    for p in points:
        pts.InsertNextPoint(*p)
    polys = vtkCellArray()
    for c in cells:
        polys.InsertNextCell(len(c))
        for idx in c:
            polys.InsertCellPoint(idx)
    poly = vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(polys)
    return poly


def torus(major: float, minor: float, nu: int = 48, nv: int = 32) -> vtkPolyData:
    pts, cells = [], []
    for i in range(nu):
        u = 2 * math.pi * i / nu
        for j in range(nv):
            v = 2 * math.pi * j / nv
            rr = major + minor * math.cos(v)
            pts.append((rr * math.cos(u), rr * math.sin(u), minor * math.sin(v)))
    for i in range(nu):
        i2 = (i + 1) % nu
        for j in range(nv):
            j2 = (j + 1) % nv
            a = i * nv + j
            cells.append((a, i2 * nv + j, i2 * nv + j2, a + 1))
    return mesh_from_points_cells(pts, cells)


def lshape() -> vtkPolyData:
    # CCW L cross-section, extruded over z in [-0.4, 0.4]
    cross = [(-1.0, -1.0), (1.0, -1.0), (1.0, -0.2), (-0.2, -0.2), (-0.2, 1.0), (-1.0, 1.0)]
    z = 0.4
    pts = [(x, y, -z) for x, y in cross] + [(x, y, z) for x, y in cross]
    n = len(cross)
    cells = []
    for i in range(n - 1, 1, -1):  # bottom cap, normal -z
        cells.append((i, i - 1, 0))
    for i in range(1, n - 1):  # top cap, normal +z
        cells.append((n, n + i, n + i + 1))
    for i in range(n):  # side quads, outward normals
        j = (i + 1) % n
        cells.append((i, j, n + j))
        cells.append((i, n + j, n + i))
    return mesh_from_points_cells(pts, cells)


TRAIN_SHAPES: dict[str, callable] = {
    "sphere_1": lambda: sphere(0.9),
    "sphere_2": lambda: sphere(0.55),
    "ellipsoid_a": lambda: ellipsoid(1.3, 0.7, 0.5),
    "ellipsoid_b": lambda: ellipsoid(0.55, 1.25, 0.8),
    "box_cubic": lambda: box(1.0, 1.0, 1.0),
    "box_tall": lambda: box(0.7, 1.6, 0.7),
    "box_wide": lambda: box(1.8, 0.8, 0.7),
    "cylinder_1": lambda: cylinder(0.6, 1.2),
    "cylinder_tall": lambda: cylinder(0.35, 1.7),
    "cone": lambda: cone(0.7, 1.3),
    "torus": lambda: torus(0.65, 0.3),
    "capsule": lambda: capsule(0.35, 0.9),
    "pyramid": lambda: cone(0.8, 1.2, res=4),
    "lshape": lshape,
}

TEST_SHAPES: dict[str, callable] = {
    "sphere_test": lambda: sphere(0.8),
    "box_test": lambda: box(1.3, 0.9, 1.5),
    "cylinder_test": lambda: cylinder(0.5, 1.4),
    "torus_test": lambda: torus(0.75, 0.25),
}


def write_binary_stl(poly: vtkPolyData, path: Path) -> None:
    writer = vtkSTLWriter()
    writer.SetInputData(poly)
    writer.SetFileName(str(path))
    writer.SetFileTypeToBinary()
    writer.Write()


def validate_stl(path: Path) -> tuple[int, tuple[float, float, float]]:
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * n:
        raise ValueError("bad binary STL size")
    edges: Counter = Counter()
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for i in range(n):
        (_nx, _ny, _nz, a, b, c, d, e, f, g, h, k, _attr) = struct.unpack_from(
            "<12fH", data, 84 + 50 * i
        )
        va, vb, vc = (a, b, c), (d, e, f), (g, h, k)
        ux, uy, uz = vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]
        wx, wy, wz = vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]
        cx, cy, cz = uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx
        if cx * cx + cy * cy + cz * cz < 1e-20:
            raise ValueError(f"degenerate triangle at index {i}")
        for u, v in ((va, vb), (vb, vc), (vc, va)):
            edges[(u, v)] += 1
        for p in (va, vb, vc):
            for axis, val in enumerate(p):
                lo[axis] = min(lo[axis], val)
                hi[axis] = max(hi[axis], val)
    bad = [e for e, c in edges.items() if c != 1]
    if bad:
        raise ValueError(f"not a closed 2-manifold ({len(bad)} bad directed edges)")
    extent = tuple(round(hi[i] - lo[i], 3) for i in range(3))
    return n, extent


def main() -> None:
    for split in ("train", "test"):
        (DATASET / split).mkdir(parents=True, exist_ok=True)

    failures = 0
    for split, shapes in (("train", TRAIN_SHAPES), ("test", TEST_SHAPES)):
        print(f"{split}:")
        for name, make in shapes.items():
            path = DATASET / split / f"{name}.stl"
            try:
                poly = make()
                if poly.GetNumberOfCells() == 0:
                    raise ValueError("mesh has no cells")
                write_binary_stl(poly, path)
                ntri, extent = validate_stl(path)
                print(f"  {path.name}: {ntri} tris, {extent[0]} x {extent[1]} x {extent[2]}")
            except Exception as exc:
                failures += 1
                print(f"  {path.name}: FAILED ({exc})", file=sys.stderr)

    if failures:
        sys.exit(f"{failures} mesh(es) failed validation")
    total = len(TRAIN_SHAPES) + len(TEST_SHAPES)
    print(f"Wrote {total} meshes to {DATASET}")


if __name__ == "__main__":
    main()