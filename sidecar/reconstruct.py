#!/usr/bin/env python3
"""Reconstruct a triangle mesh from the point cloud written by the Rust binary.

Reads the legacy-ASCII VTK point cloud produced by `cargo run -- generate`
(or any .ply/.xyz point cloud), runs Poisson surface reconstruction, and
writes a triangle mesh (STL/OBJ/PLY).

Usage (from the repo root):
    uv run --directory sidecar python reconstruct.py artifacts/out.vtk
    uv run --directory sidecar python reconstruct.py artifacts/out.vtk -o mesh.obj --method hull
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def read_points_vtk(path: Path) -> np.ndarray:
    """Parse the simple legacy-ASCII VTK point cloud written by src/inference.rs."""
    with path.open() as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == "POINTS" and len(parts) >= 2:
                n = int(parts[1])
                coords = []
                for _ in range(n):
                    line = f.readline()
                    if not line:
                        raise ValueError(f"unexpected end of file in {path}")
                    coords.append([float(x) for x in line.split()])
                return np.asarray(coords, dtype=np.float64)
    raise ValueError(f"no POINTS section found in {path}")


def read_points(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run `cargo run -- generate` first")
    if path.suffix.lower() == ".vtk":
        points = read_points_vtk(path)
    else:
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.load_file(str(path))
        points = np.asarray(ms.current_mesh().vertex_matrix())
    if len(points) == 0:
        raise ValueError(f"no points in {path}")
    return points


def poisson_mesh(points: np.ndarray, depth: int):
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=points))
    ms.compute_normal_for_point_clouds(k=15)
    ms.generate_surface_reconstruction_screened_poisson(depth=depth)
    return ms


def hull_mesh(points: np.ndarray):
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=points))
    ms.generate_convex_hull()
    return ms


def reconstruct(points: np.ndarray, method: str, depth: int):
    if method in ("auto", "poisson"):
        ms = poisson_mesh(points, depth)
        n_tri = ms.current_mesh().face_number()
        if n_tri >= 10:
            return ms, "poisson"
        if method == "poisson":
            print(
                f"warning: poisson produced only {n_tri} triangles", file=sys.stderr
            )
            return ms, "poisson"
        print(
            f"poisson produced a degenerate mesh ({n_tri} triangles); "
            "falling back to convex hull",
            file=sys.stderr,
        )
    return hull_mesh(points), "hull"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("input", type=Path, help="point cloud file (.vtk or .ply)")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output mesh path (default: <input stem>_mesh.stl)",
    )
    ap.add_argument(
        "--method",
        choices=["auto", "poisson", "hull"],
        default="auto",
        help="reconstruction method (default: auto = poisson, fall back to hull)",
    )
    ap.add_argument(
        "--depth",
        type=int,
        default=8,
        help="poisson octree depth; higher = finer detail, slower (default: 8)",
    )
    args = ap.parse_args()

    points = read_points(args.input)
    print(f"read {len(points)} points from {args.input}")

    ms, method = reconstruct(points, args.method, args.depth)
    mesh = ms.current_mesh()
    if mesh.vertex_number() == 0 or mesh.face_number() == 0:
        raise SystemExit(f"{method} produced an empty mesh; nothing to write")
    print(f"{method}: {mesh.vertex_number()} vertices, {mesh.face_number()} triangles")

    out = args.output or args.input.with_name(f"{args.input.stem}_mesh.stl")
    fmt = out.suffix.lstrip(".").lower()
    if fmt not in ("stl", "obj", "ply"):
        ap.error(f"unsupported output format '.{fmt}' (use stl, obj, or ply)")
    ms.save_current_mesh(str(out))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
