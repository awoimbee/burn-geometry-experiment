#!/usr/bin/env python3
"""Run N inferences across the latent space and assemble a GIF of the geometries.

Samples N latent vectors with `burn-mnist generate --count N`, then for each:
  1. Poisson-reconstruct the 100-point cloud (sidecar/reconstruct.py, in-process)
  2. render the mesh to a frame (fixed camera, Lambert shading)

Usage (from the repo root):
    uv run tools/latent_gif.py
    uv run tools/latent_gif.py --n 100 --dist uniform --scale 2.0 --out artifacts/latent_gif.gif
"""

# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "pillow",
#   "pymeshlab",
# ]
# ///

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sidecar"))
import reconstruct  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

LIGHT = np.array([0.4, 0.6, 0.7])
LIGHT /= np.linalg.norm(LIGHT)
BASE_COLOR = np.array([0.75, 0.85, 1.0])


def ensure_binary() -> Path:
    binary = REPO / "target" / "release" / "burn-mnist"
    if not binary.exists():
        print("building release binary...")
        subprocess.run(["cargo", "build", "--release"], cwd=REPO, check=True)
    return binary


def run_batch_inference(binary: Path, n: int, dist: str, scale: float, seed: int):
    """One binary invocation; returns (vtk paths, sampled latents)."""
    cmd = [
        str(binary), "generate",
        "--count", str(n),
        "--dist", dist,
        "--scale", str(scale),
        "--seed", str(seed),
    ]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True, text=True)
    gen_dir = REPO / "artifacts" / "generated"
    latents = np.loadtxt(gen_dir / "latents.csv", delimiter=",", ndmin=2)
    paths = [gen_dir / f"gen_{i:04}.vtk" for i in range(n)]
    return paths, latents


def mesh_arrays(points: np.ndarray, method: str, depth: int, max_tris: int):
    """Return (verts, faces) as numpy copies; safe to GC the MeshSet afterwards."""
    ms, used = reconstruct.reconstruct(points, method, depth)
    if max_tris > 0 and ms.current_mesh().face_number() > max_tris:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=max_tris)
    verts = np.array(ms.current_mesh().vertex_matrix(), dtype=np.float64)
    faces = np.array(ms.current_mesh().face_matrix(), dtype=np.int64)
    del ms
    return verts, faces, used


class Renderer:
    def __init__(self, size: int):
        self.size = size
        self.fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
        self.ax = self.fig.add_subplot(projection="3d")
        self.fig.patch.set_facecolor("#10141a")
        self.ax.set_facecolor("#10141a")
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.view_init(elev=20, azim=-60)
        self.ax.set_axis_off()
        self.collection = None

    def frame(self, verts: np.ndarray, faces: np.ndarray) -> Image.Image:
        tris = verts[faces]
        tris = tris - tris.mean(axis=(0, 1))
        scale = np.abs(tris).max()
        if scale > 0:
            tris = tris / scale * 0.92

        normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        shade = 0.30 + 0.70 * np.abs(normals / norms @ LIGHT)
        colors = np.clip(BASE_COLOR * shade[:, None], 0, 1)

        if self.collection is not None:
            self.collection.remove()
        self.collection = Poly3DCollection(
            tris, facecolors=np.concatenate([colors, np.ones((len(tris), 1))], axis=1),
            edgecolors="none",
        )
        self.ax.add_collection3d(self.collection)
        self.fig.canvas.draw()

        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        return Image.fromarray(buf[:, :, :3].copy())

    def close(self):
        plt.close(self.fig)


def overlay_text(img: Image.Image, params: np.ndarray) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    lines = [
        " ".join(f"{p:+.2f}" for p in params[:4]),
        " ".join(f"{p:+.2f}" for p in params[4:]),
    ]
    for i, line in enumerate(lines):
        draw.text((4, 4 + i * 11), line, fill=(220, 225, 235), font=font)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--n", type=int, default=100, help="number of latent samples (default: 100)")
    ap.add_argument(
        "--dist", choices=["gaussian", "uniform"], default="gaussian",
        help="latent sampling distribution (default: gaussian)",
    )
    ap.add_argument(
        "--scale", type=float, default=1.0,
        help="gaussian std or uniform half-range (default: 1.0)",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    ap.add_argument(
        "--out", type=Path, default=REPO / "artifacts" / "latent_gif.gif",
        help="output GIF path",
    )
    ap.add_argument("--size", type=int, default=320, help="frame size in px (default: 320)")
    ap.add_argument("--duration", type=int, default=120, help="ms per frame (default: 120)")
    ap.add_argument(
        "--method", choices=["auto", "poisson", "hull"], default="auto",
        help="reconstruction method (default: auto)",
    )
    ap.add_argument("--depth", type=int, default=8, help="poisson octree depth (default: 8)")
    ap.add_argument(
        "--max-tris", type=int, default=1500,
        help="decimate meshes to at most this many triangles (default: 1500, 0 = off)",
    )
    args = ap.parse_args()

    binary = ensure_binary()
    paths, latents = run_batch_inference(binary, args.n, args.dist, args.scale, args.seed)
    print(f"generated {args.n} point clouds in artifacts/generated/ "
          f"({args.dist}, scale={args.scale}, seed={args.seed})")

    renderer = Renderer(args.size)
    frames = []
    skipped = 0
    t0 = time.time()

    for i, (path, params) in enumerate(zip(paths, latents)):
        points = reconstruct.read_points_vtk(path)
        try:
            verts, faces, used = mesh_arrays(points, args.method, args.depth, args.max_tris)
        except Exception as e:
            print(f"[{i + 1}/{args.n}] reconstruction failed: {e}", file=sys.stderr)
            skipped += 1
            continue
        if len(faces) == 0:
            print(f"[{i + 1}/{args.n}] degenerate mesh, skipped", file=sys.stderr)
            skipped += 1
            continue
        frames.append(overlay_text(renderer.frame(verts, faces), params))
        print(f"[{i + 1}/{args.n}] {used}: {len(faces)} tris  ({time.time() - t0:.0f}s)")

    renderer.close()
    if not frames:
        raise SystemExit("no frames produced; nothing to write")

    frames = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
    )
    print(f"wrote {args.out} ({len(frames)} frames, {skipped} skipped, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
