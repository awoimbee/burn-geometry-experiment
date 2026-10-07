# burn-mnist

burn-mnist is a simple MNIST classifier implemented in Rust Burn deep learning library.


TODO after training:
- Encode the dataset
- Compute per-dimension min/max or std
- Use this to define a valid latent range for sampling

## Tooling

The geometry tooling is written in Rust and drives
[VTK](https://vtk.org/) directly through the [`vtk`](../rvtk) bindings; it
replaces the earlier Python (`vtk` / `pymeshlab`) scripts.

| Binary | Replaces | What it does |
| --- | --- | --- |
| `gen_dataset` | `tools/gen_dataset.py` | Builds `dataset/{train,test}` as clean, watertight binary STL meshes with `vtkSphereSource`, `vtkConeSource`, `vtkTransformFilter`, `vtkCleanPolyData`, `vtkTriangleFilter`, `vtkSTLWriter`, … and validates every file. |
| `reconstruct` | `sidecar/reconstruct.py` | Turns a point cloud into a triangle mesh (`vtkSurfaceReconstructionFilter` + `vtkContourFilter` + `vtkWindowedSincPolyDataFilter` + `vtkQuadricDecimation`), with a convex-hull fallback (`vtkDelaunay3D` + `vtkDataSetSurfaceFilter`). |

```sh
cargo run --release --bin gen_dataset          # regenerate dataset/{train,test}
cargo run --release --bin reconstruct -- artifacts/out.vtk
cargo run --release --bin reconstruct -- artifacts/out.vtk -o mesh.obj --method hull --max-tris 1500
```

`tools/latent_gif.py` (rendering only — no VTK) calls the `reconstruct` binary
for each generated point cloud:

```sh
./tools/latent_gif.py --n 100 --max-tris 1500
```

Run the bindings' tests with:

```sh
cargo test --bin gen_dataset --bin reconstruct
```
