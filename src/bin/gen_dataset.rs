//! Regenerate `dataset/{train,test}` as clean binary STL meshes using VTK.
//!
//! Rust port of `tools/gen_dataset.py`: all geometry is built with the `vtk`
//! crate instead of the Python `vtk` module.  Each mesh is watertight and free
//! of degenerate triangles; every written file is validated (binary layout,
//! zero-area triangles, closed 2-manifold) before the script exits.
//!
//! ```text
//! cargo run --release --bin gen_dataset
//! ```

use std::path::{Path, PathBuf};

use stl_validate::validate_binary_stl;
use vtk::*;

#[path = "../stl_validate.rs"]
mod stl_validate;

/// VTK's generic polygon cell type.  Everything is written as polygons and
/// triangulated later by `vtkTriangleFilter`.

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dataset = root.join("dataset");

    for split in ["train", "test"] {
        std::fs::create_dir_all(dataset.join(split)).expect("cannot create dataset directory");
    }

    let mut failures = 0usize;
    let mut total = 0usize;

    for (split, shapes) in [("train", shapes_train()), ("test", shapes_test())] {
        println!("{split}:");
        for (name, build) in shapes {
            total += 1;
            let path = dataset.join(split).join(format!("{name}.stl"));
            match build_and_write(build, &path) {
                Ok((ntri, extent)) => {
                    println!(
                        "  {name}.stl: {ntri} tris, {} x {} x {}",
                        extent[0], extent[1], extent[2]
                    );
                }
                Err(err) => {
                    failures += 1;
                    eprintln!("  {name}.stl: FAILED ({err})");
                }
            }
        }
    }

    if failures > 0 {
        eprintln!("{failures} mesh(es) failed validation");
        std::process::exit(1);
    }
    println!("Wrote {total} meshes to {}", dataset.display());
}

fn build_and_write(
    build: impl FnOnce() -> vtkPolyData,
    path: &Path,
) -> Result<(u32, [f64; 3]), String> {
    let poly = build();
    if poly.get_number_of_cells() == 0 {
        return Err("mesh has no cells".to_string());
    }
    write_binary_stl(&poly, path)?;
    validate_binary_stl(path)
}

// ---------------------------------------------------------------------------
// VTK helpers
// ---------------------------------------------------------------------------

/// Run `src`, weld points, drop degenerate cells and triangulate the result.
fn as_polydata(src: &vtkAlgorithm) -> vtkPolyData {
    src.update();

    let clean = vtkCleanPolyData::new();
    clean.set_input_connection(&src.get_output_port().expect("source has no output port"));
    clean.update();

    let triangulate = vtkTriangleFilter::new();
    triangulate.set_input_connection(
        &clean
            .get_output_port()
            .expect("clean filter has no output port"),
    );
    triangulate.update();

    triangulate
        .get_output()
        .expect("triangulate produced no output")
}

fn mesh_from_points_cells(points: &[[f64; 3]], cells: &[Vec<i64>]) -> vtkPolyData {
    let pts = vtkPoints::new();
    for p in points {
        pts.insert_next_point(*p);
    }

    // `vtkPolyData::InsertNextCell` requires the cell arrays to exist already
    // (see its "Make sure that ... vertex, line, polygon, and triangle strip
    // arrays have been supplied" note), so fill a `vtkCellArray` and attach it.
    let polys = vtkCellArray::new();
    for cell in cells {
        polys.insert_next_cell_v3(cell.len() as i32);
        for &id in cell {
            polys.insert_cell_point(id);
        }
    }

    let poly = vtkPolyData::new();
    poly.set_points(&pts);
    poly.set_polys(&polys);
    poly
}

fn write_binary_stl(poly: &vtkPolyData, path: &Path) -> Result<(), String> {
    let writer = vtkSTLWriter::new();
    writer.set_file_name(&path.to_string_lossy());
    writer.set_file_type_to_binary();
    writer.set_input_data(poly);
    if writer.write() != 1 {
        return Err(format!("failed to write {}", path.display()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// primitive shapes
// ---------------------------------------------------------------------------

fn sphere(radius: f64) -> vtkPolyData {
    let f = vtkSphereSource::new();
    f.set_radius(radius);
    f.set_theta_resolution(48);
    f.set_phi_resolution(48);
    as_polydata(&f)
}

fn ellipsoid(rx: f64, ry: f64, rz: f64) -> vtkPolyData {
    let s = vtkSphereSource::new();
    s.set_radius(1.0);
    s.set_theta_resolution(48);
    s.set_phi_resolution(48);

    let t = vtkTransform::new();
    t.scale_v3(rx, ry, rz);

    let tf = vtkTransformFilter::new();
    tf.set_transform(&t);
    tf.set_input_connection(&s.get_output_port().expect("sphere has no output port"));
    as_polydata(&tf)
}

fn cuboid(lx: f64, ly: f64, lz: f64) -> vtkPolyData {
    let c = vtkCubeSource::new();
    c.set_x_length(lx);
    c.set_y_length(ly);
    c.set_z_length(lz);
    as_polydata(&c)
}

fn cylinder(radius: f64, height: f64, res: i32) -> vtkPolyData {
    let c = vtkCylinderSource::new();
    c.set_radius(radius);
    c.set_height(height);
    c.set_resolution(res);
    c.set_capping(1);
    as_polydata(&c)
}

fn cone(radius: f64, height: f64, res: i32) -> vtkPolyData {
    let c = vtkConeSource::new();
    c.set_radius(radius);
    c.set_height(height);
    c.set_resolution(res);
    c.set_capping(1);
    as_polydata(&c)
}

// ---------------------------------------------------------------------------
// explicit meshes
// ---------------------------------------------------------------------------

/// Capsule: cylinder of straight length `height` with hemispherical caps of
/// `radius`.  Built as an explicit grid of rings (poles are single shared
/// vertices) so the result is a clean closed 2-manifold.
fn capsule(radius: f64, height: f64, nu: usize, n: usize) -> vtkPolyData {
    let (h2, r) = (height / 2.0, radius);
    let mut profile: Vec<(f64, f64)> = Vec::new();
    profile.push((0.0, -h2 - r)); // bottom pole
    for j in 1..n {
        // bottom cap
        let beta = 0.5 * std::f64::consts::PI * j as f64 / n as f64;
        profile.push((r * beta.cos(), -h2 - r * beta.sin()));
    }
    profile.push((r, -h2)); // bottom equator
    for j in 1..n {
        // side
        profile.push((r, -h2 + height * j as f64 / n as f64));
    }
    profile.push((r, h2)); // top equator
    for j in 1..n {
        // top cap
        let beta = 0.5 * std::f64::consts::PI * j as f64 / n as f64;
        profile.push((r * beta.cos(), h2 + r * beta.sin()));
    }
    profile.push((0.0, h2 + r)); // top pole

    let mut pts: Vec<[f64; 3]> = Vec::new();
    let mut row_base: Vec<usize> = Vec::new();
    for (rho, z) in &profile {
        row_base.push(pts.len());
        if *rho == 0.0 {
            pts.push([0.0, 0.0, *z]); // pole: one shared vertex
        } else {
            for i in 0..nu {
                let u = 2.0 * std::f64::consts::PI * i as f64 / nu as f64;
                pts.push([rho * u.cos(), rho * u.sin(), *z]);
            }
        }
    }

    let vid = |k: usize, i: usize| -> i64 {
        if profile[k].0 == 0.0 {
            row_base[k] as i64
        } else {
            (row_base[k] + (i % nu)) as i64
        }
    };

    let mut cells: Vec<Vec<i64>> = Vec::new();
    for k in 0..profile.len() - 1 {
        for i in 0..nu {
            let i2 = (i + 1) % nu;
            let (a, b) = (vid(k, i), vid(k, i2));
            let (c, d) = (vid(k + 1, i2), vid(k + 1, i));
            if profile[k].0 == 0.0 {
                // bottom pole row: triangle (pole, ring_i+1, ring_i)
                cells.push(vec![a, c, d]);
            } else if profile[k + 1].0 == 0.0 {
                // top pole row
                cells.push(vec![a, b, c]);
            } else {
                cells.push(vec![a, b, c, d]);
            }
        }
    }
    mesh_from_points_cells(&pts, &cells)
}

fn torus(major: f64, minor: f64, nu: usize, nv: usize) -> vtkPolyData {
    let mut pts: Vec<[f64; 3]> = Vec::new();
    for i in 0..nu {
        let u = 2.0 * std::f64::consts::PI * i as f64 / nu as f64;
        for j in 0..nv {
            let v = 2.0 * std::f64::consts::PI * j as f64 / nv as f64;
            let rr = major + minor * v.cos();
            pts.push([rr * u.cos(), rr * u.sin(), minor * v.sin()]);
        }
    }

    let mut cells: Vec<Vec<i64>> = Vec::new();
    for i in 0..nu {
        let i2 = (i + 1) % nu;
        for j in 0..nv {
            let j2 = (j + 1) % nv;
            let a = (i * nv + j) as i64;
            cells.push(vec![
                a,
                (i2 * nv + j) as i64,
                (i2 * nv + j2) as i64,
                (i * nv + j2) as i64,
            ]);
        }
    }
    mesh_from_points_cells(&pts, &cells)
}

/// CCW L cross-section, extruded over z in [-0.4, 0.4].
fn lshape() -> vtkPolyData {
    let cross = [
        (-1.0f64, -1.0f64),
        (1.0, -1.0),
        (1.0, -0.2),
        (-0.2, -0.2),
        (-0.2, 1.0),
        (-1.0, 1.0),
    ];
    let z = 0.4;
    let mut pts: Vec<[f64; 3]> = cross.iter().map(|(x, y)| [*x, *y, -z]).collect();
    pts.extend(cross.iter().map(|(x, y)| [*x, *y, z]));
    let n = cross.len();

    let mut cells: Vec<Vec<i64>> = Vec::new();
    for i in (2..n).rev() {
        // bottom cap, normal -z
        cells.push(vec![i as i64, i as i64 - 1, 0]);
    }
    for i in 1..n - 1 {
        // top cap, normal +z
        cells.push(vec![n as i64, (n + i) as i64, (n + i + 1) as i64]);
    }
    for i in 0..n {
        // side quads, outward normals
        let j = (i + 1) % n;
        cells.push(vec![i as i64, j as i64, (n + j) as i64]);
        cells.push(vec![i as i64, (n + j) as i64, (n + i) as i64]);
    }
    mesh_from_points_cells(&pts, &cells)
}

// ---------------------------------------------------------------------------
// datasets
// ---------------------------------------------------------------------------

type Shape = (&'static str, fn() -> vtkPolyData);

fn shapes_train() -> Vec<Shape> {
    vec![
        ("sphere_1", || sphere(0.9)),
        ("sphere_2", || sphere(0.55)),
        ("ellipsoid_a", || ellipsoid(1.3, 0.7, 0.5)),
        ("ellipsoid_b", || ellipsoid(0.55, 1.25, 0.8)),
        ("box_cubic", || cuboid(1.0, 1.0, 1.0)),
        ("box_tall", || cuboid(0.7, 1.6, 0.7)),
        ("box_wide", || cuboid(1.8, 0.8, 0.7)),
        ("cylinder_1", || cylinder(0.6, 1.2, 48)),
        ("cylinder_tall", || cylinder(0.35, 1.7, 48)),
        ("cone", || cone(0.7, 1.3, 48)),
        ("torus", || torus(0.65, 0.3, 48, 32)),
        ("capsule", || capsule(0.35, 0.9, 48, 8)),
        ("pyramid", || cone(0.8, 1.2, 4)),
        ("lshape", lshape),
    ]
}

fn shapes_test() -> Vec<Shape> {
    vec![
        ("sphere_test", || sphere(0.8)),
        ("box_test", || cuboid(1.3, 0.9, 1.5)),
        ("cylinder_test", || cylinder(0.5, 1.4, 48)),
        ("torus_test", || torus(0.75, 0.25, 48, 32)),
    ]
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_and_validate(label: &str, build: fn() -> vtkPolyData) -> (u32, [f64; 3]) {
        let dir = std::env::temp_dir().join("rvtk-gen-dataset-tests");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join(format!("{label}.stl"));

        let poly = build();
        assert!(poly.get_number_of_cells() > 0, "{label}: no cells");
        write_binary_stl(&poly, &path).expect("write stl");
        validate_binary_stl(&path).expect("valid binary stl")
    }

    /// Meshes built from VTK sources go through clean + triangulate.
    #[test]
    fn primitive_sources_produce_closed_manifolds() {
        let cases: [(&str, fn() -> vtkPolyData); 5] = [
            ("sphere", || sphere(0.9)),
            ("cuboid", || cuboid(1.0, 1.0, 1.0)),
            ("cylinder", || cylinder(0.6, 1.2, 48)),
            ("cone", || cone(0.7, 1.3, 48)),
            ("ellipsoid", || ellipsoid(1.3, 0.7, 0.5)),
        ];
        for (label, build) in cases {
            let (n_tri, _) = build_and_validate(label, build);
            assert!(n_tri > 0, "{label}: no triangles");
        }
    }

    /// Explicitly built meshes (points + polygons) must also be manifold.
    #[test]
    fn explicit_meshes_produce_closed_manifolds() {
        let cases: [(&str, fn() -> vtkPolyData); 3] = [
            ("torus", || torus(0.65, 0.3, 48, 32)),
            ("capsule", || capsule(0.35, 0.9, 48, 8)),
            ("lshape", lshape),
        ];
        for (label, build) in cases {
            let (n_tri, _) = build_and_validate(label, build);
            assert!(n_tri > 0, "{label}: no triangles");
        }
    }

    /// The dataset we generate must contain exactly the shapes the trainer expects.
    #[test]
    fn dataset_layout_is_stable() {
        assert_eq!(shapes_train().len(), 14);
        assert_eq!(shapes_test().len(), 4);
    }
}
