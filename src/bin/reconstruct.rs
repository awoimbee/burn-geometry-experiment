//! Reconstruct a triangle mesh from a point cloud.
//!
//! Rust port of `sidecar/reconstruct.py`: instead of driving VTK through Python
//! and pymeshlab, it talks to VTK directly with the `vtk` crate.
//!
//! ```text
//! cargo run --release --bin reconstruct -- artifacts/out.vtk
//! cargo run --release --bin reconstruct -- artifacts/out.vtk -o mesh.obj --method hull
//! ```
//!
//! The pipeline is the classic VTK point-cloud reconstruction:
//!
//! ```text
//! points -> vtkSurfaceReconstructionFilter -> vtkContourFilter(isovalue 0)
//!        -> vtkCleanPolyData -> vtkTriangleFilter
//!        -> [vtkQuadricDecimation] -> vtkSTLWriter
//! ```
//!
//! `--method hull` (or the fallback of `auto`) tetrahedralises the cloud with
//! `vtkDelaunay3D` and extracts its boundary with `vtkDataSetSurfaceFilter`.

use std::error::Error;
use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use vtk::*;

type Res<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Method {
    /// Poisson-style reconstruction, falling back to the convex hull.
    Auto,
    /// Surface reconstruction only.
    Poisson,
    /// Convex hull only.
    Hull,
}

#[derive(Parser, Debug)]
#[command(
    name = "reconstruct",
    about = "Reconstruct a triangle mesh from a point cloud using VTK"
)]
struct Args {
    /// Point cloud file (.vtk unstructured grid, .stl or .ply).
    input: PathBuf,

    /// Output mesh path (default: `<input stem>_mesh.stl`).
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Reconstruction method.
    #[arg(long, value_enum, default_value_t = Method::Auto)]
    method: Method,

    /// Implicit-function sample spacing; 0 lets VTK pick one (default: 0.05).
    #[arg(long, default_value_t = 0.05)]
    spacing: f64,

    /// Number of neighbours used to estimate the local surface (default: 15).
    #[arg(long, default_value_t = 15)]
    neighborhood: i32,

    /// Taubin smoothing iterations applied after reconstruction (0 = off).
    #[arg(long, default_value_t = 20)]
    smooth: i32,

    /// Pass band of the Taubin smoothing filter, in (0, 2).
    #[arg(long, default_value_t = 0.1)]
    pass_band: f64,

    /// Decimate to at most this many triangles (default: 0 = off).
    #[arg(long, default_value_t = 0)]
    max_tris: i64,
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Res<()> {
    let points = read_points(&args.input)?;
    if points.is_empty() {
        return Err(format!("no points in {}", args.input.display()).into());
    }
    println!("read {} points from {}", points.len(), args.input.display());

    let cloud = cloud_to_polydata(&points);
    let (mesh, method) = reconstruct(&cloud, &args);
    let mesh = clean_triangulate(&mesh);
    let mesh = if args.smooth > 0 {
        smooth(&mesh, args.smooth, args.pass_band)
    } else {
        mesh
    };

    let mesh = if args.max_tris > 0 && mesh.get_number_of_polys() > args.max_tris {
        decimate(&mesh, args.max_tris)
    } else {
        mesh
    };
    let mesh = clean_triangulate(&mesh);

    let n_tri = mesh.get_number_of_polys();
    if n_tri == 0 {
        return Err(format!("{method} produced an empty mesh").into());
    }
    println!(
        "{method}: {} vertices, {n_tri} triangles",
        mesh.get_number_of_points()
    );

    let out = args.output.unwrap_or_else(|| default_output(&args.input));
    write_mesh(&mesh, &out)?;
    println!("wrote {}", out.display());
    Ok(())
}

fn default_output(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mesh".to_string());
    input.with_file_name(format!("{stem}_mesh.stl"))
}

// ---------------------------------------------------------------------------
// reading
// ---------------------------------------------------------------------------

fn collect_points(points: &vtkPoints) -> Vec<[f64; 3]> {
    let n = points.get_number_of_points();
    let mut out = Vec::with_capacity(n as usize);
    let mut x = [0.0f64; 3];
    for i in 0..n {
        points.get_point(i, &mut x);
        out.push(x);
    }
    out
}

fn read_points(path: &Path) -> Res<Vec<[f64; 3]>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let name = path.to_string_lossy().into_owned();

    let points = match ext.as_str() {
        "vtk" => {
            let reader = vtkUnstructuredGridReader::new();
            reader.set_file_name(&name);
            reader.update();
            let grid = reader
                .get_output()
                .ok_or("the .vtk reader produced no output")?;
            let pts = grid.get_points().ok_or("the .vtk file has no points")?;
            collect_points(&pts)
        }
        "stl" => {
            let reader = vtkSTLReader::new();
            reader.set_file_name(&name);
            reader.update();
            let poly = reader
                .get_output()
                .ok_or("the .stl reader produced no output")?;
            let pts = poly.get_points().ok_or("the .stl file has no points")?;
            collect_points(&pts)
        }
        "ply" => {
            let reader = vtkPLYReader::new();
            reader.set_file_name(&name);
            reader.update();
            let poly = reader
                .get_output()
                .ok_or("the .ply reader produced no output")?;
            let pts = poly.get_points().ok_or("the .ply file has no points")?;
            collect_points(&pts)
        }
        other => {
            return Err(format!("unsupported input format '.{other}' (use vtk, stl or ply)").into());
        }
    };
    Ok(points)
}

fn cloud_to_polydata(points: &[[f64; 3]]) -> vtkPolyData {
    let pts = vtkPoints::new();
    for p in points {
        pts.insert_next_point(*p);
    }
    let poly = vtkPolyData::new();
    poly.set_points(&pts);
    poly
}

// ---------------------------------------------------------------------------
// reconstruction
// ---------------------------------------------------------------------------

fn reconstruct(cloud: &vtkPolyData, args: &Args) -> (vtkPolyData, &'static str) {
    if matches!(args.method, Method::Auto | Method::Poisson) {
        let candidate = poisson(cloud, args);
        let n_tri = candidate.get_number_of_polys();
        if n_tri >= 10 {
            return (candidate, "poisson");
        }
        if args.method == Method::Poisson {
            eprintln!("warning: poisson produced only {n_tri} triangles");
            return (candidate, "poisson");
        }
        eprintln!(
            "poisson produced a degenerate mesh ({n_tri} triangles); \
             falling back to convex hull"
        );
    }
    (hull(cloud), "hull")
}

/// `vtkSurfaceReconstructionFilter` builds a signed distance field around the
/// cloud; isosurfacing it at 0 yields the reconstructed surface.
fn poisson(cloud: &vtkPolyData, args: &Args) -> vtkPolyData {
    let surface = vtkSurfaceReconstructionFilter::new();
    surface.set_neighborhood_size(args.neighborhood);
    surface.set_sample_spacing(args.spacing);
    surface.set_input_data(cloud);
    surface.update();
    let field = surface
        .get_output()
        .expect("vtkSurfaceReconstructionFilter produced no output");

    let contour = vtkContourFilter::new();
    contour.set_value(0, 0.0);
    contour.set_compute_normals(0);
    contour.set_input_data(&field);
    contour.update();
    contour
        .get_output()
        .expect("vtkContourFilter produced no output")
}

/// Convex hull: tetrahedralise, then keep the boundary faces.
fn hull(cloud: &vtkPolyData) -> vtkPolyData {
    let delaunay = vtkDelaunay3D::new();
    delaunay.set_tolerance(0.001);
    delaunay.set_input_data(cloud);
    delaunay.update();
    let grid = delaunay
        .get_output()
        .expect("vtkDelaunay3D produced no output");

    let surface = vtkDataSetSurfaceFilter::new();
    surface.set_input_data(&grid);
    surface.update();
    surface
        .get_output()
        .expect("vtkDataSetSurfaceFilter produced no output")
}

/// A couple of Taubin (windowed sinc) iterations knock the noise off the
/// reconstructed isosurface without shrinking it the way Laplacian smoothing
/// would.
fn smooth(mesh: &vtkPolyData, iterations: i32, pass_band: f64) -> vtkPolyData {
    let filter = vtkWindowedSincPolyDataFilter::new();
    filter.set_number_of_iterations(iterations);
    filter.set_pass_band(pass_band);
    filter.set_boundary_smoothing(0);
    filter.set_feature_edge_smoothing(0);
    filter.set_normalize_coordinates(1);
    filter.set_input_data(mesh);
    filter.update();
    filter
        .get_output()
        .expect("vtkWindowedSincPolyDataFilter produced no output")
}

fn decimate(mesh: &vtkPolyData, max_tris: i64) -> vtkPolyData {
    let current = mesh.get_number_of_polys();
    let decimation = vtkQuadricDecimation::new();
    decimation.set_target_reduction(1.0 - max_tris as f64 / current as f64);
    decimation.set_input_data(mesh);
    decimation.update();
    decimation
        .get_output()
        .expect("vtkQuadricDecimation produced no output")
}

fn clean_triangulate(mesh: &vtkPolyData) -> vtkPolyData {
    let clean = vtkCleanPolyData::new();
    clean.set_input_data(mesh);
    clean.update();
    let cleaned = clean
        .get_output()
        .expect("vtkCleanPolyData produced no output");

    let triangulate = vtkTriangleFilter::new();
    triangulate.set_input_data(&cleaned);
    triangulate.update();
    triangulate
        .get_output()
        .expect("vtkTriangleFilter produced no output")
}

// ---------------------------------------------------------------------------
// writing
// ---------------------------------------------------------------------------

fn write_mesh(mesh: &vtkPolyData, out: &Path) -> Res<()> {
    let ext = out
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let name = out.to_string_lossy().into_owned();

    let written = match ext.as_str() {
        "stl" => {
            let writer = vtkSTLWriter::new();
            writer.set_file_name(&name);
            writer.set_file_type_to_binary();
            writer.set_input_data(mesh);
            writer.write()
        }
        "ply" => {
            let writer = vtkPLYWriter::new();
            writer.set_file_name(&name);
            writer.set_file_type_to_binary();
            writer.set_input_data(mesh);
            writer.write()
        }
        "obj" => {
            let writer = vtkOBJWriter::new();
            writer.set_file_name(&name);
            writer.set_input_data(mesh);
            writer.write()
        }
        other => {
            return Err(
                format!("unsupported output format '.{other}' (use stl, obj or ply)").into(),
            );
        }
    };
    if written != 1 {
        return Err(format!("failed to write {}", out.display()).into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args(method: Method, spacing: f64) -> Args {
        Args {
            input: PathBuf::from("unused.vtk"),
            output: None,
            method,
            spacing,
            neighborhood: 15,
            smooth: 10,
            pass_band: 0.1,
            max_tris: 0,
        }
    }

    /// 500 points spread over the unit sphere (Fibonacci lattice).
    fn sphere_cloud(n: usize) -> vtkPolyData {
        let points: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let t = (i as f64 + 0.5) / n as f64;
                let phi = (1.0 - 2.0 * t).acos();
                let theta = std::f64::consts::PI * (1.0 + 5.0f64.sqrt()) * i as f64;
                [phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()]
            })
            .collect();
        cloud_to_polydata(&points)
    }

    #[test]
    fn poisson_reconstruction_of_a_sphere() {
        let cloud = sphere_cloud(500);
        let (mesh, method) = reconstruct(&cloud, &test_args(Method::Auto, 0.1));
        assert_eq!(method, "poisson");
        assert!(
            mesh.get_number_of_polys() > 100,
            "got {} polys",
            mesh.get_number_of_polys()
        );
    }

    #[test]
    fn convex_hull_of_a_sphere() {
        let cloud = sphere_cloud(500);
        let (mesh, method) = reconstruct(&cloud, &test_args(Method::Hull, 0.0));
        assert_eq!(method, "hull");
        assert!(mesh.get_number_of_polys() > 100);
    }

    #[test]
    fn decimation_reduces_the_triangle_count() {
        let cloud = sphere_cloud(500);
        let (mesh, _) = reconstruct(&cloud, &test_args(Method::Auto, 0.1));
        let before = mesh.get_number_of_polys();
        let after = decimate(&mesh, 200).get_number_of_polys();
        assert!(
            after < before,
            "decimation did not reduce {before} -> {after}"
        );
        assert!(after > 0);
    }

    #[test]
    fn cleanup_keeps_the_mesh_alive() {
        let cloud = sphere_cloud(300);
        let (mesh, _) = reconstruct(&cloud, &test_args(Method::Auto, 0.15));
        let cleaned = clean_triangulate(&smooth(&clean_triangulate(&mesh), 10, 0.1));
        assert!(cleaned.get_number_of_points() > 0);
        assert!(cleaned.get_number_of_polys() > 0);
    }
}
