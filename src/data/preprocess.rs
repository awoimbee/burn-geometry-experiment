use nalgebra::VectorView3;
use rand::distr::Uniform;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use stl_io::IndexedMesh;

/// n_points: number of points to sample of the surface
pub fn preprocess_mesh(
    mesh: IndexedMesh,
    n_points: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // 1. Read STL
    // let mut file = File::open(path)?;
    // let mesh = stl_io::read_stl(&mut file)?;

    // Build cumulative triangle areas
    let mut areas = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let v0 = VectorView3::from_slice(&mesh.vertices[face.vertices[0]].0);
        let v1 = VectorView3::from_slice(&mesh.vertices[face.vertices[1]].0);
        let v2 = VectorView3::from_slice(&mesh.vertices[face.vertices[2]].0);
        let a = (v1 - v0).cross(&(v2 - v0)).norm() * 0.5;
        areas.push(a);
    }
    let dist = WeightedIndex::new(&areas)?;

    // Sample points
    let mut rng = rand::rng();
    let uniform = Uniform::new(0.0f32, 1.0)?;
    let mut points = Vec::with_capacity(n_points * 3);

    for _ in 0..n_points {
        let tri_idx = dist.sample(&mut rng);
        let face = &mesh.faces[tri_idx];
        let v0 = VectorView3::from_slice(&mesh.vertices[face.vertices[0]].0);
        let v1 = VectorView3::from_slice(&mesh.vertices[face.vertices[1]].0);
        let v2 = VectorView3::from_slice(&mesh.vertices[face.vertices[2]].0);

        let r1 = uniform.sample(&mut rng);
        let r2 = uniform.sample(&mut rng);
        let (r1, r2) = if r1 + r2 > 1.0 {
            (1.0 - r1, 1.0 - r2)
        } else {
            (r1, r2)
        };
        let p = v0 + r1 * (v1 - v0) + r2 * (v2 - v0);
        points.extend_from_slice(p.data.as_slice());
    }

    // Center and scale
    let mut cloud = ndarray::Array2::from_shape_vec((n_points, 3), points)?;
    let centroid = cloud.mean_axis(ndarray::Axis(0)).unwrap();
    cloud -= &centroid;
    let max_norm = cloud
        .rows()
        .into_iter()
        .map(|r| (r[0].powi(2) + r[1].powi(2) + r[2].powi(2)).sqrt())
        .fold(0.0f32, f32::max);
    if max_norm > 0.0 {
        cloud /= max_norm;
    }

    // Convert back to Vec<f32>
    Ok(cloud.into_raw_vec_and_offset().0)
}
