use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::DType;
use clap::ValueEnum;
use rand::distr::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::model::GeometryAutoEncoder;
use crate::training::TrainingConfig;

/// Latent dimension, hardcoded in `GeometryAutoEncoder::new`.
pub const LATENT_DIM: usize = 8;

/// Distribution used to sample latents for `generate --count`.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum LatentDist {
    Gaussian,
    Uniform,
}

fn load_model<B: Backend>(artifact_dir: &str, device: &B::Device) -> GeometryAutoEncoder<B> {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new();
    let record = record
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Trained model should exist; run train first");

    config.model.init::<B>(device).load_record(record)
}

fn decode<B: Backend>(model: &GeometryAutoEncoder<B>, params: &[f32]) -> Vec<[f32; 3]> {
    let data = TensorData::new(params.to_vec(), [params.len()]);
    let latent = Tensor::<_, 1>::from(data);
    let output = model.generate(latent).cast(DType::F32);
    let output_data = output.to_data();
    let slice: &[f32] = output_data.as_slice().unwrap();
    slice
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

/// Single latent -> `artifacts/out.vtk`
pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, params: Vec<f32>) {
    let model = load_model::<B>(artifact_dir, &device);
    let point_cloud = decode(&model, &params);
    write_vtk_legacy(&point_cloud, &Path::new(artifact_dir).join("out.vtk")).unwrap();
}

/// `count` random latents -> `artifacts/generated/gen_NNNN.vtk` + `latents.csv`
pub fn infer_batch<B: Backend>(
    artifact_dir: &str,
    device: B::Device,
    count: usize,
    dist: LatentDist,
    scale: f32,
    seed: u64,
) {
    let model = load_model::<B>(artifact_dir, &device);
    let latents = sample_latents(count, dist, scale, seed);
    let out_dir = Path::new(artifact_dir).join("generated");
    fs::create_dir_all(&out_dir).unwrap();

    let mut csv = BufWriter::new(File::create(out_dir.join("latents.csv")).unwrap());
    for (i, params) in latents.iter().enumerate() {
        let point_cloud = decode(&model, params);
        let path = out_dir.join(format!("gen_{i:04}.vtk"));
        write_vtk_legacy(&point_cloud, &path).unwrap();
        let row = params
            .iter()
            .map(|p| format!("{p:.6}"))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(csv, "{row}").unwrap();
        println!("wrote {}", path.display());
    }
}

/// Box-Muller transform; rand 0.10 has no built-in normal distribution.
fn sample_gaussian(rng: &mut StdRng, u01: &Uniform<f32>) -> f32 {
    let mut u1 = u01.sample(rng);
    if u1 <= f32::MIN_POSITIVE {
        u1 = f32::MIN_POSITIVE;
    }
    let u2 = u01.sample(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Sample `n` latent vectors in `[-scale, scale]^8` (uniform) or N(0, scale^2) (gaussian).
pub fn sample_latents(n: usize, dist: LatentDist, scale: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let u01 = Uniform::new(0.0, 1.0).expect("invalid range");
    let uniform = Uniform::new(-scale, scale)
        .expect("--scale must be positive for uniform sampling");
    (0..n)
        .map(|_| {
            (0..LATENT_DIM)
                .map(|_| match dist {
                    LatentDist::Gaussian => sample_gaussian(&mut rng, &u01) * scale,
                    LatentDist::Uniform => uniform.sample(&mut rng),
                })
                .collect()
        })
        .collect()
}

fn write_vtk_legacy(points: &[[f32; 3]], path: &Path) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);

    // --- VTK header ---
    writeln!(w, "# vtk DataFile Version 3.0")?;
    writeln!(w, "Rust point cloud")?;
    writeln!(w, "ASCII")?;
    writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

    // --- Points ---
    writeln!(w, "POINTS {} float", points.len())?;
    for &[x, y, z] in points {
        writeln!(w, "{x} {y} {z}")?;
    }

    // --- Cells (one vertex per cell) ---
    writeln!(w, "CELLS {} {}", points.len(), points.len() * 2)?;
    for i in 0..points.len() {
        writeln!(w, "1 {i}")?; // 1 = number of indices, i = vertex id
    }

    // --- Cell types (all VTK_VERTEX = 1) ---
    writeln!(w, "CELL_TYPES {}", points.len())?;
    for _ in 0..points.len() {
        writeln!(w, "1")?;
    }

    Ok(())
}
