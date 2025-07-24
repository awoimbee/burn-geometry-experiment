use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};

use crate::training::TrainingConfig;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, params: Vec<f32>) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new();
    let record = record
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = config.model.init::<B>(&device).load_record(record);

    let data_size = params.len();
    let data = TensorData::new(params, [data_size]);
    let latent = Tensor::from(data);
    let output = model.generate(latent);
    let output_data = output.to_data();
    let slice: &[f32] = output_data.as_slice().unwrap();
    let point_cloud = slice
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect::<Vec<_>>();

    write_vtk_legacy(&point_cloud, &Path::new(artifact_dir).join("out.vtk")).unwrap();
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

    // --- Cell types (all are VTK_VERTEX = 1) ---
    writeln!(w, "CELL_TYPES {}", points.len())?;
    for _ in 0..points.len() {
        writeln!(w, "1")?;
    }

    Ok(())
}
