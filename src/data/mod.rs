mod preprocess;

use std::fs::{self, File};
use std::path::Path;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use preprocess::preprocess_mesh;

/// A single point cloud sample
#[derive(Clone, Debug)]
pub struct PointCloudItem {
    pub points: Vec<f32>, // [n_points * 3] flattened
}

/// Dataset that holds all preprocessed point clouds in memory
pub struct PointCloudDataset {
    pub items: Vec<PointCloudItem>,
}

impl Dataset<PointCloudItem> for PointCloudDataset {
    fn get(&self, index: usize) -> Option<PointCloudItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

impl PointCloudDataset {
    /// Load all STL files from a directory
    pub fn from_dir<P: AsRef<Path>>(dir: P, split: &str, n_points: usize) -> Self {
        let dir = dir.as_ref().join(split);
        println!("Loading STL files from: {}", dir.display());

        let mut items = Vec::new();
        let entries = fs::read_dir(dir).expect("cannot read directory");

        for entry in entries {
            let entry = entry.expect("invalid directory entry");
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("stl") {
                println!("Processing: {}", path.display());

                let mut file = File::open(&path).unwrap();
                let mesh = stl_io::read_stl(&mut file).unwrap();

                match preprocess_mesh(mesh, n_points) {
                    Ok(points) => {
                        // let filename = path.file_name().unwrap().to_string_lossy().to_string();
                        items.push(PointCloudItem { points });
                    }
                    Err(e) => {
                        eprintln!("Failed to load {}: {}", path.display(), e);
                    }
                }
            }
        }

        println!("Loaded {} point clouds", items.len());
        Self { items }
    }
}

#[derive(Clone, Debug)]
pub struct PointCloudBatch<B: Backend> {
    pub points: Tensor<B, 3>, // shape: [batch_size, n_points, 3]
}

/// Batcher that converts PointCloudItems into batched tensors
#[derive(Clone)]
pub struct PointCloudBatcher {
    n_points: usize,
}

impl PointCloudBatcher {
    pub fn new(n_points: usize) -> Self {
        Self { n_points }
    }
}

impl<B: Backend> Batcher<B, PointCloudItem, PointCloudBatch<B>> for PointCloudBatcher {
    fn batch(&self, items: Vec<PointCloudItem>, device: &B::Device) -> PointCloudBatch<B> {
        let batch_size = items.len();

        // Flatten all points into a single vector
        let mut all_points = Vec::with_capacity(batch_size * self.n_points * 3);
        for item in items {
            all_points.extend_from_slice(&item.points);
        }
        let points_data = TensorData::new(all_points, [batch_size, self.n_points, 3]);

        PointCloudBatch {
            points: Tensor::from_data(points_data, device),
        }
    }
}
