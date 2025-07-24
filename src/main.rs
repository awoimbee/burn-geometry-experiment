#![recursion_limit = "131"]
mod data;
mod inference;
mod model;
mod training;

use std::num::NonZeroUsize;

use crate::model::ModelConfig;
use crate::training::TrainingConfig;

use burn::backend::{Autodiff, Wgpu};
use burn::data::dataset::transform::Window;
use burn::optim::AdamConfig;

/// Main function to run the training and inference.
///
/// This function initializes the WGPU device, trains the model, and then performs inference on a sample image.
fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAotudiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    let artifact_dir = "artifacts";
    let start = std::time::Instant::now();
    training::train::<MyAotudiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
    let duration = start.elapsed();
    println!("Training time: {:?}", duration);
    let random_start = rand::random::<u8>().into();
    let items = burn::data::dataset::vision::MnistDataset::test()
        .window(random_start, NonZeroUsize::new(10).unwrap())
        .unwrap();
    crate::inference::infer::<MyBackend>(artifact_dir, device, items);
}
