#![recursion_limit = "131"]
mod data;
mod inference;
mod model;
mod training;

use std::env;

use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamConfig;

use crate::model::GeometryAutoEncoderConfig;
use crate::training::TrainingConfig;

/// Main function to run the training and inference.
///
/// This function initializes the WGPU device, trains the model, and then performs inference on a sample image.
fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAotudiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    let artifact_dir = "artifacts";

    let args = env::args().collect::<Vec<_>>();
    if args[1] == "train" {
        let start = std::time::Instant::now();
        training::train::<MyAotudiffBackend>(
            artifact_dir,
            TrainingConfig::new(GeometryAutoEncoderConfig::new(1000), AdamConfig::new()),
            device.clone(),
        );
        let duration = start.elapsed();
        println!("Training time: {duration:?}");
    } else if args[1] == "generate" {
        let params: Vec<f32> = serde_json::from_str(&args[2]).unwrap();
        crate::inference::infer::<MyBackend>(artifact_dir, device, params);
    } else {
        println!("Usage: {} train|infer [PARAMS]", args[0]);
    }
}
