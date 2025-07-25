#![recursion_limit = "131"]
mod data;
mod inference;
mod model;
mod training;


use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamConfig;
use clap::{Parser, Subcommand};

use crate::model::GeometryAutoEncoderConfig;
use crate::training::TrainingConfig;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the MNIST model.
    Train {},
    /// Generate a geometry from latent parameters
    Generate {
        /// Path to the directory where the model artifacts are saved.
        #[arg(short, long)]
        parameters: Vec<f32>,
    },
}

/// Main function to run the training and inference.
///
/// This function initializes the WGPU device, trains the model, and then performs inference on a sample image.
fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAotudiffBackend = Autodiff<MyBackend>;

    let cli = Cli::parse();
    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "artifacts";

    match cli.command {
        Commands::Train {} => {
            let start = std::time::Instant::now();
            training::train::<MyAotudiffBackend>(
                artifact_dir,
                TrainingConfig::new(GeometryAutoEncoderConfig::new(1000), AdamConfig::new()),
                device.clone(),
            );
            let duration = start.elapsed();
            println!("Training time: {duration:?}");
        }
        Commands::Generate { parameters } => {
            crate::inference::infer::<MyBackend>(artifact_dir, device, parameters);
        }
    }
}
