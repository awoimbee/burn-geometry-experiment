#![recursion_limit = "131"]
mod data;
mod inference;
mod model;
mod training;

use burn::backend::{Autodiff, Wgpu};
use burn::grad_clipping::GradientClippingConfig;
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

// type MyBackend = NdArray<f32, i32>;
type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

/// Main function to run the training and inference.
///
/// This function initializes the WGPU device, trains the model, and then performs inference on a sample image.
fn main() {
    let cli = Cli::parse();
    let device = burn::backend::wgpu::WgpuDevice::default();
    // let device = burn::backend::ndarray::NdArrayDevice::default();
    let artifact_dir = "artifacts";

    match cli.command {
        Commands::Train {} => {
            let training_config = TrainingConfig::new(
                GeometryAutoEncoderConfig::new(2000),
                AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(2.0))),
            );
            let start = std::time::Instant::now();
            training::train::<MyAutodiffBackend>(artifact_dir, training_config, device);
            let duration = start.elapsed();
            println!("Training time: {duration:?}");
        }
        Commands::Generate { parameters } => {
            crate::inference::infer::<MyBackend>(artifact_dir, device, parameters);
        }
    }
}
