#![recursion_limit = "131"]
mod data;
mod inference;
mod model;
mod training;

use burn::backend::{Autodiff, Wgpu};
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;
use clap::{Parser, Subcommand};
use burn::tensor::f16;

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
        /// 8 latent parameters (omit and use --count to sample the latent space instead)
        #[arg(short, long, num_args = 1.., allow_hyphen_values = true)]
        parameters: Vec<f32>,
        /// Generate N geometries from random latents (writes artifacts/generated/)
        #[arg(short = 'n', long, default_value_t = 0)]
        count: usize,
        /// Latent sampling distribution for --count
        #[arg(long, value_enum, default_value_t = crate::inference::LatentDist::Gaussian)]
        dist: crate::inference::LatentDist,
        /// Gaussian std or uniform half-range for --count
        #[arg(long, default_value_t = 1.0)]
        scale: f32,
        /// RNG seed for --count
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

// type MyBackend = NdArray<f32, i32>;
type MyBackend = Wgpu<f16, i16>;
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
                GeometryAutoEncoderConfig::new(100),
                AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(2.0))),
            );
            let start = std::time::Instant::now();
            training::train::<MyAutodiffBackend>(artifact_dir, training_config, device);
            let duration = start.elapsed();
            println!("Training time: {duration:?}");
        }
        Commands::Generate {
            parameters,
            count,
            dist,
            scale,
            seed,
        } => {
            if !parameters.is_empty() && count > 0 {
                eprintln!("error: use either --parameters or --count, not both");
                std::process::exit(2);
            }
            if parameters.len() == crate::inference::LATENT_DIM {
                crate::inference::infer::<MyBackend>(artifact_dir, device, parameters);
            } else if count > 0 {
                crate::inference::infer_batch::<MyBackend>(artifact_dir, device, count, dist, scale, seed);
            } else {
                eprintln!(
                    "error: provide {} --parameters or --count N",
                    crate::inference::LATENT_DIM
                );
                std::process::exit(2);
            }
        }
    }
}
