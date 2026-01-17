mod decoder;
mod encoder;
mod loss;

use burn::config::Config;
use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::{InferenceStep, RegressionOutput, TrainOutput, TrainStep};
use decoder::PointCloudDecoder;
use encoder::GeometryEncoder;

use crate::data::PointCloudBatch;

#[macro_export]
macro_rules! debug_assert_not_nan {
    ($tensor:expr) => {
        debug_assert!(
            !$tensor.clone().is_nan().any().into_scalar().to_bool(),
            "Tensor {} contains NaN",
            stringify!($tensor)
        );
        debug_assert!(
            !$tensor.clone().is_inf().any().into_scalar().to_bool(),
            "Tensor {} contains inf",
            stringify!($tensor)
        );
    };
}

#[derive(Module, Debug)]
pub struct GeometryAutoEncoder<B: Backend> {
    pub encoder: GeometryEncoder<B>,
    pub decoder: PointCloudDecoder<B>,
}

impl<B: Backend> GeometryAutoEncoder<B> {
    pub fn new(num_points: usize, device: &B::Device) -> Self {
        let encoder = GeometryEncoder::new(8, device);
        let decoder = PointCloudDecoder::new(8, num_points, device);
        Self { encoder, decoder }
    }

    /// points: [B, N, 3] -> reconstructed: [B, N, 3]
    pub fn forward(&self, points: Tensor<B, 3>) -> RegressionOutput<B> {
        let latent = self.encoder.forward(points.clone());
        let reconstructed = self.decoder.forward(latent.clone());
        let loss = self.compute_loss(&points, &reconstructed, &latent);

        let batch_size = points.dims()[0];
        let flattened_size = points.dims()[1] * points.dims()[2]; // N * 3
        let output = reconstructed.reshape([batch_size, flattened_size]);
        let targets = points.reshape([batch_size, flattened_size]);

        RegressionOutput::new(loss, output, targets)
    }

    /// latent: [L] -> point cloud: [B, N, 3]
    pub fn generate(&self, latent: Tensor<B, 1>) -> Tensor<B, 2> {
        self.decoder.forward(latent.unsqueeze()).squeeze_dim(0)
    }
}

impl<B: AutodiffBackend> TrainStep
    for GeometryAutoEncoder<B>
{
    type Input = PointCloudBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: PointCloudBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward(batch.points);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}


impl<B: Backend> InferenceStep
    for GeometryAutoEncoder<B>
{
    type Input = PointCloudBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward(batch.points)
    }
}

#[derive(Config, Debug)]
pub struct GeometryAutoEncoderConfig {
    pub num_points_sampled: usize,
}

impl GeometryAutoEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GeometryAutoEncoder<B> {
        GeometryAutoEncoder::new(self.num_points_sampled, device)
    }
}
