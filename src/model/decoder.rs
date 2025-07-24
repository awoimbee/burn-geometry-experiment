use burn::module::Module;
use burn::nn::{Gelu, Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Simple MLP decoder: latent -> point cloud
#[derive(Module, Debug)]
pub struct PointCloudDecoder<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: Gelu,
    out_dim: usize,
}

impl<B: Backend> PointCloudDecoder<B> {
    pub fn new(latent_dim: usize, num_points: usize, device: &B::Device) -> Self {
        let layers = vec![
            LinearConfig::new(latent_dim, 512).init(device),
            LinearConfig::new(512, 1024).init(device),
            LinearConfig::new(1024, num_points * 3).init(device), // Flattened [B, N*3]
        ];

        Self {
            layers,
            activation: Gelu::new(),
            out_dim: num_points,
        }
    }

    /// latent: [B, 256] -> point cloud: [B, N, 3]
    pub fn forward(&self, latent: Tensor<B, 2>) -> Tensor<B, 3> {
        let mut x = latent;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i < self.layers.len() - 1 {
                x = self.activation.forward(x);
            }
        }
        let [b, _] = x.dims();
        x.reshape([b, self.out_dim, 3])
    }
}
