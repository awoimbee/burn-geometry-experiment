use burn::{
    module::Module, nn::{
        self, attention::{self, MhaInput}, conv::{Conv1d, Conv1dConfig}, pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig}, Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Relu
    }, tensor::{backend::Backend, Tensor}
};

/// k-NN helper (squared L2, brute-force for clarity)
fn knn<B: Backend>(x: Tensor<B, 3>, k: usize) -> Tensor<B, 3, Int> {
    // x: [B, N, 3]
    let xi = x.clone().unsqueeze_dim(2);           // [B, N, 1, 3]
    let xj = x.clone().unsqueeze_dim(1);           // [B, 1, N, 3]
    let dist = (xi - xj).powi_scalar(2).sum_dim(3); // [B, N, N]
    // top-k along last dim => indices [B, N, k]
    dist.topk_with_indices(k, 3).1
}

/// EdgeConv patch (local neighborhoods) embedder to produce a "local shape token"
///
/// Data preprocessing:
/// * point cloud with N points (2048 ?)
/// * center and scale to unit sphere
#[derive(Module, Debug)]
pub struct EdgeConvEmbed<B: Backend> {
    theta: Linear<B>,   // h_θ
    phi:   Linear<B>,   // h_φ
    psi:   Linear<B>,   // h_ψ
    norm:  LayerNorm<B>,
    relu:  Relu,
    k:     usize,
}

impl<B: Backend> EdgeConvEmbed<B> {
    /// k: how many nearest neighbours (20, 32, 40)
    /// in_dim: how many input channels (xyz=3, can be more if includes normals, ...)
    /// out_dim: dimension of the resulting per-point feature that will later be fed to the Transformer (128, ...)
    pub fn new(k: usize, in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        let theta = LinearConfig::new(in_dim, out_dim).init(device);
        let phi   = LinearConfig::new(in_dim, out_dim).init(device);
        let psi   = LinearConfig::new(out_dim, out_dim).init(device);
        let norm  = LayerNormConfig::new(out_dim).init(device);

        Self { theta, phi, psi, norm, relu: Relu::new(), k }
    }

    /// points: [B, N, 3]  -> tokens: [B, N, C]
    pub fn forward(&self, points: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_b, _n, d] = points.dims();
        let k = self.k;

        // 1. k-NN indices [B, N, k]
        let idx = knn(points.clone(), k);               // [B, N, k]

        // 2. Gather neighbour features
        let points_exp = points.clone().unsqueeze_dim(2); // [B, N, 1, d]
        let idx_exp = idx.unsqueeze_dim(3).repeat(&[1, 1, 1, d]); // [B, N, k, d]
        let neighbors = points_exp.gather(2, idx_exp);  // [B, N, k, d]

        // 3. Edge features: h_θ(x_i) || h_φ(x_j - x_i)
        let x_i = points;                       // [B, N, d]
        let x_j = neighbors;                            // [B, N, k, d]
        let diff = x_j - x_i.clone().unsqueeze_dim(2);              // [B, N, k, d]

        let h_theta = self.theta.forward(x_i);          // [B, N, C]
        let h_phi   = self.phi.forward(diff);           // [B, N, k, C]

        let edge = h_theta.unsqueeze_dim(2) + h_phi;        // [B, N, k, C]

        // 4. Max-pool over neighbours
        let pooled = edge.max_dim(2);                   // [B, N, C]

        // 5. Final 1×1 conv + norm
        let out = self.psi.forward(pooled);             // [B, N, C]
        self.relu.forward(self.norm.forward(out))
    }
}

/// Tiny 4-layer Transformer encoder
#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    layers: Vec<TransformerBlock<B>>,
}

#[derive(Module, Debug)]
struct TransformerBlock<B: Backend> {
    mha: attention::MultiHeadAttention<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    linear: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Transformer<B> {
    pub fn new(embed_dim: usize, num_heads: usize, device: &B::Device) -> Self {
        let mut layers = Vec::new();
        for _ in 0..4 {
            let mha = attention::MultiHeadAttentionConfig::new(embed_dim, num_heads).init(device);
            let norm1 = LayerNormConfig::new(embed_dim).init(device);
            let norm2 = LayerNormConfig::new(embed_dim).init(device);
            let linear = LinearConfig::new(embed_dim, embed_dim).init(device);
            layers.push(TransformerBlock { mha, norm1, norm2, linear, activation: Relu::new() });
        }
        Self { layers }
    }

    /// tokens: [B, N, C] -> tokens: [B, N, C]
    pub fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = tokens;
        for block in &self.layers {
            let mha_input = MhaInput::new(x.clone(), x.clone(), x.clone());
            let attn = block.mha.forward(mha_input);
            x = block.norm1.forward(x + attn.context);
            let ff = block.linear.forward(x.clone());
            x = block.norm2.forward(x + block.activation.forward(ff));
        }
        x
    }
}

/// Complete geometry encoder: point cloud -> 256-D latent
#[derive(Module, Debug)]
pub struct GeometryEncoder<B: Backend> {
    patch_embed: EdgeConvEmbed<B>,
    transformer: Transformer<B>,
    pool: AdaptiveAvgPool1d,
    proj: Linear<B>,
}

impl<B: Backend> GeometryEncoder<B> {
    pub fn new(device: &B::Device) -> Self {
        let patch_embed = EdgeConvEmbed::new(32, 3, 128, device);
        let transformer = Transformer::new(128, 8, device); // embed_dim=128, 8 heads
        let pool = AdaptiveAvgPool1dConfig::new(1).init();  // global average
        let proj = LinearConfig::new(128, 256).init(device); // 256-D latent
        Self { patch_embed, transformer, pool, proj }
    }

    /// points: [B, N, 3]  -> latent: [B, 256]
    pub fn forward(&self, points: Tensor<B, 3>) -> Tensor<B, 2> {
        let tokens = self.patch_embed.forward(points); // [B, N, 128]
        let tokens = self.transformer.forward(tokens); // [B, N, 128]
        let pooled = self.pool.forward(tokens.swap_dims(1, 2)) // [B, 128, 1]
                        .squeeze(2);                          // [B, 128]
        self.proj.forward(pooled)                               // [B, 256]
    }
}

// ------------------------

use burn::nn::Dropout;
use burn::nn::DropoutConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::Relu;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::pool::AdaptiveAvgPool2d;
use burn::nn::pool::AdaptiveAvgPool2dConfig;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::ClassificationOutput;
use burn::train::TrainOutput;
use burn::train::TrainStep;
use burn::train::ValidStep;

use crate::data::MnistBatch;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        let x = images.reshape([batch_size, 1, height, width]);
        let x = self.conv1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        // x shape
        let x = self.pool.forward(x);
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        x
    }

    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item.images, item.targets)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = 0.5)]
    dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            activation: Relu::new(),
        }
    }
}
