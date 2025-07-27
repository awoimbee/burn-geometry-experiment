use burn::module::Module;
use burn::nn::attention::{self, MhaInput};
use burn::nn::pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, Relu};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// k-NN helper (squared L2, brute-force for clarity)
// pub fn knn<B: Backend>(points: Tensor<B, 3>, k: usize) -> Tensor<B, 3, Int> {
//     let device = points.device();
//     let shape = points.shape();
//     let num_point_clouds = shape.dims[0];
//     let num_points = shape.dims[1];

//     // Expand dims for broadcasting: [B, N, 1, 3] and [B, 1, N, 3]
//     let points_i = points.clone().unsqueeze_dim::<4>(2); // [B, N, 1, 3]
//     let points_j = points.unsqueeze_dim::<4>(1); // [B, 1, N, 3]
//     // Pairwise squared distances: [B, N, N]
//     let diff = points_i.clone() - points_j; // [B, N, N, 3]
//     let mut dist2 = diff.powi_scalar(2).sum_dim(3).squeeze::<3>(3); // [B, N, N]

//     let indices = Tensor::<B, 1, Int>::arange(0..(num_points as i64), &device);
//     // diagonal mask
//     let mask = indices
//         .clone()
//         .unsqueeze::<2>() // [N, 1]
//         .expand([num_points, num_points]) // [N, N]
//         .equal(indices.unsqueeze::<1>().expand([num_points, num_points])); // [N, N]
//     let mask = mask
//         .unsqueeze::<3>()
//         .expand([num_point_clouds, num_points, num_points]); // [B, N, N]
//     dist2 = dist2.mask_fill(mask, 1e10); // Set diagonal to large value

//     // Get indices of k smallest distances (excluding self)
//     // burn's topk returns largest, so use negative distances
//     let neg_dist2 = dist2.neg();
//     let (_topk_vals, topk_indices) = neg_dist2.topk_with_indices(k, 2); // [B, N, k]

//     topk_indices
// }

fn batched_knn<B: Backend>(points: Tensor<B, 3>, k: usize) -> Tensor<B, 3, Int>
where
    B: Backend,
{
    let device = points.device();
    let [batch_size, num_points, _] = points.dims();

    // Compute pairwise distances
    let points_expanded_1 = points.clone().unsqueeze_dim::<4>(2); // (B, N, 1, 3)
    let points_expanded_2 = points.unsqueeze_dim::<4>(1); // (B, 1, N, 3)

    let diff = points_expanded_1 - points_expanded_2; // (B, N, N, 3)
    let distances = diff.powi_scalar(2).sum_dim(3).squeeze::<3>(3).sqrt(); // (B, N, N)

    // Create diagonal mask and set diagonal to infinity
    let eye = Tensor::<B, 2>::eye(num_points, &device);
    let mask = eye.unsqueeze_dim::<3>(0).repeat(&[0, batch_size]); // (B, N, N)
    // let inf_tensor = Tensor::<B, 3>::full(distances.shape(), f32::INFINITY, &device);

    // Apply mask: set diagonal elements to infinity
    let distances_masked: Tensor<B, 3> =
        distances.clone() * (1.0 - mask.clone()) + mask * f32::INFINITY;

    // Find k nearest neighbors
    let (_, indices) = distances_masked.topk_with_indices(k, 2);

    indices
}

/// EdgeConv patch (local neighborhoods) embedder to produce a "local shape token"
///
/// Data preprocessing:
/// * point cloud with N points (2048 ?)
/// * center and scale to unit sphere
#[derive(Module, Debug)]
pub struct EdgeConvEmbed<B: Backend> {
    theta: Linear<B>, // h_θ
    phi: Linear<B>,   // h_φ
    psi: Linear<B>,   // h_ψ
    norm: LayerNorm<B>,
    relu: Relu,
    k: usize,
}

impl<B: Backend> EdgeConvEmbed<B> {
    /// k: how many nearest neighbours (20, 32, 40)
    /// in_dim: how many input channels (xyz=3, can be more if includes normals, ...)
    /// out_dim: dimension of the resulting per-point feature that will later be fed to the Transformer (128, ...)
    pub fn new(k: usize, in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        let theta = LinearConfig::new(in_dim, out_dim).init(device);
        let phi = LinearConfig::new(in_dim, out_dim).init(device);
        let psi = LinearConfig::new(out_dim, out_dim).init(device);
        let norm = LayerNormConfig::new(out_dim).init(device);

        Self {
            theta,
            phi,
            psi,
            norm,
            relu: Relu::new(),
            k,
        }
    }

    /// points: [B, N, 3]  -> tokens: [B, N, K]
    pub fn forward(&self, points: Tensor<B, 3>) -> Tensor<B, 3> {
        // 1. k-NN indices [B, N, k]
        let idx = batched_knn(points.clone(), self.k); // [B, N, k]

        // 2. Gather neighbour features
        let points_exp = points
            .clone()
            .unsqueeze_dim::<4>(2)
            .repeat(&[1, 1, self.k, 1]); // [B, N, 1, 3]
        let idx_exp = idx.unsqueeze_dim::<4>(3).repeat(&[1, 1, 1, 3]); // [B, N, k, 3]
        let neighbors = points_exp.gather(1, idx_exp); // [B, N, k, 3]
        // info!("neighbors: {:?}", neighbors.shape());

        // 3. Edge features: h_θ(x_i) || h_φ(x_j - x_i)
        let x_i = points; // [B, N, 3]
        let x_j = neighbors; // [B, N, k, 3]
        let diff = x_j - x_i.clone().unsqueeze_dim::<4>(2); // [B, N, k, 3]

        let h_theta = self.theta.forward(x_i); // [B, N, C]
        let h_phi = self.phi.forward(diff); // [B, N, k, C]

        let edge = h_theta.unsqueeze_dim::<4>(2) + h_phi; // [B, N, k, C]

        // 4. Max-pool over neighbours
        let pooled = edge.max_dim(2).squeeze::<3>(2); // [B, N, C]

        // 5. Final 1×1 conv + norm
        let out = self.psi.forward(pooled); // [B, N, C]
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
            layers.push(TransformerBlock {
                mha,
                norm1,
                norm2,
                linear,
                activation: Relu::new(),
            });
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
        let pool = AdaptiveAvgPool1dConfig::new(1).init(); // global average
        let proj = LinearConfig::new(128, 256).init(device); // 256-D latent
        Self {
            patch_embed,
            transformer,
            pool,
            proj,
        }
    }

    /// points: [B, N, 3]  -> latent: [B, 256]
    pub fn forward(&self, points: Tensor<B, 3>) -> Tensor<B, 2> {
        let tokens = self.patch_embed.forward(points); // [B, N, 128]
        let tokens = self.transformer.forward(tokens); // [B, N, 128]
        let pooled = self
            .pool
            .forward(tokens.swap_dims(1, 2)) // [B, 128, 1]
            .squeeze::<2>(2); // [B, 128]
        self.proj.forward(pooled) // [B, 256]
    }
}
