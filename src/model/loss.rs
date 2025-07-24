use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use super::GeometryAutoEncoder;

impl<B: Backend> GeometryAutoEncoder<B> {
    /// original: [B, N, 3]
    /// reconstructed: [B, M, 3]
    fn chamfer_loss(&self, original: &Tensor<B, 3>, reconstructed: &Tensor<B, 3>) -> Tensor<B, 1> {
        // Compute pairwise distances: [B, N, M]
        let distances = self.pairwise_distances(original, reconstructed);

        // Forward direction: for each point in original, find nearest in reconstructed
        let min_dist_forward = distances.clone().min_dim(2); // [B, N]
        let forward_loss = min_dist_forward.mean_dim(1); // [B]

        // Backward direction: for each point in reconstructed, find nearest in original
        let min_dist_backward = distances.min_dim(1); // [B, M]
        let backward_loss = min_dist_backward.mean_dim(1); // [B]

        // Total Chamfer distance
        (forward_loss + backward_loss).mean()
    }

    /// points1: [B, N, 3]
    /// points2: [B, M, 3]
    fn pairwise_distances(&self, points1: &Tensor<B, 3>, points2: &Tensor<B, 3>) -> Tensor<B, 3> {
        // Expand dimensions for broadcasting
        let p1_expanded = points1.clone().unsqueeze_dim(2); // [B, N, 1, 3]
        let p2_expanded = points2.clone().unsqueeze_dim(1); // [B, 1, M, 3]

        // Compute squared distances
        let diff = p1_expanded - p2_expanded; // [B, N, M, 3]
        let squared_distances = diff.powf_scalar(2.0).sum_dim(3); // [B, N, M]

        squared_distances.sqrt()
    }

    fn smoothness_loss(&self, points: &Tensor<B, 3>) -> Tensor<B, 1> {
        let distances = self.pairwise_distances(points, points);
        // Use exponential weighting to focus on nearby points
        let weights = (-distances.clone()).exp(); // Closer points get higher weight
        let weighted_distances = distances * weights;

        weighted_distances.mean()
    }

    pub fn compute_loss(
        &self,
        points: &Tensor<B, 3>,
        reconstructed: &Tensor<B, 3>,
        latent: &Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        // Main reconstruction loss
        let reconstruction_loss = self.chamfer_loss(points, reconstructed);

        // Optional: L2 regularization on latent codes
        let latent_reg = latent.clone().powf_scalar(2.0).mean() * 0.001;

        // Optional: Encourage smoothness in point cloud
        let smoothness_loss = self.smoothness_loss(reconstructed) * 0.01;

        reconstruction_loss + latent_reg + smoothness_loss
    }
}
