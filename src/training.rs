use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::LossMetric;
use burn::train::{Learner, SupervisedTraining};

use crate::data::{PointCloudBatcher, PointCloudDataset};
use crate::model::GeometryAutoEncoderConfig;

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: GeometryAutoEncoderConfig,
    pub optimizer: AdamConfig,
    #[config(default = 30)]
    pub num_epochs: usize,
    #[config(default = 1)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let batcher = PointCloudBatcher::new(config.model.num_points_sampled);
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(PointCloudDataset::from_dir(
            "./dataset",
            "train",
            config.model.num_points_sampled,
        ));

    let batcher = PointCloudBatcher::new(config.model.num_points_sampled);
    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(PointCloudDataset::from_dir(
            "./dataset",
            "test",
            config.model.num_points_sampled,
        ));

    // this is supposed to get a validation dataset, not test
    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())

        .num_epochs(config.num_epochs)
        .summary();

    let result = training.launch(Learner::new(
        config.model.init::<B>(&device),
        config.optimizer.init(),
        config.learning_rate,
    ));

    // let model_trained = learner.fit(, dataloader_test);

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
