"""EKF end-to-end training script for visual odometry task."""

import pathlib

import fifteen
import tyro
from tqdm.auto import tqdm

from lib import kitti, utils, validation_tracker


def main(config: kitti.experiment_config.EkfExperimentConfig) -> None:
    experiment = fifteen.experiments.Experiment(
        data_dir=pathlib.Path("./experiments/")
        / config.experiment_identifier.format(dataset_fold=config.dataset_fold)
    ).clear()
    experiment.write_metadata("experiment_config", config)
    experiment.write_metadata("git_commit_hash", utils.get_git_commit_hash())

    # Set random seed (for everything but JAX)
    utils.set_random_seed(config.random_seed)

    # Load dataset
    train_dataloader = kitti.data_loading.make_subsequence_dataloader(
        # We use a chunk of the train set that was held out during pretraining -- this
        # will result in uncertainties that generalize better to unseen data.
        config,
        split=kitti.data_loading.DatasetSplit.TRAIN_VIRTUAL_SENSOR_HOLDOUT,
    )
    val_dataset = kitti.data_loading.make_subsequence_eval_dataset(config)

    # Helper for validation + metric-aware checkpointing
    validation = validation_tracker.ValidationTracker[kitti.training_ekf.TrainState](
        name="val",
        experiment=experiment,
        compute_metrics=kitti.validation_ekf.make_compute_metrics(val_dataset),
    )

    # Train
    train_state = kitti.training_ekf.TrainState.initialize(config, train=True)
    for epoch in tqdm(range(config.num_epochs)):
        batch: kitti.data.KittiStructNormalized
        for batch in train_dataloader:
            # Validation + checkpointing
            if train_state.steps % 50 == 0:
                validation = validation.validate_log_and_checkpoint_if_best(train_state)

            # Training step!
            train_state, log_data = train_state.training_step(batch)

            # Log to Tensorboard
            experiment.log(
                log_data,
                step=train_state.steps,
                log_scalars_every_n=10,
                log_histograms_every_n=100,
            )


if __name__ == "__main__":
    config = tyro.cli(kitti.experiment_config.EkfExperimentConfig, description=__doc__)
    main(config)
