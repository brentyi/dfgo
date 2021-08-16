"""LSTM training script for visual odometry task."""

import fannypack
from tqdm.auto import tqdm

from lib import experiment_files, kitti, utils, validation_tracker


def main(config: kitti.experiment_config.LstmExperimentConfig) -> None:
    experiment = experiment_files.ExperimentFiles(
        identifier=config.experiment_identifier.format(dataset_fold=config.dataset_fold)
    ).clear()
    experiment.write_metadata("experiment_config", config)
    experiment.write_metadata("git_commit_hash", utils.get_git_commit_hash())

    # Set random seed (for everything but JAX)
    utils.set_random_seed(config.random_seed)

    # Load dataset
    train_dataloader = kitti.data_loading.make_subsequence_dataloader(
        # Note that we use all of the training data here! Because there's no pretrained
        # virtual sensor
        config,
        split=kitti.data_loading.DatasetSplit.TRAIN_ALL,
    )
    train_partial_dataloader = kitti.data_loading.make_subsequence_dataloader(
        config,
        split=kitti.data_loading.DatasetSplit.TRAIN_VIRTUAL_SENSOR_HOLDOUT,  # For debugging, this overlaps with TRAIN_ALL
    )
    val_dataloader = kitti.data_loading.make_subsequence_dataloader(
        config, split=kitti.data_loading.DatasetSplit.VALIDATION
    )

    # Helper for validation + metric-aware checkpointing
    validation_debugging = validation_tracker.ValidationTracker[
        kitti.training_lstm.TrainState
    ](
        name="train_partial",
        experiment=experiment,
        compute_metrics=kitti.validation_lstm.make_compute_metrics(
            train_partial_dataloader  # Should overfit
        ),
    )
    validation = validation_tracker.ValidationTracker[kitti.training_lstm.TrainState](
        name="val",
        experiment=experiment,
        compute_metrics=kitti.validation_lstm.make_compute_metrics(val_dataloader),
    )

    # Train
    train_state = kitti.training_lstm.TrainState.initialize(config, train=True)
    for epoch in tqdm(range(config.num_epochs)):
        batch: kitti.data.KittiStructNormalized
        for batch in train_dataloader:
            # Validation + checkpointing
            # We intentionally do this before this first training step :)
            if train_state.steps % 2000 == 0:
                validation_debugging = (
                    validation_debugging.validate_log_and_checkpoint_if_best(
                        train_state
                    )
                )
            if train_state.steps % 200 == 0:
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
    fannypack.utils.pdb_safety_net()
    config = utils.parse_args(kitti.experiment_config.LstmExperimentConfig)
    main(config)
