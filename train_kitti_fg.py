"""Factor graph training script for visual odometry task."""

import fannypack
from tqdm.auto import tqdm

from lib import experiment_files, kitti, utils, validation_tracker


def main(config: kitti.experiment_config.FactorGraphExperimentConfig) -> None:
    experiment = experiment_files.ExperimentFiles(
        identifier=config.experiment_identifier.format(dataset_fold=config.dataset_fold)
    ).clear()
    experiment.write_metadata("experiment_config", config)
    experiment.write_metadata("git_commit_hash", utils.get_git_commit_hash())

    # Set random seed (for everything but JAX)
    utils.set_random_seed(config.random_seed)

    # Load dataset
    train_dataloader = kitti.data_loading.make_subsequence_dataloader(
        config, split=kitti.data_loading.DatasetSplit.TRAIN_VIRTUAL_SENSOR_HOLDOUT
    )
    val_dataset = kitti.data_loading.make_subsequence_eval_dataset(config)

    # Helper for validation + metric-aware checkpointing
    validation = validation_tracker.ValidationTracker[kitti.training_fg.TrainState](
        name="val",
        experiment=experiment,
        compute_metrics=kitti.validation_fg.make_compute_metrics(val_dataset),
    )

    # Train
    train_state = kitti.training_fg.TrainState.initialize(config, train=True)
    for epoch in tqdm(range(config.num_epochs)):
        batch: kitti.data.KittiStructNormalized
        for batch in train_dataloader:
            # Validation + checkpointing
            # We intentionally do this before the first training step :)
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

        # Simple early stopping
        if (
            (epoch > config.num_epochs // 3)
            and (validation.best_step is not None)
            and (validation.best_step <= int(train_state.steps) // 2)
        ):
            print(
                f"Early stopping: {validation.best_step=}, {epoch=}, {int(train_state.steps)=}"
            )
            break


if __name__ == "__main__":
    fannypack.utils.pdb_safety_net()
    config = utils.parse_args(
        kitti.experiment_config.FactorGraphExperimentConfig,
        description=__doc__,
    )
    main(config)
