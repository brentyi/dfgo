"""LSTM training script for visual tracking task."""

import pathlib

import dcargs
import fifteen
from tqdm.auto import tqdm

from lib import disk, utils, validation_tracker


def main(config: disk.experiment_config.LstmExperimentConfig) -> None:
    experiment = fifteen.experiments.Experiment(
        data_dir=pathlib.Path("./experiments/")
        / config.experiment_identifier.format(dataset_fold=config.dataset_fold)
    ).clear()
    experiment.write_metadata("experiment_config", config)
    experiment.write_metadata("git_commit_hash", utils.get_git_commit_hash())

    # Set random seed (for everything but JAX)
    utils.set_random_seed(config.random_seed)

    # Load dataset
    train_dataloader = disk.data_loading.make_subsequence_dataloader(
        config=config, train=True
    )

    # Helper for validation + metric-aware checkpointing
    validation = validation_tracker.ValidationTracker[disk.training_lstm.TrainState](
        name="val",
        experiment=experiment,
        compute_metrics=disk.validation_lstm.make_compute_metrics(
            dataset_fold=config.dataset_fold
        ),
    )

    # Train
    train_state = disk.training_lstm.TrainState.initialize(config)
    for epoch in tqdm(range(config.num_epochs)):
        batch: disk.data.DiskStructNormalized
        for batch in train_dataloader:
            # Validation + checkpointing
            if train_state.steps % 500 == 0:
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
    fifteen.utils.pdb_safety_net()
    config = dcargs.parse(
        disk.experiment_config.LstmExperimentConfig,
        description=__doc__,
    )
    main(config)
