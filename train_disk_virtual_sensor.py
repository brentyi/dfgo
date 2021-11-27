"""Pre-training script for visual tracking task virtual sensors."""

import functools

import dcargs
import fifteen
from tqdm.auto import tqdm

from lib import disk, utils, validation_tracker


def main(
    config: disk.experiment_config.VirtualSensorPretrainingExperimentConfig,
) -> None:
    experiment = fifteen.experiments.Experiment(
        identifier=config.experiment_identifier.format(dataset_fold=config.dataset_fold)
    ).clear()
    experiment.write_metadata("experiment_config", config)
    experiment.write_metadata("git_commit_hash", utils.get_git_commit_hash())

    # Set random seed (for everything but JAX)
    utils.set_random_seed(config.random_seed)

    # Load dataset
    train_dataloader = disk.data_loading.make_single_step_dataloader(config, train=True)
    val_dataloader = disk.data_loading.make_single_step_dataloader(config, train=False)

    # Helper for validation + metric-aware checkpointing
    validation = validation_tracker.ValidationTracker[
        disk.training_virtual_sensor.TrainState
    ](
        name="val",
        experiment=experiment,
        compute_metrics=functools.partial(
            disk.validation_virtual_sensor.compute_metrics,
            val_dataloader,
        ),
    )

    # Training loop
    train_state = disk.training_virtual_sensor.TrainState.initialize(config)
    for epoch in tqdm(range(config.num_epochs)):
        batch: disk.data.DiskStructNormalized
        for batch in train_dataloader:
            # Training step!
            train_state, log_data = train_state.training_step(batch)

            # Log to Tensorboard
            experiment.log(
                log_data=log_data,
                step=train_state.steps,
                log_scalars_every_n=50,
                log_histograms_every_n=300,
            )

            # Checkpoint and record validation error every 500 timesteps
            if train_state.steps % 1000 == 0:
                validation.validate_log_and_checkpoint_if_best(train_state)

    # Save finals parameters
    experiment.save_checkpoint(
        target=train_state.learnable_params,
        step=train_state.steps,
        prefix="final_params_",
    )


if __name__ == "__main__":
    fifteen.utils.pdb_safety_net()
    config = dcargs.parse(
        disk.experiment_config.VirtualSensorPretrainingExperimentConfig,
        description=__doc__,
    )
    main(config)
