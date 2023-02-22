"""Pre-training script for visual odometry task virtual sensors."""
import pathlib

import fifteen
import tyro
from tqdm.auto import tqdm

from lib import kitti, utils, validation_tracker


def main(
    config: kitti.experiment_config.VirtualSensorPretrainingExperimentConfig,
) -> None:
    experiment = fifteen.experiments.Experiment(
        data_dir=pathlib.Path("./experiments/")
        / config.experiment_identifier.format(dataset_fold=config.dataset_fold)
    ).clear()
    experiment.write_metadata("experiment_config", config)
    experiment.write_metadata("git_commit_hash", utils.get_git_commit_hash())

    # Set random seed (for everything but JAX)
    utils.set_random_seed(config.random_seed)

    # Load dataset
    train_dataloader = kitti.data_loading.make_single_step_dataloader(
        config, split=kitti.data_loading.DatasetSplit.TRAIN_VIRTUAL_SENSOR
    )

    # Helper for validation + metric-aware checkpointing
    validation = validation_tracker.ValidationTracker[
        kitti.training_virtual_sensor.TrainState
    ](
        name="val",
        experiment=experiment,
        compute_metrics=kitti.validation_virtual_sensor.make_compute_metrics(
            eval_dataloader=kitti.data_loading.make_single_step_dataloader(
                config, split=kitti.data_loading.DatasetSplit.VALIDATION
            )
        ),
    )

    # Train
    train_state = kitti.training_virtual_sensor.TrainState.initialize(
        config, train=True
    )
    for epoch in tqdm(range(config.num_epochs)):
        batch: kitti.data.KittiStructNormalized
        for batch in train_dataloader:
            # Validation + checkpointing
            if train_state.steps % 500 == 0:
                validation = validation.validate_log_and_checkpoint_if_best(train_state)

            # Training step!
            train_state, log_data = train_state.training_step(batch)
            experiment.log(
                log_data,
                step=train_state.steps,
                log_scalars_every_n=10,
                log_histograms_every_n=50,
            )

    # Back up training state
    experiment.save_checkpoint(
        target=train_state.learnable_params,
        step=train_state.steps,
        prefix="last_params_",
    )


if __name__ == "__main__":
    fifteen.utils.pdb_safety_net()
    config = tyro.cli(
        kitti.experiment_config.VirtualSensorPretrainingExperimentConfig,
        description=__doc__,
    )
    main(config)
