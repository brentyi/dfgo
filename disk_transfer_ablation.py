import dataclasses

import jax_dataclasses
from tqdm.auto import tqdm

from lib import disk, experiment_files, utils


@dataclasses.dataclass
class Args:
    fg_experiment_identifier: str = (
        "disk/fg/heteroscedastic/surrogate_loss/fold_{dataset_fold}"
    )
    ekf_experiment_identifier: str = (
        "disk/ekf/heteroscedastic/e2e_marginal_nll/fold_{dataset_fold}"
    )


def main(args: Args) -> None:
    for dataset_fold in tqdm(range(10)):
        # Experiments to transfer noise models across
        ekf_experiment = experiment_files.ExperimentFiles(
            identifier=args.ekf_experiment_identifier.format(dataset_fold=dataset_fold)
        ).assert_exists()
        fg_experiment = experiment_files.ExperimentFiles(
            identifier=args.fg_experiment_identifier.format(dataset_fold=dataset_fold)
        ).assert_exists()

        # Read experiment configurations for each experiment
        ekf_config = ekf_experiment.read_metadata(
            "experiment_config", disk.experiment_config.EkfExperimentConfig
        )
        fg_config = fg_experiment.read_metadata(
            "experiment_config", disk.experiment_config.FactorGraphExperimentConfig
        )

        # Initialize training states
        ekf_train_state = disk.training_ekf.TrainState.initialize(ekf_config)
        fg_train_state = disk.training_fg.TrainState.initialize(fg_config)

        # Load train uncertainty models... but swapped
        with jax_dataclasses.copy_and_mutate(ekf_train_state) as ekf_train_state:
            ekf_train_state.learnable_params = fg_experiment.restore_checkpoint(
                target=ekf_train_state.learnable_params,
                prefix="best_val_params_",
            )
        with jax_dataclasses.copy_and_mutate(fg_train_state) as fg_train_state:
            fg_train_state.learnable_params = ekf_experiment.restore_checkpoint(
                target=fg_train_state.learnable_params,
                prefix="best_val_params_",
            )

        # Evaluate each training state
        _, ekf_metrics = disk.validation_ekf.make_compute_metrics(
            dataset_fold=dataset_fold, progress_bar=True
        )(ekf_train_state)

        _, fg_metrics = disk.validation_fg.make_compute_metrics(
            dataset_fold=dataset_fold, progress_bar=True
        )(fg_train_state)

        # Write metrics to disk
        experiment_files.ExperimentFiles(
            identifier=f"disk/ekf/heteroscedastic/trained_on_fg/fold_{dataset_fold}"
        ).clear().write_metadata("best_val_metrics", ekf_metrics)
        experiment_files.ExperimentFiles(
            identifier=f"disk/fg/heteroscedastic/trained_on_ekf/fold_{dataset_fold}"
        ).clear().write_metadata("best_val_metrics", fg_metrics)


if __name__ == "__main__":
    args = utils.parse_args(Args)
    main(args)
