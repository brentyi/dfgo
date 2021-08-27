import dataclasses

import jax_dataclasses
from tqdm.auto import tqdm

from lib import experiment_files, kitti, utils


@dataclasses.dataclass
class Args:
    fg_experiment_identifier: str = (
        "kitti/fg/hetero/surrogate_pos/ground_truth-5/fold_{dataset_fold}"
    )
    ekf_experiment_identifier: str = "kitti/ekf/hetero/fold_{dataset_fold}"


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
            "experiment_config", kitti.experiment_config.EkfExperimentConfig
        )
        fg_config = fg_experiment.read_metadata(
            "experiment_config", kitti.experiment_config.FactorGraphExperimentConfig
        )

        # Initialize training states
        ekf_train_state = kitti.training_ekf.TrainState.initialize(
            ekf_config, train=False
        )
        fg_train_state = kitti.training_fg.TrainState.initialize(fg_config, train=False)

        # Load uncertainty models... but swapped!
        with jax_dataclasses.copy_and_mutate(
            ekf_train_state, validate=False
        ) as ekf_train_state:
            ekf_train_state.learnable_params = fg_experiment.restore_checkpoint(
                target=ekf_train_state.learnable_params,
                prefix="best_val_params_",
            )
        with jax_dataclasses.copy_and_mutate(
            fg_train_state, validate=False
        ) as fg_train_state:
            fg_train_state.learnable_params = ekf_experiment.restore_checkpoint(
                target=fg_train_state.learnable_params,
                prefix="best_val_params_",
            )

        # Evaluate each training state
        _, ekf_metrics = kitti.validation_ekf.make_compute_metrics(
            eval_dataset=kitti.data_loading.make_subsequence_eval_dataset(ekf_config)
        )(ekf_train_state)

        _, fg_metrics = kitti.validation_fg.make_compute_metrics(
            eval_dataset=kitti.data_loading.make_subsequence_eval_dataset(fg_config)
        )(fg_train_state)

        # Write metrics to kitti
        experiment_files.ExperimentFiles(
            identifier=f"kitti/ekf/hetero/trained_on_fg/fold_{dataset_fold}"
        ).clear().write_metadata("best_val_metrics", ekf_metrics)
        experiment_files.ExperimentFiles(
            identifier=f"kitti/fg/hetero/trained_on_ekf/fold_{dataset_fold}"
        ).clear().write_metadata("best_val_metrics", fg_metrics)


if __name__ == "__main__":
    args = utils.parse_args(Args)
    main(args)
