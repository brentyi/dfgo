from typing import Callable

import numpy as onp
from tqdm.auto import tqdm

from .. import validation_tracker
from . import data, data_loading, training_ekf


def make_compute_metrics(
    dataset_fold: int, progress_bar: bool = False
) -> Callable[[training_ekf.TrainState], validation_tracker.ValidationMetrics]:

    eval_trajectories = data_loading.load_trajectories(train=False, fold=dataset_fold)

    def compute_metrics(
        train_state: training_ekf.TrainState,
    ) -> validation_tracker.ValidationMetrics:

        batch: data.DiskStructNormalized
        mse: float = 0.0

        # TODO: there's no reason not to use vmap here
        for traj in tqdm(eval_trajectories) if progress_bar else eval_trajectories:
            posteriors = train_state.run_ekf(traj)

            unnorm_positions = (
                data.DiskStructNormalized(position=traj.position).unnormalize().position
            )
            mse += onp.mean((posteriors.mean.position - unnorm_positions) ** 2)

        rmse = onp.sqrt(mse / len(eval_trajectories))
        return rmse, {"rmse": rmse}

    return compute_metrics
