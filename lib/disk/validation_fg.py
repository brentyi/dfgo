from typing import Callable

import jax
import jaxfg
import numpy as onp
from jax import numpy as jnp
from tqdm.auto import tqdm

from .. import validation_tracker
from . import data, data_loading, fg_system, fg_utils, training_fg


@jax.jit
def _mse(
    train_state: training_fg.TrainState,
    traj: data.DiskStructNormalized,
    graph_template: jaxfg.core.StackedFactorGraph,
) -> jnp.ndarray:
    traj_unnorm = traj.unnormalize()
    graph = train_state.update_factor_graph(traj, graph_template=graph_template)
    initial_assignments = fg_utils.assignments_from_trajectory(
        traj_unnorm, graph.get_variables()
    )
    positions = (
        graph.solve(
            initial_assignments,
            solver=jaxfg.solvers.GaussNewtonSolver(verbose=False),
        )
        .get_stacked_value(fg_system.StateVariable)
        .position
    )
    assert positions.shape == traj_unnorm.position.shape
    return onp.mean((positions - traj_unnorm.position) ** 2)


def make_compute_metrics(
    dataset_fold: int, progress_bar: bool = False
) -> Callable[[training_fg.TrainState], validation_tracker.ValidationMetrics]:

    eval_trajectories = data_loading.load_trajectories(train=False, fold=dataset_fold)
    (trajectory_length,) = eval_trajectories[0].get_batch_axes()
    graph_template = fg_utils.make_factor_graph(trajectory_length=trajectory_length)

    def compute_metrics(
        train_state: training_fg.TrainState,
    ) -> validation_tracker.ValidationMetrics:

        batch: data.DiskStructNormalized
        mse: float = 0.0

        for traj in tqdm(eval_trajectories) if progress_bar else eval_trajectories:
            mse += float(_mse(train_state, traj, graph_template))

        rmse = onp.sqrt(mse / len(eval_trajectories))
        return rmse, {"rmse": rmse}

    return compute_metrics
