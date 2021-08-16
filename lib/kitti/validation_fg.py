from typing import Callable

import jax
import jax_dataclasses
import jaxfg
import torch
from jax import numpy as jnp

from .. import validation_tracker
from . import data, fg_utils, math_utils, training_fg


@jax_dataclasses.pytree_dataclass
class _ValidationMetrics:
    m_per_m: float
    rad_per_m: float


@jax.jit
def _compute_metrics(
    train_state: training_fg.TrainState,
    graph_template: jaxfg.core.StackedFactorGraph,
    trajectory: data.KittiStructNormalized,
) -> _ValidationMetrics:
    # Leading axes: (# timesteps,)
    assert len(trajectory.check_shapes_and_get_batch_axes()) == 1

    graph, _unused = train_state.update_factor_graph(
        graph_template=graph_template,
        trajectory=trajectory,
        prng_key=jax.random.PRNGKey(
            0
        ),  # only used for dropout -- should not matter in eval mode
    )

    initial_assignments = fg_utils.initial_assignments_from_graph(graph=graph)

    solved_assignments = graph.solve(
        initial_assignments=initial_assignments,
        solver=jaxfg.solvers.LevenbergMarquardtSolver(
            # Slower than CHOLMOD, but batchable
            # linear_solver=jaxfg.sparse.ConjugateGradientSolver(tolerance=1e-5)
            linear_solver=jaxfg.sparse.CholmodSolver(),
            verbose=False,
        ),
    )

    gt_trajectory_raw = trajectory.unnormalize()
    solved_final_state = solved_assignments.get_value(list(graph.get_variables())[-1])

    true_distance_traveled = math_utils.compute_distance_traveled(
        gt_trajectory_raw.x, gt_trajectory_raw.y
    )
    error_m = jnp.sqrt(
        (solved_final_state.x - gt_trajectory_raw.x[-1]) ** 2
        + (solved_final_state.y - gt_trajectory_raw.y[-1]) ** 2
    )
    error_rad = jnp.abs(
        math_utils.wrap_angle(solved_final_state.theta - gt_trajectory_raw.theta[-1])
    )
    assert error_m.shape == error_rad.shape == ()

    return _ValidationMetrics(
        m_per_m=error_m / true_distance_traveled,
        rad_per_m=error_rad / true_distance_traveled,
    )


graph_template: jaxfg.core.StackedFactorGraph


def make_compute_metrics(
    eval_dataset: torch.utils.data.Dataset[data.KittiStructNormalized],
) -> Callable[[training_fg.TrainState], validation_tracker.ValidationMetrics]:

    graph_template = fg_utils.make_factor_graph(sequence_length=100)

    def compute_metrics(
        train_state: training_fg.TrainState,
    ) -> validation_tracker.ValidationMetrics:
        # Eval mode
        train_state = jax_dataclasses.replace(train_state, train=False)

        metrics_summed = _ValidationMetrics(0.0, 0.0)

        # We could use vmap instead of a dumb for loop, but it'd require;
        # - Switching to a conjugate gradient linear solver (currently no support for batch
        #   axes in our cholmod solver)
        # - A lot more GPU memory... this is fine on a 11GB 1080ti but for whatever reason
        #   causes pain on the 16GB Quadro
        for i in range(len(eval_dataset)):  # type: ignore
            traj: data.KittiStructNormalized
            traj = eval_dataset[i]

            # Leading axes: (batch, # timesteps)
            (timesteps,) = traj.check_shapes_and_get_batch_axes()

            batch_metrics = _compute_metrics(
                train_state,
                graph_template,
                traj,
            )
            metrics_summed = jax.tree_map(
                lambda a, b: a + b,
                metrics_summed,
                batch_metrics,
            )

        metrics_avg: _ValidationMetrics = jax.tree_map(lambda x: x / len(eval_dataset), metrics_summed)  # type: ignore
        return metrics_avg.m_per_m, vars(metrics_avg)

    return compute_metrics
