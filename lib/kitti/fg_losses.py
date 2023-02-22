"""Loss definitions for KITTI task."""


from typing import Union

import jax
import jax_dataclasses
import jaxfg
from jax import numpy as jnp

from . import data, experiment_config, fg_system, fg_utils


def compute_loss(
    graph: jaxfg.core.StackedFactorGraph,
    trajectory_raw: data.KittiStructRaw,
    loss_config: Union[
        experiment_config.JointNllLossConfig,
        experiment_config.SurrogateLossConfig,
    ],
    prng_key: jax.random.KeyArray,
) -> jnp.ndarray:
    """Given an updated factor graph, ground-truth trajectory, and loss config, compute
    a single-trajectory loss."""
    if isinstance(loss_config, experiment_config.JointNllLossConfig):
        # Joint NLL baseline
        gt_assignments = fg_utils.assignments_from_trajectory(trajectory_raw, graph)
        return graph.compute_joint_nll(gt_assignments)
    elif isinstance(loss_config, experiment_config.SurrogateLossConfig):
        # Compute surrogate loss
        return _compute_surrogate_loss(graph, trajectory_raw, loss_config, prng_key)


def _compute_surrogate_loss(
    graph: jaxfg.core.StackedFactorGraph,
    trajectory_raw: data.KittiStructRaw,
    loss_config: experiment_config.SurrogateLossConfig,
    prng_key: jax.random.KeyArray,
) -> jnp.ndarray:
    """Compute an end-to-end loss."""
    timesteps: int = len(tuple(graph.get_variables()))

    # Switch on how the optimizer is initialized
    if (
        loss_config.gn_initialization_strategy
        is experiment_config.InitializationStrategyEnum.MIXED
    ):
        # Include both SAME_AS_EVAL and GROUND_TRUTH initialization strategies in loss
        prng_key0, prng_key1 = jax.random.split(prng_key)
        return 0.5 * (
            _compute_surrogate_loss(
                graph,
                trajectory_raw,
                loss_config=jax_dataclasses.replace(
                    loss_config,
                    gn_initialization_strategy=experiment_config.InitializationStrategyEnum.SAME_AS_EVAL,
                ),
                prng_key=prng_key0,
            )
            + _compute_surrogate_loss(
                graph,
                trajectory_raw,
                loss_config=jax_dataclasses.replace(
                    loss_config,
                    gn_initialization_strategy=experiment_config.InitializationStrategyEnum.GROUND_TRUTH,
                ),
                prng_key=prng_key1,
            )
        )  # type: ignore

    if (
        loss_config.gn_initialization_strategy
        is experiment_config.InitializationStrategyEnum.SAME_AS_EVAL
    ):
        # Use same initialization heuristics as during eval mode
        initial_assignments = fg_utils.initial_assignments_from_graph(graph=graph)
    elif (
        loss_config.gn_initialization_strategy
        is experiment_config.InitializationStrategyEnum.GROUND_TRUTH
    ):
        # Initialize assignments to ground-truth values
        # These are the main results we present in the paper!
        initial_assignments = fg_utils.assignments_from_trajectory(
            trajectory_raw=trajectory_raw, graph=graph
        )
    elif (
        loss_config.gn_initialization_strategy
        is experiment_config.InitializationStrategyEnum.NAIVE_BASELINE
    ):
        # Initialize assignments to default variable values
        initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(
            graph.get_variables()
        )
    else:
        assert False

    # Initialization noise injection for ablations
    if loss_config.gn_initialization_noise_std != 0.0:
        std = loss_config.gn_initialization_noise_std
        assert std > 0.0

        # Some type abuse, putting in floats where we expect jnp.ndarray
        unnorm_stds = data.KittiStructNormalized(
            x=std, y=std, linear_vel=std, angular_vel=std  # type: ignore
        ).unnormalize(scale_only=True)

        tangent_noise = jax.random.normal(
            key=prng_key, shape=(len(initial_assignments.get_variables()), 5)
        ) * jnp.array(
            [
                unnorm_stds.x,
                unnorm_stds.y,
                std * jnp.pi,
                unnorm_stds.linear_vel,
                unnorm_stds.angular_vel,
            ]
        )

        with jax_dataclasses.copy_and_mutate(
            initial_assignments
        ) as initial_assignments:
            initial_assignments.storage = jax.flatten_util.ravel_pytree(
                # Perturb all states
                jax.vmap(fg_system.StateVariable.manifold_retract)(
                    initial_assignments.get_stacked_value(fg_system.StateVariable),
                    tangent_noise,
                )
            )[0]

    # Solve factor graph
    solution_assignments = graph.solve(
        initial_assignments,
        solver=jaxfg.solvers.FixedIterationGaussNewtonSolver(
            unroll=True,
            iterations=loss_config.gn_steps,
            linear_solver=jaxfg.sparse.ConjugateGradientSolver(
                tolerance=loss_config.conjugate_gradient_tolerance
            ),
            verbose=False,
        ),
    )
    stacked_solved: fg_system.State = solution_assignments.get_stacked_value(
        fg_system.StateVariable
    )

    # Compute surrogate loss
    if (
        loss_config.supervision
        is experiment_config.SurrogateLossSupervisionEnum.POSITION_XY
    ):
        # Compute XY position error
        gt_translation = jnp.stack([trajectory_raw.x, trajectory_raw.y], axis=-1)
        assert (
            stacked_solved.pose.translation().shape
            == gt_translation.shape
            == (timesteps, 2)
        )
        translation_delta = stacked_solved.pose.translation() - gt_translation
        return jnp.mean(translation_delta**2)

    elif (
        loss_config.supervision
        is experiment_config.SurrogateLossSupervisionEnum.VELOCITY
    ):
        # Minimize linear, angular velocity errors
        gt_velocities = trajectory_raw.get_stacked_velocity()
        assert gt_velocities.shape == stacked_solved.velocities.shape == (timesteps, 2)
        velocity_delta: jnp.ndarray = gt_velocities - stacked_solved.velocities
        return jnp.mean(velocity_delta**2)

    else:
        assert False
