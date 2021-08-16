"""Helper functions for working with factor graphs."""

from typing import List

import jax
import jax_dataclasses
import jaxfg
import jaxlie
from jax import numpy as jnp

from . import data, fg_system, networks


def make_factor_graph(sequence_length: int) -> jaxfg.core.StackedFactorGraph:
    """Make a factor graph template."""
    variables: List[fg_system.StateVariable] = []
    factors: List[jaxfg.core.FactorBase] = []
    for t in range(sequence_length):
        variables.append(fg_system.StateVariable())

        if t == 0:
            # Add prior constraint
            factors.append(
                fg_system.PriorFactor.make(
                    variable=variables[0],
                    mu=fg_system.StateVariable.get_default_value(),  # To be populated
                    noise_model=jaxfg.noises.DiagonalGaussian(
                        sqrt_precision_diagonal=jnp.ones(5) * 1e7
                    ),
                )
            )

        else:
            # Add perception constraint
            factors.append(
                fg_system.VisionFactor.make(
                    state_variable=variables[-1],
                    predicted_velocity=jnp.zeros(2),  # To be populated
                    noise_model=jaxfg.noises.DiagonalGaussian(
                        jnp.ones(2)  # To be populated
                    ),
                )
            )

            # Add dynamics constraint
            factors.append(
                fg_system.DynamicsFactor.make(
                    before_variable=variables[-2],
                    after_variable=variables[-1],
                    noise_model=jaxfg.noises.DiagonalGaussian(
                        jnp.ones(5)  # To be populated
                    ),
                )
            )

    return jaxfg.core.StackedFactorGraph.make(factors)


def update_factor_graph(
    graph_template: jaxfg.core.StackedFactorGraph,
    trajectory_raw: data.KittiStructRaw,
    predicted_velocities: networks.RegressedVelocities,
    vision_sqrt_precision_diagonal: jnp.ndarray,
    dynamics_sqrt_precision_diagonal: jnp.ndarray,
) -> jaxfg.core.StackedFactorGraph:
    """Update the placeholder values in a factor graph template."""
    timesteps: int = len(tuple(graph_template.get_variables()))
    assert (
        predicted_velocities.linear_vel.shape
        == predicted_velocities.angular_vel.shape
        == (timesteps - 1,)
    )
    assert vision_sqrt_precision_diagonal.shape == (timesteps - 1, 2)
    assert dynamics_sqrt_precision_diagonal.shape == (5,)

    with jax_dataclasses.copy_and_mutate(graph_template, validate=True) as graph:
        # Get each stacked factor object
        prior_factor: fg_system.PriorFactor
        vision_factor: fg_system.VisionFactor
        dynamics_factor: fg_system.DynamicsFactor

        prior_factor, vision_factor, dynamics_factor = (
            stack.factor for stack in graph.factor_stacks
        )
        assert isinstance(prior_factor, fg_system.PriorFactor)
        assert isinstance(vision_factor, fg_system.VisionFactor)
        assert isinstance(dynamics_factor, fg_system.DynamicsFactor)

        # Anchor first timestep
        # Note that we need to add a batch dimension!
        prior_factor.mu = jax.tree_map(
            lambda x: x[None, ...],
            fg_system.State.make(
                pose=jaxlie.SE2.from_xy_theta(
                    x=trajectory_raw.x[0],
                    y=trajectory_raw.y[0],
                    theta=trajectory_raw.theta[0],
                ),
                linear_vel=trajectory_raw.linear_vel[0],
                angular_vel=trajectory_raw.angular_vel[0],
            ),
        )

        # Update vision factors
        vision_factor.predicted_velocity = jnp.stack(
            [
                predicted_velocities.linear_vel,
                predicted_velocities.angular_vel,
            ],
            axis=-1,
        )
        vision_factor.noise_model.sqrt_precision_diagonal = (
            vision_sqrt_precision_diagonal
        )

        # Update dynamics factors
        dynamics_factor.noise_model.sqrt_precision_diagonal = jnp.tile(
            dynamics_sqrt_precision_diagonal[None, :],
            reps=(timesteps - 1, 1),
        )

    return graph


def assignments_from_trajectory(
    trajectory_raw: data.KittiStructRaw,
    graph: jaxfg.core.StackedFactorGraph,
) -> jaxfg.core.VariableAssignments:
    """Populate variable assignments from x, y, and theta values."""
    (timesteps,) = trajectory_raw.x.shape
    assert (timesteps,) == trajectory_raw.y.shape == trajectory_raw.theta.shape

    init_pose_params = jax.vmap(jaxlie.SE2.from_xy_theta)(
        trajectory_raw.x,
        trajectory_raw.y,
        trajectory_raw.theta,
    ).unit_complex_xy
    init_velocities = trajectory_raw.get_stacked_velocity()
    assert init_pose_params.shape == (timesteps, 4)
    assert init_velocities.shape == (timesteps, 2)

    stacked_storage = jnp.concatenate(
        [init_pose_params, init_velocities],
        axis=-1,
    )
    return jaxfg.core.VariableAssignments(
        storage=stacked_storage.flatten(),
        storage_metadata=jaxfg.core.StorageMetadata.make(
            graph.get_variables(),
            local=False,
        ),
    )


def initial_assignments_from_graph(
    graph: jaxfg.core.StackedFactorGraph,
) -> jaxfg.core.VariableAssignments:
    """Compute initial assignments by forward-integrating predicted velocities."""
    # Get each stacked factor object
    prior_factor: fg_system.PriorFactor
    vision_factor: fg_system.VisionFactor
    dynamics_factor: fg_system.DynamicsFactor

    prior_factor, vision_factor, dynamics_factor = (
        stack.factor for stack in graph.factor_stacks
    )

    states: List[fg_system.State] = [
        jax.tree_map(lambda x: x.squeeze(axis=0), prior_factor.mu)
    ]
    for i in range(vision_factor.predicted_velocity.shape[0]):
        with jax_dataclasses.copy_and_mutate(states[-1].predict_next()) as next_state:
            next_state.velocities = vision_factor.predicted_velocity[i, :]
        states.append(next_state)

    return jaxfg.core.VariableAssignments.make_from_dict(
        dict(zip(graph.get_variables(), states))
    )
