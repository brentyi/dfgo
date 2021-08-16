from typing import Iterable, List

import jaxfg
import numpy as onp
from jax import numpy as jnp

from . import data, fg_system


def make_factor_graph(trajectory_length: int) -> jaxfg.core.StackedFactorGraph:
    """Make factor graph for optimization."""

    variables = []
    factors: List[jaxfg.core.FactorBase] = []
    for t in range(trajectory_length):
        variables.append(fg_system.StateVariable())

        # Add perception constraint
        factors.append(
            fg_system.VisionFactor.make(
                state_variable=variables[-1],
                predicted_position=onp.full(  # type: ignore
                    (2,), onp.nan, dtype=onp.float32
                ),  # To be populated by network
                sqrt_precision_diagonal=onp.full(  # type: ignore
                    (2,), onp.nan, dtype=onp.float32
                ),  # To be populated by network
            )
        )

        # Add dynamics constraint
        if t != 0:
            factors.append(
                fg_system.DynamicsFactor.make(
                    before_variable=variables[-2],
                    after_variable=variables[-1],
                )
            )

    return jaxfg.core.StackedFactorGraph.make(factors)


def assignments_from_trajectory(
    trajectory_unnorm: data.DiskStructRaw,
    variables: Iterable[jaxfg.core.VariableBase],
) -> jaxfg.core.VariableAssignments:
    """Convert the positions and velocities in a trajectory to an assignments object."""
    return jaxfg.core.VariableAssignments(
        storage=jnp.concatenate(
            [trajectory_unnorm.position, trajectory_unnorm.velocity], axis=-1
        ).flatten(),
        storage_metadata=jaxfg.core.StorageMetadata.make(
            variables=variables, local=False
        ),
    )
