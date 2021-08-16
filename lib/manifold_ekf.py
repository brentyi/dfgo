"""EKF implementation, with support for arbitrary state and observation manifolds."""

import dataclasses
from typing import Any, Callable, Generic, TypeVar, Union

import jax
import jax_dataclasses
import numpy as onp
from jax import flatten_util
from jax import numpy as jnp

# Manifold definition API

ManifoldPoint = TypeVar("ManifoldPoint")


# @dataclasses.dataclass(frozen=True) # <== TODO: when mypy v0.920 is released
@dataclasses.dataclass
class ManifoldDefinition(Generic[ManifoldPoint]):
    """Manifold definition."""

    boxplus: Callable[[ManifoldPoint, jnp.ndarray], ManifoldPoint]
    boxminus: Callable[[ManifoldPoint, ManifoldPoint], jnp.ndarray]
    local_dim_from_point: Callable[[ManifoldPoint], int]

    def assert_validity(self, example: ManifoldPoint) -> None:
        def _assert_allclose(x: Any, y: Any, **kwargs):
            """"""
            for a, b in zip(jax.tree_leaves(x), jax.tree_leaves(y)):
                onp.testing.assert_allclose(a, b, **kwargs)

        _assert_allclose(
            self.boxplus(example, jnp.zeros(self.local_dim_from_point(example))),
            example,
        )
        _assert_allclose(
            self.boxminus(example, example),
            onp.zeros(self.local_dim_from_point(example)),
        )
        _assert_allclose(
            self.boxminus(example, example),
            onp.zeros(self.local_dim_from_point(example)),
        )


# Default manifold: treat all values as lying within a simple Euclidean space
# Should support arbitrary Pytrees


def _linear_boxplus(x: ManifoldPoint, tangent: jnp.ndarray) -> ManifoldPoint:
    flat, unflatten = flatten_util.ravel_pytree(x)
    return unflatten(flat + tangent)


euclidean_manifold = ManifoldDefinition[Any](
    boxplus=_linear_boxplus,
    boxminus=lambda x, y: (
        flatten_util.ravel_pytree(x)[0] - flatten_util.ravel_pytree(y)[0]
    ),
    local_dim_from_point=lambda x: flatten_util.ravel_pytree(x)[0].size,
)

# EKF API

StateType = TypeVar("StateType")
ObservationType = TypeVar("ObservationType")
ControlInputType = TypeVar("ControlInputType")
Array = Union[onp.ndarray, jnp.ndarray]


@jax_dataclasses.pytree_dataclass
class MultivariateGaussian(Generic[ManifoldPoint]):
    mean: ManifoldPoint
    cov: Array


# @dataclasses.dataclass(frozen=True) # <== TODO: when mypy v0.920 is released
@dataclasses.dataclass
class EkfDefinition(Generic[StateType, ObservationType, ControlInputType]):
    """Extended Kalman filter definition."""

    dynamics_model: Callable[[StateType, ControlInputType], StateType]
    observation_model: Callable[[StateType], ObservationType]
    state_manifold: ManifoldDefinition[StateType] = euclidean_manifold
    observation_manifold: ManifoldDefinition[ObservationType] = euclidean_manifold

    def predict(
        self,
        belief: MultivariateGaussian[StateType],
        control_input: ControlInputType,
        dynamics_cov: Array,
    ) -> MultivariateGaussian[StateType]:
        """EKF prediction step."""

        # Quick shape check
        local_dim = self.state_manifold.local_dim_from_point(belief.mean)
        assert belief.cov.shape == dynamics_cov.shape == (local_dim, local_dim)

        A: jnp.ndarray = jax.jacfwd(
            # Computing jacobian of $f(x \boxplus delta) \boxminus x$
            lambda tangent: self.state_manifold.boxminus(
                self.dynamics_model(
                    self.state_manifold.boxplus(belief.mean, tangent), control_input
                ),
                belief.mean,
            )
        )(jnp.zeros(self.state_manifold.local_dim_from_point(belief.mean)))

        with jax_dataclasses.copy_and_mutate(belief, validate=True) as out:
            out.mean = self.dynamics_model(belief.mean, control_input)
            out.cov = A @ belief.cov @ A.T + dynamics_cov  # type: ignore
        return out

    def correct(
        self,
        belief: MultivariateGaussian[StateType],
        observation: MultivariateGaussian[ObservationType],
    ) -> MultivariateGaussian[StateType]:
        """EKF correction step."""

        # Quick shape checks
        local_dim = self.state_manifold.local_dim_from_point(belief.mean)
        assert belief.cov.shape == (local_dim, local_dim)
        local_dim = self.observation_manifold.local_dim_from_point(observation.mean)
        assert observation.cov.shape == (local_dim, local_dim)

        C: jnp.ndarray = jax.jacfwd(
            lambda tangent: self.observation_manifold.boxminus(
                self.observation_model(
                    self.state_manifold.boxplus(belief.mean, tangent)
                ),
                self.observation_model(belief.mean),
            )
        )(jnp.zeros(self.state_manifold.local_dim_from_point(belief.mean)))

        pred_obs_mean = self.observation_model(belief.mean)
        innovation = self.observation_manifold.boxminus(observation.mean, pred_obs_mean)
        innovation_cov = C @ belief.cov @ C.T + observation.cov

        K = belief.cov @ C.T @ jnp.linalg.inv(innovation_cov)

        with jax_dataclasses.copy_and_mutate(belief) as out:
            out.mean = self.state_manifold.boxplus(belief.mean, K @ innovation)
            out.cov = (jnp.eye(belief.cov.shape[0]) - K @ C) @ belief.cov
        return out
