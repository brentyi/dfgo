"""System definition for KITTI task. Implements variables, variable values, and factors.
"""


from typing import NamedTuple, Tuple

import jax_dataclasses
import jaxfg
import jaxlie
import numpy as onp
from jax import numpy as jnp
from overrides import overrides


@jax_dataclasses.pytree_dataclass
class State:
    """State values."""

    pose: jaxlie.SE2
    velocities: jnp.ndarray
    """Length-2 array: (linear_vel, angular_vel)."""

    @overrides
    def __repr__(self) -> str:
        return (
            "("
            + ", ".join(
                [
                    f"{k}: {onp.round(v, 5)}"
                    for k, v in {
                        "x": self.x,
                        "y": self.y,
                        "theta": self.theta,
                        "linear_vel": self.linear_vel,
                        "angular_vel": self.angular_vel,
                    }.items()
                ]
            )
            + ")"
        )

    @classmethod
    def make(
        cls,
        pose: jaxlie.SE2,
        linear_vel: jaxlie.hints.Scalar,
        angular_vel: jaxlie.hints.Scalar,
    ) -> "State":
        return cls(
            pose=pose,
            velocities=jnp.array(
                [linear_vel, angular_vel],
            ),
        )

    @property
    def x(self):
        return self.pose.translation()[..., 0]

    @property
    def y(self):
        return self.pose.translation()[..., 1]

    @property
    def theta(self):
        return self.pose.rotation().as_radians()

    @property
    def linear_vel(self):
        return self.velocities[..., 0]

    @property
    def angular_vel(self):
        return self.velocities[..., 1]

    def predict_next(self) -> "State":
        """Prediction using dynamics model."""
        # Predict the state after our dynamics update
        return State(
            pose=self.pose
            @ jaxlie.SE2.from_rotation_and_translation(
                # Note that this is ~not~ equivalent to SE.exp()
                rotation=jaxlie.SO2.from_radians(self.angular_vel),
                translation=jnp.zeros(2).at[0].set(self.linear_vel),
            ),
            velocities=self.velocities,
        )

    def manifold_minus(self, other: "State") -> jnp.ndarray:
        return jnp.concatenate(
            (
                # (1) Treat position and translation together
                -(other.pose.inverse() @ self.pose).log(),
                # (2) Treat position and translation separately
                # other.pose.translation() - self.pose.translation(),
                # (other.pose.rotation().inverse() @ self.pose.rotation()).log(),
                self.velocities - other.velocities,
            ),
            axis=-1,
        )


class StateVariable(jaxfg.core.VariableBase[State]):
    """State variables for factor graph. Represents but does not contain any values."""

    @classmethod
    @overrides
    def get_local_parameter_dim(cls) -> int:
        return 5  # (x, y, theta, linear, angular)

    @classmethod
    @overrides
    def get_default_value(cls) -> State:
        return State(pose=jaxlie.SE2.identity(), velocities=jnp.zeros(2))

    @classmethod
    @overrides
    def manifold_retract(cls, x: State, local_delta: jaxfg.hints.Array) -> State:
        return State(
            pose=x.pose @ jaxlie.SE2.exp(local_delta[:3]),
            velocities=x.velocities + local_delta[3:5],
        )


@jax_dataclasses.pytree_dataclass
class VisionFactor(jaxfg.core.FactorBase[Tuple[State]]):
    """Factor for velocity predictions from the vision system."""

    noise_model: jaxfg.noises.DiagonalGaussian  # Narrow parent annotation
    predicted_velocity: jnp.ndarray

    @staticmethod
    def make(
        state_variable: StateVariable,
        predicted_velocity: jnp.ndarray,
        noise_model: jaxfg.noises.DiagonalGaussian,
    ) -> "VisionFactor":
        assert noise_model.get_residual_dim() == 2
        return VisionFactor(
            variables=(state_variable,),
            noise_model=noise_model,
            predicted_velocity=predicted_velocity,
        )

    @overrides
    def compute_residual_vector(self, variable_values: Tuple[State]) -> jnp.ndarray:
        return variable_values[0].velocities - self.predicted_velocity


class BeforeAfterTuple(NamedTuple):
    before: State
    after: State


@jax_dataclasses.pytree_dataclass
class DynamicsFactor(jaxfg.core.FactorBase[BeforeAfterTuple]):
    """Factor containing dynamics information."""

    noise_model: jaxfg.noises.DiagonalGaussian  # Narrow parent annotation

    @staticmethod
    def make(
        before_variable: StateVariable,
        after_variable: StateVariable,
        noise_model: jaxfg.noises.DiagonalGaussian,
    ) -> "DynamicsFactor":
        assert noise_model.get_residual_dim() == 5
        return DynamicsFactor(
            variables=(before_variable, after_variable),
            noise_model=noise_model,
        )

    @overrides
    def compute_residual_vector(self, variable_values: BeforeAfterTuple) -> jnp.ndarray:
        pred_value = variable_values.before.predict_next()
        return pred_value.manifold_minus(variable_values.after)


@jax_dataclasses.pytree_dataclass
class PriorFactor(jaxfg.core.FactorBase[Tuple[State]]):
    """Factor for constraining start states."""

    noise_model: jaxfg.noises.DiagonalGaussian  # Narrow type
    mu: State

    @staticmethod
    def make(
        variable: StateVariable,
        mu: State,
        noise_model: jaxfg.noises.DiagonalGaussian,
    ) -> "PriorFactor":
        assert noise_model.get_residual_dim() == 5
        return PriorFactor(
            mu=mu,
            variables=(variable,),
            noise_model=noise_model,
        )

    @overrides
    def compute_residual_vector(self, variable_values: Tuple[State]) -> jnp.ndarray:
        return variable_values[0].manifold_minus(self.mu)
