from typing import NamedTuple, Tuple

import jax_dataclasses
import jaxfg
import numpy as onp
from jax import numpy as jnp
from overrides import overrides

SPRING_CONSTANT = 0.05
DRAG_CONSTANT = 0.0075
POSITION_NOISE_STD = 0.1
VELOCITY_NOISE_STD = 2.0

DYNAMICS_SQRT_PRECISION_DIAGONAL = 1.0 / onp.array(
    [POSITION_NOISE_STD, POSITION_NOISE_STD, VELOCITY_NOISE_STD, VELOCITY_NOISE_STD]
)
assert DYNAMICS_SQRT_PRECISION_DIAGONAL.shape == (4,)

DYNAMICS_COVARIANCE_DIAGONAL = 1.0 / DYNAMICS_SQRT_PRECISION_DIAGONAL ** 2


@jax_dataclasses.pytree_dataclass
class State:
    position: jnp.ndarray
    """Length-2 array."""

    velocity: jnp.ndarray
    """Length-2 array."""

    def predict_next(self) -> "State":
        # Predict the state after our dynamics update
        spring_force = -SPRING_CONSTANT * self.position
        drag_force = -DRAG_CONSTANT * jnp.sign(self.velocity) * (self.velocity ** 2)
        return State(
            position=self.position + self.velocity,  # type: ignore
            velocity=self.velocity + spring_force + drag_force,
        )


class StateVariable(jaxfg.core.VariableBase[State]):
    """State of our system."""

    @classmethod
    @overrides
    def get_default_value(cls) -> State:
        return State(position=onp.zeros(2), velocity=onp.zeros(2))  # type: ignore


VisionFactorVariableValues = Tuple[State]


@jax_dataclasses.pytree_dataclass
class VisionFactor(jaxfg.core.FactorBase[VisionFactorVariableValues]):
    predicted_position: jnp.ndarray
    noise_model: jaxfg.noises.DiagonalGaussian

    @staticmethod
    def make(
        state_variable: StateVariable,
        predicted_position: jnp.ndarray,
        sqrt_precision_diagonal: jnp.ndarray,
    ) -> "VisionFactor":
        assert sqrt_precision_diagonal.shape == (2,)
        return VisionFactor(
            variables=(state_variable,),
            noise_model=jaxfg.noises.DiagonalGaussian(
                sqrt_precision_diagonal=sqrt_precision_diagonal
            ),
            predicted_position=predicted_position,
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: VisionFactorVariableValues
    ) -> jnp.ndarray:
        (state,) = variable_values
        return state.position - self.predicted_position


class DynamicsFactorVariableValues(NamedTuple):
    before: State
    after: State


@jax_dataclasses.pytree_dataclass
class DynamicsFactor(jaxfg.core.FactorBase[DynamicsFactorVariableValues]):

    noise_model: jaxfg.noises.DiagonalGaussian

    @staticmethod
    def make(
        before_variable: StateVariable,
        after_variable: StateVariable,
    ) -> "DynamicsFactor":
        return DynamicsFactor(
            variables=(before_variable, after_variable),
            noise_model=jaxfg.noises.DiagonalGaussian(
                sqrt_precision_diagonal=DYNAMICS_SQRT_PRECISION_DIAGONAL
            ),
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: DynamicsFactorVariableValues
    ) -> jnp.ndarray:
        pred_value = variable_values.before.predict_next()
        actual_value = variable_values.after
        return jnp.concatenate(
            [
                pred_value.position - actual_value.position,
                pred_value.velocity - actual_value.velocity,
            ]
        )
