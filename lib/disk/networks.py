from typing import Any, Protocol, Tuple

import jax
import numpy as onp
from flax import linen as nn
from jax import numpy as jnp

from . import experiment_config

Pytree = Any

relu_layer_init = nn.initializers.kaiming_normal()  # variance = 2.0 / fan_in
linear_layer_init = nn.initializers.lecun_normal()  # variance = 1.0 / fan_in


class RegressUncertaintiesFunction(Protocol):
    def __call__(
        self,
        learnable_params: Pytree,
        visible_pixels_count: jnp.ndarray,
    ) -> jnp.ndarray:
        ...


def make_regress_uncertainties(
    noise_model: experiment_config.NoiseModelEnum,
    seed: int,
) -> Tuple[RegressUncertaintiesFunction, Pytree]:
    if noise_model is experiment_config.NoiseModelEnum.CONSTANT:

        def regress_uncertainties(
            learnable_params: Pytree,
            visible_pixels_count: jnp.ndarray,
        ) -> jnp.ndarray:
            (batch_size,) = visible_pixels_count.shape
            return jnp.ones((batch_size,)) * learnable_params

        # 3.6 is an approximate value for the virtual sensor RMSE
        return regress_uncertainties, 1.0 / 3.60

    elif noise_model is experiment_config.NoiseModelEnum.HETEROSCEDASTIC:
        mlp, learnable_params = make_uncertainty_mlp(seed=seed)

        def regress_uncertainties(
            learnable_params: Pytree,
            visible_pixels_count: jnp.ndarray,
        ) -> jnp.ndarray:
            (batch_size,) = visible_pixels_count.shape

            # Stabilize training: initial uncertainties should be loosely situated
            # around virtual sensor RMSE.
            return (
                mlp.apply(
                    learnable_params, visible_pixels_count.reshape((batch_size, 1))
                ).reshape((batch_size,))
                * 0.05
                + 1.0
            ) * (1.0 / 3.60)

        return regress_uncertainties, learnable_params

    else:
        assert False


class MLP(nn.Module):
    units: int
    layers: int
    output_dim: int

    @staticmethod
    def make(units: int, layers: int, output_dim: int):
        """Dummy constructor for type-checking."""
        return MLP(units=units, layers=layers, output_dim=output_dim)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):  # type: ignore
        x = inputs

        for i in range(self.layers):
            x = nn.Dense(features=self.units, kernel_init=relu_layer_init)(x)
            x = nn.relu(x)

        x = nn.Dense(features=self.output_dim, kernel_init=linear_layer_init)(x)
        return x


class DiskVirtualSensor(nn.Module):
    """CNN.

    Input is (N, 120, 120, 3) images, **normalized**.
    Output is (N, output_dim). Default to output_dim=2, representing **normalized** position estimate.
    """

    output_dim: int = 2

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):  # type: ignore
        x = inputs
        N = x.shape[0]
        assert x.shape == (N, 120, 120, 3), x.shape

        # Some conv layers
        for _ in range(3):
            x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=relu_layer_init)(x)
            x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=linear_layer_init)(x)

        # Channel-wise max pool
        x = jnp.max(x, axis=3, keepdims=True)

        # Spanning mean pools (to regularize X/Y coordinate regression)
        assert x.shape == (N, 120, 120, 1)
        x_horizontal = nn.avg_pool(x, window_shape=(120, 1))
        x_vertical = nn.avg_pool(x, window_shape=(1, 120))

        # Concatenate, feed through MLP
        x = jnp.concatenate(
            [x_horizontal.reshape((N, -1)), x_vertical.reshape((N, -1))], axis=1
        )
        assert x.shape == (N, 240)
        x = MLP.make(units=32, layers=3, output_dim=self.output_dim)(x)

        return x


def make_position_cnn(seed: int = 0) -> Tuple[DiskVirtualSensor, Pytree]:
    """Make CNN for disk tracking predictions."""
    model = DiskVirtualSensor()

    prng_key = jax.random.PRNGKey(seed=seed)

    N = 1
    dummy_image = onp.zeros((N, 120, 120, 3))
    return model, model.init(prng_key, dummy_image)


def make_uncertainty_mlp(seed: int = 0) -> Tuple[MLP, Pytree]:
    """Make MLP for mapping # of visible pixels => inverse standard deviation of
    position estimate."""
    model = MLP.make(units=64, layers=4, output_dim=1)

    prng_key = jax.random.PRNGKey(seed=seed)

    N = 1
    dummy_input = onp.zeros((N, 1))
    return model, model.init(prng_key, dummy_input)
