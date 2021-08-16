"""Network definition for LSTM baselines."""

from flax import linen as nn
from jax import numpy as jnp

from .. import lstm_layers
from . import data, networks


class DiskLstm(nn.Module):
    """LSTM

    Inputs are (N, T, 120, 120, 3) *normalized* images.
    Outputs are *also normalized* positions of shape (N, T, 2).
    """

    bidirectional: bool

    @nn.compact
    def __call__(self, inputs: data.DiskStructNormalized) -> jnp.ndarray:
        N, T = inputs.check_shapes_and_get_batch_axes()
        images = inputs.image
        assert images.shape == (N, T, 120, 120, 3)

        # Initial carry by encoding ground-truth initial state
        initial_carry = nn.Dense(32, kernel_init=networks.relu_layer_init)(
            jnp.concatenate(
                [
                    inputs.position[:, 0, :],
                    inputs.velocity[:, 0, :],
                ],
                axis=-1,
            )
        )
        assert initial_carry.shape == (N, 32)
        initial_carry = nn.relu(initial_carry)
        initial_carry = nn.Dense(32 * 2, kernel_init=networks.linear_layer_init)(
            initial_carry
        )
        initial_carry = (initial_carry[..., :32], initial_carry[..., 32:])

        # Image encoder
        x = networks.DiskVirtualSensor(output_dim=32)(
            images.reshape(N * T, 120, 120, 3),
        ).reshape(N, T, -1)
        assert x.shape == (N, T, 32)

        # LSTM scan
        if self.bidirectional:
            x = lstm_layers.BiLstm(hidden_size=32)(initial_carry, x)
            assert x.shape == (N, T, 32 * 2)
        else:
            _, x = lstm_layers.UniLstm()(initial_carry, x)
            assert x.shape == (N, T, 32)

        # Output
        x = nn.Dense(32, kernel_init=networks.relu_layer_init)(x)
        x = nn.relu(x)
        x = nn.Dense(2, kernel_init=networks.linear_layer_init)(x)
        assert x.shape == (N, T, 2)

        return x
