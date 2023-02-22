"""Network definitions for LSTM baselines."""

import jax
import jaxlie
from flax import linen as nn
from jax import numpy as jnp

from .. import lstm_layers
from . import data, networks


class KittiLstm(nn.Module):
    """LSTM

    Inputs are (N, T, 50, 150, 6) *normalized* images.
    Outputs are *unnormalized* SE(2) objects with (N, T) batch axes.
    """

    bidirectional: bool

    @nn.compact
    def __call__(self, inputs: data.KittiStructNormalized, train: bool) -> jaxlie.SE2:  # type: ignore
        N, T = inputs.get_batch_axes()
        stacked_images = inputs.get_stacked_image()
        assert stacked_images.shape == (N, T, 50, 150, 6)

        # Initial carry by encoding ground-truth initial state
        # initial_carry = nn.Dense(features=32, kernel_init=networks.relu_layer_init)(
        #     jnp.stack(
        #         [
        #             inputs.x[:, 0],
        #             inputs.y[:, 0],
        #             jnp.cos(inputs.theta[:, 0]),
        #             jnp.sin(inputs.theta[:, 0]),
        #             inputs.linear_vel[:, 0],
        #             inputs.angular_vel[:, 0],
        #         ],
        #         axis=-1,
        #     )
        # )
        # assert initial_carry.shape == (N, 32)
        # initial_carry = nn.relu(initial_carry)
        # initial_carry = nn.Dense(features=32, kernel_init=networks.linear_layer_init)(
        #     initial_carry
        # )
        # initial_carry = (initial_carry, initial_carry)
        initial_carry = (jnp.zeros((N, 64)), jnp.zeros((N, 64)))

        # Image encoder
        x = networks.KittiVirtualSensor(output_dim=64)(
            stacked_images.reshape(N * T, 50, 150, 6),
            train=train,
        ).reshape(N, T, -1)
        assert x.shape == (N, T, 64)

        # LSTM scan
        if self.bidirectional:
            x = lstm_layers.BiLstm(hidden_size=64)(carry=initial_carry, x=x)
            assert x.shape == (N, T, 64 * 2)
        else:
            _, x = lstm_layers.UniLstm()(carry=initial_carry, x=x)
            assert x.shape == (N, T, 64)

        # Output
        x = nn.Dense(features=64, kernel_init=networks.relu_layer_init)(x)
        x = nn.relu(x)
        x = nn.Dense(features=4, kernel_init=networks.linear_layer_init)(x)
        assert x.shape == (N, T, 4)

        unnormed_outputs = data.KittiStructNormalized(
            x=x[:, :, 0], y=x[:, :, 1]
        ).unnormalize(
            scale_only=True  # Outputs will be relative to the initial pose
        )

        # Outputs will be relative poses
        output: jaxlie.SE2 = jax.vmap(
            jax.vmap(
                jaxlie.SE2.from_xy_theta,
            )
        )(
            x=unnormed_outputs.x,
            y=unnormed_outputs.y,
            theta=jnp.arctan2(x[:, :, 2], x[:, :, 3]),
        )

        inputs_unnorm = inputs.unnormalize()
        initial_poses: jaxlie.SE2 = jax.vmap(jax.vmap(jaxlie.SE2.from_xy_theta))(
            x=inputs_unnorm.x[:, 0, None],
            y=inputs_unnorm.y[:, 0, None],
            theta=inputs_unnorm.theta[:, 0, None],
        )
        initial_poses = jax.tree_map(
            # This tile could be avoided
            lambda x: jnp.tile(x, reps=(1, T, 1)),
            initial_poses,
        )
        assert output.parameters().shape == (N, T, 4)
        assert initial_poses.parameters().shape == (N, T, 4)

        output = jax.vmap(jax.vmap(lambda a, b: a @ b))(initial_poses, output)

        assert output.parameters().shape == (N, T, 4)
        return output
