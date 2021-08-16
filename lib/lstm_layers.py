"""Network definitions for LSTM layers.

Borrows from: https://github.com/google/flax/blob/main/examples/sst2/models.py
"""

from typing import Tuple

import jax
from flax import linen as nn
from jax import numpy as jnp


class UniLstm(nn.Module):
    """A simple unidirectional LSTM."""

    @jax.partial(
        nn.transforms.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: Tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray):
        return nn.OptimizedLSTMCell()(carry, x)

    @staticmethod
    def initialize_carry_zeros(batch_dims: Tuple[int, ...], hidden_size: int):
        # Use fixed random key since default state init fn is just zeros.
        return nn.OptimizedLSTMCell.initialize_carry(
            jax.random.PRNGKey(0), batch_dims, hidden_size
        )


class BiLstm(nn.Module):
    """A simple bi-directional LSTM."""

    hidden_size: int

    def setup(self):
        self.forward_lstm = UniLstm()
        self.backward_lstm = UniLstm()

    def __call__(self, carry: Tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray):
        batch_size = x.shape[0]

        # Forward LSTM.
        initial_carry = carry
        _, forward_outputs = self.forward_lstm(initial_carry, x)

        # Backward LSTM.
        reversed_inputs = x[:, ::-1, :]
        initial_carry = UniLstm.initialize_carry_zeros((batch_size,), self.hidden_size)
        _, backward_outputs = self.backward_lstm(initial_carry, reversed_inputs)
        backward_outputs = backward_outputs[:, ::-1, :]

        # Concatenate the forward and backward representations.
        outputs = jnp.concatenate([forward_outputs, backward_outputs], -1)
        return outputs
