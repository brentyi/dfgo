from typing import TypeVar

import numpy as onp
from jax import numpy as jnp

ArrayT = TypeVar("ArrayT", jnp.ndarray, onp.ndarray)


def compute_distance_traveled(x: ArrayT, y: ArrayT) -> ArrayT:
    # TODO: revisit this...
    if isinstance(x, onp.ndarray):
        jnp = onp  # type: ignore
    else:
        jnp = globals()["jnp"]

    (timesteps,) = x.shape
    assert y.shape == (timesteps,)
    return jnp.sum(jnp.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2))


def wrap_angle(thetas: ArrayT) -> ArrayT:
    return ((thetas + onp.pi) % (2 * onp.pi)) - onp.pi  # type: ignore
