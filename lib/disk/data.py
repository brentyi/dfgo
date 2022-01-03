"""Structures (pytrees) for working with disk data."""

from typing import cast

import jax
import jax_dataclasses
import numpy as onp
from jax import numpy as jnp
from typing_extensions import Annotated

null_array = cast(jnp.ndarray, None)
# ^Placeholder value to be used as a dataclass field default, to enable structs that
# contain only a partial set of values.
#
# An intuitive solution is to populate fields with a dummy default array like
# `jnp.empty(shape=(0,))`), but this can cause silent broadcasting/tracing issues.
#
# So instead we use `None` as the default value. Which is nice because it leads to loud
# runtime errors when uninitialized values are accidentally used.
#
# Note that the correct move would be to hint fields as `Optional[jnp.ndarray]`, but
# this would result in code that's littered with `assert __ is not None` statements
# and/or casts. Which is annoying. So instead we just pretend `None` is an array,


@jax_dataclasses.pytree_dataclass
class _DiskStruct(jax_dataclasses.EnforcedAnnotationsMixin):
    """Values in our dataset."""

    image: Annotated[jnp.ndarray, (120, 120, 3), jnp.floating] = null_array
    visible_pixels_count: Annotated[jnp.ndarray, (), jnp.floating] = null_array
    position: Annotated[jnp.ndarray, (2,), jnp.floating] = null_array
    velocity: Annotated[jnp.ndarray, (2,), jnp.floating] = null_array


_DATASET_MEANS = dict(
    image=onp.array([24.30598765, 29.76503314, 29.86749727], dtype=onp.float32),  # type: ignore
    position=onp.array([-0.08499543, 0.07917813], dtype=onp.float32),  # type: ignore
    velocity=onp.array([0.02876372, 0.06096543], dtype=onp.float32),  # type: ignore
    visible_pixels_count=104.87143,  # type: ignore
)
_DATASET_STD_DEVS = dict(
    image=onp.array([74.88154621, 81.87872827, 82.00088091], dtype=onp.float32),  # type: ignore
    position=onp.array([30.53421, 30.84835], dtype=onp.float32),  # type: ignore
    velocity=onp.array([6.636913, 6.647381], dtype=onp.float32),  # type: ignore
    visible_pixels_count=47.584693544827,  # type: ignore
)


@jax_dataclasses.pytree_dataclass
class DiskStructRaw(_DiskStruct):
    def normalize(self, scale_only: bool = False) -> "DiskStructNormalized":
        """Normalize contents."""

        def _norm(value, mean, std):
            if value is null_array:
                return null_array

            if scale_only:
                return value / std
            else:
                return (value - mean) / std

        return DiskStructNormalized(
            **jax.tree_map(
                _norm,
                jax_dataclasses.asdict(self),
                _DATASET_MEANS,
                _DATASET_STD_DEVS,
            )
        )


@jax_dataclasses.pytree_dataclass
class DiskStructNormalized(_DiskStruct):
    def unnormalize(self, scale_only: bool = False) -> "DiskStructRaw":
        """Unnormalize contents."""

        def _unnorm(value, mean, std):
            if value is null_array:
                return null_array

            if scale_only:
                return value * std
            else:
                return value * std + mean

        return DiskStructRaw(
            **jax.tree_map(
                _unnorm,
                jax_dataclasses.asdict(self),
                _DATASET_MEANS,
                _DATASET_STD_DEVS,
            )
        )
