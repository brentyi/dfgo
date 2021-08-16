import jax
import jax_dataclasses
import numpy as onp
from jax import numpy as jnp
from typing_extensions import Annotated

from .. import array_struct


@jax_dataclasses.pytree_dataclass
class _DiskStruct(array_struct.ShapeAnnotatedStruct):
    """Values in our dataset."""

    image: Annotated[jnp.ndarray, (120, 120, 3)] = array_struct.null_array
    visible_pixels_count: Annotated[jnp.ndarray, ()] = array_struct.null_array
    position: Annotated[jnp.ndarray, (2,)] = array_struct.null_array
    velocity: Annotated[jnp.ndarray, (2,)] = array_struct.null_array


_DATASET_MEANS = _DiskStruct(
    image=onp.array([24.30598765, 29.76503314, 29.86749727], dtype=onp.float32),  # type: ignore
    position=onp.array([-0.08499543, 0.07917813], dtype=onp.float32),  # type: ignore
    velocity=onp.array([0.02876372, 0.06096543], dtype=onp.float32),  # type: ignore
    visible_pixels_count=104.87143,  # type: ignore
)
_DATASET_STD_DEVS = _DiskStruct(
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
            if value is array_struct.null_array:
                return array_struct.null_array

            if scale_only:
                return value / std
            else:
                return (value - mean) / std

        return DiskStructNormalized(
            **jax.tree_map(
                _norm,
                vars(self),
                vars(_DATASET_MEANS),
                vars(_DATASET_STD_DEVS),
            )
        )


@jax_dataclasses.pytree_dataclass
class DiskStructNormalized(_DiskStruct):
    def unnormalize(self, scale_only: bool = False) -> "DiskStructRaw":
        """Unnormalize contents."""

        def _unnorm(value, mean, std):
            if value is array_struct.null_array:
                return array_struct.null_array

            if scale_only:
                return value * std
            else:
                return value * std + mean

        return DiskStructRaw(
            **jax.tree_map(
                _unnorm,
                vars(self),
                vars(_DATASET_MEANS),
                vars(_DATASET_STD_DEVS),
            )
        )
