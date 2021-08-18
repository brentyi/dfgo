"""Structures (pytrees) for working with KITTI data."""

import dataclasses
from typing import List, TypeVar

import jax
import jax_dataclasses
import jaxlie
import numpy as onp
import torch
from jax import numpy as jnp
from typing_extensions import Annotated

from .. import array_struct

_KittiStructType = TypeVar("_KittiStructType", bound="_KittiStruct")


@jax_dataclasses.pytree_dataclass
class _KittiStruct(array_struct.ShapeAnnotatedStruct):
    """Base class for storing KITTI data, which can either be normalized or not."""

    # Annotated[..., ...] attaches an expected shape to each field. (which may end up
    # being prefixed by a shared set of batch axes)
    image: Annotated[jnp.ndarray, (50, 150, 3)] = array_struct.null_array
    image_diff: Annotated[jnp.ndarray, (50, 150, 3)] = array_struct.null_array
    x: Annotated[jnp.ndarray, ()] = array_struct.null_array
    y: Annotated[jnp.ndarray, ()] = array_struct.null_array
    theta: Annotated[jnp.ndarray, ()] = array_struct.null_array
    linear_vel: Annotated[jnp.ndarray, ()] = array_struct.null_array
    angular_vel: Annotated[jnp.ndarray, ()] = array_struct.null_array

    def get_stacked_velocity(self) -> jnp.ndarray:
        """Return 2-channel velocity."""
        self.linear_vel
        out = jnp.stack([self.linear_vel, self.angular_vel], axis=-1)
        assert out.shape[-1] == 2
        return out


# Constants for data normalization. We again perform a lot of type abuse...

_DATASET_MEANS = _KittiStruct(
    image=onp.array([88.91195932, 94.08863257, 92.80115751]),  # type: ignore
    image_diff=onp.array([-0.00086295, -0.00065804, -0.00133435]),  # type: ignore
    x=195.02545,  # type: ignore
    y=-142.2851,  # type: ignore
    theta=0.0,  # type: ignore
    linear_vel=0.95533156,  # type: ignore
    angular_vel=-0.000439872,  # type: ignore
)

_DATASET_STD_DEVS = _KittiStruct(
    image=onp.array([74.12011514, 76.13433045, 77.88847008]),  # type: ignore
    image_diff=onp.array([38.63185147, 39.0655375, 38.7856255]),  # type: ignore
    x=294.42093,  # type: ignore
    y=316.98776,  # type: ignore
    theta=1.0,  # type: ignore
    linear_vel=0.43558624,  # type: ignore
    angular_vel=0.017296152,  # type: ignore
)


@jax_dataclasses.pytree_dataclass
class KittiStructNormalized(_KittiStruct):
    """KITTI data that's been normalized: zero mean, unit std dev."""

    def get_stacked_image(self) -> jnp.ndarray:
        """Return 6-channel image for CNN."""
        return jnp.concatenate([self.image, self.image_diff], axis=-1)

    def unnormalize(self, scale_only: bool = False) -> "KittiStructRaw":
        """Unnormalize contents."""

        def _unnorm(value, mean, std):
            if value is array_struct.null_array:
                return array_struct.null_array

            if scale_only:
                return value * std
            else:
                return value * std + mean

        return KittiStructRaw(
            **jax.tree_map(
                _unnorm,
                vars(self),
                vars(_DATASET_MEANS),
                vars(_DATASET_STD_DEVS),
            )
        )


@jax_dataclasses.pytree_dataclass
class KittiStructRaw(_KittiStruct):
    """Raw KITTI data in original units: meters, radians, etc."""

    def fix(self) -> "KittiStructRaw":
        # Compute x, y, theta values by integrating ground-truth velocities
        x = onp.array(self.x)
        y = onp.array(self.y)
        theta = onp.array(self.theta)

        assert len(x.shape) == len(y.shape) == len(theta.shape) == 1

        pose = jaxlie.SE2.from_xy_theta(x=x[0], y=y[0], theta=theta[0])
        for t in range(len(x) - 1):
            pose = pose @ jaxlie.SE2.from_xy_theta(
                x=self.linear_vel[t], y=0.0, theta=self.angular_vel[t]
            )
            x[t + 1], y[t + 1] = pose.translation()
            theta[t + 1] = pose.rotation().as_radians()

        return dataclasses.replace(self, x=x, y=y, theta=theta)

    def mirror(self) -> "KittiStructRaw":
        """Data augmentation helper: mirror a sequence."""
        return KittiStructRaw(
            image=self.image[..., :, ::-1, :],  # (N?, rows, columns, channels)
            image_diff=self.image_diff[
                ..., :, ::-1, :
            ],  # (N?, rows, columns, channels)
            x=self.x,
            y=-self.y,  # type: ignore
            theta=-self.theta,  # type: ignore
            linear_vel=self.linear_vel,
            angular_vel=-self.angular_vel,  # type: ignore
            # ^ type ignores here are for the negative signs, which is currently not
            # correctly typed in JAX
        )

    def normalize(self, scale_only: bool = False) -> "KittiStructNormalized":
        """Normalize contents."""

        def _norm(value, mean, std):
            if value is array_struct.null_array:
                return array_struct.null_array

            if scale_only:
                return value / std
            else:
                return (value - mean) / std

        return KittiStructNormalized(
            **jax.tree_map(
                _norm,
                vars(self),
                vars(_DATASET_MEANS),
                vars(_DATASET_STD_DEVS),
            )
        )


class KittiSubsequenceDataset(torch.utils.data.Dataset[KittiStructNormalized]):
    def __init__(self, subsequences: List[KittiStructRaw]):
        self.subsequences: List[KittiStructRaw] = subsequences

    def __getitem__(self, index: int) -> KittiStructNormalized:
        raw_length = len(self.subsequences)
        if index < raw_length:
            return self.subsequences[index].normalize()
        else:
            return self.subsequences[index - raw_length].mirror().normalize()

    def __len__(self) -> int:
        return len(self.subsequences) * 2


class KittiSingleStepDataset(torch.utils.data.Dataset[KittiStructNormalized]):
    def __init__(self, trajectories: List[KittiStructRaw]):
        # We normalize on the fly to make mirroring easier
        self.concatenated_trajectories: KittiStructRaw = jax.tree_map(
            lambda *x: onp.concatenate(x, axis=0), *trajectories
        )

    def __getitem__(self, index: int) -> KittiStructNormalized:
        raw_length = self.concatenated_trajectories.x.shape[0]
        if index < raw_length:
            return jax.tree_map(
                lambda x: x[index], self.concatenated_trajectories
            ).normalize()
        else:
            # Mirror sample
            return (
                jax.tree_map(
                    lambda x: x[index - raw_length], self.concatenated_trajectories
                )
                .mirror()
                .normalize()
            )

    def __len__(self) -> int:
        return self.concatenated_trajectories.x.shape[0] * 2
