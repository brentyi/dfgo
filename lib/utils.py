"""Utilities shared by all tasks."""

import enum
import pathlib
import random
from typing import Any, Iterable, TypeVar

import fifteen
import jax
import numpy as onp
import optax
import overrides
import torch
from jax import numpy as jnp

T = TypeVar("T")
Pytree = Any
PytreeType = TypeVar("PytreeType")
DataclassType = TypeVar("DataclassType")


class StringEnum(enum.Enum):
    """Enum where all values are strings. Makes `enum.auto()` return the name of the
    constant (helps with serialization!), and adds stronger typing for `.value`."""

    @overrides.overrides
    def _generate_next_value_(name, start, count, last_values):
        return name

    @property
    def value(self) -> str:
        return super().value


def get_git_commit_hash() -> str:
    """Get current repository commit hash."""
    return fifteen.utils.get_git_commit_hash(pathlib.Path(__file__).parent)


def warmup_schedule(learning_rate: float, warmup_steps: int) -> optax.Schedule:
    """Simple linear warmup schedule for optax."""
    return lambda count: jnp.minimum(1.0, count / warmup_steps) * learning_rate


def set_random_seed(seed: int) -> None:
    """Set seeds for `random`, `torch`, and `numpy`."""
    random.seed(seed)
    torch.manual_seed(seed)
    onp.random.seed(seed)


def collate_fn(batch: Iterable[PytreeType], axis=0) -> PytreeType:
    """Collate function for torch DataLoaders."""
    return jax.tree_multimap(lambda *arrays: onp.stack(arrays, axis=axis), *batch)
