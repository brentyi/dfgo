"""Utilities shared by all tasks."""

import argparse
import dataclasses
import enum
import pathlib
import random
from typing import Any, Iterable, Optional, Type, TypeVar

import datargs
import fannypack
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
    return fannypack.utils.get_git_commit_hash(str(pathlib.Path(__file__).parent))


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


def parse_args(
    cls: Type[DataclassType], *, description: Optional[str] = None
) -> DataclassType:
    """Populates a dataclass via CLI args. Basically the same as `datargs.parse()`, but
    adds default values to helptext."""
    assert dataclasses.is_dataclass(cls)

    # Modify helptext to add default values.
    #
    # This is a little bit prettier than using the argparse helptext formatter, which
    # will include dataclass.MISSING values.
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            # Heuristic for if field has already been mutated. By default metadata will
            # resolve to a mappingproxy object.
            if isinstance(field.metadata, dict):
                continue

            # Add default value to helptext!
            if hasattr(field.default, "name"):
                # Special case for enums
                default_fmt = f"(default: {field.default.name})"
            else:
                default_fmt = "(default: %(default)s)"

            field.metadata = dict(field.metadata)
            field.metadata["help"] = (
                f"{field.metadata['help']} {default_fmt}"
                if "help" in field.metadata
                else default_fmt
            )

    return datargs.parse(cls, parser=argparse.ArgumentParser(description=description))
