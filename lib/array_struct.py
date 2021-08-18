"""Helper for constructing array structures with shape checks at runtime."""


import dataclasses
from typing import Optional, Tuple, cast

from jax import numpy as jnp
from typing_extensions import get_type_hints

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


class ShapeAnnotatedStruct:
    """Base class for dataclasses whose fields are annotated with expected shapes. Helps
    with assertions + checking batch axes.

    Example of an annotated field:

        array: Annotated[jnp.ndarray, (50, 150, 3)]

    """

    def __getattribute__(self, name):
        out = super().__getattribute__(name)
        assert out is not None
        return out

    def check_shapes_and_get_batch_axes(self) -> Tuple[int, ...]:
        """Make sure shapes of arrays are consistent with annotations, then return any
        leading batch axes (which should be shared across all contained arrays)."""

        assert dataclasses.is_dataclass(self)

        annotations = get_type_hints(type(self), include_extras=True)
        batch_axes: Optional[Tuple[int, ...]] = None

        # For each field...
        for field in dataclasses.fields(self):
            value = self.__getattribute__(field.name)
            if value is null_array:
                # Don't do anything for placeholder objects
                continue

            # Get expected shape, sans batch axes
            expected_shape = annotations[field.name].__metadata__[0]
            assert isinstance(expected_shape, tuple)

            # Get actual shape
            shape: Tuple[int, ...]
            if isinstance(value, float):
                shape = ()
            else:
                assert hasattr(value, "shape")
                shape = value.shape

            # Actual shape should be expected shape prefixed by some batch axes
            if len(expected_shape) > 0:
                assert shape[-len(expected_shape) :] == expected_shape
                field_batch_axes = shape[: -len(expected_shape)]
            else:
                field_batch_axes = shape

            if batch_axes is None:
                batch_axes = field_batch_axes
            assert batch_axes == field_batch_axes

        assert batch_axes is not None
        return batch_axes
