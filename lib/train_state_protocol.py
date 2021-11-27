"""Structural supertype for training states."""

from typing import Any, Generic, Protocol, Tuple, TypeVar

import fifteen
import jax_dataclasses

SelfType = TypeVar("SelfType")
TrainingDataType = TypeVar(
    "TrainingDataType",
    bound=jax_dataclasses.EnforcedAnnotationsMixin,
    contravariant=True,
)
Pytree = Any


class TrainStateProtocol(Protocol, Generic[TrainingDataType]):
    """Functionality that needs to be implemented by all training states."""

    learnable_params: Pytree
    steps: int

    def training_step(
        self: SelfType, batch: TrainingDataType
    ) -> Tuple[SelfType, fifteen.experiments.TensorboardLogData]:
        ...
