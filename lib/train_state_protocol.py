"""Structural supertype for training states."""

from typing import Any, Generic, Protocol, Tuple, TypeVar

from . import array_struct, experiment_files

SelfType = TypeVar("SelfType")
TrainingDataType = TypeVar(
    "TrainingDataType",
    bound=array_struct.ShapeAnnotatedStruct,
    contravariant=True,
)
Pytree = Any


class TrainStateProtocol(Protocol, Generic[TrainingDataType]):
    """Functionality that needs to be implemented by all training states."""

    learnable_params: Pytree
    steps: int

    def training_step(
        self: SelfType, batch: TrainingDataType
    ) -> Tuple[SelfType, experiment_files.TensorboardLogData]:
        ...
