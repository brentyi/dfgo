"""Experiment configurations for KITTI task.

Note: we'd likely structure these very differently if we were to rewrite this code,
particularly to replace inheritance with nested dataclasses for common fields. (the
latter is now supported in `tyro`)"""

import dataclasses
import enum
from typing import Literal, Union

from .. import utils

####################
# Base settings and mixins
####################


@dataclasses.dataclass(frozen=True)
class ConfigurationBase:
    """Settings shared by all experiment configs."""

    random_seed: int = 94305


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """Settings used for loading training data"""

    dataset_fold: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = 0
    use_overfitting_dataset: bool = False
    batch_size: int = 32


@dataclasses.dataclass(frozen=True)
class SequenceDatasetConfig(DatasetConfig):
    train_sequence_length: int = 10


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer settings."""

    num_epochs: int = 90
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    max_gradient_norm: float = 10.0


####################
# Config for CNN (pre-)training
####################


class VirtualSensorLossEnum(utils.StringEnum):
    NLL = enum.auto()
    MSE = enum.auto()


@dataclasses.dataclass(frozen=True)
class VirtualSensorPretrainingExperimentConfig(
    OptimizerConfig, DatasetConfig, ConfigurationBase
):
    experiment_identifier: str = "kitti/pretrain_virtual_sensor/fold_{dataset_fold}"
    loss: VirtualSensorLossEnum = VirtualSensorLossEnum.MSE


####################
# Config for factor graph training
####################


class NoiseModelEnum(utils.StringEnum):
    """Noise model used for perception inputs!"""

    CONSTANT = enum.auto()
    HETEROSCEDASTIC = enum.auto()


class SurrogateLossSupervisionEnum(utils.StringEnum):
    """What's our loss function on?"""

    POSITION_XY = enum.auto()
    VELOCITY = enum.auto()


class InitializationStrategyEnum(utils.StringEnum):
    """How are we initializing our nonlinear optimizer at train time? Not used for joint
    NLL supervision."""

    SAME_AS_EVAL = enum.auto()
    GROUND_TRUTH = enum.auto()
    MIXED = enum.auto()  # MIXED = both SAME_AS_EVAL and GROUND_TRUTH
    NAIVE_BASELINE = enum.auto()


@dataclasses.dataclass(frozen=True)
class JointNllLossConfig:
    """Joint NLL loss configuration.

    Empty dataclass to match syntax used for defining subparsers in tyro."""

    pass


@dataclasses.dataclass(frozen=True)
class SurrogateLossConfig:
    """End-to-end surrogate loss configuration."""

    supervision: SurrogateLossSupervisionEnum = SurrogateLossSupervisionEnum.POSITION_XY
    gn_initialization_strategy: InitializationStrategyEnum = (
        InitializationStrategyEnum.GROUND_TRUTH
    )
    gn_steps: int = 5
    gn_initialization_noise_std: float = 0.0
    conjugate_gradient_tolerance: float = 1e-5


@dataclasses.dataclass(frozen=True)
class FactorGraphExperimentConfig(
    OptimizerConfig, SequenceDatasetConfig, ConfigurationBase
):
    """Overall KITTI task experiment configuration."""

    experiment_identifier: str = "kitti/fg/default_experiment/fold_{dataset_fold}"

    # Type of noise model
    noise_model: NoiseModelEnum = NoiseModelEnum.CONSTANT

    # Training strategy
    loss_config: Union[JointNllLossConfig, SurrogateLossConfig] = SurrogateLossConfig()

    # Pretrained virtual sensor identifier
    # This is formatted with the fold number, then fed into utils.ExperimentFiles for
    # checkpoint loading
    pretrained_virtual_sensor_identifier: str = (
        "kitti/pretrain_virtual_sensor/lr1e-4/kitti-10/fold_{dataset_fold}"
    )


####################
# Config for EKF training
####################


@dataclasses.dataclass(frozen=True)
class EkfExperimentConfig(OptimizerConfig, SequenceDatasetConfig, ConfigurationBase):
    """KITTI task EKF experiment configuration."""

    experiment_identifier: str = "kitti/ekf/default_experiment/fold_{dataset_fold}"

    # Type of noise model
    noise_model: NoiseModelEnum = NoiseModelEnum.CONSTANT

    # Pretrained virtual sensor identifier
    # This is formatted with the fold number, then fed into utils.ExperimentFiles for
    # checkpoint loading
    pretrained_virtual_sensor_identifier: str = (
        "kitti/pretrain_virtual_sensor/lr1e-4/kitti-10/fold_{dataset_fold}"
    )


####################
# Config for LSTM training
####################


@dataclasses.dataclass(frozen=True)
class LstmExperimentConfig(OptimizerConfig, SequenceDatasetConfig, ConfigurationBase):
    """KITTI task LSTM experiment configuration."""

    experiment_identifier: str = "kitti/lstm/default_experiment/fold_{dataset_fold}"

    # Type of LSTM
    bidirectional: bool = False

    num_epochs: int = (
        60  # Train for longer because we don't use pretrained virtual sensors
    )
