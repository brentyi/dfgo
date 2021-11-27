"""Experiment configurations for disk task.

Note: we'd structure these very differently if we were to rewrite this code,
particularly to replace inheritance with nested dataclasses for common fields. (the
latter is now supported in `dcargs`)"""

import dataclasses
import enum
from typing import Literal

from .. import utils

####################
# Base settings and mixins
####################


@dataclasses.dataclass(frozen=True)
class ConfigurationBase:
    """Settings shared by all experiment configs."""

    experiment_identifier: str
    random_seed: int = 94305


@dataclasses.dataclass(frozen=True)
class BasicDatasetConfig:
    """Settings used for loading training data."""

    dataset_fold: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = 0
    batch_size: int = 32


@dataclasses.dataclass(frozen=True)
class SequenceDatasetConfig(BasicDatasetConfig):
    """Settings used for loading training sequences."""

    train_sequence_length: int = 20


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer settings."""

    num_epochs: int = 30
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    max_gradient_norm: float = 10.0


####################
# Config for CNN (pre-)training
####################


@dataclasses.dataclass(frozen=True)
class VirtualSensorPretrainingExperimentConfig(
    OptimizerConfig, BasicDatasetConfig, ConfigurationBase
):
    """Virtual sensor training configuration."""

    # Field already exists, but to set default value
    experiment_identifier: str = "disk/pretrain_virtual_sensor/fold_{dataset_fold}"


####################
# Config for factor graph training
####################


class NoiseModelEnum(utils.StringEnum):
    """Noise model used for perception inputs!"""

    CONSTANT = enum.auto()
    HETEROSCEDASTIC = enum.auto()


class FactorGraphLossEnum(utils.StringEnum):
    """Loss to use."""

    JOINT_NLL = enum.auto()
    SURROGATE_LOSS = enum.auto()


@dataclasses.dataclass(frozen=True)
class FactorGraphExperimentConfig(
    OptimizerConfig, SequenceDatasetConfig, ConfigurationBase
):
    """Factor graph training configuration."""

    experiment_identifier: str = "disk/fg/default_experiment/fold_{dataset_fold}"

    # Type of noise model
    noise_model: NoiseModelEnum = NoiseModelEnum.CONSTANT

    # Training strategy
    loss: FactorGraphLossEnum = FactorGraphLossEnum.SURROGATE_LOSS

    # Pretrained virtual sensor identifier
    # This is formatted with the fold number, then fed into utils.ExperimentFiles for
    # checkpoint loading
    pretrained_virtual_sensor_identifier: str = (
        "disk/pretrain_virtual_sensor/fold_{dataset_fold}"
    )


####################
# Config for EKF training
####################


class EkfLoss(utils.StringEnum):
    """Loss to use."""

    E2E_MSE = enum.auto()
    E2E_MARGINAL_NLL = enum.auto()


@dataclasses.dataclass(frozen=True)
class EkfExperimentConfig(OptimizerConfig, SequenceDatasetConfig, ConfigurationBase):
    """Factor graph training configuration."""

    experiment_identifier: str = "disk/ekf/default_experiment/fold_{dataset_fold}"

    # Type of noise model
    noise_model: NoiseModelEnum = NoiseModelEnum.CONSTANT

    # Training strategy
    loss: EkfLoss = EkfLoss.E2E_MSE

    # Pretrained virtual sensor identifier
    # This is formatted with the fold number, then fed into utils.ExperimentFiles for
    # checkpoint loading
    pretrained_virtual_sensor_identifier: str = (
        "disk/pretrain_virtual_sensor/fold_{dataset_fold}"
    )


####################
# Config for LSTM training
####################


@dataclasses.dataclass(frozen=True)
class LstmExperimentConfig(OptimizerConfig, SequenceDatasetConfig, ConfigurationBase):
    """KITTI task LSTM experiment configuration."""

    experiment_identifier: str = "disk/lstm/default_experiment/fold_{dataset_fold}"

    # Type of LSTM
    bidirectional: bool = False

    num_epochs: int = (
        60  # Train for longer because we don't use pretrained virtual sensors
    )

    # We'll be training the whole CNN, so need to shrink memory usage compared to just
    # training noise model
    batch_size: int = 16
    train_sequence_length: int = 10
