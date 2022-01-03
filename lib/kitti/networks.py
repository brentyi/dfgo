"""Neural network definitions, helpers, and related types."""

import pathlib
from typing import Any, NamedTuple, Protocol, Tuple

import fifteen
import jax
import numpy as onp
from flax import linen as nn
from jax import numpy as jnp

from . import data, experiment_config

Pytree = Any
LearnableParams = Pytree
KittiVirtualSensorParameters = Pytree
StackedImages = jnp.ndarray

relu_layer_init = nn.initializers.kaiming_normal()  # variance = 2.0 / fan_in
linear_layer_init = nn.initializers.lecun_normal()  # variance = 1.0 / fan_in


class RegressedVelocities(NamedTuple):
    linear_vel: jnp.ndarray
    angular_vel: jnp.ndarray


class RegressedUncertainties(NamedTuple):
    vision_sqrt_precision_diagonal: jnp.ndarray
    dynamics_sqrt_precision_diagonal: jnp.ndarray


class RegressVelocitiesFunction(Protocol):
    def __call__(self, stacked_images: jnp.ndarray) -> RegressedVelocities:
        ...


class RegressUncertaintiesFunction(Protocol):
    def __call__(
        self,
        learnable_params: Pytree,
        stacked_images: jnp.ndarray,
        prng_key: jax.random.KeyArray,
        train: bool,
    ) -> RegressedUncertainties:
        ...


class KittiVirtualSensor(nn.Module):
    """CNN.

    Input is (N, 50, 150, 6) images.
    Output is (N, 4). Linear and angular velocities, followed by uncertainties for each.
    """

    output_dim: int = 4

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:  # type: ignore
        x = inputs
        N = x.shape[0]
        assert x.shape == (N, 50, 150, 6), x.shape

        # conv1
        x = nn.Conv(
            features=16,
            kernel_size=(7, 7),
            strides=(1, 1),
            kernel_init=relu_layer_init,
        )(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)

        # conv2
        x = nn.Conv(
            features=16,
            kernel_size=(5, 5),
            strides=(1, 2),
            kernel_init=relu_layer_init,
        )(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)

        # conv3
        x = nn.Conv(
            features=16,
            kernel_size=(5, 5),
            strides=(1, 2),
            kernel_init=relu_layer_init,
        )(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)

        # conv4
        x = nn.Conv(
            features=16,
            kernel_size=(5, 5),
            strides=(2, 2),
            kernel_init=relu_layer_init,
        )(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)

        # Dropout
        x = nn.Dropout(rate=0.3)(x, deterministic=not train)

        # Validate shape
        # Kloss paper reports (N, 25, 18, 16), but this might be a typo? Or
        # implementation difference in how even strides/odd images are handled.
        assert x.shape == (N, 25, 19, 16)

        # Concatenate, feed through MLP
        x = x.reshape((N, -1))  # type: ignore

        # fc1
        x = nn.Dense(features=128, kernel_init=relu_layer_init)(x)
        x = nn.relu(x)

        # fc2
        x = nn.Dense(features=128, kernel_init=relu_layer_init)(x)
        x = nn.relu(x)

        # fc3
        x = nn.Dense(features=self.output_dim, kernel_init=linear_layer_init)(x)

        assert x.shape == (N, self.output_dim)
        return x


def make_observation_cnn(
    random_seed: int,
) -> Tuple[KittiVirtualSensor, KittiVirtualSensorParameters]:
    model = KittiVirtualSensor()

    dummy_image = onp.zeros((1, 50, 150, 6))
    prng_key, dropout_key = jax.random.split(jax.random.PRNGKey(random_seed))
    params = model.init(
        {"params": prng_key, "dropout": dropout_key},
        dummy_image,
        train=True,  # should not matter
    )

    return model, params


def load_pretrained_observation_cnn(
    experiment_identifier: str,
) -> Tuple[KittiVirtualSensor, KittiVirtualSensorParameters]:
    """Make CNN for processing KITTI images, and load in pre-trained weights."""

    # Note that seed does not matter, because parameters will be read from checkpoint
    model, params = make_observation_cnn(random_seed=0)

    experiment = fifteen.experiments.Experiment(
        data_dir=pathlib.Path("./experiments/") / experiment_identifier
    )
    params = experiment.restore_checkpoint(params, prefix="best_val_params_")

    return model, params


def make_regress_velocities(
    pretrained_virtual_sensor_identifier: str,
) -> RegressVelocitiesFunction:
    """Returns a helper function for predicting velocities from stacked image frames,
    using a pre-trained CNN."""
    predict_vel_model, predict_vel_params = load_pretrained_observation_cnn(
        experiment_identifier=pretrained_virtual_sensor_identifier,
    )

    def regress_velocities(
        stacked_images: jnp.ndarray,
    ) -> RegressedVelocities:
        N = stacked_images.shape[0]
        assert stacked_images.shape == (N, 50, 150, 6)

        velocities = predict_vel_model.apply(
            predict_vel_params,
            stacked_images,
            train=False,  # Never in train mode
            # rngs={"dropout": prng_key},
        )
        assert velocities.shape == (N, 4)

        # CNN outputs are normalized!
        unnorm_velocities = data.KittiStructNormalized(
            linear_vel=velocities[:, 0],
            angular_vel=velocities[:, 1],
        ).unnormalize()

        return RegressedVelocities(
            linear_vel=unnorm_velocities.linear_vel,
            angular_vel=unnorm_velocities.angular_vel,
        )

    return regress_velocities


class ConstantUncertaintyParams(NamedTuple):
    vision_sqrt_precision_diagonal: jnp.ndarray
    dynamics_sqrt_precision_diagonal: jnp.ndarray

    @staticmethod
    def handtuned() -> "ConstantUncertaintyParams":
        """Hand-tuned values from Kloss et al 2020."""
        return ConstantUncertaintyParams(
            vision_sqrt_precision_diagonal=jnp.sqrt(
                1.0
                / jnp.array(
                    [
                        # Alina numbers:
                        # 0.36,
                        # 0.36,
                        # Eyeballed from Tensorboard:
                        1.8e-1,
                        3.5e-3,
                    ]
                )
            ),
            dynamics_sqrt_precision_diagonal=jnp.sqrt(
                1.0
                / jnp.array(
                    [
                        # Alina numbers:
                        # 1e-4,
                        # 1e-4,
                        # 1e-6,
                        # 0.01,
                        # 0.16,
                        # Computed from dataset:
                        1.6513063e-02,
                        2.4398476e-01,
                        6.3744818e-07,
                        1.5382743e-01,
                        1.4825826e-02,
                    ]
                )
            ),
        )


class HeteroscedasticUncertaintyParams(NamedTuple):
    velocity_uncertainty_cnn_params: jnp.ndarray
    dynamics_sqrt_precision_diagonal: jnp.ndarray


def make_regress_uncertainties(
    noise_model: experiment_config.NoiseModelEnum,
    pretrained_virtual_sensor_identifier: str,
) -> Tuple[RegressUncertaintiesFunction, LearnableParams]:
    """Returns a function + set of learnable parameters for predicting uncertainties
    from stacked image frames."""

    regress_uncertainties: RegressUncertaintiesFunction
    learnable_params: LearnableParams

    if noise_model is experiment_config.NoiseModelEnum.CONSTANT:
        learnable_params = ConstantUncertaintyParams.handtuned()

        def regress_uncertainties(
            learnable_params: ConstantUncertaintyParams,
            stacked_images: jnp.ndarray,
            prng_key: jax.random.KeyArray,
            train: bool,
        ) -> RegressedUncertainties:
            sequence_length = stacked_images.shape[0]
            return RegressedUncertainties(
                vision_sqrt_precision_diagonal=jnp.tile(
                    learnable_params.vision_sqrt_precision_diagonal[None, :],
                    reps=(sequence_length, 1),
                ),
                dynamics_sqrt_precision_diagonal=learnable_params.dynamics_sqrt_precision_diagonal,
            )

    elif noise_model is experiment_config.NoiseModelEnum.HETEROSCEDASTIC:
        model, cnn_params = load_pretrained_observation_cnn(
            experiment_identifier=pretrained_virtual_sensor_identifier,
        )
        dynamics_sqrt_precision_diagonal = (
            ConstantUncertaintyParams.handtuned().dynamics_sqrt_precision_diagonal
        )
        learnable_params = HeteroscedasticUncertaintyParams(
            velocity_uncertainty_cnn_params=cnn_params,
            dynamics_sqrt_precision_diagonal=dynamics_sqrt_precision_diagonal,
        )

        def regress_uncertainties(
            learnable_params: HeteroscedasticUncertaintyParams,
            stacked_images: jnp.ndarray,
            prng_key: jax.random.KeyArray,
            train: bool,
        ) -> RegressedUncertainties:
            sequence_length = stacked_images.shape[0]

            assert stacked_images.shape == (sequence_length, 50, 150, 6)
            cnn_output = model.apply(
                learnable_params.velocity_uncertainty_cnn_params,
                stacked_images,
                train=train,
                rngs={"dropout": prng_key},
            )
            assert cnn_output.shape == (sequence_length, 4)

            return RegressedUncertainties(
                # Stabilize training: initial uncertainties should be loosely situated
                # around handtuned values.
                vision_sqrt_precision_diagonal=(cnn_output[:, 2:] * 0.05 + 1.0)
                * ConstantUncertaintyParams.handtuned().vision_sqrt_precision_diagonal,
                dynamics_sqrt_precision_diagonal=learnable_params.dynamics_sqrt_precision_diagonal,
            )

    else:
        assert False

    return regress_uncertainties, learnable_params
