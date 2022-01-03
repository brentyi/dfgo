from typing import Any, Tuple

import fifteen
import jax
import jax_dataclasses
import jaxlie
import optax
from jax import numpy as jnp

from .. import utils
from . import data, experiment_config, math_utils, networks_lstm

LearnableParams = Any


@jax_dataclasses.pytree_dataclass
class TrainState:
    """Everything needed for training."""

    config: experiment_config.LstmExperimentConfig = jax_dataclasses.static_field()
    optimizer: optax.GradientTransformation = jax_dataclasses.static_field()
    optimizer_state: optax.OptState

    lstm: networks_lstm.KittiLstm = jax_dataclasses.static_field()
    learnable_params: LearnableParams

    prng_key: jax.random.KeyArray
    steps: int
    train: bool = jax_dataclasses.static_field()

    @staticmethod
    def initialize(
        config: experiment_config.LstmExperimentConfig, train: bool
    ) -> "TrainState":

        # Neural network setup
        lstm = networks_lstm.KittiLstm(bidirectional=config.bidirectional)
        prng_key, dropout_key = jax.random.split(jax.random.PRNGKey(config.random_seed))
        learnable_params = lstm.init(
            {"params": prng_key, "dropout": dropout_key},
            data.KittiStructNormalized(
                image=jnp.zeros((1, 1, 50, 150, 3)),
                image_diff=jnp.zeros((1, 1, 50, 150, 3)),
                x=jnp.zeros((1, 1)),
                y=jnp.zeros((1, 1)),
                theta=jnp.zeros((1, 1)),
                linear_vel=jnp.zeros((1, 1)),
                angular_vel=jnp.zeros((1, 1)),
            ),  # batch axes are (N, T)
            train=True,
        )

        # Optimizer setup
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_gradient_norm),
            optax.adam(
                learning_rate=utils.warmup_schedule(
                    learning_rate=config.learning_rate,
                    warmup_steps=config.warmup_steps,
                )
            ),
        )
        optimizer_state = optimizer.init(learnable_params)

        # Done!
        return TrainState(
            config=config,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            lstm=lstm,
            learnable_params=learnable_params,
            prng_key=jax.random.PRNGKey(config.random_seed),
            steps=0,
            train=train,
        )

    @jax.jit
    def training_step(
        self, batch: data.KittiStructNormalized
    ) -> Tuple["TrainState", fifteen.experiments.TensorboardLogData]:
        def compute_loss(
            learnable_params: LearnableParams, prng_key: jax.random.KeyArray
        ) -> Tuple[jnp.ndarray, fifteen.experiments.TensorboardLogData]:
            """Compute average loss for all trajectories in the batch."""
            (_N, _T) = batch.get_batch_axes()

            batch_unnorm = batch.unnormalize()

            regressed_poses: jaxlie.SE2 = self.lstm.apply(
                learnable_params, batch, train=self.train, rngs={"dropout": prng_key}
            )
            gt_poses: jaxlie.SE2 = jax.vmap(jax.vmap(jaxlie.SE2.from_xy_theta))(
                batch_unnorm.x, batch_unnorm.y, batch_unnorm.theta
            )

            translation_delta = regressed_poses.translation() - gt_poses.translation()
            normalized_delta = data.KittiStructRaw(
                x=translation_delta[..., 0],
                y=translation_delta[..., 1],
            ).normalize(scale_only=True)
            translation_loss = jnp.mean(
                normalized_delta.x ** 2 + normalized_delta.y ** 2
            )

            rotation_loss = jnp.mean(
                math_utils.wrap_angle(
                    regressed_poses.rotation().as_radians()
                    - gt_poses.rotation().as_radians()
                )
                ** 2
            )

            training_loss = translation_loss + rotation_loss

            return training_loss, fifteen.experiments.TensorboardLogData(
                scalars={
                    "train/translation_loss": translation_loss,
                    "train/rotation_loss": rotation_loss,
                    "train/training_loss": training_loss,
                },
                histograms={
                    "regressed_translations_x": regressed_poses.translation()[..., 0],
                    "regressed_translations_y": regressed_poses.translation()[..., 1],
                    "regressed_radians": regressed_poses.rotation().as_radians(),
                    "gt_translations_x": gt_poses.translation()[..., 0],
                    "gt_translations_y": gt_poses.translation()[..., 1],
                    "gt_radians": gt_poses.rotation().as_radians(),
                },
            )

        # Split PRNG key
        prng_key, prng_key_new = jax.random.split(self.prng_key)

        # Compute loss + backprop => apply gradient transforms => update parameters
        (loss, compute_loss_log_data), grads = jax.value_and_grad(
            compute_loss, argnums=0, has_aux=True
        )(self.learnable_params, prng_key)
        updates, optimizer_state_new = self.optimizer.update(
            grads, self.optimizer_state, self.learnable_params
        )
        learnable_params_new = optax.apply_updates(
            self.learnable_params,
            updates,
        )

        # Data to log
        log_data = fifteen.experiments.TensorboardLogData.merge(
            compute_loss_log_data,
            fifteen.experiments.TensorboardLogData(
                scalars={
                    "train/gradient_norm": optax.global_norm(grads),
                }
            ),
        )

        # Build updated state
        with jax_dataclasses.copy_and_mutate(self) as updated_state:
            updated_state.optimizer_state = optimizer_state_new
            updated_state.learnable_params = learnable_params_new
            updated_state.prng_key = prng_key_new
            updated_state.steps = self.steps + 1
        return updated_state, log_data
