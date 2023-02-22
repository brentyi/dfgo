from typing import Optional, Tuple

import fifteen
import jax
import jax_dataclasses
import optax
from jax import numpy as jnp

from . import data, experiment_config, networks


@jax_dataclasses.pytree_dataclass
class TrainState:
    """Everything we need for training!"""

    config: experiment_config.VirtualSensorPretrainingExperimentConfig = (
        jax_dataclasses.static_field()
    )
    optimizer: optax.GradientTransformation = jax_dataclasses.static_field()
    optimizer_state: optax.OptState
    cnn_model: networks.KittiVirtualSensor = jax_dataclasses.static_field()
    learnable_params: networks.KittiVirtualSensorParameters
    prng_key: jax.random.KeyArray
    steps: int
    train: bool = jax_dataclasses.static_field()

    @staticmethod
    def initialize(
        config: experiment_config.VirtualSensorPretrainingExperimentConfig, train: bool
    ) -> "TrainState":
        cnn_model, learnable_params = networks.make_observation_cnn(
            random_seed=config.random_seed
        )

        # Optimizer setup
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_gradient_norm),
            optax.adam(learning_rate=config.learning_rate),
        )
        optimizer_state = optimizer.init(learnable_params)

        # Done!
        return TrainState(
            config=config,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            cnn_model=cnn_model,
            learnable_params=learnable_params,
            prng_key=jax.random.PRNGKey(config.random_seed),
            steps=0,
            train=train,
        )

    @jax.jit
    def compute_loss(
        self,
        batch: data.KittiStructNormalized,
        prng_key: jax.random.KeyArray,
        learnable_params: Optional[networks.KittiVirtualSensorParameters] = None,
    ) -> Tuple[float, jnp.ndarray]:
        if learnable_params is None:
            learnable_params = self.learnable_params

        # Quick shape check
        (batch_size,) = batch.x.shape
        assert batch.image.shape == batch.image_diff.shape == (batch_size, 50, 150, 3)

        cnn_output = self.cnn_model.apply(
            learnable_params,
            batch.get_stacked_image(),
            train=self.train,
            rngs={"dropout": prng_key},
        )
        pred_velocities = cnn_output[:, :2]
        pred_sqrt_covariance_diagonal = cnn_output[:, 2:]

        label_velocities = batch.get_stacked_velocity()
        assert (
            pred_velocities.shape
            == pred_sqrt_covariance_diagonal.shape
            == label_velocities.shape
            == (batch_size, 2)
        )

        if self.config.loss is experiment_config.VirtualSensorLossEnum.MSE:
            loss = jnp.mean((pred_velocities - label_velocities) ** 2)
        elif self.config.loss is experiment_config.VirtualSensorLossEnum.NLL:
            assert (
                False
            ), "Need to make this consistent with networks.make_regress_uncertainties"
            # cov_determinant_term = jnp.log(
            #     jnp.prod(pred_sqrt_covariance_diagonal, axis=-1) ** (2)
            # )
            # mahalanobis_term = jnp.sum(
            #     ((pred_velocities - label_velocities) / pred_sqrt_covariance_diagonal)
            #     ** 2,
            #     axis=-1,
            # )
            #
            # N = pred_velocities.shape[0]
            # assert cov_determinant_term.shape == mahalanobis_term.shape == (N,)
            #
            # loss = jnp.mean(mahalanobis_term + cov_determinant_term)
        else:
            assert False

        return loss, cnn_output

    @jax.jit
    def training_step(
        self,
        batch: data.KittiStructNormalized,
    ) -> Tuple["TrainState", fifteen.experiments.TensorboardLogData]:
        """Single training step."""

        # Quick shape check
        (batch_size,) = batch.x.shape
        assert batch.image.shape == batch.image_diff.shape == (batch_size, 50, 150, 3)

        # Split PRNG key
        prng_key, prng_key_new = jax.random.split(self.prng_key)

        # Compute loss + backprop => apply gradient transforms => update parameters
        (loss, cnn_output), grads = jax.value_and_grad(
            self.compute_loss, argnums=2, has_aux=True
        )(batch, prng_key, self.learnable_params)
        updates, optimizer_state_new = self.optimizer.update(
            grads, self.optimizer_state, self.learnable_params
        )
        learnable_params_new = optax.apply_updates(
            self.learnable_params,
            updates,
        )

        # Log to Tensorboard
        pred_velocities_unnorm = data.KittiStructNormalized(
            linear_vel=cnn_output[:, 0],
            angular_vel=cnn_output[:, 1],
        ).unnormalize()

        log_data = fifteen.experiments.TensorboardLogData(
            scalars={
                "train/training_loss": loss,
                "train/gradient_norm": optax.global_norm(grads),
                "train/linear_vel_rmse": jnp.sqrt(
                    jnp.mean(
                        (
                            data.KittiStructNormalized(linear_vel=batch.linear_vel)
                            .unnormalize()
                            .linear_vel
                            - pred_velocities_unnorm.linear_vel
                        )
                        ** 2
                    )
                ),
                "train/angular_vel_rmse": jnp.sqrt(
                    jnp.mean(
                        (
                            data.KittiStructNormalized(angular_vel=batch.angular_vel)
                            .unnormalize()
                            .angular_vel
                            - pred_velocities_unnorm.angular_vel
                        )
                        ** 2
                    )
                ),
            },
            histograms={
                "linear_vel": pred_velocities_unnorm.linear_vel,
                "angular_vel": pred_velocities_unnorm.angular_vel,
                "linear_uncertainty": cnn_output[:, 2],
                "angular_uncertainty": cnn_output[:, 3],
            },
        )

        # Build updated state
        with jax_dataclasses.copy_and_mutate(self) as updated_state:
            updated_state.optimizer_state = optimizer_state_new
            updated_state.learnable_params = learnable_params_new
            updated_state.prng_key = prng_key_new
            updated_state.steps = self.steps + 1
        return updated_state, log_data
