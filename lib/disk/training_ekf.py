import pathlib
from typing import Any, Optional, Tuple

import fifteen
import jax
import jax_dataclasses
import optax
from jax import numpy as jnp

from .. import manifold_ekf, utils
from . import data, experiment_config, fg_system, networks

Pytree = Any


DiskEkf = manifold_ekf.EkfDefinition[fg_system.State, jnp.ndarray, None]


@jax_dataclasses.pytree_dataclass
class TrainState:
    config: experiment_config.EkfExperimentConfig = jax_dataclasses.static_field()
    optimizer: optax.GradientTransformation = jax_dataclasses.static_field()
    optimizer_state: optax.OptState

    # CNN model and parameters. Note that these are frozen!
    cnn_model: networks.DiskVirtualSensor = jax_dataclasses.static_field()
    cnn_params: Pytree

    # Uncertainty model. This is what we're learning!
    regress_uncertainties: networks.RegressUncertaintiesFunction = (
        jax_dataclasses.static_field()
    )
    learnable_params: Pytree

    # EKF definition.
    ekf: DiskEkf = jax_dataclasses.static_field()

    steps: int

    @staticmethod
    def initialize(
        config: experiment_config.EkfExperimentConfig,
    ) -> "TrainState":
        # Load position CNN
        cnn_model, cnn_params = networks.make_position_cnn(seed=config.random_seed)
        cnn_params = fifteen.experiments.Experiment(
            data_dir=pathlib.Path("./experiments/")
            / config.pretrained_virtual_sensor_identifier.format(
                dataset_fold=config.dataset_fold
            )
        ).restore_checkpoint(cnn_params, prefix="best_val_params_")

        # Helper for computing uncertainties... this is what we're learning
        regress_uncertainties, learnable_params = networks.make_regress_uncertainties(
            noise_model=config.noise_model,
            seed=config.random_seed,
        )

        # Optimizer setup
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_gradient_norm),
            optax.adam(
                learning_rate=utils.warmup_schedule(
                    config.learning_rate, config.warmup_steps
                )
            ),
        )
        optimizer_state = optimizer.init(learnable_params)

        # Define EKF
        x: fg_system.State
        ekf: DiskEkf = manifold_ekf.EkfDefinition(
            dynamics_model=lambda x, u: fg_system.State.predict_next(x),
            observation_model=lambda x: x.position,
        )

        # Done!
        return TrainState(
            config=config,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            cnn_model=cnn_model,
            cnn_params=cnn_params,
            regress_uncertainties=regress_uncertainties,
            learnable_params=learnable_params,
            ekf=ekf,
            steps=0,
        )

    @jax.jit
    def training_step(
        self, batch: data.DiskStructNormalized
    ) -> Tuple["TrainState", fifteen.experiments.TensorboardLogData]:
        # Shape checks
        (batch_size, sequence_length) = batch.get_batch_axes()
        assert sequence_length == self.config.train_sequence_length

        def compute_loss_single(
            trajectory: data.DiskStructNormalized,
            learnable_params: Pytree,
        ) -> jnp.ndarray:
            (timesteps,) = trajectory.get_batch_axes()

            unnormed_position = (
                data.DiskStructNormalized(position=trajectory.position)
                .unnormalize()
                .position
            )

            posterior = self.run_ekf(trajectory, learnable_params)
            assert posterior.cov.shape == (timesteps, 4, 4)
            if self.config.loss is experiment_config.EkfLoss.E2E_MSE:
                return jnp.mean((posterior.mean.position - unnormed_position) ** 2)
            elif self.config.loss is experiment_config.EkfLoss.E2E_MARGINAL_NLL:

                def mahalanobis_distance(
                    pred: jnp.ndarray, label: jnp.ndarray, cov: jnp.ndarray
                ) -> jnp.ndarray:
                    """Compute scalar Mahalanobis distance."""
                    return (
                        (label - pred).reshape((1, -1))  # type: ignore
                        @ jnp.linalg.inv(cov)
                        @ (label - pred).reshape((-1, 1))  # type: ignore
                    ).reshape(())

                def gaussian_nll(
                    position: jnp.ndarray,
                    distribution: manifold_ekf.MultivariateGaussian[fg_system.State],
                ) -> jnp.ndarray:
                    """Compute NLL under a multivariate Gaussian distribution."""
                    assert position.shape == distribution.mean.position.shape == (2,)
                    assert distribution.cov.shape == (4, 4)

                    position_mean = distribution.mean.position
                    position_cov = distribution.cov[:2, :2]

                    mahalanobis_term = mahalanobis_distance(
                        pred=position, label=position_mean, cov=position_cov
                    )
                    cov_determinant_term = jnp.log(jnp.linalg.det(position_cov))

                    assert mahalanobis_term.shape == cov_determinant_term.shape == ()
                    return mahalanobis_term + cov_determinant_term

                return jnp.mean(jax.vmap(gaussian_nll)(unnormed_position, posterior))
            else:
                assert False

        def compute_loss(
            learnable_params: Pytree,
        ) -> jnp.ndarray:
            losses = jax.vmap(compute_loss_single, in_axes=(0, None))(
                batch, learnable_params
            )
            return jnp.mean(losses)

        # Compute loss + backprop => apply gradient transforms => update parameters
        loss, grads = jax.value_and_grad(compute_loss)(self.learnable_params)
        updates, optimizer_state_new = self.optimizer.update(
            grads, self.optimizer_state, self.learnable_params
        )
        learnable_params_new = optax.apply_updates(
            self.learnable_params,
            updates,
        )

        # Log data
        log_data = fifteen.experiments.TensorboardLogData(
            scalars={
                "train/training_loss": loss,
                "train/gradient_norm": optax.global_norm(grads),
            },
        )

        # Build updated state
        with jax_dataclasses.copy_and_mutate(self) as updated_state:
            updated_state.optimizer_state = optimizer_state_new
            updated_state.learnable_params = learnable_params_new
            updated_state.steps = self.steps + 1
        return updated_state, log_data

    @jax.jit
    def run_ekf(
        self,
        trajectory: data.DiskStructNormalized,
        learnable_params: Optional[Pytree] = None,
    ) -> manifold_ekf.MultivariateGaussian[fg_system.State]:
        (timesteps,) = trajectory.get_batch_axes()

        # Some type aliases
        Belief = manifold_ekf.MultivariateGaussian[fg_system.State]
        Observation = manifold_ekf.MultivariateGaussian[jnp.ndarray]

        # Pass images through virtual sensor
        observation_means = (
            data.DiskStructNormalized(
                position=self.cnn_model.apply(self.cnn_params, trajectory.image)
            )
            .unnormalize()
            .position
        )
        observation_uncertainties = self.regress_uncertainties(
            learnable_params=learnable_params
            if learnable_params is not None
            else self.learnable_params,
            visible_pixels_count=trajectory.visible_pixels_count,
        )
        observations = manifold_ekf.MultivariateGaussian(
            mean=observation_means,
            cov=(1.0 / observation_uncertainties[:, None, None] ** 2)
            * jnp.eye(2)[None, :, :],
        )

        # Initialize beliefs
        initial_state: data.DiskStructNormalized = jax.tree_map(
            lambda x: x[0, ...], trajectory
        )
        initial_state_unnorm = initial_state.unnormalize()

        initial_belief: Belief = manifold_ekf.MultivariateGaussian(
            mean=fg_system.State(
                position=initial_state_unnorm.position,
                velocity=initial_state.velocity,
            ),
            cov=jnp.eye(4) * 1e-7,  # This can probably just be zeros
        )

        def ekf_step(
            # carry
            belief: manifold_ekf.MultivariateGaussian[fg_system.State],
            # x
            observation: Observation,
        ):
            belief = self.ekf.predict(
                belief,
                control_input=None,
                dynamics_cov=jnp.diag(fg_system.DYNAMICS_COVARIANCE_DIAGONAL),
            )

            # EKF correction step
            belief = self.ekf.correct(
                belief,
                observation=observation,
            )
            return belief, belief  # (carry, y)

        final_belief: Belief
        beliefs: Belief
        final_belief, beliefs = jax.lax.scan(ekf_step, initial_belief, observations)

        return beliefs
