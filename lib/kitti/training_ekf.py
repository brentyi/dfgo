from typing import Any, Optional, Tuple

import fifteen
import jax
import jax_dataclasses
import jaxlie
import optax
from jax import numpy as jnp

from .. import manifold_ekf, utils
from . import data, experiment_config, fg_system, networks


def make_observation(
    velocities: networks.RegressedVelocities,
    vision_sqrt_precision_diagonal: jnp.ndarray,
) -> manifold_ekf.MultivariateGaussian[jnp.ndarray]:
    return manifold_ekf.MultivariateGaussian(
        mean=jnp.array(
            [
                velocities.linear_vel,
                velocities.angular_vel,
            ]
        ),
        cov=jnp.diag(1.0 / vision_sqrt_precision_diagonal**2),
    )


Pytree = Any
KittiEkf = manifold_ekf.EkfDefinition[fg_system.State, jnp.ndarray, None]


@jax_dataclasses.pytree_dataclass
class TrainState:
    config: experiment_config.EkfExperimentConfig = jax_dataclasses.static_field()
    optimizer: optax.GradientTransformation = jax_dataclasses.static_field()
    optimizer_state: optax.OptState

    regress_velocities: networks.RegressVelocitiesFunction = (
        jax_dataclasses.static_field()
    )
    regress_uncertainties: networks.RegressUncertaintiesFunction = (
        jax_dataclasses.static_field()
    )
    learnable_params: Pytree

    ekf: KittiEkf = jax_dataclasses.static_field()

    prng_key: jax.random.KeyArray
    steps: int
    train: bool = jax_dataclasses.static_field()

    @staticmethod
    def initialize(
        config: experiment_config.EkfExperimentConfig, train: bool
    ) -> "TrainState":
        """Initialize a training state."""

        # Define state manifold
        state_manifold = manifold_ekf.ManifoldDefinition(
            boxplus=fg_system.StateVariable.manifold_retract,
            boxminus=fg_system.State.manifold_minus,
            local_dim_from_point=lambda _: fg_system.StateVariable.get_local_parameter_dim(),
        )
        state_manifold.assert_validity(fg_system.StateVariable.get_default_value())

        # Define EKF
        x: fg_system.State
        u: None
        ekf: KittiEkf = manifold_ekf.EkfDefinition(
            dynamics_model=lambda x, u: fg_system.State.predict_next(x),
            observation_model=lambda state: state.velocities,
            state_manifold=state_manifold,
        )

        # Make neural network abstractions
        pretrained_virtual_sensor_identifier: str = (
            config.pretrained_virtual_sensor_identifier.format(
                dataset_fold=config.dataset_fold
            )
        )

        regress_velocities = networks.make_regress_velocities(
            pretrained_virtual_sensor_identifier
        )

        (
            regress_uncertainties,
            learnable_params,
        ) = networks.make_regress_uncertainties(
            config.noise_model,
            pretrained_virtual_sensor_identifier,
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
            regress_velocities=regress_velocities,
            regress_uncertainties=regress_uncertainties,
            learnable_params=learnable_params,
            ekf=ekf,
            prng_key=jax.random.PRNGKey(config.random_seed),
            steps=0,
            train=train,
        )

    @jax.jit
    def training_step(
        self, batched_trajectory: data.KittiStructNormalized
    ) -> Tuple["TrainState", fifteen.experiments.TensorboardLogData]:
        def compute_loss_single(
            learnable_params: Pytree,
            trajectory: data.KittiStructNormalized,
            prng_key: jax.random.KeyArray,
        ):
            trajectory_unnorm = trajectory.unnormalize()
            estimated_state = self.run_ekf(
                trajectory_unnorm, prng_key, learnable_params
            )
            return jnp.mean(
                (estimated_state.x - trajectory_unnorm.x) ** 2
                + (estimated_state.y - trajectory_unnorm.y) ** 2
            )

        def compute_loss(
            learnable_params: Pytree, prng_key: jax.random.KeyArray
        ) -> jnp.ndarray:
            batch_size: int = batched_trajectory.x.shape[0]
            losses = jax.vmap(compute_loss_single, in_axes=(None, 0, 0))(
                learnable_params,
                batched_trajectory,
                jax.random.split(prng_key, num=batch_size),
            )
            assert len(losses.shape) == 1
            return jnp.mean(losses)

        # Split PRNG key
        prng_key, prng_key_new = jax.random.split(self.prng_key)

        # Compute loss + backprop => apply gradient transforms => update parameters
        loss, grads = jax.value_and_grad(compute_loss, argnums=0)(
            self.learnable_params, prng_key
        )
        updates, optimizer_state_new = self.optimizer.update(
            grads, self.optimizer_state, self.learnable_params
        )
        learnable_params_new = optax.apply_updates(
            self.learnable_params,
            updates,
        )

        # Data to log
        log_data = fifteen.experiments.TensorboardLogData(
            scalars={
                "train/training_loss": loss,
                "train/gradient_norm": optax.global_norm(grads),
            },
            histograms={},
        )

        # Build updated state
        with jax_dataclasses.copy_and_mutate(self) as updated_state:
            updated_state.optimizer_state = optimizer_state_new
            updated_state.learnable_params = learnable_params_new
            updated_state.prng_key = prng_key_new
            updated_state.steps = self.steps + 1
        return updated_state, log_data

    def run_ekf(
        self,
        trajectory: data.KittiStructRaw,
        prng_key: jax.random.KeyArray,
        learnable_params: Optional[Pytree] = None,
    ) -> fg_system.State:
        (_timesteps,) = trajectory.get_batch_axes()

        # Some type aliases
        Belief = manifold_ekf.MultivariateGaussian[fg_system.State]
        Observation = manifold_ekf.MultivariateGaussian[jnp.ndarray]

        # Pass images through virtual sensor
        trajectory_normalized = trajectory.normalize()
        velocities = self.regress_velocities(trajectory_normalized.get_stacked_image())
        uncertainties = self.regress_uncertainties(
            self.learnable_params if learnable_params is None else learnable_params,
            trajectory_normalized.get_stacked_image(),
            prng_key=prng_key,
            train=True,
        )
        observations: Observation = jax.vmap(make_observation)(
            velocities, uncertainties.vision_sqrt_precision_diagonal
        )

        # Initialize beliefs
        initial_belief: Belief = manifold_ekf.MultivariateGaussian(
            mean=fg_system.State.make(
                pose=jaxlie.SE2.from_xy_theta(
                    x=trajectory.x[0],
                    y=trajectory.y[0],
                    theta=trajectory.theta[0],
                ),
                linear_vel=trajectory.linear_vel[0],
                angular_vel=trajectory.angular_vel[0],
            ),
            cov=jnp.eye(5) * 1e-7,  # This can probably just be zeros
        )

        (timesteps,) = trajectory.get_batch_axes()

        def ekf_step(
            # carry
            belief: manifold_ekf.MultivariateGaussian[fg_system.State],
            # x
            observation: Observation,
        ):
            belief = self.ekf.predict(
                belief,
                control_input=None,
                dynamics_cov=jnp.diag(
                    1.0 / uncertainties.dynamics_sqrt_precision_diagonal**2
                ),
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

        return beliefs.mean
