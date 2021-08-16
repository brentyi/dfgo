"""Training helpers for KITTI task."""
from typing import Any, Optional, Tuple

import jax
import jax_dataclasses
import jaxfg
import optax
from jax import numpy as jnp

from .. import experiment_files, utils
from . import data, experiment_config, fg_losses, fg_utils, networks

Pytree = Any
LearnableParams = Pytree
PRNGKey = jnp.ndarray


@jax_dataclasses.pytree_dataclass
class _TrainingPerSampleMetadata:
    """Metadata to be stacked from each sample. For logging, debugging, etc."""

    training_loss: jnp.ndarray
    regressed_velocities: networks.RegressedVelocities
    regressed_uncertainties: networks.RegressedUncertainties


@jax_dataclasses.pytree_dataclass
class TrainState:
    """Everything needed for training."""

    config: experiment_config.FactorGraphExperimentConfig = (
        jax_dataclasses.static_field()
    )
    optimizer: optax.GradientTransformation = jax_dataclasses.static_field()
    optimizer_state: optax.OptState

    regress_velocities: networks.RegressVelocitiesFunction = (
        jax_dataclasses.static_field()
    )
    regress_uncertainties: networks.RegressUncertaintiesFunction = (
        jax_dataclasses.static_field()
    )
    learnable_params: LearnableParams

    graph_template: jaxfg.core.StackedFactorGraph

    prng_key: PRNGKey
    steps: int
    train: bool = (
        jax_dataclasses.static_field()
    )  # TODO: this probably does not need to be a field. Same goes for other training state objects.

    @staticmethod
    def initialize(
        config: experiment_config.FactorGraphExperimentConfig,
        train: bool,
    ) -> "TrainState":
        """Initialize a training state."""

        # Build factor graph
        graph_template = fg_utils.make_factor_graph(
            sequence_length=config.train_sequence_length
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
            graph_template=graph_template,
            prng_key=jax.random.PRNGKey(config.random_seed),
            steps=0,
            train=train,
        )

    @jax.jit
    def update_factor_graph(
        self,
        graph_template: jaxfg.core.StackedFactorGraph,
        trajectory: data.KittiStructNormalized,
        prng_key: PRNGKey,
        *,
        trajectory_raw: Optional[data.KittiStructRaw] = None,
        learnable_params: Optional[networks.LearnableParams] = None,
    ) -> Tuple[
        jaxfg.core.StackedFactorGraph,
        Tuple[networks.RegressedVelocities, networks.RegressedUncertainties],
    ]:
        """Update a factor graph for an input trajectory. Optional arguments are
        generally not needed. Returns new factor graph + a metadata tuple."""

        if trajectory_raw is None:
            trajectory_raw = trajectory.unnormalize()
        if learnable_params is None:
            learnable_params = self.learnable_params

        # Predictions
        prng_key_velocities, prng_key_uncertainties = jax.random.split(prng_key)
        velocities = self.regress_velocities(
            trajectory.get_stacked_image()[1:, :, :, :]
        )
        uncertainties = self.regress_uncertainties(
            learnable_params,
            stacked_images=trajectory.get_stacked_image()[1:, :, :, :],
            prng_key=prng_key_uncertainties,
            train=self.train,
        )

        # Make updated factor graph
        trajectory_raw = trajectory.unnormalize()
        return (
            fg_utils.update_factor_graph(
                graph_template=graph_template,
                trajectory_raw=trajectory_raw,
                predicted_velocities=velocities,
                vision_sqrt_precision_diagonal=uncertainties.vision_sqrt_precision_diagonal,
                dynamics_sqrt_precision_diagonal=uncertainties.dynamics_sqrt_precision_diagonal,
            ),
            (velocities, uncertainties),
        )

    @jax.jit
    def training_step(
        self, batched_trajectory: data.KittiStructNormalized
    ) -> Tuple["TrainState", experiment_files.TensorboardLogData]:
        """Single training step."""

        # Shared leading axes should be (batch, timesteps)
        assert len(batched_trajectory.check_shapes_and_get_batch_axes()) == 2

        def compute_loss_single(
            learnable_params: LearnableParams,
            trajectory: data.KittiStructNormalized,
            prng_key: PRNGKey,
        ) -> Tuple[jnp.ndarray, _TrainingPerSampleMetadata]:
            """Compute training loss for a single trajectory."""

            loss_config = self.config.loss_config

            prng_key0, prng_key1 = jax.random.split(prng_key)

            # Make updated factor graph
            trajectory_raw = trajectory.unnormalize()
            graph, (velocities, uncertainties) = self.update_factor_graph(
                graph_template=self.graph_template,
                trajectory=trajectory,
                prng_key=prng_key0,
                trajectory_raw=trajectory_raw,
                learnable_params=learnable_params,
            )

            # Return loss + any metadata
            loss = fg_losses.compute_loss(graph, trajectory_raw, loss_config, prng_key1)
            metadata = _TrainingPerSampleMetadata(
                training_loss=loss,
                regressed_velocities=velocities,
                regressed_uncertainties=uncertainties,
            )
            return loss, metadata

        def compute_loss(
            learnable_params: LearnableParams, prng_key: PRNGKey
        ) -> Tuple[jnp.ndarray, _TrainingPerSampleMetadata]:
            """Compute average loss for all trajectories in the batch."""
            batch_size: int = batched_trajectory.x.shape[0]
            losses, metadata = jax.vmap(compute_loss_single, in_axes=(None, 0, 0))(
                learnable_params,
                batched_trajectory,
                jax.random.split(prng_key, num=batch_size),
            )
            assert len(losses.shape) == 1
            return jnp.mean(losses), metadata

        # Split PRNG key
        prng_key, prng_key_new = jax.random.split(self.prng_key)

        # Compute loss + backprop => apply gradient transforms => update parameters
        per_sample_metadata: _TrainingPerSampleMetadata
        (loss, per_sample_metadata), grads = jax.value_and_grad(
            compute_loss, argnums=0, has_aux=True
        )(self.learnable_params, prng_key)
        updates, optimizer_state_new = self.optimizer.update(
            grads, self.optimizer_state, self.learnable_params
        )
        learnable_params_new = optax.apply_updates(
            self.learnable_params,
            updates,
        )

        # Log data
        regressed_velocities = per_sample_metadata.regressed_velocities
        regressed_uncertainties = per_sample_metadata.regressed_uncertainties
        log_data = experiment_files.TensorboardLogData(
            scalars={
                "train/training_loss": loss,
                "train/gradient_norm": optax.global_norm(grads),
            },
            histograms={
                "training_losses": per_sample_metadata.training_loss,
                "linear_vel": regressed_velocities.linear_vel,
                "angular_vel": regressed_velocities.angular_vel,
                "linear_uncertainty": regressed_uncertainties.vision_sqrt_precision_diagonal[
                    ..., 0
                ],
                "angular_uncertainty": regressed_uncertainties.vision_sqrt_precision_diagonal[
                    ..., 1
                ],
                **{
                    f"dynamics_uncertainty_{field}": regressed_uncertainties.dynamics_sqrt_precision_diagonal[
                        ..., i
                    ]
                    for i, field in enumerate(("x", "y", "omega", "vx", "vy"))
                },
            },
        )

        # Build updated state
        with jax_dataclasses.copy_and_mutate(self) as updated_state:
            updated_state.optimizer_state = optimizer_state_new
            updated_state.learnable_params = learnable_params_new
            updated_state.prng_key = prng_key_new
            updated_state.steps = self.steps + 1
        return updated_state, log_data
