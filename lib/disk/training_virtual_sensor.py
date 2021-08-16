from typing import Any, Optional, Tuple

import jax
import jax_dataclasses
import optax
from jax import numpy as jnp

from .. import experiment_files
from . import data, experiment_config, networks

Pytree = Any
PRNGKey = jnp.ndarray


@jax_dataclasses.pytree_dataclass
class TrainState:
    config: experiment_config.VirtualSensorPretrainingExperimentConfig = (
        jax_dataclasses.static_field()
    )
    optimizer: optax.GradientTransformation = jax_dataclasses.static_field()
    optimizer_state: optax.OptState

    # CNN model and parameters. This is what we're learning!
    cnn_model: networks.DiskVirtualSensor = jax_dataclasses.static_field()
    learnable_params: Pytree

    steps: int

    @staticmethod
    def initialize(
        config: experiment_config.VirtualSensorPretrainingExperimentConfig,
    ) -> "TrainState":
        cnn_model, learnable_params = networks.make_position_cnn(
            seed=config.random_seed
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
            steps=0,
        )

    def compute_loss(
        self,
        batch: data.DiskStructNormalized,
        learnable_params: Optional[Pytree] = None,
    ) -> Tuple[float, jnp.ndarray]:
        """Compute MSE loss."""

        if learnable_params is None:
            learnable_params = self.learnable_params

        # Note that all units here are normalized
        cnn_outputs = self.cnn_model.apply(learnable_params, batch.image)
        assert cnn_outputs.shape == batch.position.shape
        loss = jnp.mean((cnn_outputs - batch.position) ** 2)
        return loss, cnn_outputs

    @jax.jit
    def training_step(
        self, batch: data.DiskStructNormalized
    ) -> Tuple["TrainState", experiment_files.TensorboardLogData]:
        """Single training step."""

        # Quick shape check
        (batch_size,) = batch.check_shapes_and_get_batch_axes()

        # Compute loss + backprop => apply gradient transforms => update parameters
        (loss, cnn_outputs), grads = jax.value_and_grad(
            self.compute_loss, argnums=1, has_aux=True
        )(batch, self.learnable_params)
        updates, optimizer_state_new = self.optimizer.update(
            grads, self.optimizer_state, self.learnable_params
        )
        learnable_params_new = optax.apply_updates(self.learnable_params, updates)

        # Log data
        log_data = experiment_files.TensorboardLogData(
            scalars={
                "train/training_loss": loss,
                "train/gradient_norm": optax.global_norm(grads),
            },
            histograms={
                "pred_x": cnn_outputs[:, 0],
                "pred_y": cnn_outputs[:, 1],
            },
        )

        # Update state and return
        with jax_dataclasses.copy_and_mutate(self) as updated_state:
            updated_state.optimizer_state = optimizer_state_new
            updated_state.learnable_params = learnable_params_new
            updated_state.steps = self.steps + 1
        return updated_state, log_data
