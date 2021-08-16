from typing import Any, Tuple

import jax
import jax_dataclasses
import optax
from jax import numpy as jnp

from .. import experiment_files, utils
from . import data, experiment_config, networks_lstm

Pytree = Any
PRNGKey = jnp.ndarray


@jax_dataclasses.pytree_dataclass
class TrainState:
    config: experiment_config.LstmExperimentConfig = jax_dataclasses.static_field()
    optimizer: optax.GradientTransformation = jax_dataclasses.static_field()
    optimizer_state: optax.OptState

    # LSTM model and parameters.
    lstm: networks_lstm.DiskLstm = jax_dataclasses.static_field()
    learnable_params: Pytree

    steps: int

    @staticmethod
    def initialize(
        config: experiment_config.LstmExperimentConfig,
    ) -> "TrainState":
        lstm = networks_lstm.DiskLstm(bidirectional=config.bidirectional)
        learnable_params = lstm.init(
            {"params": jax.random.PRNGKey(config.random_seed)},
            data.DiskStructNormalized(
                image=jnp.zeros((1, 1, 120, 120, 3)),
                visible_pixels_count=jnp.zeros((1, 1)),
                position=jnp.zeros((1, 1, 2)),
                velocity=jnp.zeros((1, 1, 2)),
            ),
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

        # Done!
        return TrainState(
            config=config,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            lstm=lstm,
            learnable_params=learnable_params,
            steps=0,
        )

    @jax.jit
    def training_step(
        self, batch: data.DiskStructNormalized
    ) -> Tuple["TrainState", experiment_files.TensorboardLogData]:

        # Shape checks
        (batch_size, sequence_length) = batch.check_shapes_and_get_batch_axes()
        assert sequence_length == self.config.train_sequence_length

        def compute_loss(
            learnable_params: Pytree,
        ) -> jnp.ndarray:
            positions = self.lstm.apply(learnable_params, batch)
            assert positions.shape == batch.position.shape
            return jnp.mean((positions - batch.position) ** 2)

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
        log_data = experiment_files.TensorboardLogData(
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
