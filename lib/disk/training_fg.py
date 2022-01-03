import pathlib
from typing import Any, Optional, Tuple

import fifteen
import jax
import jax_dataclasses
import jaxfg
import optax
from jax import numpy as jnp

from .. import utils
from . import data, experiment_config, fg_system, fg_utils, networks

Pytree = Any


@jax_dataclasses.pytree_dataclass
class TrainState:
    config: experiment_config.FactorGraphExperimentConfig = (
        jax_dataclasses.static_field()
    )
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

    graph_template: jaxfg.core.StackedFactorGraph

    steps: int

    @staticmethod
    def initialize(
        config: experiment_config.FactorGraphExperimentConfig,
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

        # Make factor graph template
        graph_template = fg_utils.make_factor_graph(
            trajectory_length=config.train_sequence_length
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
            graph_template=graph_template,
            steps=0,
        )

    def update_factor_graph(
        self,
        trajectory: data.DiskStructNormalized,
        graph_template: Optional[jaxfg.core.StackedFactorGraph] = None,
        learnable_params: Optional[Pytree] = None,
    ) -> jaxfg.core.StackedFactorGraph:
        # Shape checks
        (sequence_length,) = trajectory.get_batch_axes()

        # Optional parameters default to self.*
        if graph_template is None:
            graph_template = self.graph_template
        if learnable_params is None:
            learnable_params = self.learnable_params

        with jax_dataclasses.copy_and_mutate(
            graph_template, validate=True
        ) as graph_output:
            vision_factor: fg_system.VisionFactor
            dynamics_factor: fg_system.DynamicsFactor

            vision_factor, dynamics_factor = [
                stack.factor for stack in graph_output.factor_stacks
            ]

            assert isinstance(vision_factor, fg_system.VisionFactor)
            assert isinstance(dynamics_factor, fg_system.DynamicsFactor)

            # Regress and update positions
            vision_factor.predicted_position = (
                data.DiskStructNormalized(
                    position=self.cnn_model.apply(self.cnn_params, trajectory.image)
                )
                .unnormalize()
                .position
            )

            # Regress uncertainties: we get one scalar per timestep
            regressed_uncertainties = self.regress_uncertainties(
                learnable_params=learnable_params,
                visible_pixels_count=trajectory.visible_pixels_count,
            )
            assert regressed_uncertainties.shape == (sequence_length,)

            # ...that we then want to arrange into 2D vectors
            vision_factor.noise_model.sqrt_precision_diagonal = jnp.stack(
                [regressed_uncertainties, regressed_uncertainties], axis=-1
            )

        return graph_output

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
        ) -> Tuple[jnp.ndarray, fifteen.experiments.TensorboardLogData]:
            graph = self.update_factor_graph(
                trajectory=trajectory,
                graph_template=self.graph_template,
                learnable_params=learnable_params,
            )

            trajectory_unnorm = trajectory.unnormalize()
            gt_assignments = fg_utils.assignments_from_trajectory(
                trajectory_unnorm, graph.get_variables()
            )

            if self.config.loss is experiment_config.FactorGraphLossEnum.JOINT_NLL:
                # Compute joint NLL loss
                loss = graph.compute_joint_nll(assignments=gt_assignments)
            elif (
                self.config.loss is experiment_config.FactorGraphLossEnum.SURROGATE_LOSS
            ):

                # Compute end-to-end loss
                solved_assignments = graph.solve(
                    initial_assignments=gt_assignments,
                    solver=jaxfg.solvers.FixedIterationGaussNewtonSolver(
                        unroll=True,
                        iterations=3,
                        verbose=False,
                        linear_solver=jaxfg.sparse.ConjugateGradientSolver(
                            tolerance=1e-5
                        ),
                    ),
                )
                stacked_position = solved_assignments.get_stacked_value(
                    fg_system.StateVariable
                ).position

                position_delta: jnp.ndarray = (
                    trajectory_unnorm.position - stacked_position
                )
                loss = jnp.mean(
                    data.DiskStructRaw(position=position_delta)
                    .normalize(scale_only=True)
                    .position
                    ** 2
                )

            log_data = fifteen.experiments.TensorboardLogData(
                histograms={
                    "regressed_uncertainties": graph.factor_stacks[
                        0
                    ].factor.noise_model.sqrt_precision_diagonal
                }
            )
            return loss, log_data

        def compute_loss(
            learnable_params: Pytree,
        ) -> Tuple[jnp.ndarray, fifteen.experiments.TensorboardLogData]:
            losses, log_data = jax.vmap(compute_loss_single, in_axes=(0, None))(
                batch, learnable_params
            )
            return jnp.mean(losses), log_data

        # Compute loss + backprop => apply gradient transforms => update parameters
        log_data: fifteen.experiments.TensorboardLogData
        (loss, log_data), grads = jax.value_and_grad(compute_loss, has_aux=True)(
            self.learnable_params
        )
        updates, optimizer_state_new = self.optimizer.update(
            grads, self.optimizer_state, self.learnable_params
        )
        learnable_params_new = optax.apply_updates(
            self.learnable_params,
            updates,
        )

        # Log data
        log_data = log_data.merge_scalars(
            {
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
