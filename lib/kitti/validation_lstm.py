from typing import Callable

import jax
import jax_dataclasses
import jaxlie
import numpy as onp
import torch

from .. import validation_tracker
from . import data, math_utils, training_lstm


def make_compute_metrics(
    eval_dataloader: torch.utils.data.DataLoader[data.KittiStructNormalized],
) -> Callable[[training_lstm.TrainState], validation_tracker.ValidationMetrics]:
    def compute_metrics(
        train_state: training_lstm.TrainState,
    ) -> validation_tracker.ValidationMetrics:
        # Eval mode
        train_state = jax_dataclasses.replace(train_state, train=False)

        batch: data.KittiStructNormalized

        m_per_m_total: float = 0.0
        rad_per_m_total: float = 0.0

        for batch in eval_dataloader:
            batch_unnorm = batch.unnormalize()

            (N, T) = batch.get_batch_axes()

            regressed_poses: jaxlie.SE2 = jax.jit(
                train_state.lstm.apply, static_argnames=("train",)
            )(train_state.learnable_params, batch, train=False)
            regressed_positions = regressed_poses.translation()
            assert regressed_positions.shape == (N, T, 2)

            regressed_radians = regressed_poses.rotation().as_radians()
            assert regressed_radians.shape == (N, T)

            error_m = onp.sqrt(
                (regressed_positions[:, -1, 0] - batch_unnorm.x[:, -1]) ** 2
                + (regressed_positions[:, -1, 1] - batch_unnorm.y[:, -1]) ** 2
            )
            assert error_m.shape == (N,)

            error_rad = onp.abs(
                math_utils.wrap_angle(
                    regressed_radians[:, -1] - batch_unnorm.theta[:, -1]
                )
            )
            assert error_rad.shape == (N,)

            true_distance_traveled = jax.vmap(math_utils.compute_distance_traveled)(
                batch_unnorm.x, batch_unnorm.y
            )
            assert true_distance_traveled.shape == (N,)

            m_per_m_total += onp.mean(error_m / true_distance_traveled)
            rad_per_m_total += onp.mean(error_rad / true_distance_traveled)

        m_per_m = m_per_m_total / len(eval_dataloader)
        rad_per_m = rad_per_m_total / len(eval_dataloader)

        return m_per_m, {"m_per_m": m_per_m, "rad_per_m": rad_per_m}

    return compute_metrics
