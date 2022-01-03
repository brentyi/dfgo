from typing import Callable

import jax
import jax_dataclasses
import numpy as onp
import torch

from .. import validation_tracker
from . import data, training_virtual_sensor


def make_compute_metrics(
    eval_dataloader: torch.utils.data.DataLoader[data.KittiStructNormalized],
) -> Callable[
    [training_virtual_sensor.TrainState], validation_tracker.ValidationMetrics
]:
    def compute_metrics(
        train_state: training_virtual_sensor.TrainState,
    ) -> validation_tracker.ValidationMetrics:

        assert isinstance(eval_dataloader.dataset, data.KittiSingleStepDataset)

        # Eval mode
        train_state = jax_dataclasses.replace(train_state, train=False)

        # Sum losses across all batches
        total_loss: float = 0.0
        linear_vel_sse: float = 0.0
        angular_vel_sse: float = 0.0

        batch: data.KittiStructNormalized
        for batch in eval_dataloader:
            loss, cnn_outputs = train_state.compute_loss(
                batch,
                prng_key=jax.random.PRNGKey(0),  # Should not matter in eval mode
            )
            total_loss += loss
            linear_vel_sse += onp.mean((cnn_outputs[:, 0] - batch.linear_vel) ** 2)
            angular_vel_sse += onp.mean((cnn_outputs[:, 1] - batch.angular_vel) ** 2)

        # Unnormalize errors
        N = len(eval_dataloader)
        rmses = data.KittiStructNormalized(
            linear_vel=onp.sqrt(linear_vel_sse / N),
            angular_vel=onp.sqrt(angular_vel_sse / N),
        ).unnormalize(scale_only=True)

        # Return average
        metrics = {
            "average_loss": float(total_loss / N),
            "linear_vel_rmse": float(rmses.linear_vel),
            "angular_vel_rmse": float(rmses.angular_vel),
        }
        return metrics["average_loss"], metrics

    return compute_metrics
