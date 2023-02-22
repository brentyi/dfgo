from typing import Callable

import jax
import numpy as onp
import torch
from jax import numpy as jnp

from .. import utils, validation_tracker
from . import data, data_loading, training_lstm


def make_compute_metrics(
    dataset_fold: int,
) -> Callable[[training_lstm.TrainState], validation_tracker.ValidationMetrics]:
    dataloader = torch.utils.data.DataLoader(
        data_loading.DiskSubsequenceDataset(
            train=False,
            fold=dataset_fold,
            subsequence_length=20,  # This is the full length
        ),
        batch_size=4,
        collate_fn=utils.collate_fn,
        shuffle=True,
        drop_last=True,
    )

    @jax.jit
    def compute_mse(
        train_state: training_lstm.TrainState, batch: data.DiskStructNormalized
    ) -> jnp.ndarray:
        positions = train_state.lstm.apply(train_state.learnable_params, batch)
        assert positions.shape == batch.position.shape
        assert len(positions.shape) == 3  # (N, T, 2)
        return jnp.mean((positions - batch.position) ** 2, axis=(0, 1))

    def compute_metrics(
        train_state: training_lstm.TrainState,
    ) -> validation_tracker.ValidationMetrics:
        mse = onp.zeros(2)

        for batch in dataloader:
            mse += compute_mse(train_state, batch)

        rmse = onp.linalg.norm(
            # type ignore because we pass in a numpy array instead of a jax one
            data.DiskStructNormalized(position=onp.sqrt(mse / len(dataloader)))  # type: ignore
            .unnormalize(scale_only=True)
            .position
        )

        return rmse, {"rmse": rmse}

    return compute_metrics
