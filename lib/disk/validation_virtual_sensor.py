import torch

from .. import validation_tracker
from . import data, training_virtual_sensor


def compute_metrics(
    eval_dataloader: torch.utils.data.DataLoader[data.DiskStructNormalized],
    train_state: training_virtual_sensor.TrainState,
) -> validation_tracker.ValidationMetrics:

    batch: data.DiskStructNormalized
    total_loss: float = 0.0
    for batch in eval_dataloader:
        total_loss += train_state.compute_loss(batch)[0]
    average_loss = total_loss / len(eval_dataloader)

    return average_loss, {"average_loss": average_loss}
