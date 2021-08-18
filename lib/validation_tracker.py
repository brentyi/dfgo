"""Helper for computing and logging validation statistics, and only recording
checkpoints when they improve."""

import dataclasses
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

from . import experiment_files, train_state_protocol

Pytree = Any


TrainState = TypeVar("TrainState", bound=train_state_protocol.TrainStateProtocol)

# Validation metrics are: a primary metric that we want the best of, and any auxilliary
# metrics to log etc
ValidationMetrics = Tuple[float, Dict[str, float]]


# @dataclasses.dataclass(frozen=True) # <== TODO: when mypy v0.920 is released
@dataclasses.dataclass
class ValidationTracker(Generic[TrainState]):
    """Helper for tracking+logging validation statistics."""

    name: str
    experiment: experiment_files.ExperimentFiles
    compute_metrics: Callable[[TrainState], ValidationMetrics]

    lowest_metric: Optional[float] = dataclasses.field(default=None)
    best_step: Optional[float] = dataclasses.field(default=None)

    def validate_log_and_checkpoint_if_best(
        self, train_state: TrainState
    ) -> "ValidationTracker":
        """Compute metrics, log them, and checkpoint if best. :)

        Returns a ValidationTracker object with the `lowest_metric` and `best_step`
        fields updated*.

        *this is a little bit silly, but we have a lot of immutable dataclasses... might
        as well make this one too
        """
        primary_metric, metric_dict = self.compute_metrics(train_state)

        # Log named metrics
        for key, value in metric_dict.items():
            self.experiment.summary_writer.scalar(
                f"{self.name}/{key}", value, step=train_state.steps
            )

        # Update lowest metric
        if self.lowest_metric is None or primary_metric < self.lowest_metric:
            # Save learnable parameters
            # Alternatively, we could save the entire training state, which would
            # include optimizer parameters, step counts, etc
            self.experiment.save_checkpoint(
                target=train_state.learnable_params,
                step=train_state.steps,
                prefix=f"best_{self.name}_params_",
            )

            # Write metrics. float/int casts to avoid saving JAX or numpy arrays.
            self.experiment.write_metadata(
                f"best_{self.name}_metrics",
                {k: float(v) for k, v in metric_dict.items()},
            )
            self.experiment.write_metadata(
                f"best_{self.name}_step", int(train_state.steps)
            )

            return dataclasses.replace(
                self, lowest_metric=primary_metric, best_step=int(train_state.steps)
            )

        return self
