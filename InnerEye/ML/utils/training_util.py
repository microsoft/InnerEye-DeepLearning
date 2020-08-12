from dataclasses import dataclass
from typing import List

import torch

from InnerEye.Common import common_util
from InnerEye.Common.metrics_dict import MetricsDict


@dataclass
class ModelForwardAndBackwardsOutputs:
    loss: float
    logits: torch.Tensor
    labels: torch.Tensor


@dataclass
class ModelOutputsAndMetricsForEpoch:
    metrics: MetricsDict
    model_outputs: List[ModelForwardAndBackwardsOutputs]
    is_train: bool

    def get_logits(self) -> torch.Tensor:
        return torch.cat([x.logits for x in self.model_outputs])

    def get_labels(self) -> torch.Tensor:
        return torch.cat([x.labels for x in self.model_outputs])


@dataclass(frozen=True)
class ModelTrainingResults:
    """
    Stores the results from training, with the results on training and validation data for each training epoch.
    """
    train_results_per_epoch: List[ModelOutputsAndMetricsForEpoch]
    val_results_per_epoch: List[ModelOutputsAndMetricsForEpoch]
    learning_rates_per_epoch: List[List[float]]

    def get_logits(self, training: bool) -> torch.Tensor:
        return torch.cat([x.get_logits() for x in
                          (self.train_results_per_epoch if training else self.val_results_per_epoch)])

    def get_labels(self, training: bool) -> torch.Tensor:
        return torch.cat([x.get_labels() for x in
                          (self.train_results_per_epoch if training else self.val_results_per_epoch)])

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)

        if len(self.train_results_per_epoch) != len(self.val_results_per_epoch) != len(self.learning_rates_per_epoch):
            raise Exception("train_results_per_epoch must be the same length as val_results_per_epoch found "
                            "and learning_rates_per_epoch, found: train_metrics_per_epoch={}, "
                            "val_metrics_per_epoch={}, learning_rates_per_epoch={}"
                            .format(len(self.train_results_per_epoch), len(self.val_results_per_epoch),
                                    len(self.learning_rates_per_epoch)))
