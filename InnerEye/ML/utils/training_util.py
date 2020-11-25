#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import torch

from InnerEye.Common import common_util
from InnerEye.Common.metrics_dict import MetricType, MetricsDict


@dataclass
class ModelForwardAndBackwardsOutputs:
    """
    Dataclass to store results from a single model forward and backwards pass.
    """
    loss: float
    logits: Union[torch.Tensor, np.ndarray]
    labels: Union[torch.Tensor, np.ndarray]


@dataclass
class ModelOutputsAndMetricsForEpoch:
    """
    Dataclass to store results from a single epoch.
    """
    metrics: MetricsDict
    model_outputs: List[ModelForwardAndBackwardsOutputs]
    is_train: bool

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self, ignore=["metrics"])

    def get_logits(self) -> torch.Tensor:
        """
        Get concatenated logits from each training/validation batch in this epoch.
        """
        return torch.cat([x.logits for x in self.model_outputs])

    def get_labels(self) -> torch.Tensor:
        """
        Get concatenated labels from each training/validation batch in this epoch.
        """
        return torch.cat([x.labels for x in self.model_outputs])


@dataclass(frozen=True)
class ModelTrainingResults:
    """
    Stores the results from training, with the results on training and validation data for each training epoch.
    """
    train_results_per_epoch: List[MetricsDict]
    val_results_per_epoch: List[MetricsDict]
    optimal_temperature_scale_values_per_checkpoint_epoch: List[float] = field(default_factory=list)

    def get_metric(self, is_training: bool, metric_type: MetricType) -> List[float]:
        """
        Gets a scalar metric out of either the list of training or the list of valiation results.
        :param is_training: If True, read metrics from the `train_results_per_epoch` field, if False read from the
        `val_results_per_epoch` field.
        :param metric_type: The metric to extract.
        :return: A list of floating point numbers, with one entry per entry in the the training or validation results.
        """
        metrics = self.train_results_per_epoch if is_training else self.val_results_per_epoch
        return [m.get_single_metric(metric_type) for m in metrics]


def gather_tensor(tensor: Union[torch.Tensor, List[torch.Tensor]],
                  target_device: int = 0) -> torch.Tensor:
    """
    When using multiple GPUs, logits is a list of tensors. Concatenate them
    across the first dimension, and move them to the provided target_device.
    :param tensor: tensor to gather
    :param target_device: device to move the tensors to
    :return:
    """
    if isinstance(tensor, list):
        return torch.nn.parallel.gather(tensor, target_device=target_device)
    else:
        return tensor
