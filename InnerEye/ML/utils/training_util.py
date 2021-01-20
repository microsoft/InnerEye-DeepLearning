#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Any, List, Union

import torch

from InnerEye.Common.type_annotations import DictStrFloat


@dataclass(frozen=True)
class ModelTrainingResults:
    """
    Stores the results from training, with the results on training and validation data for each training epoch.
    """
    train_results_per_epoch: List[DictStrFloat]
    val_results_per_epoch: List[DictStrFloat]
    train_diagnostics: Any
    val_diagnostics: Any
    optimal_temperature_scale_values_per_checkpoint_epoch: List[float] = field(default_factory=list)

    def get_metric(self, is_training: bool, metric_type: str) -> List[float]:
        """
        Gets a scalar metric out of either the list of training or the list of validation results. This returns
        that value that a specific metric attains in all of the epochs.
        :param is_training: If True, read metrics from the `train_results_per_epoch` field, if False read from the
        `val_results_per_epoch` field.
        :param metric_type: The metric to extract.
        :return: A list of floating point numbers, with one entry per entry in the the training or validation results.
        """
        metrics = self.train_results_per_epoch if is_training else self.val_results_per_epoch
        return [m[metric_type] for m in metrics]

    def get_training_metric(self, metric_type: str) -> List[float]:
        """
        Gets a scalar metric from the list of training results. This returns
        the value that a specific metric attains in all of the epochs.
        :param metric_type: The metric to extract.
        :return: A list of floating point numbers, with one entry per entry in the the training results.
        """
        return self.get_metric(is_training=True, metric_type=metric_type)

    def get_validation_metric(self, metric_type: str) -> List[float]:
        """
        Gets a scalar metric from the list of validation results. This returns
        the value that a specific metric attains in all of the epochs.
        :param metric_type: The metric to extract.
        :return: A list of floating point numbers, with one entry per entry in the the validation results.
        """
        return self.get_metric(is_training=False, metric_type=metric_type)


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
