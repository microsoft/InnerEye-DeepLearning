#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import torch

from InnerEye.Common import common_util
from InnerEye.Common.metrics_dict import MetricsDict


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
    train_results_per_epoch: List[ModelOutputsAndMetricsForEpoch]
    val_results_per_epoch: List[ModelOutputsAndMetricsForEpoch]
    learning_rates_per_epoch: List[List[float]]
    optimal_temperature_scale_values_per_checkpoint_epoch: List[float] = field(default_factory=list)

    def get_logits(self, is_training: bool) -> torch.Tensor:
        """
        Get concatenated logits from each training/validation epoch.
        :param is_training: return logits from model on the training set if True otherwise form the validation set
        :return: concatenated logits from each training/validation epoch.
        """
        return torch.cat([x.get_logits() for x in
                          (self.train_results_per_epoch if is_training else self.val_results_per_epoch)])

    def get_labels(self, is_training: bool) -> torch.Tensor:
        """
        Get concatenated labels used for each training/validating epoch.
        :param is_training: return training set labels if True otherwise return validation set labels
        :return: concatenated labels used for each training/validating epoch.
        """
        return torch.cat([x.get_labels() for x in
                          (self.train_results_per_epoch if is_training else self.val_results_per_epoch)])

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)

        if len(self.train_results_per_epoch) != len(self.val_results_per_epoch) != len(self.learning_rates_per_epoch):
            raise Exception("train_results_per_epoch must be the same length as val_results_per_epoch found "
                            "and learning_rates_per_epoch, found: train_metrics_per_epoch={}, "
                            "val_metrics_per_epoch={}, learning_rates_per_epoch={}"
                            .format(len(self.train_results_per_epoch), len(self.val_results_per_epoch),
                                    len(self.learning_rates_per_epoch)))


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
