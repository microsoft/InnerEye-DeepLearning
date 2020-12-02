#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch.cuda.amp import GradScaler
# noinspection PyUnresolvedReferences
from torch.optim import Optimizer  # type: ignore

from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.models.architectures.base_model import DeviceAwareModule
from InnerEye.ML.utils import ml_util


class SegmentationForwardPass:
    """
    Pipeline that handles model forward pass operations, including loss computation and segmentation creation.
    """

    @dataclass(frozen=True)
    class Result:
        """
        Results from a single model forward pass.
        """
        posteriors: np.ndarray  # posteriors for each patch in shape: Batches x Classes x Z x Y x X
        segmentations: np.ndarray  # multi-label segmentations for each patch in shape: Batches x Z x Y x X
        loss: Optional[float]  # the criterion loss for the posteriors

        def __post_init__(self) -> None:
            ml_util.check_size_matches(arg1=self.posteriors, arg2=self.segmentations,
                                       dim1=5, dim2=4,
                                       matching_dimensions=[0, -1, -2, -3])

    def __init__(self,
                 model: DeviceAwareModule,
                 model_config: SegmentationModelBase,
                 batch_size: int,
                 optimizer: Optional[Optimizer] = None,
                 in_training_mode: Optional[bool] = False,
                 criterion: Optional[Callable] = None,
                 gradient_scaler: Optional[GradScaler] = None):
        super().__init__()
        self.model = model
        self.config = model_config
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.detect_anomaly = model_config.detect_anomaly
        self.criterion_fn = criterion
        self.gradient_scaler = gradient_scaler
        if in_training_mode and (optimizer is None or criterion is None):
            raise ValueError("When running in training mode, an optimizer and criterion must be provided.")
        self.in_training_mode = in_training_mode


def single_optimizer_step(loss: torch.Tensor,
                          optimizer: Optimizer,
                          gradient_scaler: Optional[GradScaler]) -> None:
    """
    Wrapper function to make the optimizer take a single step, given a loss tensor with gradients.
    This will update the loss tensor with auto scaling for mixed
    precision training and anomaly detection to identify NaN values in gradient updates.
    :param loss: Torch tensor representing the training loss.
    :param optimizer: The torch optimizer.
    :param gradient_scaler: The Torch gradient scaler object to handle mixed precision training.
    """
    # zero the gradients for the next optimization step as these
    # will be taken from the loss gradients
    optimizer.zero_grad()
    # compute the gradients w.r.t to the optimization variables and update the optimizer_type
    if gradient_scaler:
        # Scales the loss, and calls backward() to create scaled gradients
        gradient_scaler.scale(loss).backward()
        # Unscales gradients and calls or skips optimizer.step()
        gradient_scaler.step(optimizer)
        # Updates the scale for next iteration
        gradient_scaler.update()
    else:
        loss.backward()
        optimizer.step(closure=None)
