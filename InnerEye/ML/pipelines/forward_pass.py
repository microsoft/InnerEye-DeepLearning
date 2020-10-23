#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, autograd
from torch.cuda.amp import GradScaler
# noinspection PyUnresolvedReferences
from torch.optim import Optimizer  # type: ignore

from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.models.architectures.base_model import DeviceAwareModule
from InnerEye.ML.models.parallel.data_parallel import execute_within_autocast_if_needed
from InnerEye.ML.utils import image_util, ml_util


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

    def forward_pass_patches(self, patches: torch.Tensor,
                             labels: Optional[torch.Tensor] = None,
                             mask: Optional[torch.Tensor] = None,
                             device: Optional[torch.device] = None,
                             rank: Optional[int] = None) -> \
            SegmentationForwardPass.Result:
        """
        Wrapper function to handle model forward pass, including updating of the optimizer_type with loss gradients
        if required.
        :param patches: Images patches to be passed to the model in format Batches x Channels x Z x Y x X.
        :param labels: Labels for image patches to be used for loss computation: Batches x Classes x Z x Y x X
        :param mask: optional mask patches channel in shape Batches x Z x Y x X  to be applied to the predictions.
        :param rank: The global rank of the current process.
        :param device: The Torch device to allocate to.
        """

        # check that the patches are as expected w.r.t to the configuration
        if patches is None:
            raise Exception("Patches for forward pass cannot be None.")

        if not torch.is_tensor(patches):
            raise Exception("Input patches must be a Torch Tensor, found: {}."
                            .format(type(patches)))

        if len(patches.shape) != 5:
            raise Exception("Input expected to be 5 dimensions: Batches x Channels x Z x Y x Z, found dimensions: {}"
                            .format(len(patches.shape)))

        if patches.shape[1] != self.config.number_of_image_channels:
            raise Exception("Input expected to have {} image channels, found: {}"
                            .format(self.config.number_of_image_channels, patches.shape[1])
                            .format(len(patches.shape)))

        # mask validation if required
        if mask is not None:
            if not torch.is_tensor(mask):
                raise Exception(f"Mask must be a Torch Tensor, found: {type(mask)}.")

        # check that the patches are as expected w.r.t to the configuration
        if 0 < self.batch_size < patches.shape[0]:
            raise Exception(f"Expected batch size to be <= {self.batch_size}, found {patches.shape[0]}")

        # handle model modes
        if self.in_training_mode:
            self.model.train()
            result = self._forward_pass_with_anomaly_detection(patches=patches, mask=mask,
                                                               labels=labels, device=device)
        else:
            self.model.eval()
            # turn off autograd for memory optimizations
            with torch.no_grad():
                result = self._forward_pass_with_anomaly_detection(patches=patches, mask=mask,
                                                                   labels=labels, device=device)
            self.model.train()
        return result

    def _forward_pass_with_anomaly_detection(self,
                                             patches: torch.Tensor,
                                             mask: torch.Tensor = None,
                                             labels: torch.Tensor = None,
                                             device: torch.device = None) -> SegmentationForwardPass.Result:
        if self.detect_anomaly:
            with autograd.detect_anomaly():
                result = self._forward_pass(patches, mask, labels, device=device)
            if result.loss is not None and (math.isnan(result.loss) or math.isinf(result.loss)):
                raise RuntimeError(f"The loss computation returned {result.loss}")
            return result
        return self._forward_pass(patches, mask, labels, device=device)

    def _compute_loss(self, patches: Tensor, labels: Optional[Tensor], device: Optional[torch.device] = None
                      ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Do a forward pass on the model with the patches as input. If labels are provided, compute the loss.
        Return a tuple of (logits, loss).
        """

        def compute() -> Tuple[Any, Optional[Tensor]]:
            loss: Optional[torch.Tensor] = None
            logits = self.model(patches)
            # If labels *is* None, loss will also be None, which will stop the code below working (and
            # currently correctly triggers mypy errors).
            if labels is not None and self.criterion_fn is not None:
                loss = self.criterion_fn(logits, labels, device=device)
            return logits, loss

        return execute_within_autocast_if_needed(func=compute, use_autocast=True if self.gradient_scaler else False)

    def _forward_pass(self,
                      patches: torch.Tensor,
                      mask: torch.Tensor = None,
                      labels: torch.Tensor = None,
                      device: Optional[torch.device] = None) -> SegmentationForwardPass.Result:

        # ensure that we always have float tensors as the model is defined over floats
        # and transfer the tensors to the GPU if possible before the forward pass
        patches = self.config.get_gpu_tensor_if_possible(patches, device)
        if mask is not None:
            mask = self.config.get_gpu_tensor_if_possible(mask, device=device)

        logits, loss = self._compute_loss(patches, labels, device=device)

        if self.in_training_mode:
            if loss is None:
                raise ValueError("When running training, the labels must be present for loss computation.")
            assert self.optimizer is not None  # for mypy
            single_optimizer_step(loss, self.optimizer, self.gradient_scaler)

        # Aggregate data parallel logits if multiple hardware are used in forward pass
        if isinstance(logits, list) and not self.config.use_ddp:
            # When using multiple GPUs, logits is a list of tensors. Gather will concatenate them
            # across the first dimension, and move them to GPU0.
            logits = torch.nn.parallel.gather(logits, target_device=0)

        # apply Softmax on dimension 1 (Class) to map model output into a posterior probability distribution [0,1]
        posteriors = torch.nn.functional.softmax(logits, dim=1)

        # apply mask if required
        if not self.in_training_mode and mask is not None:
            posteriors = image_util.apply_mask_to_posteriors(posteriors=posteriors, mask=mask)

        # post process posteriors to compute result
        segmentations = image_util.posteriors_to_segmentation(posteriors=posteriors).data.cpu().numpy()
        posteriors = posteriors.data.cpu().numpy()

        return SegmentationForwardPass.Result(posteriors=posteriors, segmentations=segmentations,
                                              loss=loss.item() if loss is not None else None)


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
