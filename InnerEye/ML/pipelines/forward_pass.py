#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import math
import numpy as np
import torch
from apex import amp
from torch import autograd
# noinspection PyUnresolvedReferences
from torch.optim import Optimizer  # type: ignore

from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.models.architectures.base_model import DeviceAwareModule
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
                 criterion: Optional[Callable] = None):
        super().__init__()
        self.model = model
        self.config = model_config
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.detect_anomaly = model_config.detect_anomaly
        self.criterion_fn = criterion
        if in_training_mode and (optimizer is None or criterion is None):
            raise ValueError("When running in training mode, an optimizer and criterion must be provided.")
        self.in_training_mode = in_training_mode

    def forward_pass_patches(self, patches: torch.Tensor,
                             labels: Optional[torch.Tensor] = None,
                             mask: Optional[torch.Tensor] = None) -> \
            SegmentationForwardPass.Result:
        """
        Wrapper function to handle model forward pass, including updating of the optimizer_type with loss gradients
        if required.
        :param patches: Images patches to be passed to the model in format Batches x Channels x Z x Y x X.
        :param labels: Labels for image patches to be used for loss computation: Batches x Classes x Z x Y x X
        :param mask: optional mask patches channel in shape Batches x Z x Y x X  to be applied to the predictions.
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
            result = self._forward_pass_with_anomaly_detection(patches=patches, mask=mask, labels=labels)
        else:
            self.model.eval()
            # turn off autograd for memory optimizations
            with torch.no_grad():
                result = self._forward_pass_with_anomaly_detection(patches=patches, mask=mask, labels=labels)
            self.model.train()
        return result

    def _forward_pass_with_anomaly_detection(self,
                                             patches: torch.Tensor,
                                             mask: torch.Tensor = None,
                                             labels: torch.Tensor = None) -> SegmentationForwardPass.Result:
        if self.detect_anomaly:
            with autograd.detect_anomaly():
                result = self._forward_pass(patches, mask, labels)
            if result.loss is not None and (math.isnan(result.loss) or math.isinf(result.loss)):
                raise RuntimeError(f"The loss computation returned {result.loss}")
        return self._forward_pass(patches, mask, labels)

    def _forward_pass(self,
                      patches: torch.Tensor,
                      mask: torch.Tensor = None,
                      labels: torch.Tensor = None) -> SegmentationForwardPass.Result:

        # ensure that we always have float tensors as the model is defined over floats
        # and transfer the tensors to the GPU if possible before the forward pass
        patches = self.config.get_gpu_tensor_if_possible(patches)
        if mask is not None:
            mask = self.config.get_gpu_tensor_if_possible(mask)
        loss: Optional[torch.Tensor] = None

        # do a forward pass on the model with the patches as input
        # this will give outputs in format: Batches x Classes x Z x Y x X
        logits = self.model(patches)
        # If labels *is* None, loss will also be None, which will stop the code below working (and
        # currently correctly triggers mypy errors).
        if labels is not None and self.criterion_fn is not None:
            loss = self.criterion_fn(logits, labels)
        if self.in_training_mode:
            if loss is None:
                raise ValueError("When running training, the labels must be present for loss computation.")
            assert self.optimizer is not None  # for mypy
            single_optimizer_step(self.config, loss, self.optimizer)

        # Aggregate data parallel logits if multiple hardware are used in forward pass
        if isinstance(logits, list):
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


def single_optimizer_step(config: DeepLearningConfig,
                          loss: torch.Tensor,
                          optimizer: Optimizer) -> None:
    """
    Wrapper function to make the optimizer take a single step, given a loss tensor with gradients.
    This will update the loss tensor with auto scaling for mixed
    precision training and anomaly detection to identify NaN values in gradient updates.
    :param loss: Torch tensor representing the training loss.
    :param config: The object containing all relevant settings like use of mixed precision and anomaly detection.
    :param optimizer: The torch optimizer.
    """
    # zero the gradients for the next optimization step as these
    # will be taken from the loss gradients
    optimizer.zero_grad()
    # compute the gradients w.r.t to the optimization variables and update the optimizer_type
    if config.use_mixed_precision and config.use_gpu:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    # perform next optimization step
    optimizer.step(closure=None)
