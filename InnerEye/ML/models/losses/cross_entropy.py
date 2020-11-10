#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn.functional as F

from InnerEye.ML.utils.image_util import get_class_weights
from InnerEye.ML.utils.supervised_criterion import SupervisedLearningCriterion


class ReductionType(Enum):
    """
    Supported types of pixel reduction techniques
    """
    MEAN = "mean"
    SUM = "sum"
    NONE = "none"

    @classmethod
    def has_value(cls, value: str) -> bool:
        available_values = [item.value for item in cls]
        return value in available_values


class CrossEntropyLoss(SupervisedLearningCriterion):

    def __init__(self,
                 class_weight_power: Optional[float] = None,
                 smoothing_eps: float = 0.0,
                 focal_loss_gamma: Optional[float] = None,
                 ignore_index: int = -100):
        super().__init__(smoothing_eps)
        """
        Multi-class cross entropy loss.
        :param class_weight_power: if 1.0, weights the cross-entropy term for each class equally.
                                   Class weights are inversely proportional to the number
                                   of pixels belonging to each class, raised to class_weight_power
        :param focal_loss_gamma: Gamma term used in focal loss to weight negative log-likelihood term:
                                 https://arxiv.org/pdf/1708.02002.pdf equation(4-5).
                                 When gamma equals to zero, it is equivalent to standard
                                 CE with no class balancing. (Gamma >= 0.0)
        :param ignore_index: Specifies a target value that is ignored and does not contribute
                             to the input gradient
        """
        if class_weight_power is not None and class_weight_power < 0.0:
            raise ValueError("Class-weight power should be equal to or larger than zero")
        if isinstance(focal_loss_gamma, float):
            if not (focal_loss_gamma >= 0.0):
                raise ValueError("Gamma should be equal to or larger than zero")

        self.class_weight_power = class_weight_power
        self.focal_loss_gamma = focal_loss_gamma
        self.ignore_index = ignore_index
        self.eps = 1e-6

    @staticmethod
    def _get_class_weights(target_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Returns class weights inversely proportional to the number of pixels in each class"""

        class_weight = torch.zeros(num_classes, dtype=torch.float32)
        if target_labels.is_cuda:
            class_weight = class_weight.cuda()

        # Check which classes are available
        class_ids, class_counts = torch.unique(target_labels, sorted=False, return_counts=True)  # type: ignore
        class_weight[class_ids] = 1.0 / class_counts.float()

        return class_weight

    @torch.no_grad()
    def get_focal_loss_pixel_weights(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes weights for each pixel/sample inversely proportional to the posterior likelihood.
        :param logits: Logits tensor.
        :param target: Target label tensor in one-hot encoding.
        """
        if not (torch.sum(target == 1.0) == (target.nelement() / target.shape[1])):
            raise ValueError("Focal loss is supported only for one-hot encoded targets")

        posteriors = torch.nn.functional.softmax(logits, dim=1)
        # noinspection PyUnresolvedReferences,PyTypeChecker
        pixel_weights: torch.Tensor = (1 - posteriors + self.eps).pow(self.focal_loss_gamma)  # type: ignore
        pixel_weights = torch.sum(pixel_weights * target, dim=1)

        # Normalise the pixel weights
        scaling = pixel_weights.nelement() / (torch.sum(pixel_weights) + self.eps) 

        return pixel_weights * scaling

    @staticmethod
    def _verify_inputs(logits: torch.Tensor, target: torch.Tensor) -> None:
        """Function that verifies input data types and dimensions"""

        # Check input data types
        if not isinstance(logits, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("Logits and target must be torch.Tensors")
        if not logits.is_floating_point():
            raise TypeError("Logits must be a float tensor")

        # Check input tensor dimensions
        if len(logits.shape) < 2:
            raise ValueError("The shape of logits must be at least 2, Batch x Class x ... "
                             "(logits.shape: {})".format(logits.shape))
        if logits.shape[0] != target.shape[0]:
            raise ValueError("The logits and target must have the same batch size (logits: {}, target: {})".
                             format(logits.shape, target.shape))
        if len(logits.shape) > 2:
            if logits.shape[1:] != target.shape[1:]:
                raise ValueError("The logits and target must have same shape (logits: {}, target: {})".
                                 format(logits.shape, target.shape))

    def forward_minibatch(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Wrapper for multi-class cross entropy function implemented in PyTorch.
        The implementation supports tensors with arbitrary spatial dimension.
        Input logits are normalised internally in `F.cross_entropy` function.
        :param output: Class logits (unnormalised), e.g. in 3D : BxCxWxHxD  or in 1D BxC
        :param target: Target labels encoded in one-hot representation, e.g. in 3D BxCxWxHxD or in 1D BxC
        """

        # Convert one hot-encoded target to class indices
        class_weight = None

        # Check input tensors
        self._verify_inputs(output, target)

        # Determine class weights for unbalanced datasets
        if self.class_weight_power is not None and self.class_weight_power != 0.0:
            class_weight = get_class_weights(target, class_weight_power=self.class_weight_power) 

        # Compute negative log-likelihood
        log_prob = F.log_softmax(output, dim=1)
        if self.smoothing_eps > 0.0:
            loss = -1.0 * log_prob * target
            if class_weight is not None:
                loss = torch.einsum('bc...,c->b...', loss, class_weight)
        else:
            loss = F.nll_loss(log_prob, torch.argmax(target, dim=1), weight=class_weight, reduction='none')

        # If focal loss is specified, apply pixel weighting
        if self.focal_loss_gamma is not None:
            pixel_weights = self.get_focal_loss_pixel_weights(output, target)
            loss = loss * pixel_weights

        return torch.mean(loss)
