#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Optional

import torch

from InnerEye.ML.utils.image_util import get_class_weights
from InnerEye.ML.utils.supervised_criterion import SupervisedLearningCriterion


class SoftDiceLoss(SupervisedLearningCriterion):
    """
    Implementation of Soft-Dice Loss.
    Reference: Milletari, F., Navab, N., & Ahmadi, S. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric
               Medical Image Segmentation. In International Conference on 3D Vision (3DV).
    """

    def __init__(self,
                 eps: float = 1e-10,
                 apply_softmax: bool = True,
                 class_weight_power: Optional[float] = None):
        """
        :param eps: A small constant to smooth Sorensen-Dice Loss function. Additionally, it avoids division by zero.
        :param apply_softmax: If true, the input to the loss function will be first fed through a Softmax operation.
        If false, the input to the loss function will be used as is.
        :param class_weight_power: power to raise 1/C to, where C is the number of voxels in each class. Should be
        non-negative to help increase accuracy on small structures.
        """
        super().__init__()
        #: Small value to avoid division by zero errors.
        self.eps = eps
        self.apply_softmax = apply_softmax
        self.class_weight_power = class_weight_power

    def forward_minibatch(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Computes the forward pass of soft-dice loss. It assumes the output and target have Batch x Classes x ...
        dimensions, with the last dimensions being an arbitrary number of spatial dimensions.

        :param output: The output of the network.
        :param target: The target of the network.
        :return: The soft-dice loss.
        :raises ValueError: If the shape of the tensors is incorrect.
        :raises TypeError: If output or target are not torch.tensors.
        """
        # Check Types
        if not torch.is_tensor(output) or not torch.is_tensor(target):
            raise TypeError("Output and target must be torch.Tensors (type(output): {}, type(target): {})".
                            format(type(output), type(target)))

        # Check dimensions
        if len(output.shape) < 3:
            raise ValueError("The shape of the output and target must be at least 3, Batch x Class x ... "
                             "(output.shape: {})".format(output.shape))

        if output.shape != target.shape:
            raise ValueError("The output and target must have the same shape (output.shape: {}, target.shape: {})".
                             format(output.shape, target.shape))

        if self.apply_softmax:
            output = torch.nn.functional.softmax(output, dim=1)
        # Get the spatial dimensions; we'll sum numerator and denominator over these for efficiency.
        axes = list(range(2, len(output.shape)))

        # Eps is added to all products, avoiding division errors and problems
        # when a class does not exist in the current patch
        eps = torch.tensor([self.eps])
        if output.is_cuda:
            eps = eps.cuda(device=output.device)

        intersection = torch.sum(output * target + eps, axes)

        if self.class_weight_power is not None and self.class_weight_power != 0.0:
            # Multiply target by the class weight.
            class_weights = get_class_weights(target, self.class_weight_power)
            # noinspection PyTypeChecker
            intersection = torch.einsum("ij,j->ij", intersection, class_weights)

        output_sum_square = torch.sum(output * output + eps, axes)
        target_sum_square = torch.sum(target * target + eps, axes)
        sum_squares = output_sum_square + target_sum_square

        # Average per Batch and Class
        # noinspection PyTypeChecker
        return 1.0 - 2.0 * torch.mean(intersection / sum_squares)  # type: ignore
