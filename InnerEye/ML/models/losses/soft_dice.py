#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Optional

import torch
from pl_bolts.models.self_supervised.simclr.simclr_module import SyncFunction

from InnerEye.ML.utils.image_util import get_class_weights, get_class_weights_from_counts
from InnerEye.ML.utils.supervised_criterion import SupervisedLearningCriterion


def synchronize_across_gpus(tensor: torch.Tensor) -> torch.Tensor:
    """
    Synchronizes a tensor across all GPUs, if distributed computation is enabled. The tensors from all GPUs are stacked
    up along the batch dimension (dim=0) using torch.cat. If no distributed setup is available, return the argument
    unchanged.
    :param tensor: The tensor that should be synchronized, of size [B, ...]
    :return: If torch.distributed is enabled, return a tensor of size [B * num_GPUs, ...]. If not distributed,
    return the argument of size [B, ...] unchanged.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        synced = SyncFunction.apply(tensor)
        return synced
    return tensor


def sum_sync_and_sum(t: torch.Tensor) -> torch.Tensor:
    """
    Sums a tensor across batch and spatial dimensions, synchronizes across all GPUs, and then sums again
    across the batch dimension (that is now the number of GPUs). Returns a tensor of size [Classes]
    :param t: A tensor of shape [Batch, Classes, Z, Y, X]
    """
    # All core statistics are summed across the spatial dimensions AND the batch dimension
    sum_across = [0, *range(2, len(t.shape))]
    # Sum across the batch dimension and all spatial dimension, then add a singleton batch dimension.
    # Returns a tensor of size [1, Classes]
    sum_spatial = t.sum(dim=sum_across).unsqueeze(0)
    # Synchronize across all GPUs. This will add along the batch dimension, giving a tensor of size [GPUs, Classes]
    synced = synchronize_across_gpus(sum_spatial)
    # Sum again across batch dimension that is now equal to number of GPUs, to get a tensor of size [Classes]
    return synced.sum(dim=0)


class SoftDiceLoss(SupervisedLearningCriterion):
    """
    Implementation of Soft-Dice Loss.
    Reference: Milletari, F., Navab, N., & Ahmadi, S. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric
               Medical Image Segmentation. In International Conference on 3D Vision (3DV).
    """

    def __init__(self,
                 eps: float = 1e-5,
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

        # Intersection has size [1, classes], all the spatial dimensions are summed across.
        intersection = sum_sync_and_sum(output * target)

        if self.class_weight_power is not None and self.class_weight_power != 0.0:
            # Count classes across all batches and GPUs
            class_counts = sum_sync_and_sum(target)
            class_weights = get_class_weights_from_counts(class_counts, class_weight_power=self.class_weight_power)
            intersection = intersection * class_weights

        # All these tensors also have shape [1, classes]
        output_sum_square = sum_sync_and_sum(output * output)
        target_sum_square = sum_sync_and_sum(target * target)

        # Average across all classes, including background
        synced = 1.0 - 2.0 * torch.mean((intersection + self.eps) / (output_sum_square + target_sum_square + self.eps))
        return synced  # type: ignore
