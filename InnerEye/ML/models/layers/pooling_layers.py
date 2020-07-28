#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import torch
import torch.nn.functional as TF

from InnerEye.ML.utils.layer_util import initialise_layer_weights


class AveragePooling(torch.nn.Module):
    """
    Global average pooling operation across all spatial dimensions (e.g. 2D and 3D image grids)
    """

    def forward(self, *input: Any, **kwargs: Any) -> Any:
        kernel_size = input[0].shape[2:]
        return TF.avg_pool3d(input[0], kernel_size=kernel_size)


class MaxPooling(torch.nn.Module):
    """
    Global max pooling operation across all spatial dimensions (e.g. 2D and 3D image grids)
    """

    def forward(self, *input: Any, **kwargs: Any) -> Any:
        kernel_size = input[0].shape[2:]
        return TF.max_pool3d(input[0], kernel_size=kernel_size)


class MixPooling(torch.nn.Module):
    """
    Compute a mixture of max pooling and average pooling.
    feature = a * avg_3d + (1-a) * max_3d given a in [0, 1]
    The mixing weight is a learnable parameter.
    """

    def __init__(self) -> None:
        super().__init__()
        # noinspection PyArgumentList
        self.mixing_weight = torch.nn.Parameter(torch.zeros(1))  # type: ignore

    def forward(self, *input: Any, **kwargs: Any) -> Any:
        """
        :param input: batch of size [B, C, Z, X, Y]
        """
        kernel_size = input[0].shape[2:]
        f_avg = torch.nn.functional.avg_pool3d(input[0], kernel_size)  # B, C, 1, 1, 1
        f_max = torch.nn.functional.max_pool3d(input[0], kernel_size)  # B, C, 1, 1, 1
        return TF.sigmoid(self.mixing_weight) * f_avg + (1 - TF.sigmoid(self.mixing_weight)) * f_max  # type: ignore


class Gated3dPoolingLayer(torch.nn.Module):
    """
    Gated pooling. Flatten each volume x [1, ZYX], feed
    through a one layer NN yield one weight per image.
    This weight is used as the mixing proportion for
    max_pooling features and average pooling features similar
    to what is done in MixPooling.
    """

    def __init__(self, in_features: int) -> None:
        """
        :param in_features: should be the size of the flatten volume X*Y*Z
        """
        super().__init__()
        self.in_features = in_features

        # Create layers
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=1),
            torch.nn.Sigmoid())

        # Initialise the trainable parameters
        self.apply(initialise_layer_weights)

    def forward(self, *input: Any, **kwargs: Any) -> torch.Tensor:
        """
        :param input: batch of size [B, C, Z, X, Y
        """
        item = input[0]
        channels = item.shape[1]
        kernel_size = item.shape[2:]
        # Common gating map across all channels.
        gating_weights = self.gate(item.reshape(-1, channels, self.in_features))
        f_avg = torch.nn.functional.avg_pool3d(item, kernel_size)  # B, C, 1
        f_max = torch.nn.functional.max_pool3d(item, kernel_size)  # B, C, 1
        gating_weights = gating_weights.reshape_as(f_avg)
        # noinspection PyTypeChecker
        final = gating_weights * f_avg + (1 - gating_weights) * f_max  # type: ignore
        return final


class ZAdaptive3dAvgLayer(torch.nn.Module):
    """
    Performs 3D average pooling with custom weighting along the
    Z dimension. In short: extract the 2d average for each B-scan.
    Learn a weighting for averaging these features over all B-Scans.
    """

    def __init__(self, in_features: int) -> None:
        """
        :param in_features: number of B-scan
        """
        super().__init__()
        self.in_features = in_features
        # Create layers
        # noinspection PyArgumentList
        self.scan_weight = torch.nn.Parameter(torch.zeros(in_features, 1))  # type: ignore

        # Initialise the trainable parameters
        self.apply(initialise_layer_weights)

    def forward(self, *input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        :param input: batch of size [B, C, Z, X, Y]
        """
        item = input[0]
        B, C, Z, Y, X = item.shape
        # Average first in 2d - one feature per B-scan
        f_avg_2d = torch.nn.functional.avg_pool3d(item, [1, Y, X])  # B, C, Z, 1, 1
        # Give a custom weight to each z slice
        normalized_weight = TF.softmax(self.scan_weight, dim=0)
        custom_3d_avg = f_avg_2d.reshape(B, C, Z) @ normalized_weight
        return custom_3d_avg.reshape((B, C, 1, 1, 1))
