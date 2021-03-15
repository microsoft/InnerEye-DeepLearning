#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch

from typing import Union, Tuple
from torch import nn

# To use weights from a pretrained model, we need eps to match
# https://github.com/google-research/big_transfer/blob/0bb237d6e34ab770b56502c90424d262e565a7f3/bit_pytorch/models.py#L30
eps = 1e-10


class WeightStandardizedConv2d(nn.Conv2d):
    """
    Weight Standardization
    https://arxiv.org/pdf/1903.10520.pdf
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         padding_mode=padding_mode)

    @staticmethod
    def standardize(weights: torch.Tensor) -> torch.Tensor:
        """
        Normalize weights on a per-kernel basis for all kernels.
        """
        assert weights.ndim == 4  # type: ignore
        mean = torch.mean(weights, dim=(1, 2, 3), keepdim=True)
        variance = torch.var(weights, dim=(1, 2, 3), keepdim=True, unbiased=False)
        standardized_weights = (weights - mean) / torch.sqrt(variance + eps)
        return standardized_weights

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        standardized_weights = WeightStandardizedConv2d.standardize(self.weight)
        return self._conv_forward(input, standardized_weights, bias=None)  # type: ignore
