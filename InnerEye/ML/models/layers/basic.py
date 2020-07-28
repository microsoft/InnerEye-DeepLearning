#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Callable, Iterable, Optional, Tuple, Union

import torch

from InnerEye.ML.config import PaddingMode
from InnerEye.ML.utils.layer_util import get_padding_from_kernel_size, initialise_layer_weights


class BasicLayer(torch.nn.Module):
    """
    A Basic Layer applies a 3D convolution and BatchNorm with the given channels, kernel_size, and dilation.
    The output of BatchNorm layer is passed through an activation function and its output is returned.
    :param channels: Number of input and output channels.
    :param kernel_size: Spatial support of convolution kernels
    :param stride: Kernel stride lenght for convolution op
    :param padding: Feature map padding after convolution op {"constant/zero", "no_padding"}. When it is set to
    "no_padding", no padding is applied. For "constant", feature-map tensor size is kept the same at the output by
    padding with zeros.
    :param dilation: Kernel dilation used in convolution layer
    :param use_bias: If set to True, a bias parameter will be added to the layer. Default is set to False as
    batch normalisation layer has an affine parameter which are used applied after the bias term is added.
    :param activation: Activation layer (e.g. nonlinearity) to be used after the convolution and batch norm operations.
    """

    def __init__(self,
                 channels: Tuple[int, int],
                 kernel_size: Union[Iterable[int], int],
                 stride: Union[Iterable[int], int] = 1,
                 dilation: Union[Iterable[int], int] = 1,
                 padding: PaddingMode = PaddingMode.NoPadding,
                 use_bias: bool = False,
                 activation: Optional[Callable] = torch.nn.ReLU,
                 use_batchnorm: bool = True):
        super(BasicLayer, self).__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = get_padding_from_kernel_size(padding, kernel_size, dilation)

        # Create layers
        self.conv1 = torch.nn.Conv3d(in_channels=self.channels[0],
                                     out_channels=self.channels[1],
                                     kernel_size=self.kernel_size,  # type: ignore
                                     stride=self.stride,  # type: ignore
                                     padding=self.pad,  # type: ignore
                                     dilation=self.dilation,  # type: ignore
                                     bias=use_bias)
        self.bn1 = torch.nn.BatchNorm3d(self.channels[1]) if use_batchnorm else None
        self.activation = activation(inplace=True) if activation else None

        # Initialise the trainable parameters
        self.apply(initialise_layer_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
