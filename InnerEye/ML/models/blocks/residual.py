#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import List

import torch

from InnerEye.ML.models.layers.basic import BasicLayer


class ResidualBlock(torch.nn.Module):
    """
    A block of several convolution layers with a residual connection around them. If the channels change, then the
    number of channels must be synchronized with the expected input number of channels of the layer this residual
    is passed into. For instance, if we have an instance where
    (1) L1 (10) -> (10) L2 (20) -> (30) L3 (40) , with a residual connection L1 -> L3 then
    as L1 and L2 output only 10 + 20 = 30 channels, in which case we use another convnet that takes the feature
    responses of L1 as input and uses 30 kernels to output 30 channels that can then be passed into L3.
    """

    # noinspection PyTypeChecker
    def __init__(self, layers: List[torch.nn.Module], channels: List[int], kernel_size: int, dilations: List[int]):
        super().__init__()

        if len(channels) != len(layers) + 1:
            raise ValueError("The number of channels for n layers in a ResidualBlock must be n + 1 (channels: {},"
                             "layers: {})".format(channels, layers))

        self.kernel_size = kernel_size

        # Create layers
        self.layers: torch.nn.ModuleList = torch.nn.ModuleList()
        for i, layer in enumerate(layers):
            with_relu_cond = (i == 0 and len(layers) > 1)
            activation = torch.nn.ReLU if with_relu_cond else None
            if layer == BasicLayer:
                self.layers.append(BasicLayer(channels[i:(i + 2)],  # type: ignore
                                              kernel_size,
                                              dilation=dilations[i],
                                              use_bias=True,
                                              activation=activation))
            else:
                raise ValueError("Unknown layer found")

        if channels[0] == channels[2]:
            self.conv = None
        else:
            self.conv = BasicLayer(channels[0:-1:len(channels) - 2], kernel_size=1, dilation=1,  # type: ignore
                                   use_bias=True, activation=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Copy input
        residual = torch.tensor(x)

        for layer in self.layers:  # type: ignore
            x = layer(x)

        # The spatial size can be different because of unpadded convolutions, so we crop the difference
        shape = list(x.shape[2:])
        shape = [(residual.shape[i + 2] - s) // 2 for i, s in enumerate(shape)]
        residual = residual[:, :, shape[0]:-shape[0], shape[1]:-shape[1], shape[2]:-shape[2]]

        if self.conv is not None:
            residual = self.conv(residual)

        x += residual
        x = torch.nn.functional.relu(x, inplace=True)
        return x
