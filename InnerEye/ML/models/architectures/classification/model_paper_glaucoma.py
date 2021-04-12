#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import List

import torch

from InnerEye.ML.config import PaddingMode
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.models.layers.basic import BasicLayer
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule


class ModelFromPaper(DeviceAwareModule[ScalarItem, torch.Tensor]):
    def get_input_tensors(self, item: ScalarItem) -> List[torch.Tensor]:
        return [item.images]

    def __init__(self) -> None:
        super().__init__()

        num_classes = 1
        num_conv_layers = 5
        kernel_size = [7, 5, 3, 3, 3]
        channels = [1, 32, 32, 32, 32, 32]
        stride = [2, 1, 1, 1, 1]
        padding = [PaddingMode.Zero] * num_conv_layers

        _conv_layers = []

        # Convolution Layers
        for ii in range(num_conv_layers):
            _conv_layers.append(
                BasicLayer(channels=(channels[ii], channels[ii + 1]),
                           kernel_size=kernel_size[ii],
                           stride=stride[ii],
                           padding=padding[ii],
                           activation=torch.nn.ReLU))

        # Pooling and dense layers
        self.encoder = torch.nn.Sequential(*_conv_layers)
        self.aggregation_layer = torch.nn.functional.avg_pool3d
        self.dense_layer = torch.nn.Linear(in_features=channels[-1],
                                           out_features=num_classes,
                                           bias=True)
        self.conv_in_3d = True
        self.last_encoder_layer = ["encoder", "4", "activation"]

    def get_last_encoder_layer_names(self) -> List[str]:
        return self.last_encoder_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.encoder(x)
        x = self.aggregation_layer(input=x, kernel_size=x.shape[2:])
        x = self.dense_layer(x.view(-1, x.shape[1]))
        return x
