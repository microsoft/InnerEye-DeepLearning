#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch
from typing import List

from InnerEye.Common.type_annotations import TupleInt2
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.models.architectures.base_model import DeviceAwareModule
from InnerEye.ML.models.layers.identity import Identity


class DummyScalarModel2D(DeviceAwareModule[ScalarItem, torch.Tensor]):
    def __init__(self, expected_image_size_zyx: TupleInt2,
                 activation: torch.nn.Module = Identity()) -> None:
        super().__init__()
        self.expected_image_size_zyx = expected_image_size_zyx
        self._layers = torch.nn.ModuleList()
        fc = torch.nn.Conv2d(1, 1, kernel_size=3)
        torch.nn.init.normal_(fc.weight, 0, 0.01)
        with torch.no_grad():
            fc_out = fc(torch.zeros((1, 1) + self.expected_image_size_zyx))
            self.feature_size = fc_out.view(-1).shape[0]
        self._layers.append(fc)
        self.fc = torch.nn.Linear(self.feature_size, 1)
        self.activation = activation
        self.last_encoder_layer: List[str] = ["_layers", "0"]
        self.conv_in_3d = False

    def get_last_encoder_layer_names(self) -> List[str]:
        return self.last_encoder_layer

    def get_input_tensors(self, item: ScalarItem) -> List[torch.Tensor]:
        """
        Transforms a classification item into images
        :param item: ClassificationItem
        :return: Tensor
        """
        return [item.images]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        if x.shape[-2:] != self.expected_image_size_zyx:
            raise ValueError(f"Expected a tensor with trailing size {self.expected_image_size_zyx}, but got "
                             f"{x.shape}")

        if x.shape[1] != 1:
            raise ValueError("2D Model expects images with singleton z dimension")

        x = x.view((x.shape[0], ) + x.shape[2:])

        for layer in self._layers.__iter__():
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.activation(x)
