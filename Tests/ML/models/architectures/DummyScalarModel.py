#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import List, Union

import torch

from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.models.architectures.base_model import DeviceAwareModule
from InnerEye.ML.models.layers.identity import Identity
from InnerEye.ML.models.parallel.data_parallel import execute_within_autocast_if_needed


class DummyScalarModel(DeviceAwareModule[ScalarItem, torch.Tensor]):
    def __init__(self, expected_image_size_zyx: TupleInt3,
                 activation: torch.nn.Module = Identity(),
                 kernel_size: Union[int, TupleInt3] = 3) -> None:
        super().__init__()
        self.expected_image_size_zyx = expected_image_size_zyx
        self._layers = torch.nn.ModuleList()
        fc = torch.nn.Conv3d(1, 1, kernel_size=kernel_size)
        torch.nn.init.normal_(fc.weight, 0, 0.01)
        with torch.no_grad():
            fc_out = fc(torch.zeros((1, 1) + self.expected_image_size_zyx))
            self.feature_size = fc_out.view(-1).shape[0]
        self._layers.append(fc)
        self.fc = torch.nn.Linear(self.feature_size, 1)
        self.activation = activation
        self.last_encoder_layer: List[str] = ["_layers", "0"]
        self.conv_in_3d = False
        self.use_mixed_precision = False

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
        def _forward() -> torch.Tensor:
            # Need to copy to a local variable, because we can't re-assign x here
            x2 = x
            if x2.shape[-3:] != self.expected_image_size_zyx:
                raise ValueError(f"Expected a tensor with trailing size {self.expected_image_size_zyx}, but got "
                                 f"{x2.shape}")

            for layer in self._layers.__iter__():
                x2 = layer(x2)
            x2 = x2.view(x2.size(0), -1)
            x2 = self.fc(x2)
            return self.activation(x2)

        # Models that will be used inside of DataParallel need to do their own autocast
        return execute_within_autocast_if_needed(_forward, use_autocast=self.use_mixed_precision)
