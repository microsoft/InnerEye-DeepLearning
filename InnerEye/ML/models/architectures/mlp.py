#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import List

import torch.nn
from torch import Tensor

from InnerEye.Common.type_annotations import TupleInt2
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.models.layers.identity import Identity
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule


class MLP(DeviceAwareModule[ScalarItem, torch.Tensor]):
    """
    An implementation of a Multilayer Perceptron, with Tanh activations, and
    the ability to configure dropout and batch norm layers.
    """

    class HiddenLayer(torch.nn.Module):
        """
        An implementation of a single Multilayer Perceptron hidden layer with Tanh activations, and
        the ability to configure dropout and batch norm layers.
        """

        def __init__(self,
                     channels: TupleInt2,
                     dropout: float = 0.0,
                     use_layer_normalisation: bool = True,
                     activation: torch.nn.Module = Identity()):
            """
            :param channels: The input and output number of channels to be used for this module.
            :param dropout: The dropout probability for feature maps (default keeps all activations, no dropout).
            :param use_layer_normalisation: If set to True, it applies a layer normalisation prior to activation.
            :param activation: Torch.nn.module to be used for activation after a linear operation.
            """
            super().__init__()
            layers = [
                torch.nn.Linear(in_features=channels[0], out_features=channels[1], bias=True),
                torch.nn.LayerNorm(normalized_shape=channels[1],
                                   elementwise_affine=True) if use_layer_normalisation else Identity(),
                activation, torch.nn.Dropout(p=dropout, inplace=False)
            ]

            self.model = torch.nn.Sequential(*layers)  # type: ignore

        def forward(self, x: Tensor) -> Tensor:  # type: ignore
            return self.model(x)

    def __init__(self, hidden_layers: List[HiddenLayer]):
        super().__init__()
        self.model = torch.nn.Sequential(*hidden_layers)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.model(x)

    def get_input_tensors(self, item: ScalarItem) -> List[torch.Tensor]:
        """
        Transforms a classification item into a torch.Tensor that the forward pass can consume
        :param item: ClassificationItem
        :return: Tensor
        """
        if item.images.numel() > 0:
            return [item.images]
        else:
            return [item.get_all_non_imaging_features()]
