#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List, Optional

import torch

from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.models.architectures.base_model import BaseModel, CropSizeConstraints
from InnerEye.ML.models.blocks.residual import ResidualBlock
from InnerEye.ML.models.layers.basic import BasicLayer


class ComplexModel(BaseModel):
    """
    A general class of feed-forward convolutional neural networks that is characterised by a network definition
    (list of lists of modules). It supports residual blocks, auto-focus and atrous spatial pyramid pooling layers.
    """

    # noinspection PyTypeChecker
    def __init__(self,
                 args: SegmentationModelBase,
                 full_channels_list: List[int],
                 dilations: List[int],
                 network_definition: List[List[torch.nn.Module]],
                 crop_size_constraints: Optional[CropSizeConstraints] = None):
        """
        Creates a new instance of the class.
        :param args: The full model configuration.
        :param full_channels_list: A vector of channel sizes. First entry is the number of image channels,
        then all feature channels, then the number of classes.
        :param network_definition:
        :param crop_size_constraints: The size constraints for the training crop size.
        """
        super().__init__(name='ComplexModel',
                         input_channels=full_channels_list[0],
                         crop_size_constraints=crop_size_constraints)
        self.full_channels_list = full_channels_list
        self.kernel_size = args.kernel_size
        self.dilations = dilations

        self._layers = torch.nn.ModuleList()
        channel_i = dilation_i = 0
        for layer in network_definition:

            if isinstance(layer, list):
                n_layers = len(layer)
                model_block = ResidualBlock(layers=layer,
                                            channels=full_channels_list[channel_i:channel_i + (n_layers + 1)],
                                            kernel_size=self.kernel_size,
                                            dilations=self.dilations[dilation_i:dilation_i + n_layers])
                channel_i += n_layers
                dilation_i += n_layers
                self._layers.append(model_block)
            elif layer == BasicLayer:
                model_block = BasicLayer(full_channels_list[channel_i:channel_i + 2], self.kernel_size,  # type: ignore
                                         dilation=self.dilations[dilation_i], use_bias=True)
                channel_i += 1
                dilation_i += 1
                self._layers.append(model_block)
            else:
                raise ValueError(f"Unknown layer {layer}")

        fc = torch.nn.Conv3d(full_channels_list[channel_i], full_channels_list[channel_i + 1], kernel_size=1)
        self._layers.append(fc)

    def forward(self, x: Any) -> Any:  # type: ignore
        for layer in self._layers.children():
            x = layer(x)

        return x

    def get_all_child_layers(self) -> List[torch.nn.Module]:
        return list(self._layers.children())
