#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import List

import torch

from InnerEye.ML.config import PaddingMode
from InnerEye.ML.models.architectures.unet_3d import UNet3D


class UNet2D(UNet3D):
    """
    This class implements a UNet in 2 dimensions, with the input expected as a 3 dimensional tensor with a vanishing
    Z dimension.
    """

    def __init__(self,
                 input_image_channels: int,
                 initial_feature_channels: int,
                 num_classes: int,
                 num_downsampling_paths: int = 4,
                 downsampling_dilation: int = 2,
                 padding_mode: PaddingMode = PaddingMode.Zero):
        """
        Initializes a 2D UNet model, where the input image is expected as a 3 dimensional tensor with a vanishing
        Z dimension.
        :param input_image_channels: The number of image channels that the model should consume.
        :param initial_feature_channels: The number of feature maps used in the model in the first convolution layer.
        Subsequent layers will contain number of feature maps that are multiples of `initial_channels`
        (2^(image_level) * initial_channels)
        :param num_classes: Number of output classes
        :param num_downsampling_paths: Number of image levels used in Unet (in encoding and decoding paths)
        :param downsampling_dilation: An additional dilation that is used in the second convolution layer in each
        of the encoding blocks of the UNet. This can be used to increase the receptive field of the network. A good
        choice is (1, 2, 2), to increase the receptive field only in X and Y.
        :param padding_mode: The type of padding that should be applied.
        """
        super().__init__(input_image_channels=input_image_channels,
                         initial_feature_channels=initial_feature_channels,
                         num_classes=num_classes,
                         kernel_size=(1, 3, 3),
                         num_downsampling_paths=num_downsampling_paths,
                         downsampling_factor=(1, 2, 2),
                         downsampling_dilation=(1, downsampling_dilation, downsampling_dilation),
                         padding_mode=padding_mode,
                         name="UNet2D_depth{}".format(num_downsampling_paths))

    def get_all_child_layers(self) -> List[torch.nn.Module]:
        return super().get_all_child_layers()
