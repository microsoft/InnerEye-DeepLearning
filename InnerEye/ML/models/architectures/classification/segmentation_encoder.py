#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import List, Optional

import torch

from InnerEye.ML.config import PaddingMode
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.models.architectures.classification.image_encoder_with_mlp import create_mlp, encode_and_aggregate
from InnerEye.ML.models.layers.basic import BasicLayer
from InnerEye.ML.models.layers.pooling_layers import AveragePooling
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.ML.utils.image_util import HDF5_NUM_SEGMENTATION_CLASSES, segmentation_to_one_hot


class _AddConvAndPool(torch.nn.Module):
    """
    A module that appends the output of convolutions along XY only, and convolutions along Z only, to the input.
    The convolution outputs will appear as additional channels.
    """

    def __init__(self,
                 in_channels: int,
                 pool: Optional[torch.nn.Module] = None,
                 num_xy_convs: int = 0,
                 num_z_convs: int = 0):
        super().__init__()
        self.pool = pool
        self.conv_xy: Optional[torch.nn.Module] = None
        # BasicLayer is set up to perform convolutions, batchnorm, and activation function.
        if num_xy_convs > 0:
            self.conv_xy = BasicLayer(channels=(in_channels, num_xy_convs),
                                      kernel_size=(1, 3, 3),
                                      padding=PaddingMode.Zero)
        self.conv_z: Optional[torch.nn.Module] = None
        if num_z_convs > 0:
            self.conv_z = BasicLayer(channels=(in_channels, num_z_convs),
                                     kernel_size=(3, 1, 1),
                                     padding=PaddingMode.Zero)
        self._out_channels = in_channels + num_xy_convs + num_z_convs

    @property
    def out_channels(self) -> int:
        """
        Gets the number of channels that this model will output.
        """
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        concat = []

        def pool_if_needed(t: torch.Tensor) -> torch.Tensor:
            return self.pool(t) if self.pool else t

        if self.conv_xy:
            concat.append(pool_if_needed(self.conv_xy(x)))
        if self.conv_z:
            concat.append(pool_if_needed(self.conv_z(x)))
        if concat:
            # Add the XY and Z convolution results as additional channels, which is dim 1
            x = torch.cat([pool_if_needed(x)] + concat, dim=1)
        return x


class _ConvPoolAndShrink(torch.nn.Module):
    """
    This module packages operations that are repeatedly used: Adding a given number of XY and Z convolutions to the
    input in the form of additional channels, run max pooling, and reduce the dimensionality by performing a
    convolution with size 1.
    """

    def __init__(self,
                 in_channels: int,
                 num_xy_convs: int,
                 num_z_convs: int,
                 shrink_factor: float,
                 ):
        super().__init__()
        self.pool = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv = _AddConvAndPool(in_channels, pool=self.pool, num_xy_convs=num_xy_convs, num_z_convs=num_z_convs)
        self._out_channels = int(self.conv.out_channels * shrink_factor)
        self.shrink = BasicLayer(channels=(self.conv.out_channels, self._out_channels),
                                 kernel_size=1)
        self.layers = torch.nn.Sequential(self.conv, self.shrink)

    @property
    def out_channels(self) -> int:
        """
        Gets the number of channels that this model will output.
        """
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.layers(x)


class SegmentationEncoder(torch.nn.Module):
    """
    Implements the eye pathology classification model outlined in the following paper:
    De Fauw, Jeffrey, et al. "Clinically applicable deep learning for diagnosis and referral in retinal disease."
    Nature medicine 24.9 (2018): 1342-1350.
    The model takes segmentation maps as input and outputs its most likely corresponding semantic class.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = _AddConvAndPool(in_channels=in_channels,
                                     pool=torch.nn.MaxPool3d(kernel_size=(1, 2, 2)),
                                     num_xy_convs=2,
                                     num_z_convs=0)
        self.group1 = _ConvPoolAndShrink(self.conv1.out_channels,
                                         num_xy_convs=4,
                                         num_z_convs=2,
                                         shrink_factor=0.5,  # Graph in the paper makes that look more like 0.25?
                                         )
        self.group2 = _ConvPoolAndShrink(self.group1.out_channels,
                                         num_xy_convs=4,
                                         num_z_convs=2,
                                         shrink_factor=0.5,
                                         )
        self.group3 = _ConvPoolAndShrink(self.group2.out_channels,
                                         num_xy_convs=4,
                                         num_z_convs=2,
                                         shrink_factor=0.5,
                                         )
        self.conv2 = _AddConvAndPool(in_channels=self.group3.out_channels,
                                     pool=None,
                                     num_xy_convs=6,
                                     num_z_convs=3)
        self._out_channels = self.conv2.out_channels // 2
        self.shrink2 = BasicLayer(channels=(self.conv2.out_channels, self._out_channels),
                                  kernel_size=1)
        self.layers = torch.nn.Sequential(
            self.conv1,
            self.group1,
            self.group2,
            self.group3,
            self.conv2,
            self.shrink2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.layers(x)

    @property
    def out_channels(self) -> int:
        """
        Gets the number of channels that this model will output.
        """
        return self._out_channels


class MultiSegmentationEncoder(DeviceAwareModule[ScalarItem, torch.Tensor]):
    def __init__(self,
                 num_image_channels: int,
                 encode_channels_jointly: bool = False,
                 ) -> None:
        """
        :param encode_channels_jointly: If False, create an encoder structure separately for each channel. If True,
        encode all channels jointly (convolution will run over all channels).
        :param num_image_channels: Number of channels of the input. Input is expected to be of size
        B x (num_image_channels * 10) x Z x Y x X, where B is the batch dimension.
        """
        super().__init__()
        self.encoder_input_channels = \
            HDF5_NUM_SEGMENTATION_CLASSES * num_image_channels if encode_channels_jointly \
                else HDF5_NUM_SEGMENTATION_CLASSES
        self.encode_channels_jointly = encode_channels_jointly
        self.num_image_channels = num_image_channels
        self.encoder = SegmentationEncoder(in_channels=self.encoder_input_channels)
        num_dense_layer_inputs = \
            self.encoder.out_channels if encode_channels_jointly \
                else self.encoder.out_channels * num_image_channels
        self.dense_layer = create_mlp(num_dense_layer_inputs, dropout=0.5)

    def encode_and_aggregate(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return encode_and_aggregate(encoder=self.encoder,
                                    num_encoder_input_channels=self.encoder_input_channels,
                                    num_image_channels=self.num_image_channels,
                                    encode_channels_jointly=self.encode_channels_jointly,
                                    aggregation_layer=AveragePooling(),
                                    input_tensor=input_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        aggregated = self.encode_and_aggregate(x)
        return self.dense_layer(aggregated.view(-1, aggregated.shape[1]))

    def get_input_tensors(self, item: ScalarItem) -> List[torch.Tensor]:
        """
        Transforms a classification item into a torch.Tensor that the forward pass can consume
        :param item: ClassificationItem
        :return: Tensor
        """
        if item.segmentations is None:
            raise ValueError("Expected item.segmentations to not be None")
        use_gpu = self.is_model_on_gpu()
        return [segmentation_to_one_hot(item.segmentations, use_gpu=use_gpu)]
