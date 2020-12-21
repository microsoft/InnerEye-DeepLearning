#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Callable, List, Optional

import torch
from torch.nn.modules import Conv3d, ConvTranspose3d

from InnerEye.Common.common_util import initialize_instance_variables
from InnerEye.Common.type_annotations import IntOrTuple3, TupleInt2
from InnerEye.ML.config import PaddingMode
from InnerEye.ML.models.architectures.base_model import BaseSegmentationModel, CropSizeConstraints
from InnerEye.ML.models.layers.basic import BasicLayer
from InnerEye.ML.models.parallel.model_parallel import get_device_from_parameters, move_to_device, \
    partition_layers
from InnerEye.ML.utils.layer_util import get_padding_from_kernel_size, get_upsampling_kernel_size, \
    initialise_layer_weights


class UNet3D(BaseSegmentationModel):
    """
    Implementation of 3D UNet model.
    Ref: Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image Segmentation
    The implementation differs from the original architecture in terms of the following:
    1) Pooling layers are replaced with strided convolutions to learn the downsampling operations
    2) Upsampling layers have spatial support larger than 2x2x2 to learn interpolation as good as linear upsampling.
    3) Non-linear activation units are placed in between deconv and conv operations to avoid two redundant linear
    operations one after another.
    4) Support for more downsampling operations to capture larger image context and improve the performance.
    The network has `num_downsampling_paths` downsampling steps on the encoding side and same number upsampling steps
    on the decoding side.
    :param num_downsampling_paths: Number of downsampling paths used in Unet model (default 4 image level are used)
    :param num_classes: Number of output segmentation classes
    :param kernel_size: Spatial support of convolution kernels used in Unet model
    """

    class UNetDecodeBlock(torch.nn.Module):
        """
        Implements upsampling block for UNet architecture. The operations carried out on the input tensor are
        1) Upsampling via strided convolutions 2) Concatenating the skip connection tensor 3) Two convolution layers
        :param channels: A tuple containing the number of input and output channels
        :param upsample_kernel_size: Spatial support of upsampling kernels. If an integer is provided, the same value
        will be repeated for all three dimensions. For non-cubic kernels please pass a list or tuple with three elements
        :param upsampling_stride: Upsamling factor used in deconvolutional layer. Similar to the `upsample_kernel_size`
        parameter, if an integer is passed, the same upsampling factor will be used for all three dimensions.
        :param activation: Linear/Non-linear activation function that is used after linear deconv/conv mappings.
        :param depth: The depth inside the UNet at which the layer operates. This is only for diagnostic purposes.
        """

        @initialize_instance_variables
        def __init__(self,
                     channels: TupleInt2,
                     upsample_kernel_size: IntOrTuple3,
                     upsampling_stride: IntOrTuple3 = 2,
                     padding_mode: PaddingMode = PaddingMode.Zero,
                     activation: Callable = torch.nn.ReLU,
                     depth: Optional[int] = None):
            super().__init__()

            assert len(channels) == 2
            self.concat = False

            self.upsample_block = torch.nn.Sequential(
                ConvTranspose3d(channels[0],
                                channels[1],
                                upsample_kernel_size,  # type: ignore
                                upsampling_stride,  # type: ignore
                                get_padding_from_kernel_size(padding_mode, upsample_kernel_size)),  # type: ignore
                torch.nn.BatchNorm3d(channels[1]),
                activation(inplace=True))

        def forward(self, x: Any) -> Any:  # type: ignore
            # When using the new DataParallel of PyTorch 1.6, self.parameters would be empty. Do not attempt to move
            # the tensors in this case. If self.parameters is present, the module is used inside of a model parallel
            # construct.
            [x] = move_to_device([x], target_device=get_device_from_parameters(self))
            return self.upsample_block(x)

    class UNetEncodeBlockSynthesis(torch.nn.Module):
        """Encode block used in upsampling path of UNet Model. It differs from UNetEncodeBlock by being able to
        aggregate information coming from both skip connection and upsampled tensors. Instead of using standard
        concatenation op followed by a convolution op, this encoder block decomposes the chain of these ops into
        multiple convolutions, this way memory usage is reduced.
        """

        @initialize_instance_variables
        def __init__(self,
                     channels: TupleInt2,
                     kernel_size: IntOrTuple3,
                     dilation: IntOrTuple3 = 1,
                     padding_mode: PaddingMode = PaddingMode.Zero,
                     activation: Callable = torch.nn.ReLU,
                     depth: Optional[int] = None):
            super().__init__()

            if not len(channels) == 2:
                raise ValueError("UNetEncodeBlockSynthesis requires 2 channels (channels: {})".format(channels))

            self.concat = True
            self.conv1 = BasicLayer(channels, kernel_size, padding=padding_mode, activation=None, use_batchnorm=False)
            self.conv2 = BasicLayer(channels, kernel_size, padding=padding_mode, activation=None, use_batchnorm=False)
            self.activation_block = torch.nn.Sequential(torch.nn.BatchNorm3d(channels[1]), activation(inplace=True))
            self.block2 = BasicLayer(channels, kernel_size, padding=padding_mode, activation=activation)
            self.apply(initialise_layer_weights)

        def forward(self, x: Any, skip_connection: Any) -> Any:  # type: ignore
            # When using the new DataParallel of PyTorch 1.6, self.parameters would be empty. Do not attempt to move
            # the tensors in this case. If self.parameters is present, the module is used inside of a model parallel
            # construct.
            [x, skip_connection] = move_to_device(input_tensors=[x, skip_connection],
                                                  target_device=get_device_from_parameters(self))
            x = self.conv1(x)
            x += self.conv2(skip_connection)
            x = self.activation_block(x)
            return self.block2(x) + x

    class UNetEncodeBlock(torch.nn.Module):
        """
        Implements a EncodeBlock for UNet.
        A EncodeBlock is two BasicLayers without dilation and with same padding.
        The first of those BasicLayer can use stride > 1, and hence will downsample.
        :param channels: A list containing two elements representing the number of input and output channels
        :param kernel_size: Spatial support of convolution kernels. If an integer is provided, the same value will
        be repeated for all three dimensions. For non-cubic kernels please pass a tuple with three elements.
        :param downsampling_stride: Downsampling factor used in the first convolutional layer. If an integer is
        passed, the same downsampling factor will be used for all three dimensions.
        :param dilation: Dilation of convolution kernels - If set to > 1, kernels capture content from wider range.
        :param activation: Linear/Non-linear activation function that is used after linear convolution mappings.
        :param use_residual: If set to True, block2 learns the residuals while preserving the output of block1
        :param depth: The depth inside the UNet at which the layer operates. This is only for diagnostic purposes.
        """

        @initialize_instance_variables
        def __init__(self,
                     channels: TupleInt2,
                     kernel_size: IntOrTuple3,
                     downsampling_stride: IntOrTuple3 = 1,
                     dilation: IntOrTuple3 = 1,
                     padding_mode: PaddingMode = PaddingMode.Zero,
                     activation: Callable = torch.nn.ReLU,
                     use_residual: bool = True,
                     depth: Optional[int] = None):
            super().__init__()

            if not len(channels) == 2:
                raise ValueError("UNetEncodeBlock requires 2 channels (channels: {})".format(channels))

            self.concat = False
            self.block1 = BasicLayer(channels, kernel_size, stride=downsampling_stride, padding=padding_mode,
                                     activation=activation)
            self.block2 = BasicLayer((channels[1], channels[1]), kernel_size, stride=1, padding=padding_mode,
                                     dilation=dilation, activation=activation)

        def forward(self, x: Any) -> Any:  # type: ignore
            # When using the new DataParallel of PyTorch 1.6, self.parameters would be empty. Do not attempt to move
            # the tensors in this case. If self.parameters is present, the module is used inside of a model parallel
            # construct.
            target_device = get_device_from_parameters(self)
            [x] = move_to_device(input_tensors=[x], target_device=target_device)
            x = self.block1(x)
            return self.block2(x) + x if self.use_residual else self.block2(x)

    @initialize_instance_variables
    def __init__(self,
                 input_image_channels: int,
                 initial_feature_channels: int,
                 num_classes: int,
                 kernel_size: IntOrTuple3,
                 name: str = "UNet3D",
                 num_downsampling_paths: int = 4,
                 downsampling_factor: IntOrTuple3 = 2,
                 downsampling_dilation: IntOrTuple3 = (1, 1, 1),
                 padding_mode: PaddingMode = PaddingMode.Zero):
        if isinstance(downsampling_factor, int):
            downsampling_factor = (downsampling_factor,) * 3
        crop_size_multiple = tuple(factor ** num_downsampling_paths
                                   for factor in downsampling_factor)
        crop_size_constraints = CropSizeConstraints(multiple_of=crop_size_multiple)
        super().__init__(name=name,
                         input_channels=input_image_channels,
                         crop_size_constraints=crop_size_constraints)
        """
        Modified 3D-Unet Class
        :param input_image_channels: Number of image channels (scans) that are fed into the model.
        :param initial_feature_channels: Number of feature-maps used in the model - Subsequent layers will contain
        number
        of featuremaps that is multiples of `initial_feature_channels` (e.g. 2^(image_level) * initial_feature_channels)
        :param num_classes: Number of output classes
        :param kernel_size: Spatial support of conv kernels in each spatial axis.
        :param num_downsampling_paths: Number of image levels used in Unet (in encoding and decoding paths)
        :param downsampling_factor: Spatial downsampling factor for each tensor axis (depth, width, height). This will
        be used as the stride for the first convolution layer in each encoder block.
        :param downsampling_dilation: An additional dilation that is used in the second convolution layer in each
        of the encoding blocks of the UNet. This can be used to increase the receptive field of the network. A good
        choice is (1, 2, 2), to increase the receptive field only in X and Y.
        :param crop_size: The size of the crop that should be used for training.
        """

        self.num_dimensions = 3
        self._layers = torch.nn.ModuleList()
        self.upsampling_kernel_size = get_upsampling_kernel_size(downsampling_factor, self.num_dimensions)

        # Create forward blocks for the encoding side, including central part
        self._layers.append(UNet3D.UNetEncodeBlock((self.input_channels, self.initial_feature_channels),
                                                   kernel_size=self.kernel_size,
                                                   downsampling_stride=1,
                                                   padding_mode=self.padding_mode,
                                                   depth=0))

        current_channels = self.initial_feature_channels
        for depth in range(1, self.num_downsampling_paths + 1):  # type: ignore
            self._layers.append(UNet3D.UNetEncodeBlock((current_channels, current_channels * 2),  # type: ignore
                                                       kernel_size=self.kernel_size,
                                                       downsampling_stride=self.downsampling_factor,
                                                       dilation=self.downsampling_dilation,
                                                       padding_mode=self.padding_mode,
                                                       depth=depth))
            current_channels *= 2  # type: ignore

        # Create forward blocks and upsampling layers for the decoding side
        for depth in range(self.num_downsampling_paths + 1, 1, -1):  # type: ignore
            channels = (current_channels, current_channels // 2)  # type: ignore
            self._layers.append(UNet3D.UNetDecodeBlock(channels,
                                                       upsample_kernel_size=self.upsampling_kernel_size,
                                                       upsampling_stride=self.downsampling_factor))

            # Use negative depth to distinguish the encode blocks in the decoding pathway.
            self._layers.append(UNet3D.UNetEncodeBlockSynthesis(channels=(channels[1], channels[1]),
                                                                kernel_size=self.kernel_size,
                                                                padding_mode=self.padding_mode,
                                                                depth=-depth))

            current_channels //= 2  # type: ignore

        # Add final fc layer
        self.output_layer = Conv3d(current_channels, self.num_classes, kernel_size=1)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        skip_connections: List[torch.Tensor] = list()
        # Unet Encoder and Decoder paths
        for layer_id, layer in enumerate(self._layers):  # type: ignore
            x = layer(x, skip_connections.pop()) if layer.concat else layer(x)
            if layer_id < self.num_downsampling_paths:  # type: ignore
                skip_connections.append(x)
        # When using the new DataParallel of PyTorch 1.6, self.parameters would be empty. Do not attempt to move
        # the tensors in this case. If self.parameters is present, the module is used inside of a model parallel
        # construct.
        [x] = move_to_device(input_tensors=[x], target_device=get_device_from_parameters(self.output_layer))
        return self.output_layer(x)

    def get_all_child_layers(self) -> List[torch.nn.Module]:
        return list(self._layers.children()) + [self.output_layer]

    def partition_model(self, devices: List[torch.device]) -> None:
        if self.summary is None:
            raise RuntimeError(
                "Network summary is required to partition UNet3D. Call model.generate_model_summary() first.")

        partition_layers(self.get_all_child_layers(), summary=self.summary, target_devices=devices)
