#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from enum import Enum
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from torch.nn import ModuleList, Sequential

from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.config import PaddingMode
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.models.architectures.base_model import DeviceAwareModule
from InnerEye.ML.models.architectures.mlp import MLP
from InnerEye.ML.models.architectures.unet_3d import UNet3D
from InnerEye.ML.models.layers.basic import BasicLayer
from InnerEye.ML.models.layers.identity import Identity
from InnerEye.ML.models.layers.pooling_layers import AveragePooling, Gated3dPoolingLayer, \
    MaxPooling, MixPooling, ZAdaptive3dAvgLayer
from InnerEye.ML.scalar_config import AggregationType
from InnerEye.ML.utils.image_util import HDF5_NUM_SEGMENTATION_CLASSES, segmentation_to_one_hot


class ImagingFeatureType(Enum):
    Segmentation = "Segmentation"
    Image = "Image"
    ImageAndSegmentation = "ImageAndSegmentation"


class ImageAndNonImageFeaturesAggregator(torch.nn.Module):
    """
    Aggregator module to combine imaging and non imaging features by concatenating.
    """

    def forward(self, *item: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        image_features, non_image_features = item[0], item[1]
        x = torch.cat([image_features.flatten(1), non_image_features], dim=1)
        return x


class ImageEncoder(DeviceAwareModule[ScalarItem, torch.Tensor]):
    """
      An architecture for an image encoder that encodes the image with several UNet encoder blocks, and
      optionally appends non-imaging features to the encoder image features. This module hence creates the
      features to be used as an input for a classification or a regression module.
    """

    def __init__(self,
                 imaging_feature_type: ImagingFeatureType = ImagingFeatureType.Image,
                 encode_channels_jointly: bool = False,
                 num_image_channels: int = 1,
                 num_encoder_blocks: int = 5,
                 initial_feature_channels: int = 32,
                 num_non_image_features: int = 0,
                 padding_mode: PaddingMode = PaddingMode.NoPadding,
                 kernel_size_per_encoding_block: Union[TupleInt3, List[TupleInt3]] = (1, 3, 3),
                 stride_size_per_encoding_block: Union[TupleInt3, List[TupleInt3]] = (1, 2, 2),
                 encoder_dimensionality_reduction_factor: float = 0.8,
                 aggregation_type: AggregationType = AggregationType.Average,
                 scan_size: Optional[TupleInt3] = None,
                 use_mixed_precision: bool = True,
                 ) -> None:
        """
        Creates an image classifier that has UNet encoders sections for each image channel. The encoder output
        is fed through average pooling and an MLP.
        :param encode_channels_jointly: If False, create a UNet encoder structure separately for each channel. If True,
        encode all channels jointly (convolution will run over all channels).
        :param num_encoder_blocks: Number of UNet encoder blocks.
        :param initial_feature_channels: Number of feature channels in the first UNet encoder.
        :param num_image_channels: Number of channels of the input. Input is expected to be of size
        B x num_image_channels x Z x Y x X, where B is the batch dimension.
        :param num_non_image_features: Number of non imaging features will be used in the model.
        :param kernel_size_per_encoding_block: The size of the kernels per encoding block, assumed to be the same
        if a single tuple is provided. Otherwise the list of tuples must match num_encoder_blocks. Default
        performs convolutions only in X and Y.
        :param stride_size_per_encoding_block: The stride size for the encoding block, assumed to be the same
        if a single tuple is provided. Otherwise the list of tuples must match num_encoder_blocks. Default
        reduces spatial dimensions only in X and Y.
        :param encoder_dimensionality_reduction_factor: how to reduce the dimensionality of the image features in the
        combined model to balance with non imaging features.
        :param scan_size: should be a tuple representing 3D tensor shape and if specified it's usedd in initializing
        gated pooling or z-adaptive. The first element should be representing the z-direction for classification images
        :param use_mixed_precision: If True, assume that training happens with mixed precision. Segmentations will
        be converted to float16 tensors right away. If False, segmentations will be converted to float32 tensors.
        """
        super().__init__()
        self.num_non_image_features = num_non_image_features
        self.imaging_feature_type = imaging_feature_type

        if isinstance(kernel_size_per_encoding_block, list):
            if len(kernel_size_per_encoding_block) != num_encoder_blocks:
                raise ValueError(f"expected kernel_size_per_encoding_block to be of "
                                 f"length {num_encoder_blocks} found {len(kernel_size_per_encoding_block)}")
            self.kernel_size_per_encoding_block = kernel_size_per_encoding_block
        else:
            self.kernel_size_per_encoding_block = [kernel_size_per_encoding_block] * num_encoder_blocks

        if isinstance(stride_size_per_encoding_block, list):
            if len(stride_size_per_encoding_block) != num_encoder_blocks:
                raise ValueError(f"expected stride_size_per_encoding_block to be of "
                                 f"length {num_encoder_blocks} found {len(stride_size_per_encoding_block)}")
            self.stride_size_per_encoding_block = stride_size_per_encoding_block
        else:
            self.stride_size_per_encoding_block = [stride_size_per_encoding_block] * num_encoder_blocks
        self.conv_in_3d = np.any([k[0] != 1 for k in self.kernel_size_per_encoding_block]) \
                          or np.any([s[0] != 1 for s in self.stride_size_per_encoding_block])
        self.use_mixed_precision = use_mixed_precision
        self.padding_mode = padding_mode
        self.encode_channels_jointly = encode_channels_jointly
        self.num_image_channels = num_image_channels
        self.image_and_non_image_features_aggregator = None
        fcn_channels = [initial_feature_channels * i for i in range(1, num_encoder_blocks)]
        if encode_channels_jointly:
            # Segmentations are encoded as one-hot tensors, separately for each of the input channels.
            # 10 classes for 2 image input channels would create a tensor of size [10*2, Z, Y, X]
            if self.imaging_feature_type == ImagingFeatureType.Segmentation:
                self.encoder_input_channels = num_image_channels * HDF5_NUM_SEGMENTATION_CLASSES
            elif self.imaging_feature_type == ImagingFeatureType.ImageAndSegmentation:
                self.encoder_input_channels = num_image_channels * (HDF5_NUM_SEGMENTATION_CLASSES + 1)
            elif self.imaging_feature_type == ImagingFeatureType.Image:
                self.encoder_input_channels = num_image_channels
            else:
                raise NotImplementedError(f"Image feature type {self.imaging_feature_type} is not supported yet.")
            _encoder: ModuleList = self.create_encoder([self.encoder_input_channels] + fcn_channels)
            final_num_feature_channels = fcn_channels[-1]
        else:
            # When working with segmentations as inputs: Feed every group of 10 per-class channels through the encoder
            # When working with normal images, each image input channel is treated separately.
            if self.imaging_feature_type == ImagingFeatureType.Segmentation:
                self.encoder_input_channels = HDF5_NUM_SEGMENTATION_CLASSES
            elif self.imaging_feature_type == ImagingFeatureType.ImageAndSegmentation:
                self.encoder_input_channels = HDF5_NUM_SEGMENTATION_CLASSES + 1
            elif self.imaging_feature_type == ImagingFeatureType.Image:
                self.encoder_input_channels = 1
            else:
                raise NotImplementedError(f"Image feature type {self.imaging_feature_type} is not supported yet.")
            _encoder = self.create_encoder([self.encoder_input_channels] + fcn_channels)
            final_num_feature_channels = fcn_channels[-1] * num_image_channels

        # Name of the last layer of the encoder to use for GradCam computation
        self.last_encoder_layer: List[str] = ["encoder", f"{len([self.encoder_input_channels] + fcn_channels) - 2}",
                                              "block2"]

        if num_non_image_features > 0:
            self.image_and_non_image_features_aggregator = self.create_non_image_and_image_aggregator()
            if encoder_dimensionality_reduction_factor < 1:
                # reduce the dimensionality of the image features to be the same as the non-image features
                # so that we can balance the input representation
                reduced_num_img_features = max(int(encoder_dimensionality_reduction_factor * fcn_channels[-1]), 1)
                _encoder.append(BasicLayer(
                    channels=(fcn_channels[-1], reduced_num_img_features),
                    kernel_size=(1, 3, 3),
                    stride=(1, 2, 2),
                    activation=None,
                    padding=padding_mode
                ))
                self.last_encoder_layer = ["encoder", f"{len([self.encoder_input_channels] + fcn_channels) - 1}", "bn1"]
                if encode_channels_jointly:
                    final_num_feature_channels = reduced_num_img_features
                else:
                    final_num_feature_channels = (reduced_num_img_features * num_image_channels)
            final_num_feature_channels += num_non_image_features
        self.final_num_feature_channels = final_num_feature_channels
        self.encoder = Sequential(*_encoder)  # type: ignore

        self.aggregation_layer = self._get_aggregation_layer(aggregation_type, scan_size)

    def _get_aggregation_layer(self, aggregation_type: AggregationType, scan_size: Optional[TupleInt3]) -> Any:
        """
        Returns the aggregation layer as specified by the config
        :param aggregation_type: name of the aggregation
        :param scan_size: [Z, Y, X] size of the scans
        """
        if aggregation_type == AggregationType.Average:
            return AveragePooling()
        elif aggregation_type == AggregationType.MixPooling:
            return MixPooling()
        elif aggregation_type == AggregationType.MaxPooling:
            return MaxPooling()
        else:
            assert scan_size is not None
            input_size = [1, self.encoder_input_channels, *scan_size]
            output = self.encoder(torch.ones(input_size))
            if aggregation_type == AggregationType.GatedPooling:
                return Gated3dPoolingLayer(output.shape[2] * output.shape[3] * output.shape[4])
            elif aggregation_type == AggregationType.ZAdaptive3dAvg:
                return ZAdaptive3dAvgLayer(output.shape[2])
            else:
                raise ValueError(f"The aggregation type {aggregation_type} is not recognized")

    def get_input_tensors(self, item: ScalarItem) -> List[torch.Tensor]:
        """
        Transforms a classification item into a torch.Tensor that the forward pass can consume
        :param item: ClassificationItem
        :return: Tensor
        """
        if self.imaging_feature_type == ImagingFeatureType.Segmentation \
                or self.imaging_feature_type == ImagingFeatureType.ImageAndSegmentation:
            if item.segmentations is None:
                raise ValueError("Expected item.segmentations to not be None")
            use_gpu = self.is_model_on_gpu()
            result_dtype = torch.float16 if self.use_mixed_precision and use_gpu else torch.float32
            # Special case need for the loading of individual positions in the sequence model,
            # the images are loaded as [C, Z, X, Y] but the segmentation_to_one_hot expects [B, C, Z, X, Y]
            if item.segmentations.ndimension() == 4:
                input_tensors = [segmentation_to_one_hot(item.segmentations.unsqueeze(dim=0),
                                                         use_gpu=use_gpu,
                                                         result_dtype=result_dtype).squeeze(dim=0)]
            else:
                input_tensors = [
                    segmentation_to_one_hot(item.segmentations, use_gpu=use_gpu, result_dtype=result_dtype)]

            if self.imaging_feature_type == ImagingFeatureType.ImageAndSegmentation:
                input_tensors.append(item.images.to(dtype=result_dtype, copy=True))
                _dim = 0 if item.images.ndimension() == 4 else 1
                input_tensors = [torch.cat(input_tensors, dim=_dim)]
        else:
            input_tensors = [item.images]

        if self.image_and_non_image_features_aggregator:
            input_tensors.append(item.get_all_non_imaging_features())
        return input_tensors

    def forward(self, *item: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = item[0]
        x = self.encode_and_aggregate(x)

        # combine non image features if required
        if self.image_and_non_image_features_aggregator:
            x = self.image_and_non_image_features_aggregator(x, item[1].float())

        return x

    def encode_and_aggregate(self, x: torch.Tensor) -> torch.Tensor:
        return encode_and_aggregate(encoder=self.encoder,
                                    num_encoder_input_channels=self.encoder_input_channels,
                                    num_image_channels=self.num_image_channels,
                                    encode_channels_jointly=self.encode_channels_jointly,
                                    aggregation_layer=self.aggregation_layer,
                                    input_tensor=x)

    def create_non_image_and_image_aggregator(self) -> ImageAndNonImageFeaturesAggregator:
        return ImageAndNonImageFeaturesAggregator()

    def create_encoder(self, channels: List[int]) -> ModuleList:
        """
        Create an image encoder network.
        """
        layers = []
        for i in range(len(channels) - 1):
            layers.append(
                UNet3D.UNetEncodeBlock(
                    channels=(channels[i], channels[i + 1]),
                    kernel_size=self.kernel_size_per_encoding_block[i],
                    downsampling_stride=self.stride_size_per_encoding_block[i],
                    padding_mode=self.padding_mode,
                    use_residual=False
                )
            )
        return ModuleList(layers)


class ImageEncoderWithMlp(ImageEncoder):
    """
    An architecture for an image classifier that first encodes the image with several UNet encoder blocks,
    and then feeds the resulting features through a multi layer perceptron (MLP). The architecture can handle
    multiple input channels. Each input channels is fed either through a separate UNet encoder pathway (if
    the argument encode_channels_jointly is False) or together with all other channels (if encode_channels_jointly is
    False) The latter makes the implicit assumption that the channels are spatially aligned.
    """

    def __init__(self,
                 mlp_dropout: float = 0.5,
                 final_activation: torch.nn.Module = Identity(),
                 imaging_feature_type: ImagingFeatureType = ImagingFeatureType.Image,
                 encode_channels_jointly: bool = False,
                 num_image_channels: int = 1,
                 num_encoder_blocks: int = 5,
                 initial_feature_channels: int = 32,
                 num_non_image_features: int = 0,
                 padding_mode: PaddingMode = PaddingMode.NoPadding,
                 kernel_size_per_encoding_block: Union[TupleInt3, List[TupleInt3]] = (1, 3, 3),
                 stride_size_per_encoding_block: Union[TupleInt3, List[TupleInt3]] = (1, 2, 2),
                 encoder_dimensionality_reduction_factor: float = 0.8,
                 aggregation_type: AggregationType = AggregationType.Average,
                 scan_size: Optional[TupleInt3] = None,
                 use_mixed_precision: bool = True,
                 ) -> None:
        """
        Creates an image classifier that has UNet encoders sections for each image channel. The encoder output
        is fed through average pooling and an MLP. Extension of the ImageEncoder class using an MLP as classification
        layer.
        :param encode_channels_jointly: If False, create a UNet encoder structure separately for each channel. If True,
        encode all channels jointly (convolution will run over all channels).
        :param num_encoder_blocks: Number of UNet encoder blocks.
        :param initial_feature_channels: Number of feature channels in the first UNet encoder.
        :param num_image_channels: Number of channels of the input. Input is expected to be of size
        B x num_image_channels x Z x Y x X, where B is the batch dimension.
        :param mlp_dropout: The amount of dropout that should be applied between the two layers of the classifier MLP.
        :param final_activation: Activation function to normalize the logits default is Identity.
        :param num_non_image_features: Number of non imaging features will be used in the model.
        :param kernel_size_per_encoding_block: The size of the kernels per encoding block, assumed to be the same
        if a single tuple is provided. Otherwise the list of tuples must match num_encoder_blocks. Default
        performs convolutions only in X and Y.
        :param stride_size_per_encoding_block: The stride size for the encoding block, assumed to be the same
        if a single tuple is provided. Otherwise the list of tuples must match num_encoder_blocks. Default
        reduces spatial dimensions only in X and Y.
        :param encoder_dimensionality_reduction_factor: how to reduce the dimensionality of the image features in the
        combined model to balance with non imaging features.
        :param scan_size: should be a tuple representing 3D tensor shape and if specified it's usedd in initializing
        gated pooling or z-adaptive. The first element should be representing the z-direction for classification images
        :param use_mixed_precision: If True, assume that training happens with mixed precision. Segmentations will
        be converted to float16 tensors right away. If False, segmentations will be converted to float32 tensors.
        """
        super().__init__(imaging_feature_type=imaging_feature_type,
                         encode_channels_jointly=encode_channels_jointly,
                         num_image_channels=num_image_channels,
                         num_encoder_blocks=num_encoder_blocks,
                         initial_feature_channels=initial_feature_channels,
                         num_non_image_features=num_non_image_features,
                         padding_mode=padding_mode,
                         kernel_size_per_encoding_block=kernel_size_per_encoding_block,
                         stride_size_per_encoding_block=stride_size_per_encoding_block,
                         encoder_dimensionality_reduction_factor=encoder_dimensionality_reduction_factor,
                         aggregation_type=aggregation_type,
                         scan_size=scan_size,
                         use_mixed_precision=use_mixed_precision)
        self.classification_layer = create_mlp(self.final_num_feature_channels, mlp_dropout)
        self.final_activation = final_activation

    def forward(self, *item: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = super().forward(*item)
        # pass all the features to the MLP
        x = self.classification_layer(x.view(-1, x.shape[1]))
        return self.final_activation(x)


def encode_and_aggregate(input_tensor: torch.Tensor,
                         encoder: torch.nn.Module,
                         num_encoder_input_channels: int,
                         num_image_channels: int,
                         encode_channels_jointly: bool,
                         aggregation_layer: Callable) -> torch.Tensor:
    """
    Function that encodes a given input tensor either jointly using the encoder or separately for each channel
    in a sequential manner. Features obtained at the output encoder are then aggregated with the pooling function
    defined by `aggregation layer`.
    """
    if encode_channels_jointly:
        input_tensor = encoder(input_tensor)
        input_tensor = aggregation_layer(input_tensor)
    else:
        shape = input_tensor.shape
        channel_shape = (shape[0], num_encoder_input_channels, shape[2], shape[3], shape[4])
        encode_and_aggregate = []
        # When using multiple encoders, it is more memory efficient to aggregate the individual
        # encoder outputs and then stack those smaller results, rather than stack huge outputs and aggregate.
        for i in range(num_image_channels):
            start_index = i * num_encoder_input_channels
            end_index = start_index + num_encoder_input_channels
            encoder_output = encoder(input_tensor[:, start_index:end_index].view(channel_shape))
            aggregated = aggregation_layer(encoder_output)
            encode_and_aggregate.append(aggregated)
        input_tensor = torch.cat(encode_and_aggregate, dim=1)
    return input_tensor


def create_mlp(input_num_feature_channels: int,
               dropout: float,
               final_output_channels: int = 1,
               final_layer: Optional[torch.nn.Module] = None,
               hidden_layer_num_feature_channels: Optional[int] = None) -> MLP:
    """
    Create an MLP with 1 hidden layer.
    :param input_num_feature_channels: The number of input channels to the first MLP layer.
    :param dropout: The drop out factor that should be applied between the first and second MLP layer.
    :param final_output_channels: if provided, the final number of output channels.
    :param final_layer: if provided, the final (activation) layer to apply
    :param hidden_layer_num_feature_channels: if provided, will be used to create hidden layers, If None then
    input_num_feature_channels // 2 will be used to create the hidden layer.
    :return:
    """
    layers: List[torch.nn.Module] = []
    hidden_layer_num_feature_channels = hidden_layer_num_feature_channels \
        if hidden_layer_num_feature_channels else input_num_feature_channels // 2
    channels: List[int] = [input_num_feature_channels, hidden_layer_num_feature_channels, final_output_channels]
    dropouts: List[float] = [dropout, 0.0]
    use_layer_normalisation: List[bool] = [True, False]
    activation: List[torch.nn.Module] = [torch.nn.Tanh(), Identity()]

    for i in range(len(channels) - 1):
        layers.append(
            MLP.HiddenLayer(
                channels=(channels[i], channels[i + 1]),
                dropout=dropouts[i],
                use_layer_normalisation=use_layer_normalisation[i],
                activation=activation[i]
            )
        )

    if final_layer:
        layers.append(final_layer)

    return MLP(layers)  # type: ignore
