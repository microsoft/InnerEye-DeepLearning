#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch

from InnerEye.ML.dataset.sample import GeneralSampleMetadata
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.models.architectures.classification.segmentation_encoder import MultiSegmentationEncoder, \
    SegmentationEncoder, \
    _AddConvAndPool, _ConvPoolAndShrink
from InnerEye.ML.utils.image_util import HDF5_NUM_SEGMENTATION_CLASSES


def test_add_convs() -> None:
    channels = 5
    encoder = _AddConvAndPool(in_channels=channels, num_xy_convs=7, num_z_convs=8)
    expected_out_channels = channels + 7 + 8
    assert encoder.out_channels == expected_out_channels
    scan_size = (20, 30, 40)
    batch_size = 3
    x = torch.ones((batch_size, channels) + scan_size)
    y = encoder(x)
    assert y.size() == (batch_size, expected_out_channels) + scan_size


def test_conv_pool_shrink() -> None:
    channels = 5
    encoder = _ConvPoolAndShrink(in_channels=channels, num_xy_convs=7, num_z_convs=8, shrink_factor=0.5)
    expected_out_channels = (channels + 7 + 8) // 2
    assert encoder.out_channels == expected_out_channels
    scan_size = (21, 30, 42)
    batch_size = 3
    x = torch.ones((batch_size, channels) + scan_size)
    y = encoder(x)
    max_pooled_scan_size = tuple(i // 2 for i in scan_size)
    assert y.size() == (batch_size, expected_out_channels) + max_pooled_scan_size


def _expected_output_channels(num_input_channels: int) -> int:
    """
    Gets the expected number of output channels that the SegmentationEncoder produces for a given number
    of input channels.
    """
    channels = num_input_channels + 2
    channels = int((channels + 4 + 2) * 0.5)
    channels = int((channels + 4 + 2) * 0.5)
    channels = int((channels + 4 + 2) * 0.5)
    channels = int((channels + 6 + 3) * 0.5)
    return channels


def test_segmentation_encoder_forward() -> None:
    channels = 10
    encoder = SegmentationEncoder(in_channels=channels)
    # Full scan size is too big for CPU tests
    # scan_size = (49, 512, 496)
    scan_size = (25, 33, 65)
    batch_size = 3
    x = torch.ones((batch_size, channels) + scan_size)
    y = encoder(x)
    # Expected output size: Image reduces by factor of 2^3 in Z dimension, 2^4 in X and Y,
    # because each max pooling operation has a kernel size of (2, 2, 2) apart from the first one (1, 2, 2)
    expected_output_size = (
        scan_size[0] // 8,
        scan_size[1] // 16,
        scan_size[2] // 16,
    )
    final_output_channels = _expected_output_channels(channels)
    assert encoder.conv1.out_channels == 12
    assert encoder.group1.out_channels == 9
    assert encoder.group2.out_channels == 7
    assert encoder.group3.out_channels == 6
    assert encoder.out_channels == final_output_channels
    assert y.size() == (batch_size, final_output_channels) + expected_output_size


def test_multi_segmentation_encoder() -> None:
    scan_size = (25, 33, 65)
    batch_size = 3
    num_image_channels = 2
    encoder = MultiSegmentationEncoder(num_image_channels=num_image_channels, encode_channels_jointly=True)
    x = torch.ones((batch_size, num_image_channels * HDF5_NUM_SEGMENTATION_CLASSES) + scan_size)
    y = encoder.encode_and_aggregate(x)
    final_output_channels = _expected_output_channels(num_image_channels * HDF5_NUM_SEGMENTATION_CLASSES)
    assert y.size() == (batch_size, final_output_channels, 1, 1, 1)
    full_output = encoder(x)
    assert full_output.size() == (batch_size, 1)
    encoder = MultiSegmentationEncoder(num_image_channels=num_image_channels, encode_channels_jointly=False)
    x = torch.ones((batch_size, num_image_channels * HDF5_NUM_SEGMENTATION_CLASSES) + scan_size)
    y = encoder.encode_and_aggregate(x)
    final_output_channels = _expected_output_channels(HDF5_NUM_SEGMENTATION_CLASSES)
    # Each image channel generates 7 features, we concatenate those 7 features for the 2 image channels
    assert y.size() == (batch_size, final_output_channels * 2, 1, 1, 1)
    full_output = encoder(x)
    assert full_output.size() == (batch_size, 1)
    # Test that the encoder can correctly convert from a scalar data item to the one-hot encoded model input tensor
    scalar_item = ScalarItem(metadata=GeneralSampleMetadata(id="foo"),
                             label=torch.empty(1),
                             numerical_non_image_features=torch.empty(1),
                             categorical_non_image_features=torch.empty(1),
                             images=torch.empty(1),
                             segmentations=torch.ones((batch_size, num_image_channels, *scan_size)))
    input_tensors = encoder.get_input_tensors(scalar_item)
    assert len(input_tensors) == 1
    assert input_tensors[0].shape == (batch_size, HDF5_NUM_SEGMENTATION_CLASSES * num_image_channels, *scan_size)
