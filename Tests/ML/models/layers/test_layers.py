#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
import pytest
import torch
from torch.nn import BatchNorm3d, Conv3d, MSELoss, ReLU

from InnerEye.Common.type_annotations import TupleInt2, TupleInt3
from InnerEye.ML.config import PaddingMode
from InnerEye.ML.models.architectures.unet_2d import UNet2D
from InnerEye.ML.models.architectures.unet_3d import UNet3D
from InnerEye.ML.models.layers.basic import BasicLayer
from InnerEye.ML.models.layers.pooling_layers import Gated3dPoolingLayer, MixPooling, ZAdaptive3dAvgLayer
from InnerEye.ML.utils.layer_util import get_upsampling_kernel_size
from InnerEye.ML.utils.ml_util import set_random_seed

batch_size = 2
input_channels = 4
output_channels = 6
input_shape = (batch_size, input_channels, 10, 10, 10)
output_shape = (batch_size, output_channels, 6, 6, 6)

set_random_seed(1234)
input_tensor = torch.rand(*input_shape).float()
label_tensor = torch.rand(*output_shape).float()


def test_basic_layer_content() -> None:
    layer = BasicLayer(channels=(input_channels, output_channels),
                       kernel_size=3,
                       padding=PaddingMode.NoPadding,
                       dilation=1,
                       activation=torch.nn.ReLU)
    # Verify that BasicLayer module has a convolution layer, batch norm, and an activation function
    assert any([isinstance(ll, Conv3d) for ll in layer.children()])
    assert any([isinstance(ll, BatchNorm3d) for ll in layer.children()])
    assert any([isinstance(ll, ReLU) for ll in layer.children()])


def test_basic_layer_activation_inplace() -> None:
    layer = BasicLayer(channels=(input_channels, output_channels),
                       kernel_size=3,
                       padding=PaddingMode.NoPadding,
                       dilation=1,
                       activation=torch.nn.ReLU)

    assert layer.activation.inplace is True  # type: ignore


def test_basic_layer_forward_and_backward_pass() -> None:
    set_random_seed(1234)
    layer = BasicLayer(channels=(input_channels, output_channels),
                       kernel_size=5,
                       padding=PaddingMode.NoPadding,
                       dilation=1)

    output_tensor = layer(input_tensor)
    criterion = MSELoss()
    loss = torch.sqrt(criterion(output_tensor, label_tensor))
    loss.backward()

    # Verify that output tensor has no negative values after the ReLU operation
    assert np.all(output_tensor.detach().numpy() >= 0.0)

    # Verify the loss value (assertion value is computed without the in-place operation)
    # The loss value is verified for both relu_in_place=True and relu_in_place=False cases.
    assert loss.item() == pytest.approx(0.6774, abs=1e-04)


@pytest.mark.parametrize("num_patches", [1, 6])
@pytest.mark.parametrize("num_channels", [1, 2])
@pytest.mark.parametrize("num_output_channels", [5, 7])
@pytest.mark.parametrize("is_downsampling", [True, False])
@pytest.mark.parametrize("image_shape", [(10, 12)])
def test_unet2d_encode(num_patches: int,
                       num_channels: int,
                       num_output_channels: int,
                       is_downsampling: bool,
                       image_shape: TupleInt2) -> None:
    """
    Test if the Encode block of a Unet3D correctly works when passing in kernels that only operate in X and Y.
    """
    set_random_seed(1234)
    layer = UNet3D.UNetEncodeBlock((num_channels, num_output_channels),
                                   kernel_size=(1, 3, 3),
                                   downsampling_stride=(1, 2, 2) if is_downsampling else 1)
    input_shape = (num_patches, num_channels) + (1,) + image_shape
    input = torch.rand(*input_shape).float()
    output = layer(input)

    def output_image_size(input_image_size: int) -> int:
        # If max pooling is added, it is done with a kernel size of 2, shrinking the image by a factor of 2
        image_shrink_factor = 2 if is_downsampling else 1
        return input_image_size // image_shrink_factor

    # Expected output shape:
    # The first dimension (patches) should be retained unchanged.
    # We should get as many output channels as requested
    # Unet is defined as running over degenerate 3D images with Z=1, this should be preserved.
    # The two trailing dimensions are the adjusted image dimensions
    expected_output_shape = (num_patches, num_output_channels, 1,
                             output_image_size(image_shape[0]), output_image_size(image_shape[1]))
    assert output.shape == expected_output_shape


@pytest.mark.parametrize("num_patches", [1, 3])
@pytest.mark.parametrize("image_shape", [(12, 16, 18)])
def test_unet2d_decode(num_patches: int,
                       image_shape: TupleInt3) -> None:
    """
    Test if the Decode block of a UNet3D creates tensors of the expected size when the kernels only operate in
    X and Y.
    """
    set_random_seed(1234)
    num_input_channels = image_shape[0]
    num_output_channels = num_input_channels // 2
    upsample_layer = UNet2D.UNetDecodeBlock((num_input_channels, num_output_channels),
                                            upsample_kernel_size=(1, 4, 4),
                                            upsampling_stride=(1, 2, 2))
    encode_layer = UNet2D.UNetEncodeBlockSynthesis(channels=(num_output_channels, num_output_channels),
                                                   kernel_size=(1, 3, 3))

    dim_z = 1
    input_shape = (num_patches, num_input_channels, dim_z, image_shape[1], image_shape[2])
    input_tensor = torch.rand(*input_shape).float()
    skip_connection = torch.zeros((num_patches, num_output_channels, dim_z, image_shape[1] * 2, image_shape[2] * 2))
    output = encode_layer(upsample_layer(input_tensor), skip_connection)

    def output_image_size(i: int) -> int:
        return image_shape[i] * 2

    # Expected output shape:
    # The first dimension (patches) should be retained unchanged.
    # We should get as many output channels as requested
    # Unet is defined as running over degenerate 3D images with Z=1, this should be preserved.
    # The two trailing dimensions are the adjusted image dimensions
    expected_output_shape = (num_patches, num_output_channels, dim_z, output_image_size(1), output_image_size(2))
    assert output.shape == expected_output_shape


def test_unet3d_upsampling() -> None:
    assert get_upsampling_kernel_size(2, 3) == (4, 4, 4)
    assert get_upsampling_kernel_size(1, 3) == (1, 1, 1)
    assert get_upsampling_kernel_size((1, 2, 2), 3) == (1, 4, 4)
    assert get_upsampling_kernel_size((1, 2, 3), 3) == (1, 4, 6)
    with pytest.raises(ValueError):
        get_upsampling_kernel_size((2, 2), 3)
    with pytest.raises(ValueError):
        get_upsampling_kernel_size(0, 3)
    # Test method as integrated into the constructor
    assert UNet3D(1, 1, 1, kernel_size=3, downsampling_factor=(1, 2, 2)).upsampling_kernel_size == (1, 4, 4)


def test_mix_pooling() -> None:
    torch.manual_seed(0)
    mix = MixPooling()
    weight = torch.nn.functional.sigmoid(mix.mixing_weight)
    input = torch.tensor([0, 1, 2], dtype=torch.float32).reshape((1, 1, 3, 1, 1))
    expected = weight * 1 + (1 - weight) * 2  # type: ignore
    assert mix(input, [3, 1, 1]).squeeze() == expected.squeeze()


def test_zadaptive_pooling() -> None:
    torch.manual_seed(0)
    input = torch.tensor([0, 1, 2], dtype=torch.float32).reshape((1, 1, 3, 1, 1))
    zadap = ZAdaptive3dAvgLayer(3)
    weight = torch.nn.functional.softmax(zadap.scan_weight, dim=0)
    expected = weight[1] + 2 * weight[2]
    assert zadap(input, [3, 1, 1]).squeeze() == expected


def test_gated_pooling() -> None:
    torch.manual_seed(0)
    input = torch.tensor([0, 1, 2], dtype=torch.float32)
    gated = Gated3dPoolingLayer(3)
    weight = gated.gate(input)
    expected = weight * 1 + (1 - weight) * 2  # type: ignore
    assert gated(input.reshape((1, 1, 3, 1, 1)), [3, 1, 1]).squeeze() == expected
