#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import numpy as np

from InnerEye.ML.models.layers.weight_standardization import WeightStandardizedConv2d, eps


def test_standardize_ones() -> None:
    """
    Smoke test for normalization.
    """
    size = (5, 3, 3, 3)
    weights = torch.ones(size)
    result = WeightStandardizedConv2d.standardize(weights)
    assert result.shape == weights.shape
    assert torch.allclose(result, torch.zeros(size=size))


def test_standardize_zeros() -> None:
    """
    Make sure there are no divide-by-zero errors.
    """
    size = (5, 3, 3, 3)
    weights = torch.zeros(size)
    result = WeightStandardizedConv2d.standardize(weights)
    assert result.shape == weights.shape
    assert torch.allclose(result, torch.zeros(size=size))


def test_standardize_rows() -> None:
    """
    We find mean and variance for each filter, so a filter filled with a constant value should normalize to 0.
    """
    size = (5, 3, 3, 3)
    weights = torch.ones(size)
    for i in range(weights.shape[0]):
        weights[i] = i
    result = WeightStandardizedConv2d.standardize(weights)
    assert result.shape == weights.shape
    assert torch.allclose(result, torch.zeros(size=size))


def test_standardize_random() -> None:
    """
    Test normalization on arbitrary weights.
    """
    size = (5, 3, 3, 3)
    random = np.random.randint(low=0, high=100, size=size).astype(np.float)
    weights = torch.from_numpy(random)
    expected = np.copy(random)

    mean = expected.mean(axis=(1, 2, 3), keepdims=True)
    # this test also makes sure we are not using an unbiased estimate of variance in the convolution layer:
    # Torch uses unbiased estimates by default
    var = expected.var(axis=(1, 2, 3), ddof=0, keepdims=True)
    expected = (expected - mean)/np.sqrt(var + eps)
    expected = torch.from_numpy(expected)

    result = WeightStandardizedConv2d.standardize(weights)
    assert result.shape == weights.shape
    assert torch.allclose(result, expected)


def test_conv_constant_filter() -> None:
    """
    Test with filters filled with a constant value: they'll normalize to zero
    """
    in_channels = 3
    out_channels = 5
    batch_size = 1
    image_size = (batch_size, in_channels, 20, 20)
    kernel_size = (3, 3)
    expected_output_size = (batch_size, out_channels, image_size[2]-kernel_size[0]+1, image_size[3]-kernel_size[1]+1)

    conv_layer = WeightStandardizedConv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          bias=True)
    conv_layer.weight.data.fill_(1)
    conv_layer.bias.data.fill_(0)  # we aren't normalizing bias
    image = torch.ones(size=image_size)
    result = conv_layer(image)
    assert result.shape == expected_output_size
    assert torch.allclose(result, torch.zeros(size=expected_output_size))
