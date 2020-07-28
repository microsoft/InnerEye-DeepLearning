#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest
import torch
from torch.nn.modules import Conv2d

from InnerEye.ML.config import PaddingMode
from InnerEye.ML.utils.layer_util import get_padding_from_kernel_size


@pytest.mark.parametrize("padding_mode", [PaddingMode.NoPadding, PaddingMode.Zero, PaddingMode.Edge])
def test_get_padding_from_kernel_size(padding_mode: PaddingMode) -> None:
    def check_padding(kernel_size, dilation, num_dimensions, expected) -> None:  # type: ignore
        actual = get_padding_from_kernel_size(padding_mode, kernel_size, dilation, num_dimensions)
        if padding_mode == PaddingMode.NoPadding:
            assert actual == tuple(0 for _ in expected), "No padding should return all zeros."
        else:
            assert actual == expected

    # Scalar values for kernel size and dilation: Should expand to the given number of dimensions
    check_padding(kernel_size=1, dilation=1, num_dimensions=3, expected=(0, 0, 0))
    check_padding(kernel_size=3, dilation=1, num_dimensions=3, expected=(1, 1, 1))
    # If either kernel_size or dilation are sized, the number of dimensions should be ignored,
    # and number of dimensions should come from whatever argument has size
    check_padding(kernel_size=(3, 3), dilation=1, num_dimensions=10, expected=(1, 1))
    check_padding(kernel_size=3, dilation=(1, 1), num_dimensions=10, expected=(1, 1))
    # Non-isotropic kernels
    check_padding(kernel_size=(3, 3, 1), dilation=1, num_dimensions=10, expected=(1, 1, 0))
    # With dilation: Dimension where the kernel size is 1 should not be padded, because
    # no reduction in size is happening along that dimension (see test_degenerate_conv_with_dilation)
    check_padding(kernel_size=(3, 3, 1), dilation=5, num_dimensions=3, expected=(5, 5, 0))


@pytest.mark.parametrize("dilation1", [1, 2])
def test_degenerate_conv_with_dilation(dilation1: int) -> None:
    """
    Check if a 2D convolution with a degenerate kernel size along one dimension, and a
    dilation > 1 along the same dimension, works as expected.
    :return:
    """
    input = torch.zeros((1, 1, 10, 20))
    # Kernel is degenerate across [0], but dilation is 2
    feature_channels = 2
    conv = Conv2d(1, feature_channels, kernel_size=(1, 3), dilation=(dilation1, dilation1))
    output = conv(input)
    print("Input has size {}, output has size {}".format(input.shape, output.shape))
    # Expectation is that the image is unchanged across the first dimension, even though there is a
    # dilation specified.
    assert output.shape[0] == input.shape[0]
    assert output.shape[1] == feature_channels
    assert output.shape[2] == input.shape[2]
    assert output.shape[3] < input.shape[3]
