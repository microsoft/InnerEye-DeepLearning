#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest

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
