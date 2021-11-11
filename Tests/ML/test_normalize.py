#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Union

import numpy as np
import pytest
import torch
from pytest import approx

from InnerEye.Common.common_util import is_gpu_tensor
from InnerEye.ML import photometric_normalization
from InnerEye.ML.utils.image_util import NumpyOrTorch
from InnerEye.ML.utils.io_util import ImageDataType
from InnerEye.ML.utils.transforms import CTRange
from Tests.ML.util import no_gpu_available

shape = (4, 4, 4)
image_shape = (3, 4, 4, 4)

mask_ones = np.ones(shape, dtype=ImageDataType.MASK.value)
mask_zeros = np.zeros(shape, dtype=ImageDataType.MASK.value)
mask_half = np.ones(shape, dtype=ImageDataType.MASK.value)
mask_half[2:, :, :] = 0

valid_masks = [mask_zeros, mask_ones, mask_half]

output = (0.0, 1.0)
level = 900
window = 30
tail = 2.0
sharpen = 2.0
tail3 = [2.0, 1.5, 1.0]

use_gpu = not no_gpu_available


@pytest.fixture
def image_rand_pos() -> np.ndarray:
    torch.random.manual_seed(1)
    np.random.seed(0)
    return (np.random.rand(3, 4, 4, 4) * 1000.0).astype(ImageDataType.IMAGE.value)


@pytest.fixture
def image_rand_pos_gpu(image_rand_pos: np.ndarray) -> Union[torch.Tensor, np.ndarray]:
    return torch.tensor(image_rand_pos) if use_gpu else image_rand_pos


def assert_image_out_datatype(image_out: np.ndarray) -> None:
    """
    Check that the given image, that is coming out of photometric normalization, has the
    correct datatype.
    """
    assert image_out.dtype == ImageDataType.IMAGE.value, "The result of normalization must always have the " \
                                                         "datatype that we force images to have."


def test_simplenorm_half(image_rand_pos: np.ndarray) -> None:
    image_out = photometric_normalization.simple_norm(image_rand_pos, mask_half, debug_mode=True)
    assert np.mean(image_out, dtype=float) == approx(-0.05052318)
    for c in range(image_out.shape[0]):
        assert np.mean(image_out[c, mask_half > 0.5], dtype=float) == approx(0, abs=1e-7)
    assert_image_out_datatype(image_out)


def test_simplenorm_ones(image_rand_pos: np.ndarray) -> None:
    image_out = photometric_normalization.simple_norm(image_rand_pos, mask_ones, debug_mode=True)
    assert np.mean(image_out) == approx(0, abs=1e-7)
    assert_image_out_datatype(image_out)


def test_mriwindowhalf(image_rand_pos: np.ndarray) -> None:
    image_out, status = photometric_normalization.mri_window(image_rand_pos, mask_half, (0, 1), sharpen, tail)
    assert np.mean(image_out) == approx(0.2748852)
    assert_image_out_datatype(image_out)


def test_mriwindowones(image_rand_pos: np.ndarray) -> None:
    image_out, status = photometric_normalization.mri_window(image_rand_pos, mask_ones, (0.0, 1.0), sharpen, tail3)
    assert np.mean(image_out) == approx(0.2748852)
    assert_image_out_datatype(image_out)


def test_trimmed_norm_full(image_rand_pos: np.ndarray) -> None:
    image_out, status = photometric_normalization.normalize_trim(image_rand_pos, mask_ones,
                                                                 output_range=(-1, 1), sharpen=1,
                                                                 trim_percentiles=(1, 99))
    assert np.mean(image_out, dtype=float) == approx(-0.08756259549409151)
    assert_image_out_datatype(image_out)


def test_trimmed_norm_half(image_rand_pos: np.ndarray) -> None:
    image_out, status = photometric_normalization.normalize_trim(image_rand_pos, mask_half,
                                                                 output_range=(-1, 1), sharpen=1,
                                                                 trim_percentiles=(1, 99))
    assert np.mean(image_out, dtype=float) == approx(-0.4862089517215888)
    assert_image_out_datatype(image_out)


def test_ct_range_manual(image_rand_pos_gpu: NumpyOrTorch) -> None:
    image_out: NumpyOrTorch = CTRange.transform(
        data=image_rand_pos_gpu,
        output_range=output,
        window=window,
        level=level,
        use_gpu=use_gpu
    )
    if use_gpu:
        assert is_gpu_tensor(image_out)
        assert isinstance(image_out, torch.Tensor)
        image_out = image_out.cpu().numpy()

    assert np.mean(image_out) == approx(0.0929241235)
    assert isinstance(image_out, np.ndarray)
    assert_image_out_datatype(image_out)


def test_ct_range_liver(image_rand_pos_gpu: NumpyOrTorch) -> None:
    image_out = CTRange.transform(
        data=image_rand_pos_gpu,
        output_range=output,
        window=200,
        level=55,
        use_gpu=use_gpu
    )

    if use_gpu:
        assert is_gpu_tensor(image_out)
        assert isinstance(image_out, torch.Tensor)
        image_out = image_out.cpu().numpy()

    assert np.mean(image_out, dtype=float) == approx(0.9399296555978557)
    assert isinstance(image_out, np.ndarray)
    assert_image_out_datatype(image_out)
