#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random

import numpy as np
import pytest

import torch
from PIL import Image

from torchvision.transforms.functional import to_pil_image, to_tensor

from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.ML.augmentations.image_transforms import AddGaussianNoise, ElasticTransform, ExpandChannels, RandomGamma

image_size = (256, 256)

test_tensor_1channel_1slice = torch.ones([1, 1, *image_size], dtype=torch.float)
test_tensor_1channel_1slice[..., 100:150, 100:200] = 1 / 255.
min_test_image, max_test_image = test_tensor_1channel_1slice.min(), test_tensor_1channel_1slice.max()
test_tensor_2channels_2slices = torch.ones([2, 2, *image_size], dtype=torch.float)
test_tensor_2channels_2slices[..., 100:150, 100:200] = 1 / 255.
invalid_test_tensor = torch.ones([1, *image_size])
test_pil_image = Image.open(str(full_ml_test_data_path() / "image_and_contour.png")).convert("RGB")
test_image_as_tensor = to_tensor(test_pil_image).unsqueeze(0)  # put in a [1, C, H, W] format


def test_add_gaussian_noise() -> None:
    """
    Tests functionality of add gaussian noise
    """
    # Test case of image with 1 channel, 1 slice (2D)
    torch.manual_seed(10)
    transformed = AddGaussianNoise(std=0.05, p_apply=1)(test_tensor_1channel_1slice.clone())
    torch.manual_seed(10)
    noise = torch.randn(size=(1, *image_size)) * 0.05
    assert torch.isclose(torch.clamp(test_tensor_1channel_1slice + noise, min_test_image, max_test_image),
                         # type: ignore
                         transformed).all()

    # Test p_apply = 0
    untransformed = AddGaussianNoise(std=0.05, p_apply=0)(test_tensor_1channel_1slice.clone())
    assert torch.isclose(untransformed, test_tensor_1channel_1slice).all()

    # Check that it applies the same transform to all slices if number of slices > 1
    torch.manual_seed(10)
    transformed = AddGaussianNoise(std=0.05, p_apply=1)(test_tensor_2channels_2slices.clone())
    assert torch.isclose(torch.clamp(test_tensor_2channels_2slices + noise, min_test_image, max_test_image),
                         # type: ignore
                         transformed).all()


def test_elastic_transform() -> None:
    """
    Tests elastic transform
    """
    np.random.seed(7)
    transformed_image = ElasticTransform(sigma=4, alpha=34, p_apply=1.0)(test_image_as_tensor.clone())
    transformed_pil = to_pil_image(transformed_image.squeeze(0))
    expected_pil_image = Image.open(full_ml_test_data_path() / "elastic_transformed_image_and_contour.png").convert(
        "RGB")
    assert expected_pil_image == transformed_pil
    untransformed_image = ElasticTransform(sigma=4, alpha=34, p_apply=0.0)(test_image_as_tensor.clone())
    assert torch.isclose(test_image_as_tensor, untransformed_image).all()


def test_expand_channels() -> None:
    with pytest.raises(ValueError):
        ExpandChannels()(invalid_test_tensor)

    tensor_img = ExpandChannels()(test_tensor_1channel_1slice.clone())
    assert tensor_img.shape == torch.Size([1, 3, *image_size])


def test_random_gamma() -> None:
    # This is invalid input (expects 4 dimensions)
    with pytest.raises(ValueError):
        RandomGamma(scale=(0.3, 3))(invalid_test_tensor)

    random.seed(0)
    transformed_1 = RandomGamma(scale=(0.3, 3))(test_tensor_1channel_1slice.clone())
    assert transformed_1.shape == test_tensor_1channel_1slice.shape

    tensor_img = torch.ones([2, 3, *image_size])
    transformed_2 = RandomGamma(scale=(0.3, 3))(tensor_img)
    # If you run on 1 channel, 1 Z dimension the gamma transform applied should be the same for all slices.
    assert transformed_2.shape == torch.Size([2, 3, *image_size])
    assert torch.isclose(transformed_2[0], transformed_2[1]).all()
    assert torch.isclose(transformed_2[0, 1], transformed_2[0, 2]).all() and \
           torch.isclose(transformed_2[0, 0], transformed_2[0, 2]).all()

    human_readable_transformed = to_pil_image(RandomGamma(scale=(2, 3))(test_image_as_tensor).squeeze(0))
    expected_pil_image = Image.open(full_ml_test_data_path() / "gamma_transformed_image_and_contour.png").convert("RGB")
    assert expected_pil_image == human_readable_transformed
