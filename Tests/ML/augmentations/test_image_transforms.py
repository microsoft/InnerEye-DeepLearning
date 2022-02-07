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


@pytest.fixture(scope="module")
def tensor_1channel_1slice() -> torch.Tensor:
    torch.manual_seed(10)
    t = torch.ones([1, 1, *image_size], dtype=torch.float)
    t[..., 100:150, 100:200] = 1 / 255.
    return t


@pytest.fixture(scope="module")
def tensor_2channels_2slices() -> torch.Tensor:
    torch.manual_seed(10)
    t = torch.ones([2, 2, *image_size], dtype=torch.float)
    t[..., 100:150, 100:200] = 1 / 255.
    return t


@pytest.fixture(scope="module")
def image_as_tensor() -> torch.Tensor:
    test_pil_image = Image.open(str(full_ml_test_data_path() / "image_and_contour.png")).convert("RGB")
    return to_tensor(test_pil_image).unsqueeze(0)  # put in a [1, C, H, W] format


@pytest.fixture(scope="module")
def invalid_test_tensor() -> torch.Tensor:
    return torch.ones([1, *image_size])


def test_add_gaussian_noise(tensor_1channel_1slice: torch.Tensor,
                            tensor_2channels_2slices: torch.Tensor) -> None:
    """
    Tests functionality of add gaussian noise
    """
    min_test_image, max_test_image = tensor_1channel_1slice.min(), tensor_1channel_1slice.max()
    # Test case of image with 1 channel, 1 slice (2D)
    torch.manual_seed(10)
    transformed = AddGaussianNoise(std=0.05, p_apply=1)(tensor_1channel_1slice)
    torch.manual_seed(10)
    noise = torch.randn(size=(1, *image_size)) * 0.05
    assert torch.isclose(
        torch.clamp(tensor_1channel_1slice + noise, min_test_image, max_test_image),  # type: ignore
        transformed).all()

    # Test p_apply = 0
    untransformed = AddGaussianNoise(std=0.05, p_apply=0)(tensor_1channel_1slice)
    assert torch.isclose(untransformed, tensor_1channel_1slice).all()

    # Check that it applies the same transform to all slices if number of slices > 1
    torch.manual_seed(10)
    transformed = AddGaussianNoise(std=0.05, p_apply=1)(tensor_2channels_2slices)
    assert torch.isclose(
        torch.clamp(tensor_2channels_2slices + noise, min_test_image, max_test_image),  # type: ignore
        transformed).all()


def test_elastic_transform(image_as_tensor: torch.Tensor) -> None:
    """
    Tests elastic transform
    """
    np.random.seed(7)
    transformed_image = ElasticTransform(sigma=4, alpha=34, p_apply=1.0)(image_as_tensor)
    transformed_pil = to_pil_image(transformed_image.squeeze(0))
    expected_pil_image = Image.open(full_ml_test_data_path() / "elastic_transformed_image_and_contour.png").convert(
        "RGB")
    assert expected_pil_image == transformed_pil
    untransformed_image = ElasticTransform(sigma=4, alpha=34, p_apply=0.0)(image_as_tensor)
    assert torch.isclose(image_as_tensor, untransformed_image).all()


def test_invalid_tensors(invalid_test_tensor: torch.Tensor) -> None:
    # This is invalid input (expects 4 dimensions)
    with pytest.raises(ValueError):
        ExpandChannels()(invalid_test_tensor)
    with pytest.raises(ValueError):
        RandomGamma(scale=(0.3, 3))(invalid_test_tensor)


def test_expand_channels(tensor_1channel_1slice: torch.Tensor) -> None:
    tensor_img = ExpandChannels()(tensor_1channel_1slice)
    assert tensor_img.shape == torch.Size([1, 3, *image_size])


def test_random_gamma(tensor_1channel_1slice: torch.Tensor) -> None:
    random.seed(0)
    transformed_1 = RandomGamma(scale=(0.3, 3))(tensor_1channel_1slice.clone())
    assert transformed_1.shape == tensor_1channel_1slice.shape

    tensor_img = torch.ones([2, 3, *image_size])
    transformed_2 = RandomGamma(scale=(0.3, 3))(tensor_img)
    # If you run on 1 channel, 1 Z dimension the gamma transform applied should be the same for all slices.
    assert transformed_2.shape == torch.Size([2, 3, *image_size])
    assert torch.isclose(transformed_2[0], transformed_2[1]).all()
    assert torch.isclose(transformed_2[0, 1], transformed_2[0, 2]).all() and \
           torch.isclose(transformed_2[0, 0], transformed_2[0, 2]).all()


def test_random_gamma_image(image_as_tensor: torch.Tensor) -> None:
    human_readable_transformed = to_pil_image(RandomGamma(scale=(2, 3))(image_as_tensor).squeeze(0))
    expected_pil_image = Image.open(full_ml_test_data_path() / "gamma_transformed_image_and_contour.png").convert("RGB")
    assert expected_pil_image == human_readable_transformed
