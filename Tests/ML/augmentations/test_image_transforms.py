#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random

import numpy as np
import pytest

import torch

from scipy.ndimage import gaussian_filter, map_coordinates

from InnerEye.ML.augmentations.image_transforms import AddGaussianNoise, ElasticTransform, ExpandChannels, RandomGamma


def test_add_gaussian_noise() -> None:
    """
    Tests functionality of add gaussian noise
    """
    np.random.seed(1)
    tensor_img = torch.ones([1, 1, 256, 256], dtype=torch.float)
    tensor_img[..., 100:150, 100:200] = 1 / 255.
    # Input tensor [C, Z, H, W]
    torch.manual_seed(10)
    transformed = AddGaussianNoise(std=0.05, p_apply=1)(tensor_img)
    torch.manual_seed(10)
    noise = torch.randn(size=(1, 256, 256)) * 0.05
    assert torch.isclose(torch.clamp(tensor_img + noise, tensor_img.min(), tensor_img.max()),  # type: ignore
                         transformed).all()
    # Check that it applies the same transform to all slices
    tensor_img = torch.ones([2, 2, 256, 256], dtype=torch.float)
    tensor_img[..., 100:150, 100:200] = 1 / 255.
    # Input tensor [C, Z, H, W]
    torch.manual_seed(10)
    transformed = AddGaussianNoise(std=0.05, p_apply=1)(tensor_img)
    torch.manual_seed(10)
    noise = torch.randn(size=(1, 256, 256)) * 0.05
    assert torch.isclose(torch.clamp(tensor_img + noise, tensor_img.min(), tensor_img.max()),  # type: ignore
                         transformed).all()


def test_elastic_transform() -> None:
    """
    Tests elastic transform
    """
    image = torch.ones([2, 2, 256, 256]) * 255.
    image[..., 100:150, 100:200] = 1

    # Computed expected transform
    np.random.seed(7)
    np.random.random(1)

    shape = (256, 256)
    dx = gaussian_filter((np.random.random(shape) * 2 - 1), 4, mode="constant", cval=0) * 34
    dy = gaussian_filter((np.random.random(shape) * 2 - 1), 4, mode="constant", cval=0) * 34
    all_dimensions_axes = [np.arange(dim) for dim in image.shape]
    grid = np.meshgrid(*all_dimensions_axes, indexing='ij')
    grid[-2] = grid[-2] + dx
    grid[-1] = grid[-1] + dy
    indices = [np.reshape(grid[i], (-1, 1)) for i in range(len(grid))]
    expected_tensor = torch.tensor(map_coordinates(image, indices, order=1).reshape(image.shape))
    # Actual transform
    np.random.seed(7)
    transformed_image = ElasticTransform(sigma=4, alpha=34, p_apply=1.0)(image)
    assert torch.isclose(expected_tensor, transformed_image).all()


def test_expand_channels() -> None:
    tensor_img = torch.ones([1, 256, 256])
    with pytest.raises(ValueError):
        ExpandChannels()(tensor_img)

    tensor_img = torch.ones([1, 1, 256, 256])
    tensor_img = ExpandChannels()(tensor_img)
    assert tensor_img.shape == torch.Size([1, 3, 256, 256])


def test_random_gamma() -> None:
    # This is invalid input (expects 4 dimensions)
    tensor_img = torch.ones([1, 256, 256])
    with pytest.raises(ValueError):
        RandomGamma(scale=(0.3, 3))(tensor_img)

    random.seed(0)
    tensor_img = torch.ones([1, 1, 256, 256])
    transformed_1 = RandomGamma(scale=(0.3, 3))(tensor_img)
    assert transformed_1.shape == torch.Size([1, 1, 256, 256])

    random.seed(0)
    tensor_img = torch.ones([2, 2, 256, 256])
    transformed_2 = RandomGamma(scale=(0.3, 3))(tensor_img)
    # If you run on 1 channel, 1 Z dimension the gamma transform applied should be the same for all slices.
    assert transformed_2.shape == torch.Size([2, 2, 256, 256])
    assert torch.isclose(transformed_2[0], transformed_2[1]).all()
