#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np

import torch
from scipy.ndimage import gaussian_filter, map_coordinates
from torchvision.transforms.functional import to_pil_image

from InnerEye.ML.augmentations.image_transforms import AddGaussianNoise, ElasticTransform, ExpandChannels

input_size = [1, 256, 256]
array = np.ones(input_size) * 255.
array[0, 100:150, 100:200] = 1
test_img_as_tensor = torch.tensor(array / 255.)
test_pil_image = to_pil_image(test_img_as_tensor)


def test_add_gaussian_noise() -> None:
    """
    Tests functionality of add gaussian noise
    """
    transformation = AddGaussianNoise(p_apply=1, std=0.1)
    transformation.draw_transform(input_size)
    transformed = transformation(test_img_as_tensor)
    assert torch.isclose(torch.clamp(test_img_as_tensor + transformation.noise, 0, 1), transformed).all()


def test_elastic_transform() -> None:
    """
    Tests elastic transform
    """
    sigma = 4
    alpha = 34
    transformation = ElasticTransform(p_apply=1, sigma=sigma, alpha=alpha)
    transformation.draw_transform(input_size)
    transformation.apply = 1
    img_array = np.asarray(test_pil_image).squeeze()
    assert img_array.shape == (256, 256)
    x, y = np.meshgrid(np.arange(256), np.arange(256), indexing='ij')
    dx = gaussian_filter(transformation.dx_pertubation, sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(transformation.dy_pertubation, sigma, mode="constant", cval=0) * alpha
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
    expected_array = map_coordinates(img_array, indices, order=1).reshape((256, 256))
    transformed_image = np.asarray(transformation(test_pil_image))
    assert np.isclose(expected_array, transformed_image).all()


def test_expand_channels() -> None:
    transformation = ExpandChannels()
    # Has not effect but should not fail
    transformation.draw_transform(input_size)
    transformed = transformation(test_img_as_tensor)
    assert transformed.shape == torch.Size([3, 256, 256])
    assert torch.isclose(transformed[0], transformed[1]).all() and torch.isclose(transformed[1], transformed[2]).all()
