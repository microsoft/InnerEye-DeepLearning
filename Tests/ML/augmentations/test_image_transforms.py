#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np

import torch
import torchvision
from scipy.ndimage import gaussian_filter, map_coordinates
from torchvision.transforms.functional import to_pil_image

from InnerEye.ML.augmentations.image_transforms import AddGaussianNoise, CenterCrop, ElasticTransform, ExpandChannels, \
    ImageTransformBase, RandomAffine, RandomColorJitter, RandomHorizontalFlip

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


def test_center_crop():
    transformation = CenterCrop(center_crop_size=224)
    transformed_image = np.asarray(transformation(test_pil_image))
    assert transformed_image.shape == (224, 224)


def _check_transformation_result(image_as_tensor: torch.Tensor,
                                 transformation: ImageTransformBase,
                                 expected: torch.Tensor) -> None:
    test_tensor_pil = torchvision.transforms.functional.to_pil_image(image_as_tensor)
    transformed = torchvision.transforms.functional.to_tensor(transformation(test_tensor_pil)).squeeze()
    assert torch.isclose(transformed, expected, rtol=0.02).all()


def test_affine_transformation():
    test_image = torch.tensor([[2, 1, 3],
                               [1, 2, 3],
                               [3, 3, 2]], dtype=torch.int32)
    expected_result = torch.tensor([[1, 2, 1],
                                    [3, 2, 3],
                                    [3, 2, 3]], dtype=torch.int32)
    transformation = RandomAffine(max_angle=180)
    torch.random.manual_seed(2)
    transformation.draw_transform(test_image.shape)
    _check_transformation_result(test_image, transformation, expected_result)

    expected_result = torch.tensor([[0, 2, 1],
                                    [0, 1, 2],
                                    [0, 3, 3]], dtype=torch.int32)
    transformation = RandomAffine(max_horizontal_shift=1.0)
    transformation.draw_transform(test_image.shape)
    _check_transformation_result(test_image, transformation, expected_result)

    torch.random.manual_seed(4)
    expected_result = torch.tensor([[3, 3, 2],
                                    [0, 0, 0],
                                    [0, 0, 0]], dtype=torch.int32)
    transformation = RandomAffine(max_vertical_shift=1.0)
    transformation.draw_transform(test_image.shape)
    _check_transformation_result(test_image, transformation, expected_result)


def test_random_horizontal_flip() -> None:
    """
    Tests each individual transformation of the ImageTransformationPipeline class on a 2D input representing
    a natural image.
    """
    test_image = torch.tensor([[1, 0.5, 0.1],
                               [0.5, 1, 0.1],
                               [0.1, 0.1, 1]], dtype=torch.float32)
    expected = torch.tensor([[0.1, 0.5, 1],
                             [0.1, 1, 0.5],
                             [1, 0.1, 0.1]], dtype=torch.float32)
    transformation = RandomHorizontalFlip(p_apply=1.0)
    transformation.draw_transform(test_image.shape)
    _check_transformation_result(test_image, transformation, expected)
    transformation = RandomHorizontalFlip(p_apply=0.0)
    transformation.draw_transform(test_image.shape)
    _check_transformation_result(test_image, transformation, test_image)


def test_random_color_jitter() -> None:
    test_image = torch.tensor([[1, 0.5, 0.1],
                               [0.5, 1, 0.1],
                               [0.1, 0.1, 1]], dtype=torch.float32)
    expected = torch.tensor([[0.8510, 0.4235, 0.0824],
                             [0.4235, 0.8510, 0.0824],
                             [0.0824, 0.0824, 0.8510]])
    torch.manual_seed(0)
    transformation = RandomColorJitter(max_brightness=0.2)
    transformation.draw_transform(test_image.shape)
    _check_transformation_result(test_image, transformation, expected)

    expected = torch.tensor([[1.0000, 0.4980, 0.0353],
                             [0.4980, 1.0000, 0.0353],
                             [0.0353, 0.0353, 1.0000]])
    transformation = RandomColorJitter(max_contrast=0.2)
    transformation.draw_transform(test_image.shape)
    _check_transformation_result(test_image, transformation, expected)
