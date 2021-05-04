import random

import PIL
import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter, map_coordinates
from torchvision.transforms import ToTensor

from InnerEye.ML.SSL.datamodules_and_datasets.transforms_utils import AddGaussianNoise, CenterCrop, ElasticTransform, \
    ExpandChannels, RandomAffine, RandomColorJitter, RandomErasing, RandomGamma, RandomHorizontalFlip, RandomResizeCrop, \
    Resize, create_chest_xray_transform
from Tests.SSL.test_data_modules import cxr_augmentation_config


def test_add_gaussian_noise() -> None:
    """
    Tests functionality of add gaussian noise
    """
    np.random.seed(1)
    torch.manual_seed(10)
    array = np.ones([1, 256, 256]) * 255.
    array[0, 100:150, 100:200] = 1
    tensor_img = torch.tensor(array / 255.)
    transformed = AddGaussianNoise(cxr_augmentation_config)(tensor_img)
    torch.manual_seed(10)
    noise = torch.randn(size=(1, 256, 256)) * 0.05
    assert torch.isclose(torch.clamp(tensor_img + noise, 0, 1), transformed).all()
    with pytest.raises(AssertionError):
        AddGaussianNoise(cxr_augmentation_config)(tensor_img * 255.)


def test_elastic_transform() -> None:
    """
    Tests elastic transform
    """
    image = np.ones([256, 256]) * 255.
    image[100:150, 100:200] = 1

    # Computed expected transform
    np.random.seed(7)
    np.random.random(1)

    shape = (256, 256)
    dx = gaussian_filter((np.random.random(shape) * 2 - 1), 4, mode="constant", cval=0) * 34
    dy = gaussian_filter((np.random.random(shape) * 2 - 1), 4, mode="constant", cval=0) * 34
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
    expected_array = map_coordinates(image, indices, order=1).reshape(shape)
    # Actual transform
    np.random.seed(7)
    transformed_image = np.asarray(ElasticTransform(cxr_augmentation_config)(PIL.Image.fromarray(image)))
    assert np.isclose(expected_array, transformed_image).all()


def test_expand_channels() -> None:
    image = np.ones([1, 256, 256]) * 255.
    tensor_img = torch.tensor(image)
    tensor_img = ExpandChannels()(tensor_img)
    assert tensor_img.shape == torch.Size([3, 256, 256])
    assert torch.isclose(tensor_img[0], tensor_img[1]).all() and torch.isclose(tensor_img[1], tensor_img[2]).all()


def test_create_chest_xray_transform() -> None:
    """
    Tests that the pipeline returned by create_chest_xray_transform returns the expected transformation.
    """
    transform = create_chest_xray_transform(cxr_augmentation_config, apply_augmentations=True)
    image = np.ones([256, 256]) * 255.
    image[100:150, 100:200] = 1
    image = PIL.Image.fromarray(image).convert("L")
    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    transformed_image = transform(image)

    # Expected pipeline
    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    image = RandomAffine(cxr_augmentation_config)(image)
    image = RandomResizeCrop(cxr_augmentation_config)(image)
    image = Resize(cxr_augmentation_config)(image)
    image = RandomHorizontalFlip(cxr_augmentation_config)(image)
    image = RandomGamma(cxr_augmentation_config)(image)
    image = RandomColorJitter(cxr_augmentation_config)(image)
    image = ElasticTransform(cxr_augmentation_config)(image)
    image = CenterCrop(cxr_augmentation_config)(image)
    image = ToTensor()(image)
    image = RandomErasing(cxr_augmentation_config)(image)
    image = AddGaussianNoise(cxr_augmentation_config)(image)
    image = ExpandChannels()(image)

    assert torch.isclose(image, transformed_image).all()

    # Test the evaluation pipeline
    transform = create_chest_xray_transform(cxr_augmentation_config, apply_augmentations=False)
    image = np.ones([256, 256]) * 255.
    image[100:150, 100:200] = 1
    image = PIL.Image.fromarray(image).convert("L")
    transformed_image = transform(image)

    # Expected pipeline
    image = Resize(cxr_augmentation_config)(image)
    image = CenterCrop(cxr_augmentation_config)(image)
    image = ToTensor()(image)
    image = ExpandChannels()(image)
    assert torch.isclose(image, transformed_image).all()
