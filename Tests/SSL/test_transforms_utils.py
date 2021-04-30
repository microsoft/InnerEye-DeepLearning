import PIL
import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter, map_coordinates

from InnerEye.ML.SSL.datamodules_and_datasets.transforms_utils import AddGaussianNoise, ElasticTransform
from Tests.SSL.test_data_modules import cxr_augmentation_config


def test_add_gaussian_noise() -> None:
    """
    Tests functionality of add gaussian noise
    """
    np.random.seed(1)
    torch.manual_seed(10)
    array = np.ones([256, 256]) * 255.
    array[100:150, 100:200] = 1
    tensor_img = torch.tensor(array / 255.)
    transformed = AddGaussianNoise(cxr_augmentation_config)(tensor_img)
    torch.manual_seed(10)
    noise = torch.randn(size=(256, 256)) * 0.05
    assert torch.isclose(torch.clamp(tensor_img + noise, 0, 1), transformed).all()
    with pytest.raises(AssertionError):
        AddGaussianNoise(cxr_augmentation_config)(tensor_img * 255.)


def test_elastic_transform():
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
