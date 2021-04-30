import numpy as np
import pytest
import torch

from InnerEye.ML.SSL.datamodules_and_datasets.transforms_utils import AddGaussianNoise
from Tests.SSL.test_data_modules import cxr_augmentation_config


def test_add_gaussian_noise():
    np.random.seed(1)
    torch.manual_seed(10)
    array = np.ones([256, 256]) * 255.
    array[100:150, 100:200] = 1
    tensor_img = torch.tensor(array / 255.)
    transformed = AddGaussianNoise(cxr_augmentation_config)(tensor_img)
    torch.manual_seed(10)
    noise = torch.randn(size=(256, 256)) * 0.05
    assert torch.isclose(torch.clamp(tensor_img + noise, 0, 1), transformed, atol=1e-6).all()
    with pytest.raises(AssertionError):
        AddGaussianNoise(cxr_augmentation_config)(tensor_img * 255.)
