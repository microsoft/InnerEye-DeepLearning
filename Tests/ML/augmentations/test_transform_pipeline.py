#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random

import PIL
import pytest
import torch
from torchvision.transforms import CenterCrop, ColorJitter, Compose, RandomAffine, RandomErasing, ToTensor
from torchvision.transforms.functional import to_tensor

from InnerEye.ML.augmentations.image_transforms import AddGaussianNoise, ElasticTransform, RandomGamma
from InnerEye.ML.augmentations.transform_pipeline import ImageTransformationPipeline
# create_transform_pipeline_from_config
from Tests.SSL.test_data_modules import cxr_augmentation_config

import numpy as np


@pytest.mark.parametrize("use_different_transformation_per_channel", [True, False])
def test_torchvision_on_various_input(use_different_transformation_per_channel: bool) -> None:
    """
    This tests that we can run transformation pipeline with out of the box torchvision transforms on various types
    of input: PIL image, 3D tensor, 4D tensors. Tests that use_different_transformation_per_channel has the correct
    behavior.
    """
    image = np.ones([256, 256]) * 255.
    image[100:150, 100:200] = 1
    transform = ImageTransformationPipeline(
        [CenterCrop(224),
         RandomErasing(),
         ColorJitter(),
         RandomAffine(degrees=(10, 12), shear=15, translate=(0.1, 0.3))
         ],
        use_different_transformation_per_channel)

    # Test PIL image input
    image = PIL.Image.fromarray(image).convert("L")
    transform(image)

    # Test image as [C, H. W] tensor
    image_as_tensor = to_tensor(image)
    transform(image_as_tensor)

    # Test image as [1, 1, H, W]
    image_as_tensor = image_as_tensor.unsqueeze(0)
    assert image_as_tensor.shape == torch.Size([1, 1, 256, 256])
    assert transform(image_as_tensor).shape == torch.Size([1, 1, 224, 224])

    # Test with a fake scan [C, Z, H, W] -> [25, 34, 256, 256]
    test_4d_tensor = torch.ones([25, 34, 256, 256]) * 255.
    test_4d_tensor[..., 100:150, 100:200] = 1
    tf = transform(test_4d_tensor)
    assert tf.shape == torch.Size([25, 34, 224, 224])

    # Same transformation should be applied to all slices and channels.
    assert torch.isclose(tf[0, 0], tf[1, 1]).all() != use_different_transformation_per_channel


@pytest.mark.parametrize("use_different_transformation_per_channel", [True, False])
def test_custom_tf_on_various_input(use_different_transformation_per_channel: bool) -> None:
    """
    This tests that we can run transformation pipeline with our custom transforms on various types
    of input: PIL image, 3D tensor, 4D tensors. Tests that use_different_transformation_per_channel has the correct
    behavior. The transforms are test individually in test_image_transforms.py
    """
    image = np.ones([256, 256]) * 255.
    image[100:150, 100:200] = 1
    pipeline = ImageTransformationPipeline(
        [ElasticTransform(sigma=4, alpha=34, p_apply=1),
         AddGaussianNoise(p_apply=1, std=0.05),
         RandomGamma(scale=(0.3, 3))
         ],
        use_different_transformation_per_channel)

    # Test PIL image input
    image = PIL.Image.fromarray(image).convert("L")
    pipeline(image)

    # Test image as [C, H, W] tensor
    image_as_tensor = to_tensor(image)
    pipeline(image_as_tensor)

    # Test image as [1, 1, H, W]
    image_as_tensor = image_as_tensor.unsqueeze(0)
    assert image_as_tensor.shape == torch.Size([1, 1, 256, 256])
    assert pipeline(image_as_tensor).shape == torch.Size([1, 1, 256, 256])

    # Test with a fake scan [C, Z, H, W] -> [25, 34, 256, 256]
    test_4d_tensor = torch.ones([25, 34, 256, 256]) * 255.
    test_4d_tensor[..., 100:150, 100:200] = 1
    tf = pipeline(test_4d_tensor)
    assert tf.shape == torch.Size([25, 34, 256, 256])

    # Same transformation should be applied to all slices and channels.
    assert torch.isclose(tf[0, 0], tf[1, 1]).all() != use_different_transformation_per_channel


def test_create_transform_pipeline_from_config() -> None:
    """
    Tests that the pipeline returned by create_transform_pipeline_from_config returns the expected transformation.
    """
    transformation_pipeline = create_transform_pipeline_from_config(cxr_augmentation_config, apply_augmentations=True)
    image = np.ones([256, 256]) * 255.
    image[100:150, 100:200] = 1
    image = PIL.Image.fromarray(image).convert("L")

    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    transformed_image = transformation_pipeline(image)

    # Expected pipeline
    image = np.ones([256, 256]) * 255.
    image[100:150, 100:200] = 1
    image = PIL.Image.fromarray(image).convert("L")

    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    all_transforms = [RandomAffine(max_angle=180,
                                   max_horizontal_shift=0,
                                   max_vertical_shift=0,
                                   max_shear=40),
                      RandomResizeCrop(random_crop_scale=(0.4, 1.0),
                                       resize_size=256),
                      RandomHorizontalFlip(p_apply=0.5),
                      RandomGamma(scale=(0.5, 1.5)),
                      RandomColorJitter(max_saturation=0,
                                        max_brightness=0.2,
                                        max_contrast=0.2),
                      ElasticTransform(sigma=4, alpha=34, p_apply=0.4),
                      CenterCrop(center_crop_size=224),
                      ToTensor(),
                      RandomErasing(scale=(0.4, 1.0), ratio=(0.3, 3.3)),
                      AddGaussianNoise(std=0.05, p_apply=0.5),
                      ExpandChannels()]
    input_size = [1, 256, 256]
    for t in all_transforms:
        input_size = t.draw_transform(input_size)
    expected_transformed = image
    for t in all_transforms:
        expected_transformed = t(expected_transformed)

    assert torch.isclose(expected_transformed, transformed_image).all()

    # Test the evaluation pipeline
    transformation_pipeline = create_transform_pipeline_from_config(cxr_augmentation_config, apply_augmentations=False)
    transformed_image = transformation_pipeline(image)
    all_transforms = [Resize(resize_size=256),
                      CenterCrop(center_crop_size=224),
                      ToTensor(),
                      ExpandChannels()]
    for t in all_transforms:
        input_size = t.draw_transform(input_size)
    expected_transformed = image
    for t in all_transforms:
        expected_transformed = t(expected_transformed)
    assert torch.isclose(expected_transformed, transformed_image).all()
