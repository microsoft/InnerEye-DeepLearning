#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
from typing import Tuple

import PIL
import pytest
import torch
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    RandomAffine,
    RandomErasing,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from torchvision.transforms.functional import to_tensor

from InnerEye.ML.augmentations.image_transforms import (
    AddGaussianNoise,
    ElasticTransform,
    ExpandChannels,
    RandomGamma,
)
from InnerEye.ML.augmentations.transform_pipeline import (
    ImageTransformationPipeline,
    create_transforms_from_config,
)

from Tests.SSL.test_data_modules import cxr_augmentation_config

import numpy as np

image_size = (32, 32)
crop_size = 24


def create_test_images() -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    test_image_as_array = np.ones(list(image_size)) * 255.0
    test_image_as_array[10:15, 10:20] = 1
    image_as_pil = PIL.Image.fromarray(test_image_as_array).convert("L")
    image_2d_as_CHW_tensor = to_tensor(test_image_as_array)
    image_2d_as_ZCHW_tensor = image_2d_as_CHW_tensor.unsqueeze(0)
    scan_4d_as_tensor = torch.ones([5, 4, *image_size]) * 255.0
    scan_4d_as_tensor[..., 10:15, 10:20] = 1
    return image_as_pil, image_2d_as_CHW_tensor, image_2d_as_ZCHW_tensor, scan_4d_as_tensor


@pytest.mark.parametrize("use_different_transformation_per_channel", [True, False])
def test_torchvision_on_various_input(
        use_different_transformation_per_channel: bool,
) -> None:
    """
    This tests that we can run transformation pipeline with out of the box torchvision transforms on various types
    of input: PIL image, 3D tensor, 4D tensors. Tests that use_different_transformation_per_channel has the correct
    behavior.
    """
    image_as_pil, image_2d_as_CHW_tensor, image_2d_as_ZCHW_tensor, scan_4d_as_tensor = create_test_images()
    transform = ImageTransformationPipeline(
        [
            CenterCrop(crop_size),
            RandomErasing(),
            RandomAffine(degrees=(10, 12), shear=15, translate=(0.1, 0.3)),
        ],
        use_different_transformation_per_channel,
    )

    # Test PIL image input
    transformed = transform(image_as_pil)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == torch.Size([1, crop_size, crop_size])

    # Test image as [C, H. W] tensor
    transformed = transform(image_2d_as_CHW_tensor.clone())
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == torch.Size([1, crop_size, crop_size])

    # Test image as [1, 1, H, W]
    transformed = transform(image_2d_as_ZCHW_tensor)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == torch.Size([1, 1, crop_size, crop_size])

    # Test with a fake 4D scan [C, Z, H, W] -> [25, 34, 32, 32]
    transformed = transform(scan_4d_as_tensor)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == torch.Size([5, 4, crop_size, crop_size])

    # Same transformation should be applied to all slices and channels.
    assert (
            torch.isclose(transformed[0, 0], transformed[1, 1]).all()
            != use_different_transformation_per_channel
    )


@pytest.mark.parametrize("use_different_transformation_per_channel", [True, False])
def test_custom_tf_on_various_input(
        use_different_transformation_per_channel: bool,
) -> None:
    """
    This tests that we can run transformation pipeline with our custom transforms on various types
    of input: PIL image, 3D tensor, 4D tensors. Tests that use_different_transformation_per_channel has the correct
    behavior. The transforms are test individually in test_image_transforms.py
    """
    image_as_pil, image_2d_as_CHW_tensor, image_2d_as_ZCHW_tensor, scan_4d_as_tensor = create_test_images()
    pipeline = ImageTransformationPipeline(
        [
            ElasticTransform(sigma=4, alpha=34, p_apply=1),
            AddGaussianNoise(p_apply=1, std=0.05),
            RandomGamma(scale=(0.3, 3)),
        ],
        use_different_transformation_per_channel,
    )

    # Test PIL image input
    transformed = pipeline(image_as_pil)
    assert transformed.shape == image_2d_as_CHW_tensor.shape

    # Test image as [C, H, W] tensor
    pipeline(image_2d_as_CHW_tensor)
    assert transformed.shape == image_2d_as_CHW_tensor.shape

    # Test image as [1, 1, H, W]
    transformed = pipeline(image_2d_as_ZCHW_tensor)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == torch.Size([1, 1, *image_size])

    # Test with a fake scan [C, Z, H, W] -> [25, 34, 32, 32]
    transformed = pipeline(scan_4d_as_tensor)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == scan_4d_as_tensor.shape

    # Same transformation should be applied to all slices and channels.
    assert (
            torch.isclose(transformed[0, 0], transformed[1, 1]).all()
            != use_different_transformation_per_channel
    )


@pytest.mark.parametrize("expand_channels", [True, False])
def test_create_transform_pipeline_from_config(expand_channels: bool) -> None:
    """
    Tests that the pipeline returned by create_transform_pipeline_from_config returns the expected transformation.
    """
    transformation_pipeline = create_transforms_from_config(
        cxr_augmentation_config,
        apply_augmentations=True,
        expand_channels=expand_channels,
    )
    fake_cxr_as_array = np.ones([256, 256]) * 255.0
    fake_cxr_as_array[100:150, 100:200] = 1
    all_transforms = [
        RandomAffine(degrees=180, translate=(0, 0), shear=40),
        RandomResizedCrop(scale=(0.4, 1.0), size=256),
        RandomHorizontalFlip(p=0.5),
        RandomGamma(scale=(0.5, 1.5)),
        ColorJitter(saturation=0, brightness=0.2, contrast=0.2),
        ElasticTransform(sigma=4, alpha=34, p_apply=0.4),
        CenterCrop(size=224),
        RandomErasing(scale=(0.15, 0.4), ratio=(0.33, 3)),
        AddGaussianNoise(std=0.05, p_apply=0.5),
    ]

    if expand_channels:
        all_transforms.insert(0, ExpandChannels())
        # expand channels is used for single-channel input images
        fake_image = PIL.Image.fromarray(fake_cxr_as_array).convert("L")
        # In the pipeline the image is converted to tensor before applying the transformations. Do the same here.
        image = ToTensor()(fake_image).reshape([1, 1, 256, 256])
    else:
        fake_3d_array = np.dstack(
            [fake_cxr_as_array, fake_cxr_as_array, fake_cxr_as_array]
        )
        fake_image = PIL.Image.fromarray(fake_3d_array.astype(np.uint8)).convert("RGB")
        # In the pipeline the image is converted to tensor before applying the transformations. Do the same here.
        image = ToTensor()(fake_image).reshape([1, 3, 256, 256])

    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    transformed_image = transformation_pipeline(fake_image)
    assert isinstance(transformed_image, torch.Tensor)

    # Expected pipeline
    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    expected_transformed = image
    for t in all_transforms:
        expected_transformed = t(expected_transformed)
    # The pipeline takes as input [C, Z, H, W] and returns [C, Z, H, W]
    # But the transforms list expect [Z, C, H, W] and returns [Z, C, H, W] so need to permute dimension to compare
    expected_transformed = torch.transpose(expected_transformed, 1, 0).squeeze(1)
    assert torch.isclose(expected_transformed, transformed_image).all()

    # Test the evaluation pipeline
    transformation_pipeline = create_transforms_from_config(
        cxr_augmentation_config,
        apply_augmentations=False,
        expand_channels=expand_channels,
    )
    transformed_image = transformation_pipeline(image)
    assert isinstance(transformed_image, torch.Tensor)
    all_transforms = [Resize(size=256), CenterCrop(size=224)]
    if expand_channels:
        all_transforms.insert(0, ExpandChannels())

    expected_transformed = image
    for t in all_transforms:
        expected_transformed = t(expected_transformed)
    expected_transformed = torch.transpose(expected_transformed, 1, 0).squeeze(1)
    assert torch.isclose(expected_transformed, transformed_image).all()
