#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random

import PIL
import torch

from InnerEye.ML.augmentations.image_transforms import AddGaussianNoise, CenterCrop, ElasticTransform, ExpandChannels, \
    RandomAffine, \
    RandomColorJitter, \
    RandomErasing, RandomGamma, \
    RandomHorizontalFlip, \
    RandomResizeCrop, Resize, ToTensor
from InnerEye.ML.augmentations.transform_pipeline import create_transform_pipeline_from_config
from Tests.SSL.test_data_modules import cxr_augmentation_config

import numpy as np


def test_create_transform_pipeline() -> None:
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
    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    all_transforms = [RandomAffine(cxr_augmentation_config),
                      RandomResizeCrop(cxr_augmentation_config),
                      Resize(cxr_augmentation_config),
                      RandomHorizontalFlip(cxr_augmentation_config),
                      RandomGamma(cxr_augmentation_config),
                      RandomColorJitter(cxr_augmentation_config),
                      ElasticTransform(cxr_augmentation_config),
                      CenterCrop(cxr_augmentation_config),
                      ToTensor(),
                      RandomErasing(cxr_augmentation_config),
                      AddGaussianNoise(cxr_augmentation_config),
                      ExpandChannels()]
    input_size = [256, 256]
    for t in all_transforms:
        input_size = t.draw_transform(input_size)
    expected_transformed = image
    for t in all_transforms:
        expected_transformed = t(expected_transformed)

    assert torch.isclose(expected_transformed, transformed_image).all()

    # Test the evaluation pipeline
    transformation_pipeline = create_transform_pipeline_from_config(cxr_augmentation_config, apply_augmentations=False)
    transformed_image = transformation_pipeline(image)
    all_transforms = [Resize(cxr_augmentation_config),
                      CenterCrop(cxr_augmentation_config),
                      ToTensor(),
                      ExpandChannels()]
    for t in all_transforms:
        input_size = t.draw_transform(input_size)
    expected_transformed = image
    for t in all_transforms:
        expected_transformed = t(expected_transformed)
    assert torch.isclose(expected_transformed, transformed_image).all()
