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

# TODO
# Add a test for initialization of pipeline directly from a list of configs. Same as below but simpler maybe.
# Need a test with actually 4D inputs.
# Need a test for RBG images and join_channel transforms
# Need to test throws an error if use_joint_transform and the image is not RGB or 1-channel
# Need to test if use same transform for all channels and use one transform per channel works as expected

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
