#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
from typing import Any, Callable, List

import numpy as np
import pytest
import torch
from torchvision.transforms import functional as TF

from InnerEye.ML.dataset.sample import Sample
from InnerEye.ML.utils import augmentation, ml_util
from InnerEye.ML.utils.augmentation import ImageTransformationBase
from Tests.ML.util import DummyPatientMetadata

ml_util.set_random_seed(1)

image_size = (8, 8, 8)
valid_image_4d = np.random.uniform(size=((5,) + image_size)) * 10
valid_mask = np.random.randint(2, size=image_size)
number_of_classes = 5
class_assignments = np.random.randint(2, size=image_size)
valid_labels = np.zeros((number_of_classes,) + image_size)
for c in range(number_of_classes):
    valid_labels[c, class_assignments == c] = 1
valid_crop_size = (2, 2, 2)
valid_full_crop_size = image_size
valid_class_weights = [0.5] + [0.5 / (number_of_classes - 1)] * (number_of_classes - 1)
crop_size_requires_padding = (9, 8, 12)


# Random Crop Tests
def test_valid_full_crop() -> None:
    metadata = DummyPatientMetadata
    sample, _ = augmentation.random_crop(sample=Sample(image=valid_image_4d,
                                                       labels=valid_labels,
                                                       mask=valid_mask,
                                                       metadata=metadata),
                                         crop_size=valid_full_crop_size,
                                         class_weights=valid_class_weights)

    assert np.array_equal(sample.image, valid_image_4d)
    assert np.array_equal(sample.labels, valid_labels)
    assert np.array_equal(sample.mask, valid_mask)
    assert sample.metadata == metadata


@pytest.mark.parametrize("image", [None, list(), valid_image_4d])
@pytest.mark.parametrize("labels", [None, list(), valid_labels])
@pytest.mark.parametrize("mask", [None, list(), valid_mask])
@pytest.mark.parametrize("class_weights", [[0, 0, 0], [0], [-1, 0, 1], [-1, -2, -3], valid_class_weights])
def test_invalid_arrays(image: Any, labels: Any, mask: Any, class_weights: Any) -> None:
    """
    Tests failure cases of the random_crop function for invalid image, labels, mask or class
    weights arguments.
    """
    # Skip the final combination, because it is valid
    if not (np.array_equal(image, valid_image_4d) and np.array_equal(labels, valid_labels)
            and np.array_equal(mask, valid_mask) and class_weights == valid_class_weights):
        with pytest.raises(Exception):
            augmentation.random_crop(Sample(metadata=DummyPatientMetadata, image=image, labels=labels, mask=mask),
                                     valid_crop_size, class_weights)


@pytest.mark.parametrize("crop_size", [None, ["a"], 5])
def test_invalid_crop_arg(crop_size: Any) -> None:
    with pytest.raises(Exception):
        augmentation.random_crop(
            Sample(metadata=DummyPatientMetadata, image=valid_image_4d, labels=valid_labels, mask=valid_mask),
            crop_size, valid_class_weights)


@pytest.mark.parametrize("crop_size", [[2, 2], [2, 2, 2, 2], [10, 10, 10]])
def test_invalid_crop_size(crop_size: Any) -> None:
    with pytest.raises(Exception):
        augmentation.random_crop(
            Sample(metadata=DummyPatientMetadata, image=valid_image_4d, labels=valid_labels, mask=valid_mask),
            crop_size, valid_class_weights)


def test_random_crop_no_fg() -> None:
    with pytest.raises(Exception):
        augmentation.random_crop(Sample(metadata=DummyPatientMetadata, image=valid_image_4d, labels=valid_labels,
                                        mask=np.zeros_like(valid_mask)),
                                 valid_crop_size, valid_class_weights)

    with pytest.raises(Exception):
        augmentation.random_crop(Sample(metadata=DummyPatientMetadata, image=valid_image_4d,
                                        labels=np.zeros_like(valid_labels), mask=valid_mask),
                                 valid_crop_size, valid_class_weights)


@pytest.mark.parametrize("crop_size", [valid_crop_size])
def test_random_crop(crop_size: Any) -> None:
    labels = valid_labels
    # create labels such that there are no foreground voxels in a particular class
    # this should ne handled gracefully (class being ignored from sampling)
    labels[0] = 1
    labels[1] = 0
    sample, _, _ = augmentation.random_crop(Sample(
        image=valid_image_4d,
        labels=valid_labels,
        mask=valid_mask,
        metadata=DummyPatientMetadata
    ), crop_size, valid_class_weights)

    expected_img_crop_size = (valid_image_4d.shape[0], *crop_size)
    expected_labels_crop_size = (valid_labels.shape[0], *crop_size)

    assert sample.image.shape == expected_img_crop_size
    assert sample.labels.shape == expected_labels_crop_size
    assert sample.mask.shape == tuple(crop_size)


@pytest.mark.parametrize("class_weights",
                         [None, [0, 0.5, 0.5, 0, 0], [0.1, 0.45, 0.45, 0, 0], [0.5, 0.25, 0.25, 0, 0],
                          [1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0.04, 0.12, 0.20, 0.28, 0.36], [0, 0.5, 0, 0.5, 0]])
def test_valid_class_weights(class_weights: List[float]) -> None:
    """
    Produce a large number of crops and make sure the crop center class proportions respect class weights
    """
    ml_util.set_random_seed(1)
    num_classes = len(valid_labels)
    image = np.zeros_like(valid_image_4d)
    labels = np.zeros_like(valid_labels)
    class0, class1, class2 = non_empty_classes = [0, 2, 4]
    labels[class0] = 1
    labels[class0][3, 3, 3] = 0
    labels[class0][3, 2, 3] = 0
    labels[class1][3, 3, 3] = 1
    labels[class2][3, 2, 3] = 1

    mask = np.ones_like(valid_mask)
    sample = Sample(image=image, labels=labels, mask=mask, metadata=DummyPatientMetadata)

    crop_size = (1, 1, 1)
    total_crops = 200
    sampled_label_center_distribution = np.zeros(num_classes)

    # If there is no class that has a non-zero weight and is present in the sample, there is no valid
    # way to select a class, so we expect an exception to be thrown.
    if class_weights is not None and sum(class_weights[c] for c in non_empty_classes) == 0:
        with pytest.raises(ValueError):
            augmentation.random_crop(sample, crop_size, class_weights)
        return

    for _ in range(0, total_crops):
        crop_sample, center = augmentation.random_crop(sample, crop_size, class_weights)
        sampled_class = list(labels[:, center[0], center[1], center[2]]).index(1)
        sampled_label_center_distribution[sampled_class] += 1

    sampled_label_center_distribution /= total_crops

    if class_weights is None:
        weight = 1.0 / len(non_empty_classes)
        expected_label_center_distribution = [weight if c in non_empty_classes else 0.0
                                              for c in range(number_of_classes)]
    else:
        total = sum(class_weights[c] for c in non_empty_classes)
        expected_label_center_distribution = [class_weights[c] / total if c in non_empty_classes else 0.0
                                              for c in range(number_of_classes)]
    assert np.allclose(sampled_label_center_distribution, expected_label_center_distribution, atol=0.1)


def _check_transformation_result(image_as_tensor: torch.Tensor,
                                 transformation: Callable,
                                 expected: torch.Tensor) -> None:
    test_tensor_pil = TF.to_pil_image(image_as_tensor)
    transformed = TF.to_tensor(transformation(test_tensor_pil)).squeeze()
    np.testing.assert_allclose(transformed, expected, rtol=0.02)


# Augmentation pipeline tests

@pytest.mark.parametrize(["transformation", "expected"],
                         [(ImageTransformationBase.horizontal_flip(), torch.tensor([[0, 1, 2],
                                                                                    [0, 2, 1],
                                                                                    [2, 0, 0]])),
                          (ImageTransformationBase.rotate(45), torch.tensor([[1, 0, 0],
                                                                             [2, 2, 2],
                                                                             [1, 0, 0]])),
                          (ImageTransformationBase.translateX(0.3), torch.tensor([[0, 2, 1],
                                                                                  [0, 1, 2],
                                                                                  [0, 0, 0]])),
                          (ImageTransformationBase.translateY(0.3), torch.tensor([[0, 0, 0],
                                                                                  [2, 1, 0],
                                                                                  [1, 2, 0]])),
                          (ImageTransformationBase.identity(), torch.tensor([[2, 1, 0],
                                                                             [1, 2, 0],
                                                                             [0, 0, 2]]))])
def test_transformations_for_segmentations(transformation: Callable, expected: torch.Tensor) -> None:
    """
    Tests each individual transformation of the ImageTransformationBase class on a 2D input representing
    a segmentation map.
    """
    image_as_tensor = torch.tensor([[2, 1, 0],
                                    [1, 2, 0],
                                    [0, 0, 2]], dtype=torch.int32)
    _check_transformation_result(image_as_tensor, transformation, expected)


def test_invalid_segmentation_type() -> None:
    """
    Validates the necessity of converting segmentation maps to int before PIL
    conversion.
    """
    image_as_tensor = torch.tensor([[2, 1, 0],
                                    [1, 2, 0],
                                    [0, 0, 2]], dtype=torch.float32)
    expected = torch.tensor([[1, 0, 0], [2, 2, 2], [1, 0, 0]])
    with pytest.raises(AssertionError):
        _check_transformation_result(image_as_tensor, ImageTransformationBase.rotate(45), expected)


@pytest.mark.parametrize(["transformation", "expected"],
                         [(ImageTransformationBase.horizontal_flip(), torch.tensor([[0.1, 0.5, 1],
                                                                                    [0.1, 1, 0.5],
                                                                                    [1, 0.1, 0.1]])),
                          (ImageTransformationBase.adjust_contrast(2), torch.tensor([[1., 0.509804, 0.],
                                                                                     [0.509804, 1., 0.],
                                                                                     [0., 0., 1.]])),
                          (ImageTransformationBase.adjust_brightness(2), torch.tensor([[1.0000, 0.9961, 0.1961],
                                                                                       [0.9961, 1.0000, 0.1961],
                                                                                       [0.1961, 0.1961, 1.0000]])),
                          (ImageTransformationBase.adjust_contrast(0), torch.tensor([[0.4863, 0.4863, 0.4863],
                                                                                     [0.4863, 0.4863, 0.4863],
                                                                                     [0.4863, 0.4863, 0.4863]])),
                          (ImageTransformationBase.adjust_brightness(0), torch.tensor([[0.0, 0.0, 0.0],
                                                                                       [0.0, 0.0, 0.0],
                                                                                       [0.0, 0.0, 0.0]]))])
def test_transformation_image(transformation: Callable, expected: torch.Tensor) -> None:
    """
    Tests each individual transformation of the ImageTransformationBase class on a 2D input representing
    a natural image.
    """
    image_as_tensor = torch.tensor([[1, 0.5, 0.1],
                                    [0.5, 1, 0.1],
                                    [0.1, 0.1, 1]], dtype=torch.float32)
    _check_transformation_result(image_as_tensor, transformation, expected)


def test_apply_transformations() -> None:
    """
    Testing the function applying a series of transformations to a given image
    """
    operations = [ImageTransformationBase.identity(), ImageTransformationBase.translateX(0.3),
                  ImageTransformationBase.horizontal_flip()]

    # Case 1 on segmentations
    image_as_tensor = torch.tensor([[[2, 1, 0], [1, 2, 0], [0, 0, 2]],
                                    [[2, 1, 0], [1, 2, 0], [0, 0, 2]]], dtype=torch.int32)
    transformed_tensor = ImageTransformationBase.apply_transform_on_3d_image(image=image_as_tensor,
                                                                             transforms=operations)
    expected = torch.tensor([[[1, 2, 0], [2, 1, 0], [0, 0, 0]],
                             [[1, 2, 0], [2, 1, 0], [0, 0, 0]]])
    assert torch.all(expected == transformed_tensor)

    # Case 2 on image
    image_as_tensor = torch.tensor([[[1, 0.5, 0.1], [0.5, 1, 0.1], [0.1, 0.1, 1]],
                                    [[1, 0.5, 0.1], [0.5, 1, 0.1], [0.1, 0.1, 1]]], dtype=torch.float32)
    transformed_tensor = ImageTransformationBase.apply_transform_on_3d_image(image=image_as_tensor,
                                                                             transforms=operations)
    expected = torch.tensor([[[0.5, 1, 0], [1, 0.5, 0], [0.1, 0.1, 0]],
                             [[0.5, 1, 0], [1, 0.5, 0], [0.1, 0.1, 0]]])
    np.testing.assert_allclose(transformed_tensor, expected, rtol=0.02)


def _compute_expected_pipeline_result(transformations: List[List[Callable]],
                                      input_image: torch.Tensor) -> torch.Tensor:
    expected = input_image.clone()
    expected[0] = ImageTransformationBase.apply_transform_on_3d_image(expected[0],
                                                                      transformations[0])
    expected[1] = ImageTransformationBase.apply_transform_on_3d_image(expected[1],
                                                                      transformations[1])
    return expected


def test_RandAugment_pipeline() -> None:
    """
    Test the RandAugment transformation pipeline for online data augmentation.
    """
    # Set random seeds for transformations
    np.random.seed(1)
    random.seed(0)

    # Get inputs
    one_channel_image = torch.tensor([[[2, 1, 0], [1, 2, 0], [0, 0, 2]],
                                      [[2, 1, 0], [1, 2, 0], [0, 0, 2]]], dtype=torch.int32)
    two_channel_image = torch.stack((one_channel_image, one_channel_image), dim=0)

    # Case no transformation applied
    pipeline = augmentation.RandAugmentSlice(magnitude=3,
                                             n_transforms=0,
                                             is_transformation_for_segmentation_maps=True)
    transformed = pipeline(two_channel_image)
    assert torch.all(two_channel_image == transformed)

    # Case separate transformation per channel
    pipeline = augmentation.RandAugmentSlice(magnitude=3,
                                             n_transforms=1,
                                             is_transformation_for_segmentation_maps=True,
                                             use_joint_channel_transformation=False)
    expected_sampled_ops_channel_1 = [ImageTransformationBase.translateY(-0.3 * 0.2)]
    expected_sampled_ops_channel_2 = [ImageTransformationBase.horizontal_flip()]
    expected = _compute_expected_pipeline_result(transformations=[expected_sampled_ops_channel_1,
                                                                  expected_sampled_ops_channel_2],
                                                 input_image=two_channel_image)
    transformed = pipeline(two_channel_image)
    assert torch.all(transformed == expected)

    # Case same transformation for all channels
    pipeline = augmentation.RandAugmentSlice(magnitude=5,
                                             n_transforms=2,
                                             is_transformation_for_segmentation_maps=True,
                                             use_joint_channel_transformation=True)
    transformed = pipeline(two_channel_image)

    expected_sampled_ops_channel = [ImageTransformationBase.rotate(0.5 * 30),
                                    ImageTransformationBase.translateY(0.5 * 0.2)]

    expected = _compute_expected_pipeline_result(transformations=[expected_sampled_ops_channel,
                                                                  expected_sampled_ops_channel],
                                                 input_image=two_channel_image)
    assert torch.all(transformed == expected)

    # Case for images
    two_channel_image = two_channel_image / 2.0
    pipeline = augmentation.RandAugmentSlice(magnitude=3, n_transforms=1,
                                             use_joint_channel_transformation=True,
                                             is_transformation_for_segmentation_maps=False)
    transformed = pipeline(two_channel_image)
    expected_sampled_ops_channel = [ImageTransformationBase.adjust_contrast(1 - 0.3)]
    expected = _compute_expected_pipeline_result(transformations=[expected_sampled_ops_channel,
                                                                  expected_sampled_ops_channel],
                                                 input_image=two_channel_image)
    assert torch.all(transformed == expected)


def test_RandomSliceTransformation_pipeline() -> None:
    """
    Test the RandomSerial transformation pipeline for online data augmentation.
    """
    # Set random seeds for transformations
    np.random.seed(1)
    random.seed(0)

    one_channel_image = torch.tensor([[[2, 1, 0], [1, 2, 0], [0, 0, 2]],
                                      [[2, 1, 0], [1, 2, 0], [0, 0, 2]]], dtype=torch.int32)
    image_with_two_channels = torch.stack((one_channel_image, one_channel_image), dim=0)

    # Case no transformation applied
    pipeline = augmentation.RandomSliceTransformation(probability_transformation=0,
                                                      is_transformation_for_segmentation_maps=True)
    transformed = pipeline(image_with_two_channels)
    assert torch.all(image_with_two_channels == transformed)

    # Case separate transformation per channel
    pipeline = augmentation.RandomSliceTransformation(probability_transformation=1,
                                                      is_transformation_for_segmentation_maps=True,
                                                      use_joint_channel_transformation=False)
    transformed = pipeline(image_with_two_channels)
    expected_transformations_channel_1 = [ImageTransformationBase.rotate(-7),
                                          ImageTransformationBase.translateX(0.011836899667533166),
                                          ImageTransformationBase.translateY(-0.04989873172751189),
                                          ImageTransformationBase.horizontal_flip()]
    expected_transformations_channel_2 = [ImageTransformationBase.rotate(-1),
                                          ImageTransformationBase.translateX(-0.04012366553408523),
                                          ImageTransformationBase.translateY(-0.08525151327425498),
                                          ImageTransformationBase.horizontal_flip()]
    expected = _compute_expected_pipeline_result(transformations=[expected_transformations_channel_1,
                                                                  expected_transformations_channel_2],
                                                 input_image=image_with_two_channels)
    assert torch.all(transformed == expected)

    # Case same transformation for all channels
    pipeline = augmentation.RandomSliceTransformation(probability_transformation=1,
                                                      is_transformation_for_segmentation_maps=True,
                                                      use_joint_channel_transformation=True)
    transformed = pipeline(image_with_two_channels)
    expected_transformations_channel_1 = [ImageTransformationBase.rotate(9),
                                          ImageTransformationBase.translateX(-0.0006422133534675356),
                                          ImageTransformationBase.translateY(0.07352055509855618),
                                          ImageTransformationBase.horizontal_flip()]
    expected = _compute_expected_pipeline_result(transformations=[expected_transformations_channel_1,
                                                                  expected_transformations_channel_1],
                                                 input_image=image_with_two_channels)
    assert torch.all(transformed == expected)

    # Case for images - convert to range 0-1 first
    image_with_two_channels = image_with_two_channels / 4.0
    pipeline = augmentation.RandomSliceTransformation(probability_transformation=1,
                                                      is_transformation_for_segmentation_maps=False,
                                                      use_joint_channel_transformation=True)
    transformed = pipeline(image_with_two_channels)
    expected_transformations_channel_1 = [ImageTransformationBase.rotate(8),
                                          ImageTransformationBase.translateX(-0.02782961037858135),
                                          ImageTransformationBase.translateY(0.06066901082924045),
                                          ImageTransformationBase.adjust_contrast(0.2849887878176489),
                                          ImageTransformationBase.adjust_brightness(1.0859799153800245)]
    expected = _compute_expected_pipeline_result(transformations=[expected_transformations_channel_1,
                                                                  expected_transformations_channel_1],
                                                 input_image=image_with_two_channels)
    assert torch.all(transformed == expected)
