#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List

import monai
import numpy as np
import pytest

from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.augmentations.augmentation_for_segmentation_utils import BasicAugmentations, random_crop
from InnerEye.ML.dataset.sample import Sample
from InnerEye.ML.utils import io_util, ml_util
from InnerEye.ML.utils.io_util import ImageWithHeader, load_nifti_image

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


def test_basic_augmentation_segmentation(test_output_dirs: OutputFolderForTests) -> None:
    def _load_and_scale_image(name: str) -> ImageWithHeader:
        image_with_header = load_nifti_image(full_ml_test_data_path(name))
        return ImageWithHeader(
            image=np.where(image_with_header.image < 200, 0, 1),
            header=image_with_header.header
        )

    monai.utils.misc.set_determinism(seed=123, additional_settings=None)
    transforms = BasicAugmentations()
    metadata = DummyPatientMetadata
    image_with_header = load_nifti_image(full_ml_test_data_path("posterior_bladder.nii.gz"))
    image = image_with_header.image
    labels = _load_and_scale_image("posterior_bladder.nii.gz").image
    image_with_channel = np.expand_dims(image, axis=0)
    labels_with_channel = np.expand_dims(labels, axis=0)
    sample = Sample(image=image_with_channel,
                    labels=labels_with_channel,
                    mask=labels_with_channel,
                    metadata=metadata)
    no_transform = 0
    for i in range(5):
        transformed_sample = transforms(sample)

        if np.array_equal(transformed_sample.image[0], image):
            no_transform += 1

        image_path = f"{test_output_dirs.root_dir}/test{i}.nii.gz"
        io_util.store_as_nifti(transformed_sample.image[0], image_with_header.header,
                               file_name=image_path, image_type=np.short)

        seg_path = f"{test_output_dirs.root_dir}/test_seg{i}.nii.gz"
        io_util.store_as_nifti(transformed_sample.labels[0], image_with_header.header,
                               file_name=seg_path, image_type=np.short)

        expected_image_transform = load_nifti_image(full_ml_test_data_path(f"augmentation_baselines/test{i}.nii.gz"),
                                                    np.short)
        expected_seg_transform = load_nifti_image(full_ml_test_data_path(f"augmentation_baselines/test_seg{i}.nii.gz"),
                                                  np.short)
        print(i)
        print(expected_image_transform.image.max())
        print(transformed_sample.image[0].astype(np.short).max())
        assert np.array_equal(expected_image_transform.image,
                              transformed_sample.image[0].astype(np.short))
        assert np.array_equal(expected_seg_transform.image,
                              transformed_sample.labels[0].astype(np.short))
    # Check that some samples are the same image
    print(no_transform)
    assert no_transform == 1


def test_valid_full_crop() -> None:
    metadata = DummyPatientMetadata
    sample, _ = random_crop(sample=Sample(image=valid_image_4d,
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
            random_crop(Sample(metadata=DummyPatientMetadata, image=image, labels=labels, mask=mask),
                        valid_crop_size, class_weights)


@pytest.mark.parametrize("crop_size", [None, ["a"], 5])
def test_invalid_crop_arg(crop_size: Any) -> None:
    with pytest.raises(Exception):
        random_crop(
            Sample(metadata=DummyPatientMetadata, image=valid_image_4d, labels=valid_labels, mask=valid_mask),
            crop_size, valid_class_weights)


@pytest.mark.parametrize("crop_size", [[2, 2], [2, 2, 2, 2], [10, 10, 10]])
def test_invalid_crop_size(crop_size: Any) -> None:
    with pytest.raises(Exception):
        random_crop(
            Sample(metadata=DummyPatientMetadata, image=valid_image_4d, labels=valid_labels, mask=valid_mask),
            crop_size, valid_class_weights)


def test_random_crop_no_fg() -> None:
    with pytest.raises(Exception):
        random_crop(Sample(metadata=DummyPatientMetadata, image=valid_image_4d, labels=valid_labels,
                           mask=np.zeros_like(valid_mask)),
                    valid_crop_size, valid_class_weights)

    with pytest.raises(Exception):
        random_crop(Sample(metadata=DummyPatientMetadata, image=valid_image_4d,
                           labels=np.zeros_like(valid_labels), mask=valid_mask),
                    valid_crop_size, valid_class_weights)


@pytest.mark.parametrize("crop_size", [valid_crop_size])
def test_random_crop(crop_size: Any) -> None:
    labels = valid_labels
    # create labels such that there are no foreground voxels in a particular class
    # this should ne handled gracefully (class being ignored from sampling)
    labels[0] = 1
    labels[1] = 0
    sample, _ = random_crop(Sample(
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
            random_crop(sample, crop_size, class_weights)
        return

    for _ in range(0, total_crops):
        crop_sample, center = random_crop(sample, crop_size, class_weights)
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
