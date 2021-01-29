#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List, Optional

import numpy as np
import pytest
import torch

from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.config import PaddingMode, SegmentationModelBase, SliceExclusionRule, SummedProbabilityRule
from InnerEye.ML.utils import image_util
from InnerEye.ML.utils.image_util import compute_uncertainty_map_from_posteriors, gaussian_smooth_posteriors, \
    get_class_weights, get_largest_z_slice
from InnerEye.ML.utils.io_util import ImageWithHeader, load_nifti_image
from InnerEye.ML.utils.transforms import LinearTransform
from Tests.ML.configs.DummyModel import DummyModel


@pytest.mark.parametrize("image_size", [None, (4, 4, 5), (2, 4, 4, 5)])
@pytest.mark.parametrize("crop_size", [None, (4, 3, 3)])
@pytest.mark.parametrize("output_size", [None, (4, 3, 3), (5, 6, 6)])
def test_pad_images_for_inference_invalid(image_size: Any, crop_size: Any, output_size: Any) -> None:
    """
    Test to make sure that pad_images_for_inference raises errors in case of invalid inputs.
    """
    with pytest.raises(Exception):
        assert image_util.pad_images_for_inference(images=np.random.uniform(size=image_size),
                                                   crop_size=crop_size,
                                                   output_size=output_size)


@pytest.mark.parametrize("image_size", [(4, 4, 5), (2, 4, 4, 5)])
def test_pad_images_for_inference(image_size: TupleInt3) -> None:
    """
    Test to make sure the correct padding is performed for crop_size and output_size
    that are == , >, and > by 1 in each dimension.
    """
    image = np.random.uniform(size=image_size)

    padded_shape = image_util.pad_images_for_inference(images=image, crop_size=image_size[-3:],
                                                       output_size=(4, 3, 1)).shape
    expected_shape = (4, 5, 9) if len(image_size) == 3 else (2, 4, 5, 9)
    assert padded_shape == expected_shape


@pytest.mark.parametrize("image_size", [(4, 4, 5), (2, 4, 4, 5)])
def test_pad_images_for_training(image_size: TupleInt3) -> None:
    """
    Test to make sure the correct padding is performed for crop_size and output_size
    that are == , >, and > by 1 in each dimension.
    """
    image = np.random.uniform(size=image_size)
    expected_pad_value = np.min(image)

    padded_image = image_util.pad_images(images=image, output_size=(8, 7, 6),
                                         padding_mode=PaddingMode.Minimum)
    expected_shape = (8, 7, 6) if len(image_size) == 3 else (2, 8, 7, 6)
    assert padded_image.shape == expected_shape
    assert np.all(padded_image[..., 8:4, 8:4, 8:4] == expected_pad_value)


@pytest.mark.parametrize("image", [None, np.random.uniform((4, 4, 4)), np.random.uniform((1, 4, 4, 4))])
@pytest.mark.parametrize("crop_shape", [None, (8, 8, 8)])
def test_get_center_crop_invalid(image: Any, crop_shape: Any) -> None:
    """
    Test that get_center_crop corectly raises an error for invalid arguments
    """
    with pytest.raises(Exception):
        assert image_util.get_center_crop(image=image, crop_shape=crop_shape)


def test_get_center_crop() -> None:
    """
    Test to make sure the center crop is extracted correctly from a given image.
    """
    image = np.random.uniform(size=(4, 4, 4))
    crop = image_util.get_center_crop(image=image, crop_shape=(2, 2, 2))
    expected = image[1:3, 1:3, 1:3]
    assert np.array_equal(crop, expected)


def test_merge_masks() -> None:
    """
    Test to make sure mask merging is as expected.
    """
    with pytest.raises(Exception):
        # noinspection PyTypeChecker
        assert image_util.merge_masks(masks=None)
    with pytest.raises(Exception):
        assert image_util.merge_masks(masks=np.zeros(shape=(2, 2, 2)))
    with pytest.raises(Exception):
        assert image_util.merge_masks(masks=np.zeros(shape=(2, 2, 2, 2, 2)))

    image = np.zeros(shape=(2, 2, 2, 2))
    image[0, 0, 0, 0] = 1
    image[1, 1, 0, 0] = 1

    actual = image_util.merge_masks(masks=image)
    expected = np.zeros(shape=(2, 2, 2))
    expected[0, 0, 0] = 0
    expected[1, 0, 0] = 1
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("image_size", [[2, 3, 4, 4, 4], [2, 4, 4, 4]])
def test_apply_mask_to_posteriors(image_size: List[int]) -> None:
    """
    Test to make sure masks are being applied as expected.
    """
    image = np.ones(image_size)
    mask_size = image_size[:1] + image_size[-3:] if len(image_size) == 5 else image_size[-3:]
    mask = np.zeros(shape=mask_size)
    mask[0] = 1

    with pytest.raises(Exception):
        # noinspection PyTypeChecker
        image_util.apply_mask_to_posteriors(posteriors=None, mask=None)
    with pytest.raises(Exception):
        # noinspection PyTypeChecker
        image_util.apply_mask_to_posteriors(posteriors=image, mask=None)
    with pytest.raises(Exception):
        # noinspection PyTypeChecker
        image_util.apply_mask_to_posteriors(posteriors=None, mask=image)
    with pytest.raises(Exception):
        # noinspection PyTypeChecker
        image_util.apply_mask_to_posteriors(posteriors=image, mask=image)

    image = image_util.apply_mask_to_posteriors(posteriors=image, mask=mask)
    assert np.all(image[:, 0, ...] == 1)
    assert np.all(image[1:, 1:, ...] == 0)


def test_posteriors_to_segmentation() -> None:
    """
    Test to make sure the posterior to segmentation conversion is as expected
    """
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        image_util.posteriors_to_segmentation(posteriors=None)
    with pytest.raises(ValueError):
        image_util.posteriors_to_segmentation(posteriors=np.zeros(shape=(3, 3, 3)))
    with pytest.raises(ValueError):
        image_util.posteriors_to_segmentation(posteriors=np.zeros(shape=(3, 3, 3, 3, 3, 3)))

    image = np.zeros(shape=(2, 3, 3, 3))
    image[1] = 0.6
    # class 1 dominates entire image
    assert np.array_equal(np.ones(shape=(3, 3, 3)), image_util.posteriors_to_segmentation(image))
    # no dominating class (first index with this argmax chosen chosen by default)
    image[...] = 0.5
    assert np.array_equal(np.zeros(shape=(3, 3, 3)), image_util.posteriors_to_segmentation(image))


def test_multi_map_to_binary_maps() -> None:
    """
    Test the multi_label_array_to_binary conversion
    """
    array = np.zeros((3, 3))
    # The binary mask for class assignment to class 0
    result0 = np.ones_like(array)
    # The binary mask for class assignment to class 1
    result1 = np.zeros_like(array)
    # Wherever class is 1, the binary mask for class 0 is 0
    array[1, 1] = 1
    result0[1, 1] = 0
    result1[1, 1] = 1
    # Assign a class 2, that should not show up in any of the returned masks.
    array[2, 2] = 2
    result0[2, 2] = 0
    result1[2, 2] = 0
    # We only expect binary masks for classes 0 and 1
    num_classes = 2
    binary = image_util.multi_label_array_to_binary(array, num_classes)
    assert binary.shape == (num_classes, 3, 3)
    expected = np.stack([result0, result1], 0)
    assert np.array_equal(binary, expected)


@pytest.mark.parametrize(["input_array", "expected"],
                         [(np.array([0, 2, 4]), False),
                          (np.array([[0, 1]]), True)])
def test_is_binary_array(input_array: np.ndarray, expected: bool) -> None:
    """
    Test is_binary_array function
    """
    assert image_util.is_binary_array(input_array) == expected


def test_check_input_range() -> None:
    """
    Test the `check_array_range` function in particular for arrays with missing
    values.
    """
    image = np.array([1, 2, 3, 4])
    image_nan = np.array([1, 2, 3, np.nan, np.nan])
    image_inf = np.array([1, 2, 3, np.inf, np.inf])
    image_nan_inf = np.array([1, 2, 3, np.nan, np.inf])
    # All values are in the range, this should work
    image_util.check_array_range(image, (1, 4))
    # When not providing a range, it should only check for NaN and Inf, but there are none.
    image_util.check_array_range(image, None)
    # Using a smaller range than is present in the array: This should fail, and print the interval in the error message
    with pytest.raises(ValueError) as err:
        image_util.check_array_range(image, (1, 2))
    assert "within [1, 2]" in err.value.args[0]
    assert "invalid values: 3, 4" in err.value.args[0]
    # Now try all the arrays that contain NaN and/or Inf. None should pass the test, with or without an interval given.
    for data in [image_inf, image_nan_inf]:
        with pytest.raises(ValueError) as err:
            image_util.check_array_range(data)
        assert "finite" in err.value.args[0]
        assert "inf" in err.value.args[0]
        with pytest.raises(ValueError) as err:
            image_util.check_array_range(data, (1, 4))
        assert "within [1, 4]" in err.value.args[0]
        assert "inf" in err.value.args[0]
    for data in [image_nan, image_nan_inf]:
        with pytest.raises(ValueError) as err:
            image_util.check_array_range(data)
        assert "finite" in err.value.args[0]
        assert "nan" in err.value.args[0]
        with pytest.raises(ValueError) as err:
            image_util.check_array_range(data, (1, 4))
        assert "within [1, 4]" in err.value.args[0]
        assert "nan" in err.value.args[0]
    # Case where there are values outside of the expected range and NaN:
    with pytest.raises(ValueError) as err:
        image_util.check_array_range(image_nan_inf, (1, 2))
    assert "within [1, 2]" in err.value.args[0]
    assert "nan, inf, 3.0" in err.value.args[0]
    # Degenerate interval with a single value
    single_value = np.array([2, 2])
    image_util.check_array_range(single_value, (2, 2))
    with pytest.raises(ValueError) as err:
        image_util.check_array_range(single_value, (3, 3))
    assert "within [3, 3]" in err.value.args[0]
    assert "2" in err.value.args[0]


def test_check_input_range_with_tolerance() -> None:
    """
    Test `check_array_range` for cases where values are only *just* outside the range.
    """
    tolerance = image_util.VALUE_RANGE_TOLERANCE
    low_value = 0.0
    high_value = 1.0
    allowed_range = (low_value, high_value)
    values1 = np.array([low_value - 1.1 * tolerance, high_value + 1.1 * tolerance])
    with pytest.raises(ValueError):
        image_util.check_array_range(values1, allowed_range)
    values2 = np.array([low_value - 0.9 * tolerance, high_value + 1.1 * tolerance])
    with pytest.raises(ValueError):
        image_util.check_array_range(values2, allowed_range)
    values3 = np.array([low_value - 1.1 * tolerance, high_value + 0.9 * tolerance])
    with pytest.raises(ValueError):
        image_util.check_array_range(values3, allowed_range)
    values4 = np.array([low_value - 0.9 * tolerance, high_value + 0.9 * tolerance])
    image_util.check_array_range(values4, allowed_range)
    assert values4[0] == low_value
    assert values4[1] == high_value


def test_get_largest_z_slice() -> None:
    """
    Tests `get_largest_z_slice`
    """
    # An image with 2 z slices
    mask = np.zeros((2, 3, 3))
    # Initially there are no non-zero elements. argmax will return the lowest index
    # that attains the maximum, which is 0
    assert get_largest_z_slice(mask) == 0
    # Now set the whole plane z==1 plane to 1, making it the largest plane
    mask[1] = np.ones((3, 3))
    assert get_largest_z_slice(mask) == 1
    # Remove one of the non-zero entries at z == 1 to 0, and set all of the z == 0 plane to 1,
    # making z == 0 the largest plane.
    mask[1, 0, 0] = 0
    mask[0] = np.ones((3, 3))
    assert get_largest_z_slice(mask) == 0
    with pytest.raises(ValueError):
        get_largest_z_slice(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        get_largest_z_slice(np.zeros((3, 3, 3, 3)))


def test_one_hot_to_class_indices() -> None:
    """
    Test decoding from one_hot to multi-map
    """
    labels = torch.tensor([[[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [1, 1, 0, 1, 1]]], dtype=torch.float32)
    class_ids = image_util.one_hot_to_class_indices(labels=labels)
    assert torch.eq(class_ids, torch.tensor([[2, 2, 1, 2, 2]])).all()

    # Check if it throws an error when there are multiple ones in channels
    with pytest.raises(ValueError) as err:
        problematic_labels = torch.tensor([[[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 0, 1, 1]]], dtype=torch.float32)
        image_util.one_hot_to_class_indices(labels=problematic_labels)
        assert "multiple foreground pixels" in err.value.args[0]


@pytest.mark.parametrize("restrict_classes", [None, [1], [2], [1, 2]])
@pytest.mark.parametrize("has_fg_component", [True, False])
def test_extract_largest_foreground_connected_component(restrict_classes: Optional[List[int]],
                                                        has_fg_component: bool) -> None:
    segmentation = np.zeros((4, 4, 4))
    if has_fg_component:
        segmentation[0, :, :] = 1
        segmentation[2, 2, 2] = 1
        segmentation[1, :, :] = 2
        segmentation[3, 3, 3] = 2

    # extract the largest components for all foreground classes
    restrict_pairs = None
    if restrict_classes is not None:
        restrict_pairs = [(idx, None) for idx in restrict_classes]

    largest_fg_classes = image_util.extract_largest_foreground_connected_component(
        multi_label_array=segmentation, restrictions=restrict_pairs)  # type: ignore

    expected = segmentation.copy()
    if restrict_classes is None or restrict_classes == [1, 2]:
        expected[2, 2, 2] = 0
        expected[3, 3, 3] = 0
    elif restrict_classes == [1]:
        expected[2, 2, 2] = 0
    elif restrict_classes == [2]:
        expected[3, 3, 3] = 0

    assert np.array_equal(largest_fg_classes, expected)


def test_extract_largest_foreground_connected_component_with_threshold() -> None:
    # Class 1 is 17 voxels in two components, of 16 voxels and 1 voxel (so latter is < 0.1 as a proportion)
    # Class 2 is 20 voxels in two components, of 16 voxels and 4 voxel (so latter is > 0.1 as a proportion)
    segmentation = np.zeros((4, 4, 4))
    segmentation[0, :, :] = 1  # 16 voxels
    segmentation[2, 2, 2] = 1  # 1 voxel
    segmentation[1, :, :] = 2  # 16 voxels
    segmentation[3, 3, :] = 2  # 4 voxels

    # extract the largest components for all foreground classes. A threshold
    # of 0.1 is greater than 1/17 but not greater than 4/20, so only the lone voxel
    # at 2,2,2 should be deleted.
    largest_fg_classes = image_util.extract_largest_foreground_connected_component(
        multi_label_array=segmentation, restrictions=[(1, 0.1), (2, 0.1)])

    expected = segmentation.copy()
    # We expected only the 1 disconnected voxel in class 1 to disappear, not the 4 in class 2.
    expected[2, 2, 2] = 0
    assert np.array_equal(largest_fg_classes, expected)


def test_compute_uncertainty_map_from_posteriors() -> None:
    posteriors = np.zeros((3, 1, 1, 1))
    posteriors[0] = 0.1
    posteriors[1] = 0.2
    posteriors[2] = 0.7
    expected_uncertainty = np.array([[[0.7298467]]])
    # check computation is as expected for an arbitrary posterior distribution
    actual_uncertainty = compute_uncertainty_map_from_posteriors(posteriors)
    assert np.allclose(expected_uncertainty, actual_uncertainty)

    # check minimum uncertainty case is handled as expected
    posteriors[0] = 1
    posteriors[1:] = 0
    assert np.array_equal(np.zeros_like(expected_uncertainty),
                          compute_uncertainty_map_from_posteriors(posteriors))

    # check maximum uncertainty case is handled as expected
    posteriors[...] = 1 / 3
    assert np.array_equal(np.ones_like(expected_uncertainty),
                          compute_uncertainty_map_from_posteriors(posteriors))

    # check invalid posteriors are handled as expected
    with pytest.raises(ValueError):
        compute_uncertainty_map_from_posteriors(np.zeros_like(posteriors))
        compute_uncertainty_map_from_posteriors(np.ones_like(posteriors))
        # all posteriors sum to 1 but not in the class dimension (as required)
        random_posteriors = np.zeros((2, 2, 1, 1))
        random_posteriors[0, 0] = 0.8
        random_posteriors[0, 1] = 0.2
        random_posteriors[1, 0] = 0.8
        random_posteriors[1, 1] = 0.2

        compute_uncertainty_map_from_posteriors(random_posteriors)
        # posteriors with values < 0
        compute_uncertainty_map_from_posteriors(np.ones_like(posteriors) * -1)
        # posteriors with values > 1
        compute_uncertainty_map_from_posteriors(np.ones_like(posteriors) * 2)


def test_posterior_smoothing() -> None:
    def _load_and_scale_image(name: str) -> ImageWithHeader:
        image_with_header = load_nifti_image(full_ml_test_data_path(name))
        return ImageWithHeader(
            image=LinearTransform.transform(data=image_with_header.image, input_range=(0, 255), output_range=(0, 1)),
            header=image_with_header.header
        )

    original_posterior_fg = _load_and_scale_image("posterior_bladder.nii.gz")
    expected_posterior_fg = _load_and_scale_image("smoothed_posterior_bladder.nii.gz").image

    # create the BG/FG posterior pair from single posterior and apply smoothing
    smoothed_posterior_fg = gaussian_smooth_posteriors(
        posteriors=np.stack([1 - original_posterior_fg.image, original_posterior_fg.image], axis=0),
        kernel_size_mm=(2, 2, 2),
        voxel_spacing_mm=original_posterior_fg.header.spacing
    )[1]

    # smooth and check if as expected (tolerance required due to scaling from (0, 255) <=> (0, 1))
    assert np.allclose(expected_posterior_fg, smoothed_posterior_fg, atol=1e-2)


def test_get_class_weights() -> None:
    """
    Test get_class_weights for segmentation models.
    """
    # 5 voxels are class 0, 3 are class 1, none are class 2.
    target = torch.zeros((2, 3, 4))
    target[0, 1, 0] = 1
    target[0, 0, 1] = 1
    target[0, 0, 2] = 1
    target[0, 0, 3] = 1
    target[1, 0, 0] = 1
    target[1, 0, 1] = 1
    target[1, 1, 2] = 1
    target[1, 1, 3] = 1
    voxel_counts = target.sum((0, 2))
    voxel_counts[voxel_counts == 0] = 1  # empty classes are treated as if they had one voxel
    # noinspection PyTypeChecker
    inverses: torch.Tensor = 1.0 / voxel_counts  # type: ignore
    counts = get_class_weights(target, class_weight_power=1.0)
    expected = target.shape[1] * inverses / inverses.sum()
    assert torch.allclose(expected, counts, atol=0.001)
    counts = get_class_weights(target, class_weight_power=2.0)
    inverses *= inverses
    expected = target.shape[1] * inverses / inverses.sum()
    assert torch.allclose(expected, counts, atol=0.001)


ground_truth_ids = ["region1", "region2", "region3"]

# z coordinates increase from bottom to top of segmentation

# replace voxels of the non-dominant class by the dominant class in single region of overlap
segmentation_single_overlap = np.array([[[1, 3], [1, 1]], [[3, 2], [1, 1]], [[1, 3], [1, 2]], [[2, 2], [2, 2]]])
expected_segmentation_single_overlap_r1_dominates = np.array([[[1, 3], [1, 1]], [[3, 1], [1, 1]],
                                                              [[1, 3], [1, 1]], [[2, 2], [2, 2]]])
expected_segmentation_single_overlap_r2_dominates = np.array([[[1, 3], [1, 1]], [[3, 2], [2, 2]],
                                                              [[2, 3], [2, 2]], [[2, 2], [2, 2]]])

# replace voxels of the non-dominant class by the dominant class in the region between the slice of first overlap
# and the slice of last overlap
segmentation_multiple_overlap = np.array([[[1, 3], [1, 1]], [[3, 2], [1, 1]], [[1, 3], [1, 1]], [[2, 1], [2, 2]]])
expected_segmentation_multiple_overlap_r1_dominates = np.array([[[1, 3], [1, 1]], [[3, 1], [1, 1]],
                                                                [[1, 3], [1, 1]], [[1, 1], [1, 1]]])
expected_segmentation_multiple_overlap_r2_dominates = np.array([[[1, 3], [1, 1]], [[3, 2], [2, 2]],
                                                                [[2, 3], [2, 2]], [[2, 2], [2, 2]]])

# No overlap in these situations, segmentation should not change
segmentation_no_overlap = np.array([[[1, 3], [1, 1]], [[3, 1], [1, 1]], [[1, 3], [1, 1]], [[2, 2], [3, 2]]])
segmentation_class_not_present = np.array([[[1, 3], [1, 1]], [[3, 1], [1, 1]], [[1, 3], [1, 1]], [[1, 1], [3, 1]]])

slice_exclusion_rule_r1_dominates = [SliceExclusionRule("region2", "region1", False)]
slice_exclusion_rule_r2_dominates = [SliceExclusionRule("region2", "region1", True)]


@pytest.fixture
def model_config(slice_exclusion_rules: List[SliceExclusionRule],
                 summed_probability_rules: List[SummedProbabilityRule]) -> SegmentationModelBase:
    test_config = DummyModel()
    test_config.slice_exclusion_rules = slice_exclusion_rules
    test_config.summed_probability_rules = summed_probability_rules
    test_config.ground_truth_ids = ground_truth_ids
    return test_config


# noinspection PyTestParametrized
@pytest.mark.parametrize("slice_exclusion_rules, summed_probability_rules, expected_segmentation",
                         [(slice_exclusion_rule_r1_dominates, [], expected_segmentation_single_overlap_r1_dominates),
                          (slice_exclusion_rule_r2_dominates, [], expected_segmentation_single_overlap_r2_dominates)])
def test_slice_exclusion_rules_single_overlap(model_config: SegmentationModelBase,
                                              expected_segmentation: np.ndarray) -> None:
    """
    Test `apply_slice_exclusion_rules` in the single overlap case
    """
    # create a copy as apply_slice_exclusion_rules modifies in place
    segmentation_copy = np.copy(segmentation_single_overlap)
    image_util.apply_slice_exclusion_rules(model_config, segmentation_copy)
    assert np.array_equal(segmentation_copy, expected_segmentation)


# noinspection PyTestParametrized
@pytest.mark.parametrize("slice_exclusion_rules, summed_probability_rules, expected_segmentation",
                         [(slice_exclusion_rule_r1_dominates, [], expected_segmentation_multiple_overlap_r1_dominates),
                          (slice_exclusion_rule_r2_dominates, [], expected_segmentation_multiple_overlap_r2_dominates)])
def test_slice_exclusion_rules_multiple_overlap(model_config: SegmentationModelBase,
                                                expected_segmentation: np.ndarray) -> None:
    """
    Test `apply_slice_exclusion_rules` in the multiple overlap case
    """
    # create a copy as apply_slice_exclusion_rules modifies in place
    segmentation_copy = np.copy(segmentation_multiple_overlap)
    image_util.apply_slice_exclusion_rules(model_config, segmentation_copy)
    assert np.array_equal(segmentation_copy, expected_segmentation)


# noinspection PyTestParametrized
@pytest.mark.parametrize("slice_exclusion_rules, summed_probability_rules",
                         [(slice_exclusion_rule_r1_dominates, []),
                          (slice_exclusion_rule_r2_dominates, [])])
def test_slice_exclusion_rules_no_overlap(model_config: SegmentationModelBase) -> None:
    """
    Test `apply_slice_exclusion_rules` in the no overlap case
    """
    # create a copy as apply_slice_exclusion_rules modifies in place
    segmentation_copy = np.copy(segmentation_no_overlap)
    image_util.apply_slice_exclusion_rules(model_config, segmentation_copy)
    assert np.array_equal(segmentation_copy, segmentation_no_overlap)


# noinspection PyTestParametrized
@pytest.mark.parametrize("slice_exclusion_rules, summed_probability_rules",
                         [(slice_exclusion_rule_r1_dominates, []),
                          (slice_exclusion_rule_r2_dominates, [])])
def test_slice_exclusion_rules_class_not_present(model_config: SegmentationModelBase) -> None:
    """
    Test `apply_slice_exclusion_rules` if the class to exclude is not present
    """
    # create a copy as apply_slice_exclusion_rules modifies in place
    segmentation_copy = np.copy(segmentation_class_not_present)
    image_util.apply_slice_exclusion_rules(model_config, segmentation_copy)
    assert np.array_equal(segmentation_copy, segmentation_class_not_present)


# noinspection PyTestParametrized
@pytest.mark.parametrize("slice_exclusion_rules, summed_probability_rules",
                         [([], [])])
def test_slice_exclusion_rules_none(model_config: SegmentationModelBase) -> None:
    """
    Test `apply_slice_exclusion_rules` if no rule is provided
    """
    # create a copy as apply_slice_exclusion_rules modifies in place
    segmentation_copy = np.copy(segmentation_single_overlap)
    image_util.apply_slice_exclusion_rules(model_config, segmentation_copy)
    assert np.array_equal(segmentation_copy, segmentation_single_overlap)


# noinspection PyTestParametrized
@pytest.mark.parametrize("slice_exclusion_rules, summed_probability_rules",
                         [([], [SummedProbabilityRule("region1", "region2", "region3")])])
def test_apply_summed_probability_rules_incorrect_input(model_config: SegmentationModelBase) -> None:
    """
    Test `apply_summed_probability_rules` with invalid inputs
    """
    posteriors = np.zeros(shape=(2, 4, 3, 3, 3))
    segmentation = image_util.posteriors_to_segmentation(posteriors)

    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        image_util.apply_summed_probability_rules(model_config, posteriors=posteriors,
                                                  segmentation=None)

    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        image_util.apply_summed_probability_rules(model_config, posteriors=None,
                                                  segmentation=segmentation)

    with pytest.raises(ValueError):
        image_util.apply_summed_probability_rules(model_config, posteriors=posteriors,
                                                  segmentation=np.zeros(shape=(3, 3, 3)))

    with pytest.raises(ValueError):
        image_util.apply_summed_probability_rules(model_config, posteriors=posteriors,
                                                  segmentation=np.zeros(shape=(3, 3, 3, 3)))


# noinspection PyTestParametrized
@pytest.mark.parametrize("slice_exclusion_rules, summed_probability_rules",
                         [([], [SummedProbabilityRule("region1", "region2", "region3")])])
@pytest.mark.parametrize("is_batched", [True, False])
def test_apply_summed_probability_rules_change(model_config: SegmentationModelBase, is_batched: bool) -> None:
    """
    Test `apply_summed_probability_rules` with valid inputs and an expected change
    """
    posteriors = np.zeros(shape=(2, 4, 3, 3, 3))
    posteriors[:, 3] = 0.4
    posteriors[:, 1, :1] = 0.35
    posteriors[:, 2, :1] = 0.25
    posteriors[:, 1, 1:2] = 0.25
    posteriors[:, 2, 1:2] = 0.35

    # class 1 and class 2 together have a larger probability than class 3 in half the image
    # In this region, we replace external pixels (class 3) with class 1 or class 2, whichever has the higher probability
    expected_segmentation = np.full(shape=(2, 3, 3, 3), fill_value=3)
    expected_segmentation[:, :1] = 1
    expected_segmentation[:, 1:2] = 2

    if not is_batched:
        posteriors = posteriors[0]
        expected_segmentation = expected_segmentation[0]

    segmentation = image_util.posteriors_to_segmentation(posteriors)

    # test for both np arrays and tensors
    # we make a copy of segmentation as apply_summed_probability_rules modifies it in place
    assert np.array_equal(expected_segmentation,
                          image_util.apply_summed_probability_rules(model_config, posteriors, np.copy(segmentation)))
    # noinspection PyTypeChecker
    assert torch.equal(torch.from_numpy(expected_segmentation).type(torch.LongTensor),  # type: ignore
                       image_util.apply_summed_probability_rules(model_config, torch.from_numpy(posteriors),
                                                                 torch.from_numpy(np.copy(segmentation))))


# noinspection PyTestParametrized
@pytest.mark.parametrize("slice_exclusion_rules, summed_probability_rules",
                         [([], [SummedProbabilityRule("region1", "region2", "region3")])])
@pytest.mark.parametrize("is_batched", [True, False])
def test_apply_summed_probability_rules_no_change(model_config: SegmentationModelBase, is_batched: bool) -> None:
    """
    Test `apply_summed_probability_rules` with valid inputs and no expected change
    """
    posteriors = np.zeros(shape=(2, 4, 3, 3, 3))
    posteriors[:, 3] = 0.6
    posteriors[:, 1, :2] = 0.2
    posteriors[:, 2, :2] = 0.2

    # class 1 and class 2 together do not have a larger probability than external (class 3).
    expected_segmentation = np.full(shape=(2, 3, 3, 3), fill_value=3)

    if not is_batched:
        posteriors = posteriors[0]
        expected_segmentation = expected_segmentation[0]

    segmentation = image_util.posteriors_to_segmentation(posteriors)

    # test for both np arrays and tensors
    # we make a copy of segmentation as apply_summed_probability_rules modifies it in place
    assert np.array_equal(expected_segmentation,
                          image_util.apply_summed_probability_rules(model_config, posteriors, np.copy(segmentation)))
    # noinspection PyTypeChecker
    assert torch.equal(torch.from_numpy(expected_segmentation).type(torch.LongTensor),  # type: ignore
                       image_util.apply_summed_probability_rules(model_config, torch.from_numpy(posteriors),
                                                                 torch.from_numpy(np.copy(segmentation))))
