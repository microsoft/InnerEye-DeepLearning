#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
import string
from typing import List

import pytest

from InnerEye.ML.config import SliceExclusionRule, SummedProbabilityRule
from InnerEye.ML.configs.segmentation.HeadAndNeckBase import HeadAndNeckBase
from InnerEye.ML.configs.segmentation.HeadAndNeckPaper import HeadAndNeckPaper, \
    STRUCTURE_LIST as DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
from InnerEye.ML.configs.segmentation.ProstatePaper import ProstatePaper, \
    fg_classes as DEFAULT_PROSTATE_GROUND_TRUTH_IDS
from InnerEye.ML.utils.model_metadata_util import generate_random_colours_list

RANDOM_COLOUR_GENERATOR = random.Random(0)


def generate_random_string(N: int) -> str:
    '''
    Generate a random string (upper case or digits) of length N
    :param N: length of string to generate.
    '''
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))


def generate_random_display_ids(N: int) -> List[str]:
    return [generate_random_string(6) for i in range(N)]


def generate_random_fill_holes(N: int) -> List[bool]:
    return [bool(random.getrandbits(1)) for i in range(N)]


def generate_random_class_weights(N: int) -> List[float]:
    '''
    Generate a list of random class weights of length N.
    '''
    class_weights = [random.random() for i in range(0, N)]
    total = sum(class_weights)
    scaled_class_weights = [w / total for w in class_weights]
    return scaled_class_weights


def test_head_and_neck_base_with_3_ground_truth_ids() -> None:
    ground_truth_ids = ["parotid_r", "parotid_l", "larynx"]
    config = HeadAndNeckBase(ground_truth_ids)
    assert config.ground_truth_ids == ground_truth_ids


def test_head_and_neck_paper_with_no_ground_truth_ids() -> None:
    '''
    Check that passing num_structures = default generates all default structures.
    '''
    ground_truth_ids = DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
    config = HeadAndNeckPaper()
    assert config.ground_truth_ids == ground_truth_ids


def test_head_and_neck_paper_with_0_ground_truth_ids() -> None:
    '''
    Check that passing num_structures = 0 raises ValueError exception.
    '''
    with pytest.raises(ValueError):
        _ = HeadAndNeckPaper(num_structures=0)


@pytest.mark.parametrize("ground_truth_count", list(range(1, len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS), 3)))
def test_head_and_neck_paper_with_some_ground_truth_ids(
        ground_truth_count: int) -> None:
    '''
    Check that passing a num_structures between 1 and len(defaults) generates the correct subset.
    '''
    ground_truth_ids = DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS[:ground_truth_count]
    config = HeadAndNeckPaper(num_structures=ground_truth_count)
    assert config.ground_truth_ids == ground_truth_ids


def test_head_and_neck_paper_with_too_many_ground_truth_ids() -> None:
    '''
    Check that passing num_structures larger than len(defaults) raises ValueError exception.
    '''
    ground_truth_count = len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS) + 2
    with pytest.raises(ValueError):
        _ = HeadAndNeckPaper(num_structures=ground_truth_count)


@pytest.mark.parametrize("ground_truth_count", list(range(1, len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS), 3)))
def test_head_and_neck_paper_with_optional_params(
        ground_truth_count: int) -> None:
    '''
    Check that optional parameters can be passed in.
    '''
    ground_truth_ids = DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS[:ground_truth_count]
    ground_truth_ids_display_names = generate_random_display_ids(ground_truth_count)
    colours = generate_random_colours_list(RANDOM_COLOUR_GENERATOR, ground_truth_count)
    fill_holes = generate_random_fill_holes(ground_truth_count)
    class_weights = generate_random_class_weights(ground_truth_count + 1)
    num_feature_channels = random.randint(1, ground_truth_count)
    slice_exclusion_rules = [SliceExclusionRule("brainstem", "spinal_cord", False)]
    summed_probability_rules = [SummedProbabilityRule("spinal_cord", "brainstem", "external")]
    config = HeadAndNeckPaper(
        num_structures=ground_truth_count,
        ground_truth_ids_display_names=ground_truth_ids_display_names,
        colours=colours,
        fill_holes=fill_holes,
        class_weights=class_weights,
        num_feature_channels=num_feature_channels,
        slice_exclusion_rules=slice_exclusion_rules,
        summed_probability_rules=summed_probability_rules)
    assert config.ground_truth_ids == ground_truth_ids
    assert config.ground_truth_ids_display_names == ground_truth_ids_display_names
    assert config.colours == colours
    assert config.fill_holes == fill_holes
    assert config.class_weights == class_weights
    assert config.feature_channels == [num_feature_channels]
    assert config.slice_exclusion_rules == slice_exclusion_rules
    assert config.summed_probability_rules == summed_probability_rules


def test_head_and_neck_paper_with_mismatched_display_names_raises() -> None:
    '''
    Check that passing too many colours raises ValueError exception.
    '''
    ground_truth_count = len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS) - 2
    ground_truth_ids_display_names = generate_random_display_ids(ground_truth_count - 1)
    with pytest.raises(ValueError):
        _ = HeadAndNeckPaper(num_structures=ground_truth_count,
                             ground_truth_ids_display_names=ground_truth_ids_display_names)


def test_head_and_neck_paper_with_mismatched_colours_raises() -> None:
    '''
    Check that passing too many colours raises ValueError exception.
    '''
    ground_truth_count = len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS) - 2
    colours = generate_random_colours_list(RANDOM_COLOUR_GENERATOR, ground_truth_count - 1)
    with pytest.raises(ValueError):
        _ = HeadAndNeckPaper(num_structures=ground_truth_count, colours=colours)


def test_head_and_neck_paper_with_mismatched_fill_holes_raises() -> None:
    '''
    Check that passing too many colours raises ValueError exception.
    '''
    ground_truth_count = len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS) - 2
    fill_holes = generate_random_fill_holes(ground_truth_count - 1)
    with pytest.raises(ValueError):
        _ = HeadAndNeckPaper(num_structures=ground_truth_count, fill_holes=fill_holes)


def test_head_and_neck_paper_with_mismatched_class_weights_raises() -> None:
    '''
    Check that passing too many colours raises ValueError exception.
    '''
    ground_truth_count = len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS) - 2
    class_weights = generate_random_class_weights(ground_truth_count - 1)
    with pytest.raises(ValueError):
        _ = HeadAndNeckPaper(num_structures=ground_truth_count, class_weights=class_weights)


def test_prostate_paper() -> None:
    ground_truth_ids = DEFAULT_PROSTATE_GROUND_TRUTH_IDS
    config = ProstatePaper()
    assert config.ground_truth_ids == ground_truth_ids


def test_prostate_paper_with_optional_params() -> None:
    '''
    Check that optional parameters can be passed in.
    '''
    ground_truth_ids = DEFAULT_PROSTATE_GROUND_TRUTH_IDS
    ground_truth_count = len(ground_truth_ids)
    ground_truth_ids_display_names = generate_random_display_ids(ground_truth_count)
    colours = generate_random_colours_list(RANDOM_COLOUR_GENERATOR, ground_truth_count)
    fill_holes = generate_random_fill_holes(ground_truth_count)
    class_weights = generate_random_class_weights(ground_truth_count + 1)
    largest_connected_component_foreground_classes = DEFAULT_PROSTATE_GROUND_TRUTH_IDS[1:3]
    config = ProstatePaper(
        ground_truth_ids_display_names=ground_truth_ids_display_names,
        colours=colours,
        fill_holes=fill_holes,
        class_weights=class_weights,
        largest_connected_component_foreground_classes=largest_connected_component_foreground_classes)
    assert config.ground_truth_ids == ground_truth_ids
    assert config.ground_truth_ids_display_names == ground_truth_ids_display_names
    assert config.colours == colours
    assert config.fill_holes == fill_holes
    assert config.class_weights == class_weights
    expected_lccfc = [(c, None) for c in largest_connected_component_foreground_classes]
    assert config.largest_connected_component_foreground_classes == expected_lccfc
