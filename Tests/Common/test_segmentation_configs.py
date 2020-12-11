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
from InnerEye.ML.configs.segmentation.ProstateBase import ProstateBase
from InnerEye.ML.configs.segmentation.ProstatePaper import ProstatePaper, \
    fg_classes as DEFAULT_PROSTATE_GROUND_TRUTH_IDS
from InnerEye.ML.utils.model_metadata_util import generate_random_colours_list

RANDOM_COLOUR_GENERATOR = random.Random(0)


def generate_random_string(size: int) -> str:
    """
    Generate a random string (upper case or digits) of length size
    :param size: length of string to generate.
    """
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=size))


def generate_random_display_ids(size: int) -> List[str]:
    """
    Generate a list of random display ids of length size.
    :param size: length of list of random display ids to generate.
    """
    return [generate_random_string(6) for i in range(size)]


def generate_random_fill_holes(size: int) -> List[bool]:
    """
    Generate a list of random bools of length size.
    :param size: length of list of random booleans to generate.
    """
    return [bool(random.getrandbits(1)) for i in range(size)]


def generate_random_class_weights(size: int) -> List[float]:
    """
    Generate a list of random class weights of length size.
    """
    class_weights = [random.random() for i in range(size)]
    total = sum(class_weights)
    scaled_class_weights = [w / total for w in class_weights]
    return scaled_class_weights


def generate_random_slice_exclusion_rules(ground_truth_ids: List[str]) -> List[SliceExclusionRule]:
    """
    Generate a list of random slice exclusion rules, if possible.
    """
    if len(ground_truth_ids) < 2:
        return []

    index0 = random.randint(1, len(ground_truth_ids) - 1)
    index1 = random.randint(0, index0 - 1)

    return [SliceExclusionRule(ground_truth_ids[index0], ground_truth_ids[index1], False)]


def generate_random_summed_probability_rules(ground_truth_ids: List[str]) -> List[SummedProbabilityRule]:
    """
    Generate a list of random summed probability rules, if possible.
    """
    if len(ground_truth_ids) < 3:
        return []

    index0 = random.randint(2, len(ground_truth_ids) - 1)
    index1 = random.randint(1, index0 - 1)
    index2 = random.randint(0, index1 - 1)

    return [SummedProbabilityRule(ground_truth_ids[index1], ground_truth_ids[index0], ground_truth_ids[index2])]


def test_head_and_neck_base() -> None:
    """
    Check we can instantiate HeadAndNeckBase class.
    """
    ground_truth_ids = DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
    config = HeadAndNeckBase(ground_truth_ids)
    assert config.ground_truth_ids == ground_truth_ids


def test_head_and_neck_base_with_optional_params() -> None:
    """
    Check that optional parameters can be passed in to HeadAndNeckBase class.
    """
    ground_truth_ids = DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
    ground_truth_count = len(ground_truth_ids)
    ground_truth_ids_display_names = generate_random_display_ids(ground_truth_count)
    colours = generate_random_colours_list(RANDOM_COLOUR_GENERATOR, ground_truth_count)
    fill_holes = generate_random_fill_holes(ground_truth_count)
    class_weights = generate_random_class_weights(ground_truth_count + 1)
    num_feature_channels = random.randint(1, ground_truth_count)
    slice_exclusion_rules = generate_random_slice_exclusion_rules(ground_truth_ids)
    summed_probability_rules = generate_random_summed_probability_rules(ground_truth_ids)
    config = HeadAndNeckBase(
        ground_truth_ids=ground_truth_ids,
        ground_truth_ids_display_names=ground_truth_ids_display_names,
        colours=colours,
        fill_holes=fill_holes,
        class_weights=class_weights,
        slice_exclusion_rules=slice_exclusion_rules,
        summed_probability_rules=summed_probability_rules,
        num_feature_channels=num_feature_channels)
    assert config.ground_truth_ids == ground_truth_ids
    assert config.ground_truth_ids_display_names == ground_truth_ids_display_names
    assert config.colours == colours
    assert config.fill_holes == fill_holes
    assert config.class_weights == class_weights
    assert config.feature_channels == [num_feature_channels]
    assert config.slice_exclusion_rules == slice_exclusion_rules
    assert config.summed_probability_rules == summed_probability_rules


def test_head_and_neck_base_with_invalid_slice_exclusion_rule() -> None:
    """
    Check that an invalid slice exclusion rule raises an Exception.
    """
    ground_truth_ids = DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
    slice_exclusion_rules = [SliceExclusionRule("brainstem2", "spinal_cord", False)]
    with pytest.raises(Exception):
        _ = HeadAndNeckBase(
            ground_truth_ids=ground_truth_ids,
            slice_exclusion_rules=slice_exclusion_rules)


def test_head_and_neck_base_with_invalid_slice_exclusion_rule2() -> None:
    """
    Check that an invalid slice exclusion rule raises a Exception.
    """
    ground_truth_ids = DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
    slice_exclusion_rules = [SliceExclusionRule("brainstem", "spinal_cord2", False)]
    with pytest.raises(Exception):
        _ = HeadAndNeckBase(
            ground_truth_ids=ground_truth_ids,
            slice_exclusion_rules=slice_exclusion_rules)


def test_head_and_neck_base_with_invalid_summed_probability_rule() -> None:
    """
    Check that an invalid summed probability rule raises a ValueError.
    """
    ground_truth_ids = DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
    summed_probability_rules = [SummedProbabilityRule("spinal_cord2", "brainstem", "external")]
    with pytest.raises(ValueError):
        _ = HeadAndNeckBase(
            ground_truth_ids=ground_truth_ids,
            summed_probability_rules=summed_probability_rules)


def test_head_and_neck_base_with_invalid_summed_probability_rule2() -> None:
    """
    Check that an invalid summed probability rule raises a ValueError.
    """
    ground_truth_ids = DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
    summed_probability_rules = [SummedProbabilityRule("spinal_cord2", "brainstem2", "external")]
    with pytest.raises(ValueError):
        _ = HeadAndNeckBase(
            ground_truth_ids=ground_truth_ids,
            summed_probability_rules=summed_probability_rules)


def test_head_and_neck_base_with_invalid_summed_probability_rule3() -> None:
    """
    Check that an invalid summed probability rule raises a ValueError.
    """
    ground_truth_ids = DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
    summed_probability_rules = [SummedProbabilityRule("spinal_cord", "brainstem", "external2")]
    with pytest.raises(ValueError):
        _ = HeadAndNeckBase(
            ground_truth_ids=ground_truth_ids,
            summed_probability_rules=summed_probability_rules)


def test_head_and_neck_paper_with_no_ground_truth_ids() -> None:
    """
    Check that passing num_structures = default generates all default structures.
    """
    ground_truth_ids = DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
    config = HeadAndNeckPaper()
    assert config.ground_truth_ids == ground_truth_ids


def test_head_and_neck_paper_with_0_ground_truth_ids() -> None:
    """
    Check that passing num_structures = 0 raises ValueError exception.
    """
    with pytest.raises(ValueError):
        _ = HeadAndNeckPaper(num_structures=0)


@pytest.mark.parametrize("ground_truth_count", list(range(1, len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS), 3)))
def test_head_and_neck_paper_with_some_ground_truth_ids(
        ground_truth_count: int) -> None:
    """
    Check that passing a num_structures between 1 and len(defaults) generates the correct subset.
    """
    ground_truth_ids = DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS[:ground_truth_count]
    config = HeadAndNeckPaper(num_structures=ground_truth_count)
    assert config.ground_truth_ids == ground_truth_ids


def test_head_and_neck_paper_with_too_many_ground_truth_ids() -> None:
    """
    Check that passing num_structures larger than len(defaults) raises ValueError exception.
    """
    ground_truth_count = len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS) + 2
    with pytest.raises(ValueError):
        _ = HeadAndNeckPaper(num_structures=ground_truth_count)


@pytest.mark.parametrize("ground_truth_count", list(range(1, len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS), 3)))
def test_head_and_neck_paper_with_optional_params(
        ground_truth_count: int) -> None:
    """
    Check that optional parameters can be passed in.
    """
    ground_truth_ids = DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS[:ground_truth_count]
    ground_truth_ids_display_names = generate_random_display_ids(ground_truth_count)
    colours = generate_random_colours_list(RANDOM_COLOUR_GENERATOR, ground_truth_count)
    fill_holes = generate_random_fill_holes(ground_truth_count)
    class_weights = generate_random_class_weights(ground_truth_count + 1)
    num_feature_channels = random.randint(1, ground_truth_count)
    slice_exclusion_rules = generate_random_slice_exclusion_rules(ground_truth_ids)
    summed_probability_rules = generate_random_summed_probability_rules(ground_truth_ids)
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
    """
    Check that passing too many colours raises ValueError exception.
    """
    ground_truth_count = len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS) - 2
    ground_truth_ids_display_names = generate_random_display_ids(ground_truth_count - 1)
    with pytest.raises(ValueError):
        _ = HeadAndNeckPaper(num_structures=ground_truth_count,
                             ground_truth_ids_display_names=ground_truth_ids_display_names)


def test_head_and_neck_paper_with_mismatched_colours_raises() -> None:
    """
    Check that passing too many colours raises ValueError exception.
    """
    ground_truth_count = len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS) - 2
    colours = generate_random_colours_list(RANDOM_COLOUR_GENERATOR, ground_truth_count - 1)
    with pytest.raises(ValueError):
        _ = HeadAndNeckPaper(num_structures=ground_truth_count, colours=colours)


def test_head_and_neck_paper_with_mismatched_fill_holes_raises() -> None:
    """
    Check that passing too many colours raises ValueError exception.
    """
    ground_truth_count = len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS) - 2
    fill_holes = generate_random_fill_holes(ground_truth_count - 1)
    with pytest.raises(ValueError):
        _ = HeadAndNeckPaper(num_structures=ground_truth_count, fill_holes=fill_holes)


def test_head_and_neck_paper_with_mismatched_class_weights_raises() -> None:
    """
    Check that passing too many colours raises ValueError exception.
    """
    ground_truth_count = len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS) - 2
    class_weights = generate_random_class_weights(ground_truth_count - 1)
    with pytest.raises(ValueError):
        _ = HeadAndNeckPaper(num_structures=ground_truth_count, class_weights=class_weights)


def test_prostate_base() -> None:
    """
    Check that ProstateBase class can be instantiated.
    """
    ground_truth_ids = DEFAULT_PROSTATE_GROUND_TRUTH_IDS
    config = ProstateBase(ground_truth_ids)
    assert config.ground_truth_ids == ground_truth_ids


def test_prostate_base_with_optional_params() -> None:
    """
    Check that optional parameters can be passed in to ProstateBase class.
    """
    ground_truth_ids = DEFAULT_PROSTATE_GROUND_TRUTH_IDS
    ground_truth_count = len(ground_truth_ids)
    ground_truth_ids_display_names = generate_random_display_ids(ground_truth_count)
    colours = generate_random_colours_list(RANDOM_COLOUR_GENERATOR, ground_truth_count)
    fill_holes = generate_random_fill_holes(ground_truth_count)
    class_weights = generate_random_class_weights(ground_truth_count + 1)
    largest_connected_component_foreground_classes = DEFAULT_PROSTATE_GROUND_TRUTH_IDS[1:3]
    config = ProstateBase(
        ground_truth_ids,
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


def test_prostate_paper() -> None:
    """
    Check that ProstatePaper class can be instantiated.
    """
    ground_truth_ids = DEFAULT_PROSTATE_GROUND_TRUTH_IDS
    config = ProstatePaper()
    assert config.ground_truth_ids == ground_truth_ids


def test_prostate_paper_with_optional_params() -> None:
    """
    Check that optional parameters can be passed in to ProstatePaper class.
    """
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
