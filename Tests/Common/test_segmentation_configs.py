#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List, Optional

import pytest
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.configs.segmentation.HeadAndNeckBase import HeadAndNeckBase
from InnerEye.ML.configs.segmentation.HeadAndNeckPaper import STRUCTURE_LIST as DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS, HeadAndNeckPaper
from InnerEye.ML.configs.segmentation.ProstateBase import ProstateBase
from InnerEye.ML.configs.segmentation.ProstatePaper import fg_classes as DEFAULT_PROSTATE_GROUND_TRUTH_IDS, ProstatePaper

def check_hn_ground_truths(config,expected_ground_truth_ids:List[str]):
    assert len(config.class_weights) == len(expected_ground_truth_ids) + 1
    assert config.ground_truth_ids == expected_ground_truth_ids
    assert len(config.ground_truth_ids_display_names) == len(expected_ground_truth_ids)
    assert len(config.colours) == len(expected_ground_truth_ids)
    assert len(config.fill_holes) == len(expected_ground_truth_ids)

def test_head_and_neck_for_testing_with_0_ground_truth_ids() -> None:
    ground_truth_ids=DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
    config=HeadAndNeckBase(None)
    check_hn_ground_truths(config, ground_truth_ids)

def test_head_and_neck_for_testing_with_3_ground_truth_ids() -> None:
    ground_truth_ids=["parotid_r", "parotid_l", "larynx"]
    config=HeadAndNeckBase(ground_truth_ids)
    check_hn_ground_truths(config, ground_truth_ids)

def test_head_and_neck_paper_with_0_ground_truth_ids() -> None:
    ground_truth_ids=DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
    config=HeadAndNeckPaper(num_structures=0)
    check_hn_ground_truths(config, ground_truth_ids)

def test_head_and_neck_paper_with_2_ground_truth_ids() -> None:
    ground_truth_ids=DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS[:2]
    config=HeadAndNeckPaper(num_structures=2)
    check_hn_ground_truths(config, ground_truth_ids)

def test_head_and_neck_paper_with_many_ground_truth_ids() -> None:
    ground_truth_count = len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS) - 2
    ground_truth_ids=DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS[:ground_truth_count]
    config=HeadAndNeckPaper(num_structures=ground_truth_count)
    check_hn_ground_truths(config, ground_truth_ids)

def test_head_and_neck_paper_with_too_many_ground_truth_ids() -> None:
    ground_truth_count = len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS) + 2
    ground_truth_ids=DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
    config=HeadAndNeckPaper(num_structures=ground_truth_count)
    check_hn_ground_truths(config, ground_truth_ids)

def test_prostate_paper() -> None:
    ground_truth_ids=DEFAULT_PROSTATE_GROUND_TRUTH_IDS
    config=ProstatePaper()
    check_hn_ground_truths(config, ground_truth_ids)
