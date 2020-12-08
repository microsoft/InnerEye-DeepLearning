#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List, Optional

import pytest
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.configs.segmentation.HeadAndNeckBase import STRUCTURE_LIST as DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS, HeadAndNeckBase
from InnerEye.ML.configs.segmentation.ProstateBase import ProstateBase

DEFAULT_PROSTATE_GROUND_TRUTH_IDS = (ProstateBase()).ground_truth_ids

class HeadNeckForTesting(HeadAndNeckBase):
    '''
    Prototype base class for Head and Neck models.
    '''
    def __init__(self,
                 ground_truth_ids: List[str],
                 ground_truth_ids_display_names: Optional[List[str]] = None,
                 fill_holes: Optional[List[bool]] = None,
                 class_weights: Optional[List[float]] = None,
                 colours: Optional[List[TupleInt3]] = None,
                 **kwargs: Any) -> None:
        '''
        Creates a new instance of the class.
        :param ground_truth_ids: List of ground truth ids.
        :param ground_truth_ids_display_names: Optional list of ground truth id display names. If
        present then must be of the same length as ground_truth_ids.
        :param fill_holes: Optional list of fill hole flags. If
        present then must be of the same length as ground_truth_ids.
        :param class_weights: Optional list of class weights. If
        present then must be of the same length as ground_truth_ids + 1.
        :param colours: Optional list of colours. If
        present then must be of the same length as ground_truth_ids.
        :param kwargs: Additional arguments that will be passed through to the SegmentationModelBase constructor.
        '''
        super().__init__(
            ground_truth_ids=ground_truth_ids)

class HeadNeckPaper(HeadAndNeckBase):
    '''
    Head and Neck model, as used in the paper.
    '''
    def __init__(self, num_structures: int = 0, **kwargs: Any) -> None:
        '''
        Creates a new instance of the class.
        :param num_structures: Optional number of structures to include. Defaults to all.
        :param kwargs: Additional arguments that will be passed through to the SegmentationModelBase constructor.
        '''
        super().__init__(
            num_structures,
            kwargs)

class ProstateForTesting(ProstateBase):
    '''
    Prototype base class for Prostate models.
    '''
    def __init__(self,
                 fg_classes: List[str],
                 fg_display_names: Optional[List[str]] = None,
                 fill_holes: Optional[List[bool]] = None,
                 class_weights: Optional[List[float]] = None,
                 colours: Optional[List[TupleInt3]] = None,
                 **kwargs: Any) -> None:
        '''
        Creates a new instance of the class.
        :param fg_classes: List of ground truth ids.
        :param fg_display_names: Optional list of ground truth id display names. If
        present then must be of the same length as fg_classes.
        :param fill_holes: Optional list of fill hole flags. If
        present then must be of the same length as fg_classes.
        :param class_weights: Optional list of class weights. If
        present then must be of the same length as fg_classes + 1.
        :param colours: Optional list of colours. If
        present then must be of the same length as fg_classes.
        :param kwargs: Additional arguments that will be passed through to the SegmentationModelBase constructor.
        '''

class ProstatePaper(ProstateBase):
    '''
    Prostate model, as used in the paper.
    '''
    def __init__(self, **kwargs: Any) -> None:
        '''
        Creates a new instance of the class.
        :param kwargs: Additional arguments that will be passed through to the SegmentationModelBase constructor.
        '''
        super().__init__(
            kwargs)

def check_hn_ground_truths(config,expected_ground_truth_ids:List[str]):
    assert len(config.class_weights) == len(expected_ground_truth_ids) + 1
    assert config.ground_truth_ids == expected_ground_truth_ids
    assert len(config.ground_truth_ids_display_names) == len(expected_ground_truth_ids)
    assert len(config.colours) == len(expected_ground_truth_ids)
    assert len(config.fill_holes) == len(expected_ground_truth_ids)

def test_head_and_neck_for_testing_with_0_ground_truth_ids() -> None:
    ground_truth_ids=DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
    config=HeadNeckForTesting(None)
    check_hn_ground_truths(config, ground_truth_ids)

def test_head_and_neck_for_testing_with_3_ground_truth_ids() -> None:
    ground_truth_ids=["parotid_r", "parotid_l", "larynx"]
    config=HeadNeckForTesting(ground_truth_ids)
    check_hn_ground_truths(config, ground_truth_ids)

def test_head_and_neck_paper_with_0_ground_truth_ids() -> None:
    ground_truth_ids=DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
    config=HeadNeckPaper(num_structures=0)
    check_hn_ground_truths(config, ground_truth_ids)

def test_head_and_neck_paper_with_2_ground_truth_ids() -> None:
    ground_truth_ids=DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS[:2]
    config=HeadNeckPaper(num_structures=2)
    check_hn_ground_truths(config, ground_truth_ids)

def test_head_and_neck_paper_with_many_ground_truth_ids() -> None:
    ground_truth_count = len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS) - 2
    ground_truth_ids=DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS[:ground_truth_count]
    config=HeadNeckPaper(num_structures=ground_truth_count)
    check_hn_ground_truths(config, ground_truth_ids)

def test_head_and_neck_paper_with_too_many_ground_truth_ids() -> None:
    ground_truth_count = len(DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS) + 2
    ground_truth_ids=DEFAULT_HEAD_AND_NECK_GROUND_TRUTH_IDS
    config=HeadNeckPaper(num_structures=ground_truth_count)
    check_hn_ground_truths(config, ground_truth_ids)

def test_prostate_paper_with_0_ground_truth_ids() -> None:
    ground_truth_ids=DEFAULT_PROSTATE_GROUND_TRUTH_IDS
    config=ProstatePaper(num_structures=0)
    check_hn_ground_truths(config, ground_truth_ids)

def test_prostate_paper_with_2_ground_truth_ids() -> None:
    ground_truth_ids=DEFAULT_PROSTATE_GROUND_TRUTH_IDS[:2]
    config=ProstatePaper(num_structures=2)
    check_hn_ground_truths(config, ground_truth_ids)

def test_prostate_paper_with_many_ground_truth_ids() -> None:
    ground_truth_count = len(DEFAULT_PROSTATE_GROUND_TRUTH_IDS) - 2
    ground_truth_ids=DEFAULT_PROSTATE_GROUND_TRUTH_IDS[:ground_truth_count]
    config=ProstatePaper(num_structures=ground_truth_count)
    check_hn_ground_truths(config, ground_truth_ids)

def test_prostate_paper_with_too_many_ground_truth_ids() -> None:
    ground_truth_count = len(DEFAULT_PROSTATE_GROUND_TRUTH_IDS) + 2
    ground_truth_ids=DEFAULT_PROSTATE_GROUND_TRUTH_IDS
    config=ProstatePaper(num_structures=ground_truth_count)
    check_hn_ground_truths(config, ground_truth_ids)

