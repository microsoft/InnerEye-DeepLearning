#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

from InnerEye.ML.config import equally_weighted_classes
from .ProstateBase import ProstateBase

fg_classes = ["external", "femur_r", "femur_l", "rectum", "prostate", "bladder", "seminalvesicles"]
fg_display_names = ["External", "Femur_R", "Femur_L", "Rectum", "Prostate", "Bladder", "SeminalVesicles"]


class ProstatePaper(ProstateBase):
    """
    Prostate radiotherapy image segmentation model, as in the paper.
    """

    def __init__(self, **kwargs: Any) -> None:
        '''
        Creates a new instance of the class.
        :param kwargs: Additional arguments that will be passed through to the SegmentationModelBase constructor.
        '''
        ground_truth_ids = fg_classes
        ground_truth_ids_display_names = kwargs.pop("ground_truth_ids_display_names",
                                                    [f"zz_{name}" for name in fg_display_names])
        colours = kwargs.pop("colours", [(255, 0, 0)] * len(ground_truth_ids))
        fill_holes = kwargs.pop("fill_holes", [True, True, True, True, True, False, True])
        class_weights = kwargs.pop("class_weights", equally_weighted_classes(ground_truth_ids, background_weight=0.02))
        largest_connected_component_foreground_classes = kwargs.pop("largest_connected_component_foreground_classes",
                                                                    [name for name in ground_truth_ids if
                                                                     name != "seminalvesicles"])
        super().__init__(
            ground_truth_ids=ground_truth_ids,
            ground_truth_ids_display_names=ground_truth_ids_display_names,
            colours=colours,
            fill_holes=fill_holes,
            class_weights=class_weights,
            largest_connected_component_foreground_classes=largest_connected_component_foreground_classes,
            **kwargs
        )
