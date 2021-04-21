#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List, Optional

import pandas as pd

from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.config import PhotometricNormalizationMethod, SegmentationModelBase, equally_weighted_classes
from InnerEye.ML.deep_learning_config import OptimizerType
from InnerEye.ML.utils.split_dataset import DatasetSplits


class ProstateBase(SegmentationModelBase):
    """
    Prostate radiotherapy image segmentation model.
    """

    def __init__(self,
                 ground_truth_ids: List[str],
                 ground_truth_ids_display_names: Optional[List[str]] = None,
                 colours: Optional[List[TupleInt3]] = None,
                 fill_holes: Optional[List[bool]] = None,
                 roi_interpreted_types: Optional[List[str]] = None,
                 class_weights: Optional[List[float]] = None,
                 largest_connected_component_foreground_classes: Optional[List[str]] = None,
                 **kwargs: Any) -> None:
        """
        Creates a new instance of the class.
        :param ground_truth_ids: List of ground truth ids.
        :param ground_truth_ids_display_names: Optional list of ground truth id display names. If
        present then must be of the same length as ground_truth_ids.
        :param colours: Optional list of colours. If
        present then must be of the same length as ground_truth_ids.
        :param fill_holes: Optional list of fill hole flags. If
        present then must be of the same length as ground_truth_ids.
        :param interpreted_types: Optional list of interpreted_types. If
        present then must be of the same length as ground_truth_ids.
        :param class_weights: Optional list of class weights. If
        present then must be of the same length as ground_truth_ids + 1.
        :param kwargs: Additional arguments that will be passed through to the SegmentationModelBase constructor.
        """
        ground_truth_ids_display_names = ground_truth_ids_display_names or [f"zz_{name}" for name in ground_truth_ids]
        colours = colours or [(255, 0, 0)] * len(ground_truth_ids)
        fill_holes = fill_holes or [True] * len(ground_truth_ids)
        roi_interpreted_types = roi_interpreted_types or ["ORGAN"] * len(ground_truth_ids)
        class_weights = class_weights or equally_weighted_classes(ground_truth_ids, background_weight=0.02)
        largest_connected_component_foreground_classes = largest_connected_component_foreground_classes or \
                                                         ground_truth_ids
        super().__init__(
            should_validate=False,
            adam_betas=(0.9, 0.999),
            architecture="UNet3D",
            class_weights=class_weights,
            crop_size=(64, 224, 224),
            feature_channels=[32],
            ground_truth_ids=ground_truth_ids,
            ground_truth_ids_display_names=ground_truth_ids_display_names,
            colours=colours,
            fill_holes=fill_holes,
            roi_interpreted_types=roi_interpreted_types,
            image_channels=["ct"],
            inference_batch_size=1,
            inference_stride_size=(64, 256, 256),
            kernel_size=3,
            l_rate=1e-3,
            min_l_rate=1e-5,
            l_rate_polynomial_gamma=0.9,
            largest_connected_component_foreground_classes=largest_connected_component_foreground_classes,
            level=50,
            momentum=0.9,
            monitoring_interval_seconds=0,
            norm_method=PhotometricNormalizationMethod.CtWindow,
            num_dataload_workers=2,
            num_epochs=120,
            opt_eps=1e-4,
            optimizer_type=OptimizerType.Adam,
            test_crop_size=(128, 512, 512),
            train_batch_size=2,
            use_mixed_precision=True,
            use_model_parallel=True,
            weight_decay=1e-4,
            window=600,
            posterior_smoothing_mm=(2.0, 2.0, 3.0),
        )
        self.add_and_validate(kwargs)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        """
        Return an adjusted split
        """
        return DatasetSplits.from_proportions(dataset_df, proportion_train=0.8, proportion_val=0.05,
                                              proportion_test=0.15,
                                              random_seed=0)
