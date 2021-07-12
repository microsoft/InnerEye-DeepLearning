#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import pandas as pd

from InnerEye.ML.config import PhotometricNormalizationMethod, SegmentationModelBase, equally_weighted_classes
from InnerEye.ML.configs.segmentation.Lung import AZURE_DATASET_ID
from InnerEye.ML.deep_learning_config import LRSchedulerType
from InnerEye.ML.utils.split_dataset import DatasetSplits

fg_classes = ["spinalcord", "lung_r", "lung_l"]

default_single_comparison_blob = "refs_pull_483_merge_1624269679_90b1d23c/outputs/best_validation_epoch/Test"


class BasicModel2Epochs(SegmentationModelBase):
    def __init__(self, **kwargs: Any) -> None:
        comparison_blob_storage_paths = kwargs.pop("comparison_blob_storage_paths",
                                                   [("Single", default_single_comparison_blob)])
        super().__init__(
            should_validate=False,
            architecture="Basic",
            feature_channels=[2] * 8,
            crop_size=(64, 64, 64),
            image_channels=["ct", "heart"],
            # Test with multiple channels, even though the "heart" is clearly nonsense
            ground_truth_ids=fg_classes,
            ground_truth_ids_display_names=fg_classes,
            colours=[(255, 255, 255)] * len(fg_classes),
            fill_holes=[False] * len(fg_classes),
            roi_interpreted_types=["ORGAN"] * len(fg_classes),
            mask_id="heart",
            norm_method=PhotometricNormalizationMethod.CtWindow,
            level=50,
            window=200,
            class_weights=equally_weighted_classes(fg_classes),
            num_dataload_workers=1,
            train_batch_size=8,
            num_epochs=2,
            recovery_checkpoint_save_interval=1,
            use_mixed_precision=True,
            azure_dataset_id=AZURE_DATASET_ID,
            comparison_blob_storage_paths=comparison_blob_storage_paths,
            inference_on_test_set=True,
            inference_on_val_set=True,
            dataset_mountpoint="/tmp/innereye",
            # Use an LR scheduler with a pronounced and clearly visible decay, to be able to easily see if that
            # is applied correctly in run recovery.
            l_rate=1e-4,
            l_rate_scheduler=LRSchedulerType.Step,
            l_rate_step_step_size=1,
            l_rate_step_gamma=0.9,
            # Necessary to avoid https://github.com/pytorch/pytorch/issues/45324
            max_num_gpus=2,
        )
        self.add_and_validate(kwargs)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_subject_ids(
            df=dataset_df,
            train_ids=['0', '1'],
            test_ids=['5'],
            val_ids=['2']
        )
