#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import pandas as pd

from InnerEye.ML.config import ModelArchitectureConfig, PhotometricNormalizationMethod, SegmentationModelBase, \
    equally_weighted_classes, SegmentationLoss
from InnerEye.ML.deep_learning_config import OptimizerType
from InnerEye.ML.utils.split_dataset import DatasetSplits

MODALITY = "mri"
LABELS = ["hippocampus_L", "hippocampus_R"]

class Hippocampus(SegmentationModelBase):
    """
    Hippocampus segmentation in MR scans.
    """

    def __init__(self, **kwargs: Any) -> None:
        fg_classes = LABELS
        super().__init__(
            should_validate=False,
            azure_dataset_id="adni_hippocampus_L-R_split_minus_holdout",
            architecture=ModelArchitectureConfig.UNet3D,
            feature_channels=[16], 
            crop_size=(128, 176, 176), 
            inference_batch_size=1,
            image_channels=[MODALITY],
            ground_truth_ids=fg_classes,
            ground_truth_ids_display_names=fg_classes,
            colours=[(255, 255, 255)] * len(fg_classes),
            fill_holes=[False] * len(fg_classes),
            roi_interpreted_types=["ORGAN"] * len(fg_classes),
            num_dataload_workers=12,
            mask_id=None,
            norm_method=PhotometricNormalizationMethod.MriWindow,
            trim_percentiles=(1, 99),
            sharpen=2.5,
            tail=[1.0],
            class_weights=equally_weighted_classes(fg_classes, background_weight=0.0),
            train_batch_size=2, 
            num_epochs=200,
            l_rate=1e-3,
            l_rate_polynomial_gamma=0.9,
            optimizer_type=OptimizerType.Adam,
            opt_eps=1e-4,
            adam_betas=(0.9, 0.999),
            momentum=0.9,
            loss_type=SegmentationLoss.SoftDice,
            weight_decay=1e-4,
            use_mixed_precision=True,
            use_model_parallel=True,
        )
        self.add_and_validate(kwargs)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_proportions(
            dataset_df,
            proportion_train=0.8,
            proportion_val=0.1,
            proportion_test=0.1,
            subject_column="subject",
            shuffle=True,
            group_column="groupID",
        )
