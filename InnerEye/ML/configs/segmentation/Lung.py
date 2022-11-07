#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import numpy
import pandas as pd

from InnerEye.ML.config import PhotometricNormalizationMethod, SegmentationLoss, \
    SegmentationModelBase, equally_weighted_classes
from InnerEye.ML.deep_learning_config import OptimizerType
from InnerEye.ML.utils.split_dataset import DatasetSplits

# Change this string to the name of your dataset on Azure blob storage.
LUNG_AZURE_DATASET_ID = "2339eba2-8ec5-4ccb-86ff-c170470ac6e2_geonorm_with_train_test_split_2020_05_26"


class Lung(SegmentationModelBase):
    """
    Lung CT image segmentation model
    Target structures are: Esophagus, Heart, Left and Right Lungs, and Spinal cord
    """

    def __init__(self, **kwargs: Any) -> None:
        fg_classes = ["spinalcord", "lung_r", "lung_l", "heart", "esophagus"]
        fg_display_names = ["SpinalCord", "Lung_R", "Lung_L", "Heart", "Esophagus"]

        azure_dataset_id = kwargs.pop("azure_dataset_id", LUNG_AZURE_DATASET_ID)

        super().__init__(
            adam_betas=(0.9, 0.999),
            architecture="UNet3D",
            azure_dataset_id=azure_dataset_id,
            check_exclusive=False,
            class_weights=equally_weighted_classes(fg_classes, background_weight=0.02),
            colours=[(255, 255, 255)] * len(fg_classes),
            crop_size=(64, 224, 224),
            feature_channels=[32],
            fill_holes=[False] * len(fg_classes),
            ground_truth_ids_display_names=fg_display_names,
            ground_truth_ids=fg_classes,
            image_channels=["ct"],
            inference_batch_size=1,
            inference_stride_size=(64, 256, 256),
            kernel_size=3,
            l_rate_polynomial_gamma=0.9,
            l_rate=1e-3,
            largest_connected_component_foreground_classes=["lung_l", "lung_r", "heart"],
            level=-500,
            loss_type=SegmentationLoss.SoftDice,
            min_l_rate=1e-5,
            momentum=0.9,
            monitoring_interval_seconds=0,
            norm_method=PhotometricNormalizationMethod.CtWindow,
            num_dataload_workers=2,
            num_epochs=300,
            opt_eps=1e-4,
            optimizer_type=OptimizerType.Adam,
            roi_interpreted_types=["ORGAN"] * len(fg_classes),
            test_crop_size=(112, 512, 512),
            train_batch_size=3,
            use_mixed_precision=True,
            use_model_parallel=True,
            weight_decay=1e-4,
            window=2200,
        )
        self.add_and_validate(kwargs)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        test = list(map(str, range(0, 9)))
        train_val = list(dataset_df[~dataset_df.subject.isin(test)].subject.unique())

        val = list(map(str, numpy.random.choice(train_val, int(len(train_val) * 0.1), replace=False)))
        train = [str(x) for x in train_val if x not in val]

        return DatasetSplits.from_subject_ids(
            df=dataset_df,
            test_ids=test,
            val_ids=val,
            train_ids=train
        )
