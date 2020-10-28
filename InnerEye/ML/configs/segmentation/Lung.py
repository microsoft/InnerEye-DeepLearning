#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import numpy
import pandas as pd

from InnerEye.ML.config import MixtureLossComponent, PhotometricNormalizationMethod, SegmentationLoss, \
    SegmentationModelBase, equally_weighted_classes
from InnerEye.ML.deep_learning_config import OptimizerType
from InnerEye.ML.utils.split_dataset import DatasetSplits

# Change this string to the name of your dataset on Azure blob storage.
AZURE_DATASET_ID = "2339eba2-8ec5-4ccb-86ff-c170470ac6e2_geonorm_with_train_test_split_2020_05_26"


class Lung(SegmentationModelBase):
    """
    Lung CT image segmentation model
    Target structures are: Esophagus, Heart, Left and Right Lungs, and Spinal cord
    """

    def __init__(self, **kwargs: Any) -> None:
        fg_classes = ["spinalcord", "lung_r", "lung_l", "heart", "esophagus"]
        fg_display_names = ["SpinalCord", "Lung_R", "Lung_L", "Heart", "Esophagus"]
        super().__init__(
            architecture="UNet3D",
            feature_channels=[32],
            kernel_size=3,
            azure_dataset_id=AZURE_DATASET_ID,
            crop_size=(64, 224, 224),
            test_crop_size=(128, 512, 512),
            image_channels=["ct"],
            ground_truth_ids=fg_classes,
            ground_truth_ids_display_names=fg_display_names,
            colours=[(255, 255, 255)] * len(fg_classes),
            fill_holes=[False] * len(fg_classes),
            largest_connected_component_foreground_classes=["lung_r", "lung_l", "heart"],
            num_dataload_workers=8,
            norm_method=PhotometricNormalizationMethod.CtWindow,
            level=40,
            window=400,
            class_weights=equally_weighted_classes(fg_classes, background_weight=0.02),
            train_batch_size=8,
            inference_batch_size=1,
            inference_stride_size=(64, 256, 256),
            start_epoch=0,
            num_epochs=140,
            l_rate=1e-3,
            min_l_rate=1e-5,
            l_rate_polynomial_gamma=0.9,
            optimizer_type=OptimizerType.Adam,
            opt_eps=1e-4,
            adam_betas=(0.9, 0.999),
            momentum=0.9,
            weight_decay=1e-4,
            save_start_epoch=100,
            save_step_epochs=20,
            test_start_epoch=140,
            use_mixed_precision=True,
            use_model_parallel=True,
            monitoring_interval_seconds=0,
            test_diff_epochs=1,
            test_step_epochs=1,
            loss_type=SegmentationLoss.Mixture,
            mixture_loss_components=[MixtureLossComponent(0.5, SegmentationLoss.Focal, 0.2),
                                     MixtureLossComponent(0.5, SegmentationLoss.SoftDice, 0.1)],
        )
        self.add_and_validate(kwargs)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        # The first 24 subject IDs are the designated test subjects in this dataset.
        test = list(map(str, range(0, 24)))
        train_val = list(dataset_df[~dataset_df.subject.isin(test)].subject.unique())

        val = list(map(str, numpy.random.choice(train_val, int(len(train_val) * 0.1), replace=False)))
        train = [str(x) for x in train_val if x not in val]

        return DatasetSplits.from_subject_ids(
            df=dataset_df,
            test_ids=test,
            val_ids=val,
            train_ids=train
        )
