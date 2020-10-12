#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import pandas as pd

from InnerEye.ML.config import PhotometricNormalizationMethod, SegmentationModelBase, equally_weighted_classes
from InnerEye.ML.deep_learning_config import OptimizerType
from InnerEye.ML.utils.split_dataset import DatasetSplits


class ProstateBase(SegmentationModelBase):
    """
    Prostate radiotherapy image segmentation model.
    """

    def __init__(self, **kwargs: Any) -> None:
        fg_classes = ["external", "femur_r", "femur_l", "rectum", "prostate", "bladder", "seminalvesicles"]
        fg_display_names = ["External", "Femur_R", "Femur_L", "Rectum", "Prostate", "Bladder", "SeminalVesicles"]
        colors = [(255, 0, 0)] * len(fg_display_names)
        fill_holes = [True, True, True, True, True, False, True]
        super().__init__(
            should_validate=False,
            adam_betas=(0.9, 0.999),
            architecture="UNet3D",
            class_weights=equally_weighted_classes(fg_classes, background_weight=0.02),
            crop_size=(64, 224, 224),
            feature_channels=[32],
            ground_truth_ids=fg_classes,
            ground_truth_ids_display_names=[f"zz_{name}" for name in fg_display_names],
            colours=colors,
            fill_holes=fill_holes,
            image_channels=["ct"],
            inference_batch_size=1,
            inference_stride_size=(64, 256, 256),
            kernel_size=3,
            l_rate=1e-3,
            min_l_rate=1e-5,
            l_rate_polynomial_gamma=0.9,
            largest_connected_component_foreground_classes=[name for name in fg_classes if name != "seminalvesicles"],
            level=50,
            momentum=0.9,
            monitoring_interval_seconds=0,
            norm_method=PhotometricNormalizationMethod.CtWindow,
            num_dataload_workers=4,
            num_epochs=120,
            opt_eps=1e-4,
            optimizer_type=OptimizerType.Adam,
            save_step_epochs=20,
            start_epoch=0,
            test_crop_size=(128, 512, 512),
            test_diff_epochs=1,
            test_start_epoch=120,
            test_step_epochs=1,
            train_batch_size=8,
            use_mixed_precision=True,
            use_model_parallel=True,
            weight_decay=1e-4,
            window=600,
            posterior_smoothing_mm=(2.0, 2.0, 3.0),
            save_start_epoch=100,
        )
        self.add_and_validate(kwargs)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        """
        Return an adjusted split
        """
        return DatasetSplits.from_proportions(dataset_df, proportion_train=0.8, proportion_val=0.05,
                                              proportion_test=0.15,
                                              random_seed=0)
