#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import pandas as pd
from azureml.core import ScriptRunConfig
from azureml.train.hyperdrive import HyperDriveConfig

from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.ML.config import PhotometricNormalizationMethod, SegmentationModelBase
from InnerEye.ML.deep_learning_config import OptimizerType
from InnerEye.ML.utils.split_dataset import DatasetSplits


class DummyModel(SegmentationModelBase):
    fg_ids = ["region"]
    train_subject_ids = ['1', '2', '3']
    test_subject_ids = ['4', '7']
    val_subject_ids = ['5', '6']

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            should_validate=False,  # will validate after adding the overrides in kwargs.
            random_seed=42,
            architecture="Basic",
            feature_channels=[3, 3, 4, 4, 4, 4, 5, 5],
            kernel_size=3,
            local_dataset=full_ml_test_data_path(),
            crop_size=(55, 55, 55),
            # This speeds up loading dramatically. Multi-process data loading is tested via BasicModel2Epochs
            num_dataload_workers=0,
            # Disable monitoring so that we can use VS Code remote debugging
            monitoring_interval_seconds=0,
            shuffle=True,
            image_channels=["channel1", "channel2"],
            ground_truth_ids=self.fg_ids,
            ground_truth_ids_display_names=self.fg_ids,
            colours=[(255, 255, 255)] * len(self.fg_ids),
            fill_holes=[False] * len(self.fg_ids),
            roi_interpreted_types=["Organ"] * len(self.fg_ids),
            mask_id="mask",
            dataset_expected_spacing_xyz=(1.0, 1.0, 1.0),
            norm_method=PhotometricNormalizationMethod.CtWindow,
            output_range=(-1.0, 1.0),
            level=0,
            window=400,
            debug_mode=False,
            tail=[1.0],
            sharpen=1.9,
            trim_percentiles=(1, 99),
            inference_batch_size=1,
            train_batch_size=2,
            num_epochs=2,
            l_rate=1e-3,
            l_rate_polynomial_gamma=0.9,
            optimizer_type=OptimizerType.RMSprop,
            opt_eps=1e-4,
            rms_alpha=0.9,
            adam_betas=(0.9, 0.999),
            momentum=0.6,
            weight_decay=1e-4,
            class_weights=[0.5, 0.5],
            detect_anomaly=False,
            use_mixed_precision=False)
        self.add_and_validate(kwargs)
        # Trying to run DDP from the test suite hangs, hence restrict to single GPU.
        self.max_num_gpus = 1

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits(train=dataset_df[dataset_df.subject.isin(self.train_subject_ids)],
                             test=dataset_df[dataset_df.subject.isin(self.test_subject_ids)],
                             val=dataset_df[dataset_df.subject.isin(self.val_subject_ids)])

    def get_parameter_search_hyperdrive_config(self, run_config: ScriptRunConfig) -> HyperDriveConfig:
        return super().get_parameter_search_hyperdrive_config(run_config)

    def get_cross_validation_hyperdrive_config(self, run_config: ScriptRunConfig) -> HyperDriveConfig:
        return super().get_cross_validation_hyperdrive_config(run_config)
