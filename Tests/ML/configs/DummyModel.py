#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pandas as pd
from typing import Any
from azureml.train.estimator import Estimator
from azureml.train.hyperdrive import HyperDriveConfig

from InnerEye.ML.config import PhotometricNormalizationMethod, SegmentationModelBase
from InnerEye.ML.deep_learning_config import OptimizerType
from InnerEye.ML.utils.split_dataset import DatasetSplits
from Tests.fixed_paths_for_tests import full_ml_test_data_path


class DummyModel(SegmentationModelBase):
    fg_ids = ["region"]

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
            train_batch_size=10,
            start_epoch=0,
            num_epochs=2,
            l_rate=1e-3,
            l_rate_polynomial_gamma=0.9,
            optimizer_type=OptimizerType.RMSprop,
            opt_eps=1e-4,
            rms_alpha=0.9,
            adam_betas=(0.9, 0.999),
            momentum=0.6,
            weight_decay=1e-4,
            save_start_epoch=1,
            save_step_epochs=100,
            class_weights=[0.5, 0.5],
            detect_anomaly=False,
            use_mixed_precision=False,
            test_start_epoch=1,
            test_diff_epochs=1,
            test_step_epochs=1,
        )
        self.add_and_validate(kwargs)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits(train=dataset_df[dataset_df.subject.isin(['1', '2'])],
                             test=dataset_df[dataset_df.subject.isin(['3', '4'])],
                             val=dataset_df[dataset_df.subject.isin(['5', '6'])])

    def get_parameter_search_hyperdrive_config(self, estimator: Estimator) -> HyperDriveConfig:
        return super().get_parameter_search_hyperdrive_config(estimator)

    def get_cross_validation_hyperdrive_config(self, estimator: Estimator) -> HyperDriveConfig:
        return super().get_cross_validation_hyperdrive_config(estimator)
