#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import pandas as pd
from azureml.train.estimator import Estimator
from azureml.train.hyperdrive import BanditPolicy, HyperDriveConfig, PrimaryMetricGoal, RandomParameterSampling, uniform

from InnerEye.ML.common import TrackedMetrics
from InnerEye.ML.config import ModelArchitectureConfig, PhotometricNormalizationMethod, SegmentationModelBase, \
    equally_weighted_classes
from InnerEye.ML.deep_learning_config import OptimizerType
from InnerEye.ML.utils.split_dataset import DatasetSplits


class GbmBase(SegmentationModelBase):
    """
    Glioblastoma malignant brain tumour segmentation in MR scans.
    """

    def __init__(self, **kwargs: Any) -> None:
        fg_classes = ["tumour_mass"]
        super().__init__(
            should_validate=False,
            architecture=ModelArchitectureConfig.UNet3D,
            feature_channels=[32],
            crop_size=(64, 192, 160),
            kernel_size=3,
            test_crop_size=(256, 320, 320),  # This encloses all images in the dataset.
            inference_stride_size=(128, 160, 160),
            inference_batch_size=1,
            image_channels=["mr"],
            ground_truth_ids=fg_classes,
            ground_truth_ids_display_names=fg_classes,
            colours=[(255, 255, 255)] * len(fg_classes),
            fill_holes=[False] * len(fg_classes),
            num_dataload_workers=8,
            mask_id=None,
            norm_method=PhotometricNormalizationMethod.MriWindow,
            trim_percentiles=(1, 99),
            sharpen=2.5,
            tail=[1.0],
            class_weights=equally_weighted_classes(fg_classes),
            train_batch_size=8,
            start_epoch=0,
            num_epochs=200,
            l_rate=1e-3,
            l_rate_polynomial_scheduler_gamma=0.9,
            optimizer_type=OptimizerType.Adam,
            opt_eps=1e-4,
            adam_betas=(0.9, 0.999),
            momentum=0.9,
            weight_decay=1e-4,
            save_start_epoch=50,
            save_step_epochs=10,
            test_start_epoch=50,
            test_diff_epochs=20,
            test_step_epochs=10,
            use_mixed_precision=True,
            use_model_parallel=True,
        )
        self.add_and_validate(kwargs)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_institutions(
            df=dataset_df,
            proportion_train=0.6,
            proportion_test=0.2,
            proportion_val=0.2,
            shuffle=True
        )

    def get_cross_validation_hyperdrive_config(self, estimator: Estimator) -> HyperDriveConfig:
        return super().get_cross_validation_hyperdrive_config(estimator)

    def get_parameter_search_hyperdrive_config(self, estimator: Estimator) -> HyperDriveConfig:
        """
        Specify an Azure Hyperdrive configuration.
        Further details are described in the tutorial
        https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-tune-hyperparameters
        A reference is provided at https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train
        .hyperdrive?view=azure-ml-py
        :param estimator: The estimator (configured PyTorch environment) of the experiment.
        :return: An Azure Hyperdrive run configuration (configured PyTorch environment).
        """
        parameter_space = {
            'l_rate': uniform(0.0005, 0.01)
        }

        param_sampling = RandomParameterSampling(parameter_space)

        # early terminate poorly performing runs
        early_termination_policy = BanditPolicy(slack_factor=0.15, evaluation_interval=1, delay_evaluation=10)

        config = HyperDriveConfig(estimator=estimator,
                                  hyperparameter_sampling=param_sampling,
                                  policy=early_termination_policy,
                                  primary_metric_name=TrackedMetrics.Val_Loss.value,
                                  primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
                                  max_total_runs=64,
                                  max_concurrent_runs=8
                                  )

        return config
