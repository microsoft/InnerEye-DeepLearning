#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import numpy
import pandas as pd
from azureml.train.estimator import Estimator
from azureml.train.hyperdrive import BanditPolicy, HyperDriveConfig, PrimaryMetricGoal, RandomParameterSampling, uniform

from InnerEye.ML.common import TrackedMetrics
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
            num_dataload_workers=8,
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
        test = list(dataset_df[dataset_df.tags.str.contains("ContinuousLearning")].subject.unique())
        train_val = list(dataset_df[~dataset_df.subject.isin(test)].subject.unique())

        val = numpy.random.choice(train_val, int(len(train_val) * 0.1), replace=False)
        train = [x for x in train_val if x not in val]

        return DatasetSplits.from_subject_ids(
            df=dataset_df,
            test_ids=test,
            val_ids=val,
            train_ids=train
        )

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

    def get_cross_validation_hyperdrive_config(self, estimator: Estimator) -> HyperDriveConfig:
        return super().get_cross_validation_hyperdrive_config(estimator)
