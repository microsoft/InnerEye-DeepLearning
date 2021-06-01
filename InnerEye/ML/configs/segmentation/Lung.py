#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any
from azureml.train.hyperdrive import BayesianParameterSampling, NoTerminationPolicy, choice
import numpy
import pandas as pd
from azureml.core import ScriptRunConfig
from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal

from InnerEye.Common.metrics_constants import TrackedMetrics
from InnerEye.ML.config import PhotometricNormalizationMethod, SegmentationLoss, \
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
            roi_interpreted_types=["ORGAN"] * len(fg_classes),
            largest_connected_component_foreground_classes=["lung_r", "lung_l", "heart"],
            num_dataload_workers=0,
            norm_method=PhotometricNormalizationMethod.CtWindow,
            level=0,
            window=4000,
            class_weights=equally_weighted_classes(fg_classes, background_weight=0.02),
            train_batch_size=8,
            inference_batch_size=1,
            inference_stride_size=(64, 256, 256),
            num_epochs=140,
            l_rate=1e-3,
            min_l_rate=1e-5,
            l_rate_polynomial_gamma=0.9,
            check_exclusive=False,
            optimizer_type=OptimizerType.Adam,
            opt_eps=1e-4,
            adam_betas=(0.9, 0.999),
            momentum=0.9,
            weight_decay=1e-4,
            recovery_checkpoint_save_interval=10,
            use_mixed_precision=True,
            use_model_parallel=True,
            monitoring_interval_seconds=0,
            loss_type=SegmentationLoss.SoftDice,
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

    def get_parameter_search_hyperdrive_config(self, run_config: ScriptRunConfig) -> HyperDriveConfig:
        """
        Specify an Azure HyperDrive configuration.
        Further details are described in the tutorial
        https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-tune-hyperparameters
        A reference is provided at https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train
        .hyperdrive?view=azure-ml-py
        :param run_config: The configuration for running an individual experiment.
        :return: An Azure HyperDrive run configuration (configured PyTorch environment).
        """
        crop_sizes = [f'{z}, {x}, {x}' for x in range(128, 512, 32) for z in [64, 112]]

        parameter_space = {
            'crop_size': choice(crop_sizes),
            'feature_channels': choice(16, 32),
            'l_rate_step_step_size': choice(20, 40, 75),
            'l_rate': choice(0.0005, 0.001, 0.01),
            'train_batch_size': choice(1, 2, 4, 8, 16, 32),
            'num_epochs': choice(50, 80, 150),
            'loss_type': choice("SoftDice", "Focal")
        }

        param_sampling = BayesianParameterSampling(parameter_space)

        return HyperDriveConfig(
            run_config=run_config,
            hyperparameter_sampling=param_sampling,
            policy=NoTerminationPolicy(),
            primary_metric_name=TrackedMetrics.Val_Loss.value,
            primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
            max_total_runs=12,
            max_concurrent_runs=2,
            max_duration_minutes=30,  # Max 2 hours per experiment
        )
