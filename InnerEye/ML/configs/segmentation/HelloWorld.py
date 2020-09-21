#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

from azureml.train.estimator import Estimator
from azureml.train.hyperdrive import BanditPolicy, HyperDriveConfig, PrimaryMetricGoal, RandomParameterSampling, uniform

from InnerEye.ML.common import TrackedMetrics
from InnerEye.ML.config import PhotometricNormalizationMethod, SegmentationModelBase
from InnerEye.ML.utils.split_dataset import DatasetSplits
from Tests.fixed_paths_for_tests import full_ml_test_data_path


class HelloWorld(SegmentationModelBase):
    """
    This is a very basic model that is pre-configured to train on the CPU for 2 epochs on a dummy dataset
    ../Tests/ML/test_data/dataset.csv

    The aim of this config is to demonstrate how to:
    1) Subclass SegmentationModelBase which is the base config for all segmentation model configs
    2) Configure the UNet3D implemented in this package
    3) Configure Azure HyperDrive based parameter search

    - This model can be trained from the commandline: ../InnerEye/runner.py --model=HelloWorld --train=True
    - If you have setup AzureML then parameter search can be performed for this model by running:
    ../InnerEye/runner.py --model=HelloWorld --hyperdrive=True
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            architecture="UNet3D",
            local_dataset=full_ml_test_data_path(),
            feature_channels=[2] * 8,
            crop_size=(64, 64, 64),
            image_channels=["channel1", "channel2"],
            ground_truth_ids=["region", "region_1"],
            mask_id="mask",
            norm_method=PhotometricNormalizationMethod.CtWindow,
            level=50,
            window=200,
            num_dataload_workers=0,
            train_batch_size=2,
            start_epoch=0,
            num_epochs=2,
            save_start_epoch=1,
            save_step_epochs=1,
            test_start_epoch=2,
            test_diff_epochs=1,
            test_step_epochs=1,
            use_mixed_precision=True
        )
        self.add_and_validate(kwargs)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_subject_ids(
            df=dataset_df,
            train_ids=[1, 2, 3],
            val_ids=[4, 5],
            test_ids=[6],
        )

    def get_parameter_search_hyperdrive_config(self, estimator: Estimator) -> HyperDriveConfig:
        """
        Specify an Azure HyperDrive configuration.
        Further details are described in the tutorial
        https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-tune-hyperparameters
        A reference is provided at https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py
        :param estimator: The estimator (configured PyTorch environment) of the experiment.
        :return: An Azure HyperDrive run configuration (configured PyTorch environment).
        """
        parameter_space = {
            'l_rate': uniform(0.0005, 0.01)
        }

        param_sampling = RandomParameterSampling(parameter_space)

        # early terminate poorly performing runs
        early_termination_policy = BanditPolicy(slack_factor=0.15, evaluation_interval=1, delay_evaluation=10)

        return HyperDriveConfig(
            estimator=estimator,
            hyperparameter_sampling=param_sampling,
            policy=early_termination_policy,
            primary_metric_name=TrackedMetrics.Val_Loss.value,
            primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
            max_total_runs=10,
            max_concurrent_runs=2
        )
