#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import abc
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd
from azureml.core import ScriptRunConfig
from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, PrimaryMetricGoal, choice
from pandas import DataFrame

from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY
from InnerEye.Common.common_util import ModelProcessing
from InnerEye.Common.metrics_constants import TrackedMetrics
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode, STORED_CSV_FILE_NAMES
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.utils.split_dataset import DatasetSplits


class ModelConfigBaseMeta(type(DeepLearningConfig), abc.ABCMeta):  # type: ignore
    """
    Metaclass to make the hierarchy explicit for ModelConfigBase
    """
    pass


class ModelConfigBase(DeepLearningConfig, abc.ABC, metaclass=ModelConfigBaseMeta):

    def __init__(self, **params: Any):
        super().__init__(**params)

    def read_dataset_into_dataframe_and_pre_process(self) -> None:
        """
        Loads a dataset from a file or other source, and saves it into the model's data_frame property.
        Applying any pre-processing functions defined in pre_process_dataset_dataframe
        The data frame should contain all of the training, test, and validation data.
        """
        # This method is factually an abstract method. We don't want to mark at as such
        # because this would prevent us from easily instantiating this class in tests.
        raise NotImplementedError("read_dataset_into_dataframe must be overridden")

    def pre_process_dataset_dataframe(self) -> None:
        """
        Performs any dataframe pre-processing functions, default is identity
        :return:
        """
        pass

    def get_parameter_search_hyperdrive_config(self, run_config: ScriptRunConfig) -> HyperDriveConfig:
        """
        Returns a configuration for AzureML Hyperdrive that should be used when running hyperparameter
        tuning.
        This is an abstract method that each specific model should override.
        :param run_config: The AzureML estimator object that runs model training.
        :return: A hyperdrive configuration object.
        """
        # This method is factually an abstract method. We don't want to mark at as such
        # because this would prevent us from easily instantiating this class in tests.
        raise NotImplementedError("get_parameter_search_hyperdrive_config must be overridden")

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        """
        Computes the training, validation and test splits for the model, from a dataframe that contains
        the full dataset.
        :param dataset_df: A dataframe that contains the full dataset that the model is using.
        :return: An instance of DatasetSplits with dataframes for training, validation and testing.
        """
        # This method is factually an abstract method. We don't want to mark at as such
        # because this would prevent us from easily instantiating this class in tests.
        raise NotImplementedError("get_model_train_test_dataset_splits must be overridden")

    def create_and_set_torch_datasets(self, for_training: bool = True, for_inference: bool = True) -> None:
        """
        Creats and sets torch datasets for training and validation, and stores them in the self._datasets_for_training
        field. Similarly, create torch datasets in the form required for model inference, for all of the
        3 splits of the full data, and stored them in the self._datasets_for_training and/or
        self._datasets_for_inference fields.
        This is an abstract method that each specific model should override. If for_training and/or for_inference
        are False, the derived method *may* still create the corresponding datasets, but should not assume that
        the relevant splits (train/test/val) are non-empty. If either or both is True, they *must* create the
        corresponding datasets, and should be able to make the assumption.
        :param for_training: whether to create the datasets required for training.
        :param for_inference: whether to create the datasets required for inference.
        """
        # This method is factually an abstract method. We don't want to mark at as such
        # because this would prevent us from easily instantiating this class in tests.
        raise NotImplementedError("create_datasets must be overridden")

    def read_dataset_if_needed(self) -> DataFrame:
        """
        If the present object already stores a data frame, return it. Otherwise, read it from file.
        :return: The data frame that the model uses.
        """
        if self.dataset_data_frame is None:
            self.read_dataset_into_dataframe_and_pre_process()
        return self.dataset_data_frame

    def get_torch_dataset_for_inference(self, mode: ModelExecutionMode) -> Any:
        """
        Returns a torch Dataset for running the model in inference mode, on the given split of the full dataset.
        The torch dataset must return data in the format required for running the model in inference mode.
        :return: A torch Dataset object.
        """
        if self._datasets_for_inference is None:
            self.create_and_set_torch_datasets(for_training=False)
        assert self._datasets_for_inference is not None  # for mypy
        return self._datasets_for_inference[mode]

    def create_data_loaders(self) -> Dict[ModelExecutionMode, Any]:
        """
        Creates the torch DataLoaders that supply the training and the validation set during training only.
        :return: A dictionary, with keys ModelExecutionMode.TRAIN and ModelExecutionMode.VAL, and their respective
        data loaders.
        """
        logging.info("Starting to read and parse the datasets.")
        if self._datasets_for_training is None:
            self.create_and_set_torch_datasets(for_inference=False)
        assert self._datasets_for_training is not None  # for mypy
        if self._datasets_for_training == {}:
            return {}
        logging.info("Creating the data loader for the training set.")
        train_loader = self._datasets_for_training[ModelExecutionMode.TRAIN] \
            .as_data_loader(shuffle=self.shuffle,
                            use_imbalanced_sampler=self.use_imbalanced_sampler_for_training,
                            drop_last_batch=self.drop_last_batch_in_training,
                            max_repeats=self.get_total_number_of_training_epochs())
        logging.info("Creating the data loader for the validation set.")

        val_loader = self._datasets_for_training[ModelExecutionMode.VAL].as_data_loader(
            shuffle=False,
            max_repeats=self.get_total_number_of_validation_epochs()
        )
        logging.info("Finished creating the data loaders.")
        return {
            ModelExecutionMode.TRAIN: train_loader,
            ModelExecutionMode.VAL: val_loader
        }

    def create_model(self) -> Any:
        """
        Creates a PyTorch model from the settings stored in the present object.
        This is an abstract method that each model class (segmentation, regression) should override.
        Return type is LightningModule, not Any - but we want to avoid importing torch at this point.
        """
        # This method is factually an abstract method. We don't want to mark at as such
        # because this would prevent us from easily instantiating this class in tests.
        raise NotImplementedError("create_model must be overridden")

    def get_total_number_of_cross_validation_runs(self) -> int:
        """
        Returns the total number of HyperDrive/offline runs required to sample the entire
        cross validation parameter space.
        """
        return self.number_of_cross_validation_splits

    def get_cross_validation_hyperdrive_sampler(self) -> GridParameterSampling:
        """
        Returns the cross validation sampler, required to sample the entire parameter space for cross validation.
        """
        return GridParameterSampling(parameter_space={
            CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY: choice(list(range(self.number_of_cross_validation_splits))),
        })

    def get_cross_validation_hyperdrive_config(self, run_config: ScriptRunConfig) -> HyperDriveConfig:
        """
        Returns a configuration for AzureML Hyperdrive that varies the cross validation split index.
        :param run_config: The AzureML run configuration object that training for an individual model.
        :return: A hyperdrive configuration object.
        """
        return HyperDriveConfig(
            run_config=run_config,
            hyperparameter_sampling=self.get_cross_validation_hyperdrive_sampler(),
            primary_metric_name=TrackedMetrics.Val_Loss.value,
            primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
            max_total_runs=self.get_total_number_of_cross_validation_runs()
        )

    def get_cross_validation_dataset_splits(self, dataset_split: DatasetSplits) -> DatasetSplits:
        """
        When running cross validation, this method returns the dataset split that should be used for the
        currently executed cross validation split.
        :param dataset_split: The full dataset, split into training, validation and test section.
        :return: The dataset split with training and validation sections shuffled according to the current
        cross validation index.
        """
        splits = dataset_split.get_k_fold_cross_validation_splits(self.number_of_cross_validation_splits)
        return splits[self.cross_validation_split_index]

    def get_hyperdrive_config(self, run_config: ScriptRunConfig) -> HyperDriveConfig:
        """
        Returns the HyperDrive config for either parameter search or cross validation
        (if number_of_cross_validation_splits > 1).
        :param run_config: AzureML estimator
        :return: HyperDriveConfigs
        """
        if self.perform_cross_validation:
            return self.get_cross_validation_hyperdrive_config(run_config)
        else:
            return self.get_parameter_search_hyperdrive_config(run_config)

    def get_dataset_splits(self) -> DatasetSplits:
        """
        The Train/Val/Test dataset splits. If number_of_cross_validation_splits > 1 then the
        corresponding cross_validation_split_index will be used to get the cross validation split for Train/Val/Test.
        :return: DatasetSplits
        """
        dataset_df = self.read_dataset_if_needed()

        splits = self.get_model_train_test_dataset_splits(dataset_df)

        if self.perform_cross_validation:
            splits = self.get_cross_validation_dataset_splits(splits)

        if self.restrict_subjects:
            splits = splits.restrict_subjects(self.restrict_subjects)

        return splits

    def write_dataset_files(self, root: Optional[Path] = None) -> None:
        """
        Writes to disk the dataset.csv and the train and test files
        :return: None
        """
        root = root or self.outputs_folder
        if root and not root.exists():
            root.mkdir(parents=True)
        self.read_dataset_if_needed().to_csv(root / DATASET_CSV_FILE_NAME, index=False)
        # get datasets for the model
        dataset_splits = self.get_dataset_splits()
        logging.info(str(dataset_splits))

        for mode, split_name in zip([ModelExecutionMode.TRAIN, ModelExecutionMode.VAL, ModelExecutionMode.TEST],
                                    ["Training", "Validation", "Test"]):
            dataframe = dataset_splits[mode]
            subjects = dataframe[dataset_splits.subject_column].unique()
            logging.debug(f"{split_name} set has {len(subjects)} items: {subjects}")
            dst = root / STORED_CSV_FILE_NAMES[mode]
            dataframe.to_csv(dst, mode='w', index=False)

    def set_derived_model_properties(self, model: Any) -> None:
        """
        A hook to adjust the model configuration that is stored in the present object to match
        the torch model given in the argument. This hook is called after adjusting the model for
        mixed precision and parallel training.
        :param model: The torch model.
        """
        pass

    def generate_custom_report(self, report_dir: Path, model_proc: ModelProcessing) -> Path:
        """
        Enables creating a custom results report, given the metrics files written during model training and inference.
        By default, this method is a no-op.

        :param report_dir: The output directory where the generated report should be saved.
        :param model_proc: The type of model that is registered (single or ensemble)
        :return: The path to the generated report file.
        """
        pass


class ModelTransformsPerExecutionMode:
    """
    This is a container class used to store transformations
    to apply to each sample depending on each execution mode (train, validation and test)
    """

    def __init__(self, train: Optional[Callable] = None,
                 val: Optional[Callable] = None,
                 test: Optional[Callable] = None):
        """

        :param train: the transformation(s) to apply to the training set.
        Should be a function that takes a sample as input and outputs sample.
        :param val: the transformation(s) to apply to the validation set
        :param test: the transformation(s) to apply to the test set
        """
        self.train = train
        self.val = val
        self.test = test
