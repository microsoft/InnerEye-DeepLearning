#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import copy
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch.multiprocessing
from azureml.core import Run
from azureml.core.model import Model
from azureml.data import FileDataset

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_runner import INPUT_DATA_KEY, get_or_create_dataset
from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, \
    CROSS_VALIDATION_SUB_FOLD_SPLIT_INDEX_TAG_KEY, DEFAULT_CROSS_VALIDATION_SPLIT_INDEX, \
    EFFECTIVE_RANDOM_SEED_KEY_NAME, IS_ENSEMBLE_KEY_NAME, MODEL_ID_KEY_NAME, \
    NUMBER_OF_CROSS_VALIDATION_SPLITS_PER_FOLD_KEY_NAME, PARENT_RUN_CONTEXT, \
    PARENT_RUN_ID_KEY_NAME, RUN_CONTEXT, RUN_RECOVERY_FROM_ID_KEY_NAME, RUN_RECOVERY_ID_KEY_NAME, \
    create_run_recovery_id, get_results_blob_path, has_input_datasets, is_offline_run_context, merge_conda_files, \
    update_run_tags
from InnerEye.Common import fixed_paths
from InnerEye.Common.build_config import ExperimentResultLocation, build_information_to_dot_net_json_file
from InnerEye.Common.common_util import ModelProcessing, is_windows, logging_section, print_exception
from InnerEye.Common.fixed_paths import INNEREYE_PACKAGE_NAME, PROJECT_SECRETS_FILE
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.deep_learning_config import CHECKPOINT_FOLDER, FINAL_MODEL_FOLDER, MultiprocessingStartMethod
from InnerEye.ML.metrics import InferenceMetrics, InferenceMetricsForSegmentation
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_inference_config import ModelInferenceConfig
from InnerEye.ML.model_testing import model_test
from InnerEye.ML.model_training import model_train
from InnerEye.ML.runner import ModelDeploymentHookSignature, Runner, get_all_environment_files
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.utils import ml_util
from InnerEye.ML.utils.blobxfer_util import download_blobs
from InnerEye.ML.utils.checkpoint_handling import CheckpointHandler
from InnerEye.ML.utils.ml_util import make_pytorch_reproducible
from InnerEye.ML.visualizers import activation_maps
from InnerEye.ML.visualizers.plot_cross_validation import \
    get_config_and_results_for_offline_runs, plot_cross_validation_from_files


def try_to_mount_input_dataset(run_context: Any) -> Optional[Path]:
    """
    If run_context has an input_datasets attribute with an INPUT_DATA_KEY key, return
    the corresponding value as a Path. Otherwise, warn and return None, so that a backup
    strategy can be tried.
    """
    if has_input_datasets(run_context):
        try:
            return Path(run_context.input_datasets[INPUT_DATA_KEY])
        except KeyError:
            logging.warning(f"Run context input_datasets has no {INPUT_DATA_KEY} entry")
            logging.warning("Attempting to download dataset instead")
    return None


def download_dataset_via_blobxfer(dataset_id: str,
                                  azure_config: AzureConfig,
                                  target_folder: Path) -> Optional[Path]:
    """
    Attempts to downloads a dataset from the Azure storage account for datasets, with download happening via
    blobxfer. This is only possible if the datasets storage account and keyword are present in the `azure_config`.
    The function returns None if the required settings were not present.
    :param dataset_id: The folder of the dataset, expected in the container given by azure_config.datasets_container.
    :param azure_config: The object with all Azure-related settings.
    :param target_folder: The local folder into which the dataset should be downloaded.
    :return: The folder that contains the downloaded dataset. Returns None if the datasets account name or password
    were not present.
    """
    datasets_account_key = azure_config.get_dataset_storage_account_key()
    if not datasets_account_key:
        logging.info("No account key for the dataset storage account was found.")
        logging.info(f"We checked in environment variables and in the file {PROJECT_SECRETS_FILE}")
        return None
    if (not azure_config.datasets_container) or (not azure_config.datasets_storage_account):
        logging.info("Datasets storage account or container missing.")
        return None
    target_folder.mkdir(exist_ok=True)
    result_folder = target_folder / dataset_id
    # only download if hasn't already been downloaded
    if result_folder.is_dir():
        logging.info(f"Folder already exists, skipping download: {result_folder}")
        return result_folder
    with logging_section(f"Downloading dataset {dataset_id}"):
        download_blobs(
            account=azure_config.datasets_storage_account,
            account_key=datasets_account_key,
            # When specifying the blobs root path, ensure that there is a slash at the end, otherwise
            # all datasets with that dataset_id as a prefix get downloaded.
            blobs_root_path=f"{azure_config.datasets_container}/{dataset_id}/",
            destination=result_folder
        )
    return result_folder


def download_dataset(azure_dataset_id: str,
                     target_folder: Path,
                     azure_config: AzureConfig) -> Path:
    """
    Downloads or checks for an existing dataset on the executing machine. If a local_dataset is supplied and the
    directory is present, return that. Otherwise, download the dataset specified by the azure_dataset_id from the
    AzureML dataset attached to the given AzureML workspace. The dataset is downloaded into the `target_folder`,
    in a subfolder that has the same name as the dataset. If there already appears to be such a folder, and the folder
    contains a dataset.csv file, no download is started.
    :param local_dataset: The path to an existing local dataset.
    :param azure_dataset_id: The name of a dataset that is registered in the AzureML workspace.
    :param target_folder: The folder in which to download the dataset from Azure.
    :param azure_config: All Azure-related configuration options.
    :return: A path on the local machine that contains the dataset.
    """
    try:
        downloaded_via_blobxfer = download_dataset_via_blobxfer(dataset_id=azure_dataset_id,
                                                                azure_config=azure_config,
                                                                target_folder=target_folder)
        if downloaded_via_blobxfer:
            return downloaded_via_blobxfer
    except Exception as ex:
        print_exception(ex, message="Unable to download dataset via blobxfer.")
    logging.info("Trying to download dataset via AzureML datastore now.")
    azure_dataset = get_or_create_dataset(azure_config, azure_dataset_id)
    if not isinstance(azure_dataset, FileDataset):
        raise ValueError(f"Expected to get a FileDataset, but got {type(azure_dataset)}")
    # The downloaded dataset may already exist from a previous run.
    expected_dataset_path = target_folder / azure_dataset_id
    expected_dataset_file = expected_dataset_path / DATASET_CSV_FILE_NAME
    logging.info(f"Model training will use dataset '{azure_dataset_id}' in Azure.")
    if expected_dataset_path.is_dir() and expected_dataset_file.is_file():
        logging.info(f"The dataset appears to be downloaded already in {expected_dataset_path}. Skipping.")
        return expected_dataset_path
    logging.info("Starting to download the dataset - WARNING, this could take very long!")
    with logging_section("Downloading dataset"):
        azure_dataset.download(target_path=str(expected_dataset_path), overwrite=False)
    logging.info(f"Azure dataset '{azure_dataset_id}' is now available in {expected_dataset_path}")
    return expected_dataset_path


def log_metrics(val_metrics: Optional[InferenceMetricsForSegmentation],
                test_metrics: Optional[InferenceMetricsForSegmentation],
                train_metrics: Optional[InferenceMetricsForSegmentation],
                run_context: Run) -> None:
    """
    Log metrics for each split to the provided run, or the current run context if None provided
    :param val_metrics: Inference results for the validation split
    :param test_metrics: Inference results for the test split
    :param train_metrics: Inference results for the train split
    :param run_context: Run for which to log the metrics to, use the current run context if None provided
    """
    for split in [x for x in [val_metrics, test_metrics, train_metrics] if x]:
        split.log_metrics(run_context)


class MLRunner:

    def __init__(self,
                 model_config: ModelConfigBase,
                 azure_config: Optional[AzureConfig] = None,
                 project_root: Optional[Path] = None,
                 model_deployment_hook: Optional[ModelDeploymentHookSignature] = None) -> None:
        """
        Driver class to run a ML experiment. Note that the project root argument MUST be supplied when using InnerEye
        as a package!
        :param model_config: Model related configurations
        :param azure_config: Azure related configurations
        :param project_root: Project root. This should only be omitted if calling run_ml from the test suite. Supplying
        it is crucial when using InnerEye as a package or submodule!
        :param model_deployment_hook: optional function for deploying a model in an application-specific way
        """
        self.model_config = model_config
        self.azure_config: AzureConfig = azure_config or AzureConfig()
        self.project_root: Path = project_root or fixed_paths.repository_root_directory()
        self.model_deployment_hook = model_deployment_hook

    def is_offline_cross_val_parent_run(self) -> bool:
        """
        Returns true if the current run is an offline run with cross validation splits > 0
        and cross_validation_split_index == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX (ie: a parent)
        """
        return self.model_config.cross_validation_split_index == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX and \
               self.model_config.perform_cross_validation and self.model_config.is_offline_run

    def spawn_offline_cross_val_classification_child_runs(self) -> None:
        """
        Trains and Tests k models based on their respective data splits sequentially.
        Stores the results on the Validation set to the outputs directory of the parent run.
        """
        _config = self.model_config
        assert isinstance(_config, ScalarModelBase)
        parent_run_file_system = _config.file_system_config

        def _spawn_run(cross_val_split_index: int, cross_val_sub_fold_split_index: int) -> None:
            split_model_config = copy.deepcopy(_config)
            assert isinstance(split_model_config, ScalarModelBase)
            split_model_config.cross_validation_split_index = cross_val_split_index
            split_model_config.cross_validation_sub_fold_split_index = cross_val_sub_fold_split_index

            if cross_val_sub_fold_split_index == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX:
                _local_split_folder_name = str(cross_val_split_index)
            else:
                _local_split_folder_name = \
                    str((cross_val_split_index * split_model_config.number_of_cross_validation_splits_per_fold)
                        + cross_val_sub_fold_split_index)

            split_model_config.file_system_config = parent_run_file_system.add_subfolder(_local_split_folder_name)

            logging.info(f"Running model train and test on cross validation split: {x}")
            split_ml_runner = MLRunner(split_model_config, self.azure_config, self.project_root,
                                       self.model_deployment_hook)
            split_ml_runner.run()

        cv_fold_indices = [list(range(_config.number_of_cross_validation_splits_per_fold))
                           if _config.perform_sub_fold_cross_validation else [DEFAULT_CROSS_VALIDATION_SPLIT_INDEX]]
        cv_fold_indices *= _config.number_of_cross_validation_splits

        for i, x in enumerate(cv_fold_indices):
            for y in x:
                _spawn_run(i, int(y))

        config_and_files = get_config_and_results_for_offline_runs(self.model_config)
        plot_cross_validation_from_files(config_and_files, Path(config_and_files.config.outputs_directory))

    def set_run_tags_from_parent(self) -> None:
        """
        Set metadata for the run
        """
        assert PARENT_RUN_CONTEXT, "This function should only be called in a Hyperdrive run."
        run_tags_parent = PARENT_RUN_CONTEXT.get_tags()
        tags_to_copy = [
            "tag",
            "model_name",
            "execution_mode",
            "recovered_from",
            "friendly_name",
            "build_number",
            "build_user",
            "source_repository",
            "source_branch",
            "source_id",
            "source_message",
            "source_author",
            "source_dirty",
            RUN_RECOVERY_FROM_ID_KEY_NAME
        ]
        new_tags = {tag: run_tags_parent.get(tag, "") for tag in tags_to_copy}
        new_tags[RUN_RECOVERY_ID_KEY_NAME] = create_run_recovery_id(run=RUN_CONTEXT)
        new_tags[CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY] = str(self.model_config.cross_validation_split_index)
        new_tags[EFFECTIVE_RANDOM_SEED_KEY_NAME] = str(self.model_config.get_effective_random_seed())
        if isinstance(self.model_config, ScalarModelBase):
            new_tags[NUMBER_OF_CROSS_VALIDATION_SPLITS_PER_FOLD_KEY_NAME] = str(
                self.model_config.number_of_cross_validation_splits_per_fold)
            new_tags[CROSS_VALIDATION_SUB_FOLD_SPLIT_INDEX_TAG_KEY] = str(
                self.model_config.cross_validation_sub_fold_split_index)
        RUN_CONTEXT.set_tags(new_tags)

    def run(self) -> None:
        """
        Driver function to run a ML experiment. If an offline cross validation run is requested, then
        this function is recursively called for each cross validation split.
        """
        if self.is_offline_cross_val_parent_run():
            if self.model_config.is_segmentation_model:
                raise NotImplementedError("Offline cross validation is only supported for classification models.")
            self.spawn_offline_cross_val_classification_child_runs()
            return

        # Get the AzureML context in which the script is running
        if not self.model_config.is_offline_run and PARENT_RUN_CONTEXT is not None:
            logging.info("Setting tags from parent run.")
            self.set_run_tags_from_parent()

        self.save_build_info_for_dotnet_consumers()

        # Set data loader start method
        self.set_multiprocessing_start_method()

        # configure recovery container if provided
        checkpoint_handler = CheckpointHandler(model_config=self.model_config,
                                               azure_config=self.azure_config,
                                               project_root=self.project_root,
                                               run_context=RUN_CONTEXT)
        checkpoint_handler.discover_and_download_checkpoints_from_previous_runs()
        # do training and inference, unless the "only register" switch is set (which requires a run_recovery
        # to be valid).
        if not self.azure_config.register_model_only_for_epoch:
            # Set local_dataset to the mounted path specified in azure_runner.py, if any, or download it if that fails
            # and config.local_dataset was not already set.
            self.model_config.local_dataset = self.mount_or_download_dataset()
            self.model_config.write_args_file()
            logging.info(str(self.model_config))
            # Ensure that training runs are fully reproducible - setting random seeds alone is not enough!
            make_pytorch_reproducible()

            # Check for existing dataset.csv file in the correct locations. Skip that if a dataset has already been
            # loaded (typically only during tests)
            if self.model_config.dataset_data_frame is None:
                assert self.model_config.local_dataset is not None
                ml_util.validate_dataset_paths(self.model_config.local_dataset)

            # train a new model if required
            if self.azure_config.train:
                with logging_section("Model training"):
                    model_train(self.model_config, checkpoint_handler)
            else:
                self.model_config.write_dataset_files()
                self.create_activation_maps()

            # log the number of epochs used for model training
            RUN_CONTEXT.log(name="Train epochs", value=self.model_config.num_epochs)

        # We specify the ModelProcessing as DEFAULT here even if the run_recovery points to an ensemble run, because
        # the current run is a single one. See the documentation of ModelProcessing for more details.
        best_epoch = self.run_inference_and_register_model(checkpoint_handler, ModelProcessing.DEFAULT)

        # Generate report
        if best_epoch:
            Runner.generate_report(self.model_config, best_epoch, ModelProcessing.DEFAULT)
        elif self.model_config.is_scalar_model and len(self.model_config.get_test_epochs()) == 1:
            # We don't register scalar models but still want to create a report if we have run inference.
            Runner.generate_report(self.model_config, self.model_config.get_test_epochs()[0], ModelProcessing.DEFAULT)

    def run_inference_and_register_model(self, checkpoint_handler: CheckpointHandler,
                                         model_proc: ModelProcessing) -> Optional[int]:
        """
        Run inference as required, and register the model, but not necessarily in that order:
        if we can identify the epoch to register at without running inference, we register first.
        :param checkpoint_handler: Checkpoint handler object to find checkpoint paths for model initialization
        :param model_proc: whether we are running an ensemble model from within a child run with index 0. If we are,
        then outputs will be written to OTHER_RUNS/ENSEMBLE under the main outputs directory.
        """
        registration_epoch = self.decide_registration_epoch_without_evaluating()
        if registration_epoch is not None:
            model_description = f"Registering model for epoch {registration_epoch} without considering metrics."
            checkpoint_paths = checkpoint_handler.get_checkpoint_from_epoch(registration_epoch).checkpoint_paths
            self.register_model_for_epoch(checkpoint_paths,model_description,model_proc)
            if self.azure_config.register_model_only_for_epoch is not None:
                return self.azure_config.register_model_only_for_epoch

        # run full image inference on existing or newly trained model on the training, and testing set
        test_metrics, val_metrics, _ = self.model_inference_train_and_test(checkpoint_handler=checkpoint_handler,
                                                                           model_proc=model_proc)

        # register the generated model from the run if we haven't already done so
        if self.model_config.is_segmentation_model and (not self.model_config.is_offline_run):
            if registration_epoch is None:
                if self.should_register_model():
                    assert test_metrics is None or isinstance(test_metrics, InferenceMetricsForSegmentation)
                    assert val_metrics is None or isinstance(val_metrics, InferenceMetricsForSegmentation)
                    registration_epoch = self.register_model_for_best_epoch(checkpoint_handler,
                                                                            test_metrics,
                                                                            val_metrics,
                                                                            model_proc)
            self.try_compare_scores_against_baselines(model_proc)
        else:
            logging.warning("Couldn't register model in offline mode")

        return registration_epoch

    def should_register_model(self) -> bool:
        """
        Whether we should register a model at all. If no training has taken place, an equivalent
        model (from the run we recovered) should already have been registered, so we should only
        do so if this run is specifically for that purpose.
        """
        return self.azure_config.train or self.azure_config.register_model_only_for_epoch is not None

    def decide_registration_epoch_without_evaluating(self) -> Optional[int]:
        """
        In general we need to do evaluations to discover the best test epoch to register the model
        for. But there are two exceptions, which allow us to register first: (1) the switch
        register_model_only_for_epoch is set; (2) there is only one test epoch.
        :return: the epoch to register, or None if it cannot be decided or if registration is not needed.
        """
        if not self.should_register_model():
            return None
        if self.azure_config.register_model_only_for_epoch is not None:
            return self.azure_config.register_model_only_for_epoch
        candidate_best_epochs = self.model_config.get_test_epochs()
        if len(candidate_best_epochs) == 1:
            return candidate_best_epochs[0]
        return None

    def create_activation_maps(self) -> None:
        if self.model_config.is_segmentation_model and self.model_config.activation_map_layers is not None:
            logging.info("Extracting activation maps for layer")
            activation_maps.extract_activation_maps(self.model_config)
            logging.info("Successfully extracted and saved activation maps")

    def mount_or_download_dataset(self) -> Path:
        """
        Makes the dataset that the model uses available on the executing machine. If the present training run is outside
        of AzureML, it expects that either the model has a `local_dataset` field set, in which case no action will be
        taken. If a dataset is specified in `azure_dataset_id`, it will attempt to download the dataset from Azure
        into the local repository, in the "datasets" folder.
        If the training run is inside of AzureML, the dataset that was specified at job submission time will be
        mounted or downloaded.
        Returns the path of the dataset on the executing machine.
        """
        azure_dataset_id = self.model_config.azure_dataset_id

        if is_offline_run_context(RUN_CONTEXT):
            # The present run is outside of AzureML: If local_dataset is set, use that as the path to the data.
            # Otherwise, download the dataset specified by the azure_dataset_id
            local_dataset = self.model_config.local_dataset
            if (not azure_dataset_id) and (local_dataset is None):
                raise ValueError("The model must contain either local_dataset or azure_dataset_id.")
            if local_dataset:
                expected_dir = Path(local_dataset)
                if not expected_dir.is_dir():
                    raise FileNotFoundError(f"The model uses a dataset in {expected_dir}, but that does not exist.")
                logging.info(f"Model training will use the local dataset provided in {expected_dir}")
                return expected_dir
            return download_dataset(azure_dataset_id=azure_dataset_id,
                                    target_folder=self.project_root / fixed_paths.DATASETS_DIR_NAME,
                                    azure_config=self.azure_config)

        # Inside of AzureML, datasets can be either mounted or downloaded.
        if not azure_dataset_id:
            raise ValueError("The model must contain azure_dataset_id for running on AML")
        mounted = try_to_mount_input_dataset(RUN_CONTEXT)
        if not mounted:
            raise ValueError("Unable to mount or download input dataset.")
        return mounted

    def register_model_for_best_epoch(self,
                                      checkpoint_handler: CheckpointHandler,
                                      test_metrics: Optional[InferenceMetricsForSegmentation],
                                      val_metrics: Optional[InferenceMetricsForSegmentation],
                                      model_proc: ModelProcessing) -> int:
        if val_metrics is not None:
            best_epoch = val_metrics.get_best_epoch()
            num_epochs = len(val_metrics.epochs)
            model_description = f"Epoch {best_epoch} has best validation set metrics (out of {num_epochs} epochs " \
                                f"available). Validation set Dice: {val_metrics.epochs[best_epoch]}. "
            if test_metrics:
                model_description += f"Test set Dice: {test_metrics.epochs[best_epoch]}."
            else:
                model_description += "Test set metrics not available."
        elif test_metrics is not None:
            # We should normally not get here. We presently always run inference on both validation and test set
            # together.
            best_epoch = test_metrics.get_best_epoch()
            num_epochs = len(test_metrics.epochs)
            model_description = f"Epoch {best_epoch} has best test set metrics (out of {num_epochs} epochs " \
                                f"available). Test set Dice: {test_metrics.epochs[best_epoch]}"
        else:
            raise ValueError("At least one of val_metrics, test_metrics should be available.")
        checkpoint_paths = checkpoint_handler.get_checkpoint_from_epoch(best_epoch).checkpoint_paths
        self.register_model_for_epoch(checkpoint_paths, model_description, model_proc)
        return best_epoch

    def save_build_info_for_dotnet_consumers(self) -> None:
        results_container = get_results_blob_path(RUN_CONTEXT.id)
        result_location = ExperimentResultLocation(
            azure_job_name=RUN_CONTEXT.id,
            dataset_folder=self.model_config.azure_dataset_id,
            results_container_name=results_container,
            commandline_overrides=str(self.model_config.overrides),
            dataset_uri=self.model_config.azure_dataset_id,
            results_uri="",
        )
        # Fill in the missing information in the build config (everything that is not available at the time
        # of evoking the runner), and then save in the format needed for the .NET consumers
        build_information_to_dot_net_json_file(
            self.azure_config, result_location, folder=self.model_config.outputs_folder)

    def set_multiprocessing_start_method(self) -> None:
        """
        Set the (PyTorch) multiprocessing start method.
        """
        method = self.model_config.multiprocessing_start_method
        if is_windows():
            if method != MultiprocessingStartMethod.spawn:
                logging.warning(f"Cannot set multiprocessing start method to '{method.name}' "
                                "because only 'spawn' is available in Windows")
        else:
            logging.info(f"Setting multiprocessing start method to '{method.name}'")
            torch.multiprocessing.set_start_method(method.name, force=True)

    def register_model_for_epoch(self,
                                 checkpoint_paths: List[Path],
                                 model_description: str,
                                 model_proc: ModelProcessing) -> None:
        """
        Registers the model in AzureML, with the given set of checkpoints. The AzureML run's tags are updated
        to describe with information about ensemble creation and the parent run ID.
        :param checkpoint_paths: The set of Pytorch checkpoints that should be included.
        :param model_description: A string description of the model, usually containing accuracy numbers.
        :param model_proc: The type of model that is registered (single or ensemble)
        """
        if not checkpoint_paths:
            # No point continuing, since no checkpoints were found
            logging.warning("Abandoning model registration - no valid checkpoint paths found")
            return

        if not self.model_config.is_offline_run:
            split_index = RUN_CONTEXT.get_tags().get(CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, None)
            if split_index == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX:
                update_run_tags(RUN_CONTEXT, {IS_ENSEMBLE_KEY_NAME: model_proc == ModelProcessing.ENSEMBLE_CREATION})
            elif PARENT_RUN_CONTEXT is not None:
                update_run_tags(RUN_CONTEXT, {PARENT_RUN_ID_KEY_NAME: PARENT_RUN_CONTEXT.id})
        with logging_section(f"Registering {model_proc.value} model"):
            self.register_segmentation_model(
                checkpoint_paths=checkpoint_paths,
                model_description=model_description,
                model_proc=model_proc)

    def try_compare_scores_against_baselines(self, model_proc: ModelProcessing) -> None:
        """
        Attempt comparison of scores against baseline scores and scatterplot creation if possible.
        """
        if not isinstance(self.model_config, SegmentationModelBase):  # keep type checker happy
            return
        try:
            from InnerEye.ML.baselines_util import compare_scores_against_baselines
            with logging_section("Comparing scores against baselines"):
                compare_scores_against_baselines(self.model_config, self.azure_config, model_proc)
        except Exception as ex:
            print_exception(ex, "Model baseline comparison failed.")

    def register_segmentation_model(self,
                                    checkpoint_paths: List[Path],
                                    model_description: str,
                                    model_proc: ModelProcessing) -> Tuple[Model, Optional[Any]]:
        """
        Registers a new model in the workspace's model registry to be deployed further,
        and creates a model zip for portal deployment (if required). This model is the
        model checkpoint with the highest test accuracy.
        :param model_description: A string description that is added to the deployed model. It would usually contain
        the test set performance and information at which epoch the result was achieved.
        :param checkpoint_paths: Checkpoint paths to use to upload model checkpoints to AML.
        :param model_proc: whether it's a single or ensemble model.
        :returns Tuple element 1: AML model object. Tuple element 2: The result of running the
        model_deployment_hook, or None if no hook was supplied.
        """
        if not isinstance(self.model_config, SegmentationModelBase):
            raise ValueError(f"This function can only register segmentation models, but got {type(self.model_config)}")
        is_offline_run = is_offline_run_context(RUN_CONTEXT)
        final_model_folder = self.model_config.final_model_folder
        # Copy all code from project and InnerEye into the model folder, and copy over checkpoints.
        # This increases the size of the data stored for the run. The other option would be to store all checkpoints
        # right in the final model folder - however, then that would also contain any other checkpoints that the model
        # produced or downloaded for recovery, bloating the final model file.
        self.copy_child_paths_to_folder(final_model_folder, checkpoint_paths)
        logging.info("Registering the model on the workspace.")
        if is_offline_run:
            model_description = model_description + f"\nModel built by {self.azure_config.build_user} outside AzureML"
            model = Model.register(
                workspace=self.azure_config.get_workspace(),
                model_name=self.model_config.model_name,
                model_path=str(final_model_folder),
                description=model_description
            )
        else:
            # The files for the final model can't live in the outputs folder. If they do: when registering the model,
            # the files are not yet uploaded by hosttools, and may (or not) cause errors. Hence, place the folder
            # for the final models outside of "outputs", and upload manually.
            artifacts_path = FINAL_MODEL_FOLDER
            logging.info(f"Uploading files in {final_model_folder} to the run with prefix '{artifacts_path}'")
            final_model_folder_relative = final_model_folder.relative_to(Path.cwd())
            RUN_CONTEXT.upload_folder(name=artifacts_path, path=str(final_model_folder_relative))
            logging.info(f"Registering the model on run {RUN_CONTEXT.id}")
            # When registering the model on the run, we need to provide a relative path inside of the run's output
            # folder in `model_path`
            model = RUN_CONTEXT.register_model(
                model_name=self.model_config.model_name,
                model_path=artifacts_path,
                tags=RUN_CONTEXT.get_tags(),
                description=model_description
            )

        logging.info(f"Registered {model_proc.value} model: {model.name}, with Id: {model.id}")

        # update the run's tags with the registered model information
        if not is_offline_run:
            update_run_tags(RUN_CONTEXT, {MODEL_ID_KEY_NAME: model.id})

        # create a version of the model for deployment if the hook is provided
        deployment_result = None
        if self.model_deployment_hook is not None:
            assert isinstance(self.model_config, SegmentationModelBase)
            deployment_result = self.model_deployment_hook(
                self.model_config, self.azure_config, model, model_proc)
        return model, deployment_result

    @staticmethod
    def tags_with_run_information(run: Run, tags: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        extra_tags = {"experiment_name": run.experiment.name,
                      "run_id": run.id,
                      "run_number": run.number}
        # Let values already in tags take priority:
        return {**extra_tags, **(tags or {})}

    def copy_child_paths_to_folder(self,
                                   model_folder: Path,
                                   checkpoint_paths: List[Path]) -> None:
        """
        Gets the files that are required to register a model for inference. The necessary files are copied from
        the current folder structure into the given temporary folder.
        The folder will contain all source code in the InnerEye folder, possibly additional source code from the
        extra_code_directory, and all checkpoints in a newly created "checkpoints" folder inside the model.
        :param model_folder: The folder into which all files should be copied.
        :param checkpoint_paths: A list with absolute paths to checkpoint files. They are expected to be
        inside of the model's checkpoint folder.
        """

        def copy_folder(source_folder: Path, destination_folder: str = "") -> None:
            logging.info(f"Copying folder for registration: {source_folder}")
            destination_folder = destination_folder or source_folder.name
            shutil.copytree(str(source_folder), str(model_folder / destination_folder),
                            ignore=shutil.ignore_patterns('*.pyc'))

        def copy_file(source: Path, destination_file: str) -> None:
            logging.info(f"Copying file for registration: {source} to {destination_file}")
            destination = model_folder / destination_file
            if destination.is_file():
                # This could happen if there is score.py inside of the InnerEye package and also inside the calling
                # project. The latter will have precedence
                logging.warning(f"Overwriting existing {source.name} with {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(source), str(destination))

        relative_checkpoint_paths = []
        for checkpoint in checkpoint_paths:
            if checkpoint.is_absolute():
                try:
                    # Checkpoints live in a folder structure in the checkpoint folder. There can be multiple of
                    # them, with identical names, coming from an ensemble run. Hence, preserve their folder structure.
                    checkpoint_relative = checkpoint.relative_to(self.model_config.checkpoint_folder)
                except ValueError:
                    raise ValueError(f"Checkpoint file {checkpoint} was expected to be in a subfolder of "
                                     f"{self.model_config.checkpoint_folder}")
                # Checkpoints go into a newly created folder "checkpoints" inside of the model folder
                relative_checkpoint_paths.append(str(Path(CHECKPOINT_FOLDER) / checkpoint_relative))
            else:
                raise ValueError(f"Expected an absolute path to a checkpoint file, but got: {checkpoint}")
        model_folder.mkdir(parents=True, exist_ok=True)
        model_inference_config = ModelInferenceConfig(model_name=self.model_config.model_name,
                                                      structure_names=self.model_config.ground_truth_ids_display_names,
                                                      colours=self.model_config.colours,
                                                      fill_holes=self.model_config.fill_holes,
                                                      model_configs_namespace=self.model_config.__class__.__module__,
                                                      checkpoint_paths=relative_checkpoint_paths)
        # Inference configuration must live in the root folder of the registered model
        full_path_to_config = model_folder / fixed_paths.MODEL_INFERENCE_JSON_FILE_NAME
        full_path_to_config.write_text(model_inference_config.to_json(), encoding='utf-8')  # type: ignore
        # Merge the conda files into one merged environment file at the root of the model
        merged_conda_file = model_folder / fixed_paths.ENVIRONMENT_YAML_FILE_NAME
        merge_conda_files(get_all_environment_files(self.project_root), result_file=merged_conda_file)
        # InnerEye package: This can be either in Python's package folder, or a plain folder. In both cases,
        # we can identify it by going up the folder structure off a known file (repository_root does exactly that)
        repository_root = fixed_paths.repository_root_directory()
        copy_folder(repository_root / INNEREYE_PACKAGE_NAME)
        # Extra code directory is expected to be relative to the project root folder.
        if self.azure_config.extra_code_directory:
            extra_code_folder = self.project_root / self.azure_config.extra_code_directory
            if extra_code_folder.is_dir():
                copy_folder(extra_code_folder)
            else:
                logging.warning(f"The `extra_code_directory` is set to '{self.azure_config.extra_code_directory}', "
                                "but this folder does not exist in the project root folder.")
        # All files at project root should be copied as-is. Those should be essential things like score.py that
        # are needed for inference to run. First try to find them at repository root (but they might not be there
        # if InnerEye is used as a package), then at project root.
        files_to_copy = list(repository_root.glob("*.py"))
        if repository_root != self.project_root:
            files_to_copy.extend(self.project_root.glob("*.py"))
        for f in files_to_copy:
            copy_file(f, destination_file=f.name)
        for (checkpoint_source, checkpoint_destination) in zip(checkpoint_paths, relative_checkpoint_paths):
            if checkpoint_source.is_file():
                copy_file(checkpoint_source, destination_file=str(checkpoint_destination))
            else:
                raise ValueError(f"Checkpoint file {checkpoint_source} does not exist")

    def model_inference_train_and_test(self,
                                       checkpoint_handler: CheckpointHandler,
                                       model_proc: ModelProcessing = ModelProcessing.DEFAULT) -> \
            Tuple[Optional[InferenceMetrics], Optional[InferenceMetrics], Optional[InferenceMetrics]]:
        train_metrics = None
        val_metrics = None
        test_metrics = None

        config = self.model_config

        def run_model_test(data_split: ModelExecutionMode) -> Optional[InferenceMetrics]:
            return model_test(config, data_split=data_split, checkpoint_handler=checkpoint_handler,
                              model_proc=model_proc)

        if config.perform_validation_and_test_set_inference:
            # perform inference on test set
            test_metrics = run_model_test(ModelExecutionMode.TEST)
            # perform inference on validation set
            val_metrics = run_model_test(ModelExecutionMode.VAL)

        if config.perform_training_set_inference:
            # perform inference on training set if required
            train_metrics = run_model_test(ModelExecutionMode.TRAIN)

        # log the metrics to AzureML experiment if possible
        if config.is_segmentation_model and not is_offline_run_context(RUN_CONTEXT):
            log_metrics(val_metrics=val_metrics, test_metrics=test_metrics,  # type: ignore
                        train_metrics=train_metrics, run_context=RUN_CONTEXT)  # type: ignore

        return test_metrics, val_metrics, train_metrics
