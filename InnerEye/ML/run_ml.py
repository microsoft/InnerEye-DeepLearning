#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch.multiprocessing
from azureml.core import Run, Workspace
from azureml.core.model import Model
from azureml.data import FileDataset

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_runner import INPUT_DATA_KEY, get_or_create_dataset
from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, \
    CROSS_VALIDATION_SUB_FOLD_SPLIT_INDEX_TAG_KEY, DEFAULT_CROSS_VALIDATION_SPLIT_INDEX, \
    EFFECTIVE_RANDOM_SEED_KEY_NAME, IS_ENSEMBLE_KEY_NAME, MODEL_ID_KEY_NAME, \
    NUMBER_OF_CROSS_VALIDATION_SPLITS_PER_FOLD_KEY_NAME, PARENT_RUN_CONTEXT, \
    PARENT_RUN_ID_KEY_NAME, RUN_CONTEXT, RUN_RECOVERY_FROM_ID_KEY_NAME, RUN_RECOVERY_ID_KEY_NAME, \
    create_run_recovery_id, get_results_blob_path, has_input_datasets, is_offline_run_context, update_run_tags
from InnerEye.Common import fixed_paths
from InnerEye.Common.build_config import ExperimentResultLocation, build_information_to_dot_net_json_file
from InnerEye.Common.common_util import ModelProcessing, is_windows, logging_section, print_exception
from InnerEye.Common.fixed_paths import ENVIRONMENT_YAML_FILE_NAME, INNEREYE_PACKAGE_NAME, PROJECT_SECRETS_FILE
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.deep_learning_config import MultiprocessingStartMethod
from InnerEye.ML.metrics import InferenceMetrics, InferenceMetricsForSegmentation
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_inference_config import ModelInferenceConfig
from InnerEye.ML.model_testing import model_test
from InnerEye.ML.model_training import model_train
from InnerEye.ML.runner import ModelDeploymentHookSignature, Runner
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.utils import ml_util
from InnerEye.ML.utils.blobxfer_util import download_blobs
from InnerEye.ML.utils.ml_util import make_pytorch_reproducible
from InnerEye.ML.utils.checkpoint_handling import CheckpointHandler
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
                 model_deployment_hook: Optional[ModelDeploymentHookSignature] = None,
                 innereye_submodule_name: Optional[str] = None):
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
        self.innereye_submodule_name = innereye_submodule_name

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
                                       self.model_deployment_hook, self.innereye_submodule_name)
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
            self.register_model_for_epoch(RUN_CONTEXT, checkpoint_handler, registration_epoch, np.nan, model_proc)
            if self.azure_config.register_model_only_for_epoch is not None:
                return self.azure_config.register_model_only_for_epoch

        # run full image inference on existing or newly trained model on the training, and testing set
        test_metrics, val_metrics, _ = self.model_inference_train_and_test(run_context=RUN_CONTEXT,
                                                                           checkpoint_handler=checkpoint_handler,
                                                                           model_proc=model_proc)

        # register the generated model from the run if we haven't already done so
        if self.model_config.is_segmentation_model and (not self.model_config.is_offline_run):
            if registration_epoch is None:
                if self.should_register_model():
                    assert test_metrics is None or isinstance(test_metrics, InferenceMetricsForSegmentation)
                    assert val_metrics is None or isinstance(val_metrics, InferenceMetricsForSegmentation)
                    registration_epoch = self.register_model_for_best_epoch(checkpoint_handler, test_metrics, val_metrics,
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

    def register_model_for_best_epoch(self, checkpoint_handler: CheckpointHandler,
                                      test_metrics: Optional[InferenceMetricsForSegmentation],
                                      val_metrics: Optional[InferenceMetricsForSegmentation],
                                      model_proc: ModelProcessing) -> int:
        if val_metrics is not None:
            best_epoch = val_metrics.get_best_epoch()
        elif test_metrics is not None:
            best_epoch = test_metrics.get_best_epoch()
        else:
            best_epoch = self.model_config.get_test_epochs()[-1]
        if test_metrics is not None:
            best_epoch_dice = test_metrics.epochs[best_epoch]
        else:
            best_epoch_dice = 0.0  # dummy value
        assert isinstance(self.model_config, SegmentationModelBase)
        self.register_model_for_epoch(RUN_CONTEXT, checkpoint_handler, best_epoch, best_epoch_dice, model_proc)
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
                                 run_context: Run,
                                 checkpoint_handler: CheckpointHandler,
                                 best_epoch: int,
                                 best_epoch_dice: float,
                                 model_proc: ModelProcessing) -> None:

        checkpoint_path_and_epoch = checkpoint_handler.get_checkpoint_from_epoch(epoch=best_epoch)
        if not checkpoint_path_and_epoch or not checkpoint_path_and_epoch.checkpoint_paths:
            # No point continuing, since no checkpoints were found
            logging.warning("Abandoning model registration - no valid checkpoint paths found")
            return

        if not self.model_config.is_offline_run:
            split_index = run_context.get_tags().get(CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, None)
            if split_index == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX:
                update_run_tags(run_context, {IS_ENSEMBLE_KEY_NAME: model_proc == ModelProcessing.ENSEMBLE_CREATION})
            elif PARENT_RUN_CONTEXT is not None:
                update_run_tags(run_context, {PARENT_RUN_ID_KEY_NAME: PARENT_RUN_CONTEXT.id})
        with logging_section(f"Registering {model_proc.value} model"):
            self.register_segmentation_model(
                run=run_context,
                best_epoch=best_epoch,
                best_epoch_dice=best_epoch_dice,
                checkpoint_paths=checkpoint_path_and_epoch.checkpoint_paths,
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
                                    best_epoch: int,
                                    best_epoch_dice: float,
                                    checkpoint_paths: List[Path],
                                    model_proc: ModelProcessing,
                                    run: Optional[Run] = None,
                                    workspace: Optional[Workspace] = None,
                                    tags: Optional[Dict[str, str]] = None) -> \
            Tuple[Optional[Model], Optional[Path], Any]:
        """
        Registers a new model in the workspace's model registry to be deployed further,
        and creates a model zip for portal deployment (if required). This model, is the
        model checkpoint with the highest test accuracy.
        :param best_epoch: The training epoch that resulted in the highest validation score.
        :param best_epoch_dice: Dice metric for the best epoch
        :param checkpoint_paths: Checkpoint paths to use to upload model checkpoints to AML.
        :param model_proc: whether it's a single or ensemble model.
        :param run: If provided then the run's workspace and tags will be used to register the model.
        :param workspace: If provided, then this workspace will be used to register the model instead of the
        workspace associated with the provided run.
        :param tags: If provided, then these will be used instead of the tags found in the provided run.
        :returns AML model object, the path to the specially-deployed model if any, and a further object
        relating to model deployment; if model_deployment_hook is None, the last two are also None.
        However if a model cannot be registered because the run is an _OfflineRun, or the model_config is not
        for a segmentation model, None is returned instead of a model.
        """
        if not isinstance(self.model_config, SegmentationModelBase):
            logging.warning("Non-segmentation models cannot be registered")
            return None, None, None
        if (run is None) == (workspace is None):
            raise ValueError("Either a run or a workspace must be provided but not both")
        elif run:
            if not hasattr(run, 'experiment'):
                logging.warning("Not registering a model, because the run has no associated experiment")
                return None, None, None
            workspace = run.experiment.workspace
            tags = run.get_tags()

        relative_checkpoint_paths = [x.relative_to(self.project_root) if x.is_absolute() else x for x in
                                     checkpoint_paths]
        model_inference_config = ModelInferenceConfig(model_name=self.model_config.model_name,
                                                      structure_names=self.model_config.ground_truth_ids_display_names,
                                                      colours=self.model_config.colours,
                                                      fill_holes=self.model_config.fill_holes,
                                                      model_configs_namespace=self.model_config.__class__.__module__,
                                                      checkpoint_paths=list(map(str, relative_checkpoint_paths)))
        full_path_to_config = self.project_root / fixed_paths.MODEL_INFERENCE_JSON_FILE_NAME
        full_path_to_config.write_text(model_inference_config.to_json(), encoding='utf-8')  # type: ignore
        relative_child_paths = self.get_child_paths(checkpoint_paths)

        # Add experiment and run ID to tags
        if run is not None:
            tags = self.tags_with_run_information(run, tags)
        model = Model.register(
            workspace=workspace,
            model_path=str(self.project_root),
            child_paths=relative_child_paths,
            model_name=self.model_config.model_name,
            tags=tags,
            description="Best epoch: {}, Accuracy : {}".format(best_epoch, best_epoch_dice)
        )
        logging.info(f"Registered {model_proc.value} model: {model.name}, with Id: {model.id}")

        # update the run's tags with the registered model information
        if not self.model_config.is_offline_run:
            update_run_tags(run, {MODEL_ID_KEY_NAME: model.id})

        # create a version of the model for deployment if the hook is provided
        if self.model_deployment_hook is not None:
            assert isinstance(self.model_config, SegmentationModelBase)
            deployment_model_path, deployment_model_spec = self.model_deployment_hook(
                self.model_config, self.azure_config, model, model_proc)
            return model, deployment_model_path, deployment_model_spec
        return model, None, None

    @staticmethod
    def tags_with_run_information(run: Run, tags: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        extra_tags = {"experiment_name": run.experiment.name,
                      "run_id": run.id,
                      "run_number": run.number}
        # Let values already in tags take priority:
        return {**extra_tags, **(tags or {})}

    def get_child_paths(self, checkpoint_paths: List[Path]) -> List[str]:
        """
        Gets the files that are required to register a model for inference
        :param checkpoint_paths: Path(s) to checkpoint (multiple if model is an ensemble).
        These need to be under path_to_current_model
        :return: a list of relative paths to the model directory to register the model
        """
        path_to_current_model = self.project_root
        extra_code_directory = Path(
            self.azure_config.extra_code_directory) if self.azure_config.extra_code_directory else None
        model_name = self.model_config.model_name
        full_child_paths_package = list((path_to_current_model / INNEREYE_PACKAGE_NAME).rglob('*.py'))
        submodule_path: Optional[Path] = None
        submodule_package_path: Optional[Path] = None
        if self.innereye_submodule_name is not None:
            submodule_path = path_to_current_model / self.innereye_submodule_name
            submodule_package_path = submodule_path / INNEREYE_PACKAGE_NAME
            # Paths under submodule/InnerEye
            full_child_paths_submodule = list(submodule_package_path.rglob('*.py'))
            # Paths matching submodule/*.py
            full_child_paths_submodule += list(submodule_path.glob('*.py'))
            # submodule/environment.yml
            full_child_paths_submodule += [submodule_path / ENVIRONMENT_YAML_FILE_NAME]
        else:
            full_child_paths_submodule = []
        if extra_code_directory not in [None, Path(INNEREYE_PACKAGE_NAME), submodule_package_path]:
            full_child_paths_extra = list((path_to_current_model / extra_code_directory).rglob('*.py'))  # type: ignore
        else:
            full_child_paths_extra = []
        full_child_paths = (full_child_paths_package + full_child_paths_submodule + full_child_paths_extra
                            + checkpoint_paths)
        full_child_paths.append(path_to_current_model / Path(fixed_paths.ENVIRONMENT_YAML_FILE_NAME))
        full_child_paths.append(path_to_current_model / Path(fixed_paths.MODEL_INFERENCE_JSON_FILE_NAME))
        top_level_paths = list(path_to_current_model.glob("*.py"))
        full_child_paths += top_level_paths
        relative_child_path_names = [str(x.relative_to(path_to_current_model)) if x.is_absolute()
                                     else str(x) for x in full_child_paths]
        logging.info(f"Registering model {model_name} with {len(relative_child_path_names)} paths")
        logging.info(f"  {len(full_child_paths_package)} of the paths are under {INNEREYE_PACKAGE_NAME}/")
        logging.info(f"  {len(full_child_paths_submodule)} of the paths are under {submodule_path}/")
        if extra_code_directory:
            logging.info(f"  {len(full_child_paths_extra)} of the paths are under {extra_code_directory}/")
        else:
            logging.info("   Parameter extra_code_directory is unset, so no paths there")
        logging.info(f"  {len(top_level_paths)} of the paths are *.py files at top level")
        logging.debug("The paths are:")
        for path_name in relative_child_path_names:
            logging.debug(f"  {path_name}")
        return relative_child_path_names

    def model_inference_train_and_test(self,
                                       checkpoint_handler: CheckpointHandler,
                                       run_context: Optional[Run] = None,
                                       model_proc: ModelProcessing = ModelProcessing.DEFAULT) -> \
            Tuple[Optional[InferenceMetrics], Optional[InferenceMetrics], Optional[InferenceMetrics]]:
        train_metrics = None
        val_metrics = None
        test_metrics = None

        config = self.model_config

        def run_model_test(data_split: ModelExecutionMode) -> Optional[InferenceMetrics]:
            return model_test(config, data_split=data_split, checkpoint_handler=checkpoint_handler, model_proc=model_proc)

        if config.perform_validation_and_test_set_inference:
            # perform inference on test set
            test_metrics = run_model_test(ModelExecutionMode.TEST)
            # perform inference on validation set
            val_metrics = run_model_test(ModelExecutionMode.VAL)

        if config.perform_training_set_inference:
            # perform inference on training set if required
            train_metrics = run_model_test(ModelExecutionMode.TRAIN)

        # log the metrics to AzureML experiment if possible
        if config.is_segmentation_model and run_context is not None:
            log_metrics(val_metrics=val_metrics, test_metrics=test_metrics,  # type: ignore
                        train_metrics=train_metrics, run_context=run_context)  # type: ignore

        return test_metrics, val_metrics, train_metrics
