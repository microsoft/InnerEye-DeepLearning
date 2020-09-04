#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import copy
import logging
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch.multiprocessing
from azureml.core import Run, Workspace  # , Dataset
from azureml.core.model import Model

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_runner import INPUT_DATA_KEY
from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, DEFAULT_CROSS_VALIDATION_SPLIT_INDEX, \
    IS_ENSEMBLE_KEY_NAME, MODEL_ID_KEY_NAME, PARENT_RUN_CONTEXT, PARENT_RUN_ID_KEY_NAME, RUN_CONTEXT, \
    RUN_RECOVERY_FROM_ID_KEY_NAME, \
    RUN_RECOVERY_ID_KEY_NAME, \
    create_run_recovery_id, get_results_blob_path, has_input_datasets, storage_account_from_full_name, \
    update_run_tags
from InnerEye.Common import fixed_paths
from InnerEye.Common.build_config import ExperimentResultLocation, build_information_to_dot_net_json_file
from InnerEye.Common.common_util import ModelProcessing, is_windows, logging_section, print_exception
from InnerEye.Common.fixed_paths import ENVIRONMENT_YAML_FILE_NAME, INNEREYE_PACKAGE_NAME
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.deep_learning_config import MultiprocessingStartMethod
from InnerEye.ML.metrics import InferenceMetrics, InferenceMetricsForSegmentation
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_inference_config import ModelInferenceConfig
from InnerEye.ML.model_testing import model_test
from InnerEye.ML.model_training import model_train
from InnerEye.ML.runner import ModelDeploymentHookSignature
from InnerEye.ML.utils import ml_util
from InnerEye.ML.utils.blobxfer_util import download_blobs
from InnerEye.ML.utils.ml_util import make_pytorch_reproducible
from InnerEye.ML.utils.run_recovery import RunRecovery
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
        parent_run_file_system = self.model_config.file_system_config
        for x in range(self.model_config.number_of_cross_validation_splits):
            split_model_config = copy.deepcopy(self.model_config)
            split_model_config.cross_validation_split_index = x
            split_model_config.file_system_config = parent_run_file_system.add_subfolder(str(x))
            logging.info(f"Running model train and test on cross validation split: {x}")
            split_ml_runner = MLRunner(split_model_config, self.azure_config, self.project_root,
                                       self.model_deployment_hook, self.innereye_submodule_name)
            split_ml_runner.run()

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

        if self.azure_config.storage_account:
            self.save_build_info_for_dotnet_consumers()

        # Set data loader start method
        self.set_multiprocessing_start_method()

        # configure recovery container if provided
        run_recovery: Optional[RunRecovery] = None
        if self.azure_config.run_recovery_id:
            run_recovery = RunRecovery.download_checkpoints_from_recovery_run(
                self.azure_config, self.model_config, RUN_CONTEXT)
        # do training and inference, unless the "only register" switch is set (which requires a run_recovery
        # to be valid).
        if self.azure_config.register_model_only_for_epoch is None or run_recovery is None:
            # Set local_dataset to the mounted path specified in azure_runner.py, if any, or download it if that fails
            # and config.local_dataset was not already set.
            self.mount_or_download_dataset()
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
            if self.azure_config.is_train:
                with logging_section("model training"):
                    model_train(self.model_config, run_recovery)
            else:
                self.model_config.write_dataset_files()
                self.create_activation_maps()

            # log the number of epochs used for model training
            RUN_CONTEXT.log(name="Train epochs", value=self.model_config.num_epochs)

        # We specify the ModelProcessing as DEFAULT here even if the run_recovery points to an ensemble run, because
        # the current run is a single one. See the documentation of ModelProcessing for more details.
        self.run_inference_and_register_model(run_recovery, ModelProcessing.DEFAULT)

    def run_inference_and_register_model(self, run_recovery: Optional[RunRecovery], model_proc: ModelProcessing) -> None:
        """
        Run inference as required, and register the model, but not necessarily in that order:
        if we can identify the epoch to register at without running inference, we register first.
        :param run_recovery: details of run specified by run_recovery_id
        :param model_proc: whether we are running an ensemble model from within a child run with index 0. If we are,
        then outputs will be written to OTHER_RUNS/ENSEMBLE under the main outputs directory.
        """
        registration_epoch = self.decide_registration_epoch_without_evaluating()
        if registration_epoch is not None:
            self.register_model_for_epoch(RUN_CONTEXT, run_recovery, registration_epoch, np.nan, model_proc)
            if self.azure_config.register_model_only_for_epoch is not None:
                return
        # run full image inference on existing or newly trained model on the training, and testing set
        test_metrics, val_metrics, _ = self.model_inference_train_and_test(RUN_CONTEXT, run_recovery, model_proc)
        # register the generated model from the run if we haven't already done so
        if self.model_config.is_segmentation_model and (not self.model_config.is_offline_run):
            if registration_epoch is None:
                if self.should_register_model():
                    assert test_metrics is None or isinstance(test_metrics, InferenceMetricsForSegmentation)
                    assert val_metrics is None or isinstance(val_metrics, InferenceMetricsForSegmentation)
                    self.register_model_for_best_epoch(run_recovery, test_metrics, val_metrics, model_proc)
            self.try_compare_scores_against_baselines(model_proc)
        else:
            logging.warning("Couldn't register model in offline mode")

    def should_register_model(self) -> bool:
        """
        Whether we should register a model at all. If no training has taken place, an equivalent
        model (from the run we recovered) should already have been registered, so we should only
        do so if this run is specifically for that purpose.
        """
        return self.azure_config.is_train or self.azure_config.register_model_only_for_epoch is not None

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

    def mount_or_download_dataset(self) -> None:
        if self.model_config.azure_dataset_id:
            mounted = try_to_mount_input_dataset(RUN_CONTEXT)
            if mounted:
                self.model_config.local_dataset = mounted
        if self.model_config.local_dataset is None:
            # We are not running inside AzureML: Try to download a dataset from blob storage.
            # The downloaded dataset may already exist from a previous run.
            dataset_path = self.project_root / fixed_paths.DATASETS_DIR_NAME
            self.model_config.local_dataset = self.download_dataset(RUN_CONTEXT, dataset_path=dataset_path)

    def register_model_for_best_epoch(self, run_recovery: Optional[RunRecovery],
                                      test_metrics: Optional[InferenceMetricsForSegmentation],
                                      val_metrics: Optional[InferenceMetricsForSegmentation],
                                      model_proc: ModelProcessing) -> None:
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
        self.register_model_for_epoch(RUN_CONTEXT, run_recovery, best_epoch, best_epoch_dice, model_proc)

    def save_build_info_for_dotnet_consumers(self) -> None:
        results_container = storage_account_from_full_name(self.azure_config.storage_account) \
                            + "/" + get_results_blob_path(RUN_CONTEXT.id)
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
                                 run_recovery: Optional[RunRecovery],
                                 best_epoch: int,
                                 best_epoch_dice: float,
                                 model_proc: ModelProcessing) -> None:
        checkpoint_paths = [self.model_config.get_path_to_checkpoint(best_epoch)] if not run_recovery \
            else run_recovery.get_checkpoint_paths(best_epoch)
        if not self.model_config.is_offline_run:
            split_index = run_context.get_tags().get(CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, None)
            if split_index == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX:
                update_run_tags(run_context, {IS_ENSEMBLE_KEY_NAME: model_proc == ModelProcessing.ENSEMBLE_CREATION})
            elif PARENT_RUN_CONTEXT is not None:
                update_run_tags(run_context, {PARENT_RUN_ID_KEY_NAME: PARENT_RUN_CONTEXT.id})
        # Discard any checkpoint paths that do not exist - they will make registration fail. This can happen
        # when some child runs fail; it may still be worth registering the model.
        valid_checkpoint_paths = []
        for path in checkpoint_paths:
            if path.exists():
                valid_checkpoint_paths.append(path)
            else:
                logging.warning(f"Discarding non-existent checkpoint path {path}")
        if not valid_checkpoint_paths:
            # No point continuing
            logging.warning("Abandoning model registration - no valid checkpoint paths found")
            return
        with logging_section(f"registering {model_proc.value} model"):
            self.register_segmentation_model(
                run=run_context,
                best_epoch=best_epoch,
                best_epoch_dice=best_epoch_dice,
                checkpoint_paths=valid_checkpoint_paths,
                model_proc=model_proc)

    def try_compare_scores_against_baselines(self, model_proc: ModelProcessing) -> None:
        """
        Attempt comparison of scores against baseline scores and scatterplot creation if possible.
        """
        if not isinstance(self.model_config, SegmentationModelBase):  # keep type checker happy
            return
        try:
            from InnerEye.ML.baselines_util import compare_scores_against_baselines
            with logging_section("comparing scores against baselines"):
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

    def model_inference_train_and_test(self, run_context: Optional[Run] = None,
                                       run_recovery: Optional[RunRecovery] = None,
                                       model_proc: ModelProcessing = ModelProcessing.DEFAULT) -> \
            Tuple[Optional[InferenceMetrics], Optional[InferenceMetrics], Optional[InferenceMetrics]]:
        train_metrics = None
        val_metrics = None
        test_metrics = None

        config = self.model_config

        def run_model_test(data_split: ModelExecutionMode) -> Optional[InferenceMetrics]:
            return model_test(config, data_split=data_split, run_recovery=run_recovery, model_proc=model_proc)

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

    def download_dataset(self, run_context: Optional[Run] = None,
                         dataset_path: Path = Path.cwd()) -> Optional[Path]:
        """
        Configures the dataset for model training/testing. The dataset is downloaded into dataset_path, only if the
        dataset
        does not exist in the given path.
        Returns a path to the folder that contains the dataset.
        """

        # check if the dataset needs to be downloaded from Azure
        config = self.model_config
        if config.azure_dataset_id:
            # log the dataset id being used for this run
            if run_context is not None:
                run_context.tag("dataset_id", config.azure_dataset_id)
            target_folder = dataset_path / config.azure_dataset_id
            # only download if hasn't already been downloaded
            if target_folder.is_dir():
                logging.info("Using cached dataset in folder %s", target_folder)
            else:
                logging.info("Starting to download dataset from Azure to folder %s", target_folder)
                start_time = timer()
                # download the dataset blobs from Azure to local
                download_blobs(
                    account=self.azure_config.datasets_storage_account,
                    account_key=self.azure_config.get_dataset_storage_account_key(),
                    # When specifying the blobs root path, ensure that there is a slash at the end, otherwise
                    # all datasets with that dataset_id as a prefix get downloaded.
                    blobs_root_path="{}/{}/".format(self.azure_config.datasets_container, config.azure_dataset_id),
                    destination=target_folder
                )
                elapsed_seconds = timer() - start_time
                logging.info("Downloading dataset from Azure took {} sec".format(elapsed_seconds))
            return target_folder
        return config.local_dataset
