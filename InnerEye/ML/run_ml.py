#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import copy
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import stopit
import torch.multiprocessing
from azureml._restclient.constants import RunStatus
from azureml.core import Environment, Run
from azureml.core.model import Model
from azureml.data import FileDataset
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.utilities.cloud_io import load as pl_load
from torch.utils.data import DataLoader

from InnerEye.Azure import azure_util
from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_runner import ENVIRONMENT_VERSION, INPUT_DATA_KEY, get_or_create_dataset
from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, \
    DEFAULT_CROSS_VALIDATION_SPLIT_INDEX, EFFECTIVE_RANDOM_SEED_KEY_NAME, IS_ENSEMBLE_KEY_NAME, \
    MODEL_ID_KEY_NAME, PARENT_RUN_CONTEXT, PARENT_RUN_ID_KEY_NAME, RUN_CONTEXT, RUN_RECOVERY_FROM_ID_KEY_NAME, \
    RUN_RECOVERY_ID_KEY_NAME, create_run_recovery_id, is_offline_run_context, merge_conda_files
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import BASELINE_COMPARISONS_FOLDER, BASELINE_WILCOXON_RESULTS_FILE, \
    CROSSVAL_RESULTS_FOLDER, ENSEMBLE_SPLIT_NAME, METRICS_AGGREGATES_FILE, ModelProcessing, \
    OTHER_RUNS_SUBDIR_NAME, SCATTERPLOTS_SUBDIR_NAME, SUBJECT_METRICS_FILE_NAME, \
    get_epoch_results_path, is_windows, logging_section, logging_to_file, print_exception, remove_file_or_directory
from InnerEye.Common.fixed_paths import INNEREYE_PACKAGE_NAME, PYTHON_ENVIRONMENT_NAME
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.deep_learning_config import CHECKPOINT_FOLDER, DeepLearningConfig, FINAL_ENSEMBLE_MODEL_FOLDER, \
    FINAL_MODEL_FOLDER, ModelCategory, MultiprocessingStartMethod
from InnerEye.ML.lightning_base import InnerEyeContainer
from InnerEye.ML.lightning_container import LightningContainer, LightningInference
from InnerEye.ML.metrics import InferenceMetrics, InferenceMetricsForSegmentation
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_inference_config import ModelInferenceConfig
from InnerEye.ML.model_testing import model_test
from InnerEye.ML.model_training import create_lightning_trainer, model_train
from InnerEye.ML.reports.notebook_report import generate_classification_multilabel_notebook, \
    generate_classification_notebook, generate_segmentation_notebook, get_ipynb_report_name, reports_folder
from InnerEye.ML.runner import LOG_FILE_NAME, ModelDeploymentHookSignature, PostCrossValidationHookSignature, \
    get_all_environment_files
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils.checkpoint_handling import CheckpointHandler
from InnerEye.ML.visualizers import activation_maps
from InnerEye.ML.visualizers.plot_cross_validation import \
    get_config_and_results_for_offline_runs, plot_cross_validation_from_files


def try_to_mount_input_dataset() -> Optional[Path]:
    """
    Checks if the AzureML run context has a field for input datasets. If yes, the dataset stored there is
    returned as a Path. Returns None if no input datasets was found.
    """
    if hasattr(RUN_CONTEXT, "input_datasets"):
        try:
            return Path(RUN_CONTEXT.input_datasets[INPUT_DATA_KEY])
        except KeyError:
            logging.warning(f"Run context field input_datasets has no {INPUT_DATA_KEY} entry.")
    return None


def download_dataset(azure_dataset_id: str,
                     target_folder: Path,
                     dataset_csv: str,
                     azure_config: AzureConfig) -> Path:
    """
    Downloads or checks for an existing dataset on the executing machine. If a local_dataset is supplied and the
    directory is present, return that. Otherwise, download the dataset specified by the azure_dataset_id from the
    AzureML dataset attached to the given AzureML workspace. The dataset is downloaded into the `target_folder`,
    in a subfolder that has the same name as the dataset. If there already appears to be such a folder, and the folder
    contains a dataset csv file, no download is started.
    :param azure_dataset_id: The name of a dataset that is registered in the AzureML workspace.
    :param target_folder: The folder in which to download the dataset from Azure.
    :param dataset_csv: Name of the csv file describing the dataset. This is only used to check if the dataset has been
    downloaded already.
    :param azure_config: All Azure-related configuration options.
    :return: A path on the local machine that contains the dataset.
    """
    logging.info("Trying to download dataset via AzureML datastore now.")
    azure_dataset = get_or_create_dataset(azure_config, azure_dataset_id)
    if not isinstance(azure_dataset, FileDataset):
        raise ValueError(f"Expected to get a FileDataset, but got {type(azure_dataset)}")
    # The downloaded dataset may already exist from a previous run.
    expected_dataset_path = target_folder / azure_dataset_id
    logging.info(f"Model training will use dataset '{azure_dataset_id}' in Azure.")
    if expected_dataset_path.is_dir():
        if dataset_csv:
            if (expected_dataset_path / dataset_csv).is_file():
                logging.info(f"The file {dataset_csv} is already downloaded in {expected_dataset_path}. Skipping.")
                return expected_dataset_path
        else:
            existing_files = sum(1 for _ in expected_dataset_path.rglob("*"))
            if existing_files > 1:
                logging.info(f"There are already {existing_files} files in {expected_dataset_path}. Skipping.")
                return expected_dataset_path

    logging.info("Starting to download the dataset - WARNING, this could take very long!")
    with logging_section("Downloading dataset"):
        t0 = time.perf_counter()
        azure_dataset.download(target_path=str(expected_dataset_path), overwrite=False)
        t1 = time.perf_counter() - t0
        logging.info(f"Azure dataset '{azure_dataset_id}' downloaded in {t1} seconds")
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
                 model_config: Optional[DeepLearningConfig],
                 azure_config: Optional[AzureConfig] = None,
                 container: Optional[LightningContainer] = None,
                 project_root: Optional[Path] = None,
                 post_cross_validation_hook: Optional[PostCrossValidationHookSignature] = None,
                 model_deployment_hook: Optional[ModelDeploymentHookSignature] = None) -> None:
        """
        Driver class to run a ML experiment. Note that the project root argument MUST be supplied when using InnerEye
        as a package!
        :param model_config: If None, run the training as per the `container` argument (bring-your-own-model). If not
        None, this is the model configuration for a built-in InnerEye model.
        :param container: The LightningContainer object to use for training. If None, assume that the training is
        for a built-in InnerEye model.
        :param azure_config: Azure related configurations
        :param project_root: Project root. This should only be omitted if calling run_ml from the test suite. Supplying
        it is crucial when using InnerEye as a package or submodule!
        :param post_cross_validation_hook: A function to call after waiting for completion of cross validation runs.
        The function is called with the model configuration and the path to the downloaded and merged metrics files.
        :param model_deployment_hook: an optional function for deploying a model in an application-specific way.
        If present, it should take a model config (SegmentationModelBase), an AzureConfig, and an AzureML
        Model as arguments, and return an optional Path and a further object of any type.
        """
        if model_config is not None and container is not None:
            raise ValueError("One of the two arguments 'model_config', 'container' must be provided.")
        self.model_config = model_config
        if container is None:
            assert isinstance(model_config, ModelConfigBase), \
                "When using a built-in InnerEye model, the configuration should be an instance of ModelConfigBase"
            container = InnerEyeContainer(model_config)
        self.container = container
        self.azure_config: AzureConfig = azure_config or AzureConfig()
        self.project_root: Path = project_root or fixed_paths.repository_root_directory()
        self.post_cross_validation_hook = post_cross_validation_hook
        self.model_deployment_hook = model_deployment_hook
        self._has_setup_run = False

    def setup(self) -> None:
        """
        If the present object is using one of the InnerEye built-in models, create a (fake) container for it
        and call the setup method. It sets the random seeds, and then creates the actual Lightning modules.
        """
        if self._has_setup_run:
            return
        if not self.azure_config.only_register_model:
            # Set local_dataset to the mounted path specified in azure_runner.py, if any, or download it if that fails
            # and config.local_dataset was not already set.
            # This must happen before container setup because that could already read datasets.
            self.container.local_dataset = self.mount_or_download_dataset()
        # Ensure that we use fixed seeds before initializing the PyTorch models
        seed_everything(self.container.get_effective_random_seed())
        # This call needs to happen before the LightningModule is created, because the output parameters of the
        # container will be copied into the module.
        self.container.create_filesystem(self.project_root)
        # A lot of the code for the built-in InnerEye models expects the output paths directly in the config files.
        if isinstance(self.container, InnerEyeContainer):
            self.container.config.local_dataset = self.container.local_dataset
            self.container.config.file_system_config = self.container.file_system_config
        self.container.setup()
        self.container.create_lightning_module_and_store()
        self._has_setup_run = True

    def start_logging_to_file(self) -> None:
        if self.container is None:
            self.setup()
        logging_to_file(self.container.logs_folder / LOG_FILE_NAME)

    @property
    def is_offline_run(self) -> bool:
        """
        Returns True if the present run is outside of AzureML, and False if it is inside of AzureML.
        :return:
        """
        return is_offline_run_context(RUN_CONTEXT)

    def is_offline_cross_val_parent_run(self) -> bool:
        """
        Returns true if the current run is an offline run with cross validation splits > 0
        and cross_validation_split_index == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX (ie: a parent)
        """
        return self.container.cross_validation_split_index == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX and \
               self.container.perform_cross_validation and self.is_offline_run

    def spawn_offline_cross_val_classification_child_runs(self) -> None:
        """
        Trains and Tests k models based on their respective data splits sequentially.
        Stores the results on the Validation set to the outputs directory of the parent run.
        """
        _config = self.model_config
        assert isinstance(_config, ScalarModelBase)
        parent_run_file_system = _config.file_system_config

        def _spawn_run(cross_val_split_index: int) -> None:
            split_config = copy.deepcopy(_config)
            split_config.cross_validation_split_index = cross_val_split_index
            _local_split_folder_name = str(cross_val_split_index)
            split_config.file_system_config = parent_run_file_system.add_subfolder(_local_split_folder_name)
            logging.info(f"Running model train and test on cross validation split: {cross_val_split_index}")
            split_ml_runner = MLRunner(model_config=split_config,
                                       container=None,
                                       azure_config=self.azure_config,
                                       project_root=self.project_root,
                                       post_cross_validation_hook=self.post_cross_validation_hook,
                                       model_deployment_hook=self.model_deployment_hook)
            split_ml_runner.run()

        for i in range(_config.number_of_cross_validation_splits):
            _spawn_run(i)

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
        new_tags[CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY] = str(self.container.cross_validation_split_index)
        new_tags[EFFECTIVE_RANDOM_SEED_KEY_NAME] = str(self.container.get_effective_random_seed())
        RUN_CONTEXT.set_tags(new_tags)

    def run(self) -> None:
        """
        Driver function to run a ML experiment. If an offline cross validation run is requested, then
        this function is recursively called for each cross validation split.
        """
        self.setup()
        if self.is_offline_cross_val_parent_run():
            if self.model_config.is_segmentation_model:
                raise NotImplementedError("Offline cross validation is only supported for classification models.")
            self.spawn_offline_cross_val_classification_child_runs()
            return

        # Get the AzureML context in which the script is running
        if not self.is_offline_run and PARENT_RUN_CONTEXT is not None:
            logging.info("Setting tags from parent run.")
            self.set_run_tags_from_parent()

        # Set data loader start method
        self.set_multiprocessing_start_method()

        # configure recovery container if provided
        checkpoint_handler = CheckpointHandler(container=self.container,
                                               azure_config=self.azure_config,
                                               project_root=self.project_root,
                                               run_context=RUN_CONTEXT)
        checkpoint_handler.download_recovery_checkpoints_or_weights()
        trainer: Optional[Trainer] = None
        # do training and inference, unless the "only register" switch is set (which requires a run_recovery
        # to be valid).
        if not self.azure_config.only_register_model:
            # train a new model if required
            if self.azure_config.train:
                with logging_section("Model training"):
                    trainer, _ = model_train(checkpoint_handler,
                                             container=self.container,
                                             num_nodes=self.azure_config.num_nodes)
                # log the number of epochs used for model training
                RUN_CONTEXT.log(name="Train epochs", value=self.container.num_epochs)
            elif isinstance(self.container, InnerEyeContainer):
                self.model_config.write_dataset_files()
                self.create_activation_maps()

        if isinstance(self.container, InnerEyeContainer):
            # Inference for the InnerEye built-in models
            # We specify the ModelProcessing as DEFAULT here even if the run_recovery points to an ensemble run, because
            # the current run is a single one. See the documentation of ModelProcessing for more details.
            self.run_inference_and_register_model(checkpoint_handler, ModelProcessing.DEFAULT)

            if self.container.generate_report:
                self.generate_report(ModelProcessing.DEFAULT)

            # If this is an cross validation run, and the present run is child run 0, then wait for the sibling runs,
            # build the ensemble model, and write a report for that.
            if self.container.number_of_cross_validation_splits > 0:
                should_wait_for_other_child_runs = (not self.is_offline_run) and \
                                                   self.container.cross_validation_split_index == 0
                if should_wait_for_other_child_runs:
                    self.wait_for_runs_to_finish()
                    self.create_ensemble_model_and_run_inference()
        else:
            # Inference for all models that are specified via LightningContainers
            self.run_inference_for_lightning_models(checkpoint_handler.get_checkpoints_to_test(), trainer)

    def run_inference_for_lightning_models(self, checkpoint_paths: List[Path], trainer: Optional[Trainer]) -> None:
        """
        Run inference on the test set for all models that are specified via a LightningContainer.
        """
        if len(checkpoint_paths) != 1:
            raise ValueError(f"This method expects exactly 1 checkpoint for inference, but got {len(checkpoint_paths)}")
        lightning_model = self.container.model
        if isinstance(lightning_model, LightningInference) and \
                lightning_model.inference_step != LightningInference.inference_step:
            logging.info("Running inference via the LightningInference.inference_step method")
            data = self.container.get_inference_data_module()
            dataloaders: List[Tuple[DataLoader, ModelExecutionMode]] = []
            if self.container.perform_validation_and_test_set_inference:
                dataloaders.append((data.test_dataloader(), ModelExecutionMode.TEST))
                dataloaders.append((data.val_dataloader(), ModelExecutionMode.VAL))
            if self.container.perform_training_set_inference:
                dataloaders.append((data.train_dataloader(), ModelExecutionMode.TRAIN))
            map_location = "gpu" if self.container.use_gpu else "cpu"
            checkpoint = pl_load(checkpoint_paths[0], map_location=map_location)
            lightning_model.load_state_dict(checkpoint['state_dict'])
            lightning_model.eval()
            lightning_model.on_inference_start()
            for loader, split in dataloaders:
                logging.info(f"Starting inference on {split.value} set")
                lightning_model.on_inference_epoch_start(dataset_split=split, is_ensemble_model=False)
                for batch_idx, item in enumerate(loader):
                    model_output = lightning_model.forward(item[0])
                    lightning_model.inference_step(item, batch_idx, model_output=model_output)
                lightning_model.on_inference_epoch_end()
            lightning_model.on_inference_end()
        elif lightning_model.test_step != LightningModule.test_step:
            logging.info("Running inference via the LightningModule.test_step method")
            trainer = trainer or create_lightning_trainer(self.container)
            trainer.test(self.container.model,
                         test_dataloaders=self.container.get_data_module().test_dataloader(),
                         ckpt_path=str(checkpoint_paths[0]))
            logging.info("Finished inference.")
        else:
            logging.warning("None of the suitable test methods is overridden. Skipping inference completely.")

    def run_inference_and_register_model(self, checkpoint_handler: CheckpointHandler,
                                         model_proc: ModelProcessing) -> None:
        """
        Run inference as required, and register the model, but not necessarily in that order:
        if we can identify the epoch to register at without running inference, we register first.
        :param checkpoint_handler: Checkpoint handler object to find checkpoint paths for model initialization
        :param model_proc: whether we are running an ensemble model from within a child run with index 0. If we are,
        then outputs will be written to OTHER_RUNS/ENSEMBLE under the main outputs directory.
        """

        if self.should_register_model():
            checkpoint_paths = checkpoint_handler.get_checkpoints_to_test()
            if not checkpoint_paths:
                raise ValueError("Model registration failed: No checkpoints found")

            model_description = "Registering model."
            checkpoint_paths = checkpoint_paths
            self.register_model(checkpoint_paths, model_description, model_proc)

        if not self.azure_config.only_register_model:
            # run full image inference on existing or newly trained model on the training, and testing set
            test_metrics, val_metrics, _ = self.model_inference_train_and_test(checkpoint_handler=checkpoint_handler,
                                                                               model_proc=model_proc)

            self.try_compare_scores_against_baselines(model_proc)

    def should_register_model(self) -> bool:
        """
        Returns True if we should register a model at all. No models should be registered when running outside
        AzureML. Inside AzureML, if no training has taken place, an equivalent
        model (from the run we recovered) should already have been registered, so we should only
        do so if this run is specifically for that purpose.
        """
        if self.is_offline_run:
            return False
        return self.azure_config.train or self.azure_config.only_register_model

    def create_activation_maps(self) -> None:
        if self.model_config.is_segmentation_model and self.model_config.activation_map_layers is not None:
            logging.info("Extracting activation maps for layer")
            activation_maps.extract_activation_maps(self.model_config)
            logging.info("Successfully extracted and saved activation maps")

    def mount_or_download_dataset(self) -> Optional[Path]:
        """
        Makes the dataset that the model uses available on the executing machine. If the present training run is outside
        of AzureML, it expects that either the model has a `local_dataset` field set, in which case no action will be
        taken. If a dataset is specified in `azure_dataset_id`, it will attempt to download the dataset from Azure
        into the local repository, in the "datasets" folder.
        If the training run is inside of AzureML, the dataset that was specified at job submission time will be
        mounted or downloaded.
        Returns the path of the dataset on the executing machine.
        """
        azure_dataset_id = self.container.azure_dataset_id
        local_dataset = self.container.local_dataset
        # A dataset, either local or in Azure, is required for the built-in InnerEye models. When models are
        # specified via a LightningContainer, these dataset fields are optional, because the container datasets
        # could be downloaded even from the web.
        is_dataset_required = isinstance(self.container, InnerEyeContainer)
        if self.is_offline_run:
            # The present run is outside of AzureML: If local_dataset is set, use that as the path to the data.
            # Otherwise, download the dataset specified by the azure_dataset_id
            if is_dataset_required:
                if (not azure_dataset_id) and (local_dataset is None):
                    raise ValueError("The model must contain either local_dataset or azure_dataset_id.")
            if local_dataset:
                expected_dir = Path(local_dataset)
                if not expected_dir.is_dir():
                    raise FileNotFoundError(f"The model uses a dataset in {expected_dir}, but that does not exist.")
                logging.info(f"Model training will use the local dataset provided in {expected_dir}")
                return expected_dir
            if azure_dataset_id:
                dataset_csv = ""
                if isinstance(self.model_config, DeepLearningConfig):
                    dataset_csv = self.model_config.dataset_csv
                return download_dataset(azure_dataset_id=azure_dataset_id,
                                        target_folder=self.project_root / fixed_paths.DATASETS_DIR_NAME,
                                        dataset_csv=dataset_csv, azure_config=self.azure_config)
            return None

        # Inside of AzureML, datasets can be either mounted or downloaded.
        if is_dataset_required and not azure_dataset_id:
            raise ValueError("The model must contain azure_dataset_id for running on AML")
        if azure_dataset_id:
            mounted = try_to_mount_input_dataset()
            if not mounted:
                raise ValueError("Unable to mount or download input dataset.")
            return mounted
        return None

    def set_multiprocessing_start_method(self) -> None:
        """
        Set the (PyTorch) multiprocessing start method.
        """
        method = self.container.multiprocessing_start_method
        if is_windows():
            if method != MultiprocessingStartMethod.spawn:
                logging.warning(f"Cannot set multiprocessing start method to '{method.name}' "
                                "because only 'spawn' is available in Windows")
        else:
            logging.info(f"Setting multiprocessing start method to '{method.name}'")
            torch.multiprocessing.set_start_method(method.name, force=True)

    def register_model(self,
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

        if not self.is_offline_run:
            split_index = RUN_CONTEXT.get_tags().get(CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, None)
            if split_index == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX:
                RUN_CONTEXT.tag(IS_ENSEMBLE_KEY_NAME, str(model_proc == ModelProcessing.ENSEMBLE_CREATION))
            elif PARENT_RUN_CONTEXT is not None:
                RUN_CONTEXT.tag(PARENT_RUN_ID_KEY_NAME, str(PARENT_RUN_CONTEXT.id))
        if isinstance(self.model_config, SegmentationModelBase):
            with logging_section(f"Registering {model_proc.value} model"):
                self.register_segmentation_model(
                    checkpoint_paths=checkpoint_paths,
                    model_description=model_description,
                    model_proc=model_proc)
        else:
            logging.info(f"No deployment done for this type of model: {type(self.model_config)}")

    def try_compare_scores_against_baselines(self, model_proc: ModelProcessing) -> None:
        """
        Attempt comparison of scores against baseline scores and scatterplot creation if possible.
        """
        if not isinstance(self.model_config, SegmentationModelBase):  # keep type checker happy
            return
        from InnerEye.ML.baselines_util import compare_scores_against_baselines
        with logging_section("Comparing scores against baselines"):
            compare_scores_against_baselines(self.model_config, self.azure_config, model_proc)

    def register_segmentation_model(self,
                                    checkpoint_paths: List[Path],
                                    model_description: str,
                                    model_proc: ModelProcessing) -> Tuple[Model, Any]:
        """
        Registers a new model in the workspace's model registry to be deployed further,
        and creates a model zip for portal deployment (if required).
        :param model_description: A string description that is added to the deployed model. It would usually contain
        the test set performance and information at which epoch the result was achieved.
        :param checkpoint_paths: Checkpoint paths to use to upload model checkpoints to AML.
        :param model_proc: whether it's a single or ensemble model.
        :returns Tuple element 1: AML model object, or None if no model could be registered.
        Tuple element 2: The result of running the model_deployment_hook, or None if no hook was supplied.
        """
        # The files for the final model can't live in the outputs folder. If they do: when registering the model,
        # the files may not yet uploaded by hosttools, and that may (or not) cause errors. Hence, place the folder
        # for the final models outside of "outputs", and upload manually.
        model_subfolder = FINAL_MODEL_FOLDER if model_proc == ModelProcessing.DEFAULT else FINAL_ENSEMBLE_MODEL_FOLDER
        # This is the path under which AzureML will know the files: Either "final_model" or "final_ensemble_model"
        artifacts_path = model_subfolder
        final_model_folder = self.model_config.file_system_config.run_folder / model_subfolder
        # Copy all code from project and InnerEye into the model folder, and copy over checkpoints.
        # This increases the size of the data stored for the run. The other option would be to store all checkpoints
        # right in the final model folder - however, then that would also contain any other checkpoints that the model
        # produced or downloaded for recovery, bloating the final model file.
        self.copy_child_paths_to_folder(final_model_folder, checkpoint_paths)
        # If the present run is a child run of a Hyperdrive parent run, and we are building an ensemble model,
        # register it the model on the parent run.
        if PARENT_RUN_CONTEXT and model_proc == ModelProcessing.ENSEMBLE_CREATION:
            run_to_register_on = PARENT_RUN_CONTEXT
            logging.info(f"Registering the model on the parent run {run_to_register_on.id}")
        else:
            run_to_register_on = RUN_CONTEXT
            logging.info(f"Registering the model on the current run {run_to_register_on.id}")
        logging.info(f"Uploading files in {final_model_folder} with prefix '{artifacts_path}'")
        final_model_folder_relative = final_model_folder.relative_to(Path.cwd())
        run_to_register_on.upload_folder(name=artifacts_path, path=str(final_model_folder_relative))
        # When registering the model on the run, we need to provide a relative path inside of the run's output
        # folder in `model_path`
        model = run_to_register_on.register_model(
            model_name=self.model_config.model_name,
            model_path=artifacts_path,
            tags=RUN_CONTEXT.get_tags(),
            description=model_description
        )
        # Add the name of the Python environment as a model tag, because we need it when running inference
        # on the model. We could add that as an immutable property, but with tags we have the option to modify
        # to a custom environment later.
        python_environment = RUN_CONTEXT.get_environment()
        assert python_environment.version == ENVIRONMENT_VERSION, \
            f"Expected all Python environments to have version '{ENVIRONMENT_VERSION}', but got: " \
            f"'{python_environment.version}"
        model.add_tags({PYTHON_ENVIRONMENT_NAME: python_environment.name})
        # update the run's tags with the registered model information
        run_to_register_on.tag(MODEL_ID_KEY_NAME, model.id)

        deployment_result = None
        logging.info(f"Registered {model_proc.value} model: {model.name}, with Id: {model.id}")
        # create a version of the model for deployment if the hook is provided
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
                                   checkpoint_paths: List[Path],
                                   python_environment: Optional[Environment] = None) -> None:
        """
        Gets the files that are required to register a model for inference. The necessary files are copied from
        the current folder structure into the given temporary folder.
        The folder will contain all source code in the InnerEye folder, possibly additional source code from the
        extra_code_directory, and all checkpoints in a newly created "checkpoints" folder inside the model.
        In addition, the name of the present AzureML Python environment will be written to a file, for later use
        in the inference code.
        :param model_folder: The folder into which all files should be copied.
        :param checkpoint_paths: A list with absolute paths to checkpoint files. They are expected to be
        inside of the model's checkpoint folder.
        :param python_environment: The Python environment that is used in the present AzureML run.
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

        # log the metrics to AzureML experiment if possible. When doing ensemble runs, log to the Hyperdrive parent run,
        # so that we get the metrics of child run 0 and the ensemble separated.
        if config.is_segmentation_model and not self.is_offline_run:
            run_for_logging = PARENT_RUN_CONTEXT if model_proc.ENSEMBLE_CREATION else RUN_CONTEXT
            log_metrics(val_metrics=val_metrics, test_metrics=test_metrics,  # type: ignore
                        train_metrics=train_metrics, run_context=run_for_logging)  # type: ignore

        return test_metrics, val_metrics, train_metrics

    @stopit.threading_timeoutable()
    def wait_for_runs_to_finish(self, delay: int = 60) -> None:
        """
        Wait for cross val runs (apart from the current one) to finish and then aggregate results of all.
        :param delay: How long to wait between polls to AML to get status of child runs
        """
        with logging_section("Waiting for sibling runs"):
            while not self.are_sibling_runs_finished():
                time.sleep(delay)

    def are_sibling_runs_finished(self) -> bool:
        """
        Checks if all child runs (except the current run) of the current run's parent are completed or failed.
        :return: True if all sibling runs of the current run have finished (they either completed successfully,
        or failed). False if any of them is still pending (running or queued).
        """
        if (not self.is_offline_run) \
                and (azure_util.is_cross_validation_child_run(RUN_CONTEXT)):
            n_splits = self.model_config.get_total_number_of_cross_validation_runs()
            child_runs = azure_util.fetch_child_runs(PARENT_RUN_CONTEXT,
                                                     expected_number_cross_validation_splits=n_splits)
            pending_runs = [x.id for x in child_runs
                            if (x.id != RUN_CONTEXT.id)
                            and (x.get_status() not in [RunStatus.COMPLETED, RunStatus.FAILED])]
            all_runs_finished = len(pending_runs) == 0
            if not all_runs_finished:
                logging.info(f"Waiting for sibling run(s) to finish: {pending_runs}")
            return all_runs_finished
        else:
            raise NotImplementedError("are_sibling_runs_finished only works for cross validation runs in AzureML.")

    def create_ensemble_model_and_run_inference(self) -> None:
        """
        Create an ensemble model from the results of the sibling runs of the present run. The present run here will
        be cross validation child run 0.
        """
        assert PARENT_RUN_CONTEXT, "This function should only be called in a Hyperdrive run"
        with logging_section("Downloading checkpoints from sibling runs"):
            checkpoint_handler = CheckpointHandler(container=self.container,
                                                   azure_config=self.azure_config,
                                                   project_root=self.project_root,
                                                   run_context=PARENT_RUN_CONTEXT)
            checkpoint_handler.download_checkpoints_from_hyperdrive_child_runs(PARENT_RUN_CONTEXT)

        self.run_inference_and_register_model(checkpoint_handler=checkpoint_handler,
                                              model_proc=ModelProcessing.ENSEMBLE_CREATION)

        crossval_dir = self.plot_cross_validation_and_upload_results()
        if self.model_config.generate_report:
            self.generate_report(ModelProcessing.ENSEMBLE_CREATION)
        # CrossValResults should have been uploaded to the parent run, so we don't need it here.
        remove_file_or_directory(crossval_dir)
        # We can also remove OTHER_RUNS under the root, as it is no longer useful and only contains copies of files
        # available elsewhere. However, first we need to upload relevant parts of OTHER_RUNS/ENSEMBLE.
        other_runs_dir = self.model_config.outputs_folder / OTHER_RUNS_SUBDIR_NAME
        other_runs_ensemble_dir = other_runs_dir / ENSEMBLE_SPLIT_NAME
        if PARENT_RUN_CONTEXT is not None:
            if other_runs_ensemble_dir.exists():
                # Only keep baseline Wilcoxon results and scatterplots and reports
                for subdir in other_runs_ensemble_dir.glob("*"):
                    if subdir.name not in [BASELINE_WILCOXON_RESULTS_FILE,
                                           SCATTERPLOTS_SUBDIR_NAME,
                                           reports_folder]:
                        remove_file_or_directory(subdir)
                PARENT_RUN_CONTEXT.upload_folder(name=BASELINE_COMPARISONS_FOLDER, path=str(other_runs_ensemble_dir))
            else:
                logging.warning(f"Directory not found for upload: {other_runs_ensemble_dir}")
        remove_file_or_directory(other_runs_dir)

    def plot_cross_validation_and_upload_results(self) -> Path:
        from InnerEye.ML.visualizers.plot_cross_validation import crossval_config_from_model_config, \
            plot_cross_validation, unroll_aggregate_metrics
        # perform aggregation as cross val splits are now ready
        plot_crossval_config = crossval_config_from_model_config(self.model_config)
        plot_crossval_config.run_recovery_id = PARENT_RUN_CONTEXT.tags[RUN_RECOVERY_ID_KEY_NAME]
        plot_crossval_config.outputs_directory = self.model_config.outputs_folder
        plot_crossval_config.azure_config = self.azure_config
        cross_val_results_root = plot_cross_validation(plot_crossval_config, is_ensemble_run=True)
        if self.post_cross_validation_hook:
            self.post_cross_validation_hook(self.model_config, cross_val_results_root)
        # upload results to the parent run's outputs so that the files are visible inside the AzureML UI.
        PARENT_RUN_CONTEXT.upload_folder(name=CROSSVAL_RESULTS_FOLDER, path=str(cross_val_results_root))
        if self.model_config.is_scalar_model:
            try:
                aggregates = pd.read_csv(cross_val_results_root / METRICS_AGGREGATES_FILE)
                unrolled_aggregate_metrics = unroll_aggregate_metrics(aggregates)
                for m in unrolled_aggregate_metrics:
                    PARENT_RUN_CONTEXT.log(m.metric_name, m.metric_value)
            except Exception as ex:
                print_exception(ex, "Unable to log metrics to Hyperdrive parent run.", logger_fn=logging.warning)
        return cross_val_results_root

    def generate_report(self, model_proc: ModelProcessing) -> None:
        config = self.model_config
        if config.model_category not in [ModelCategory.Segmentation, ModelCategory.Classification]:
            logging.info(f"No reporting available for a model with category {config.model_category}")
            return
        logging.info("Saving report in HTML")
        try:
            def get_epoch_path(mode: ModelExecutionMode) -> Path:
                p = get_epoch_results_path(mode=mode, model_proc=model_proc)
                return config.outputs_folder / p / SUBJECT_METRICS_FILE_NAME

            path_to_best_epoch_train = get_epoch_path(ModelExecutionMode.TRAIN)
            path_to_best_epoch_val = get_epoch_path(ModelExecutionMode.VAL)
            path_to_best_epoch_test = get_epoch_path(ModelExecutionMode.TEST)

            output_dir = config.outputs_folder / OTHER_RUNS_SUBDIR_NAME / ENSEMBLE_SPLIT_NAME \
                if model_proc == ModelProcessing.ENSEMBLE_CREATION else config.outputs_folder

            reports_dir = output_dir / reports_folder
            if not reports_dir.exists():
                reports_dir.mkdir(exist_ok=False)

            if config.model_category == ModelCategory.Segmentation:
                generate_segmentation_notebook(
                    result_notebook=reports_dir / get_ipynb_report_name(config.model_category.value),
                    train_metrics=path_to_best_epoch_train,
                    val_metrics=path_to_best_epoch_val,
                    test_metrics=path_to_best_epoch_test)
            else:
                if isinstance(config, ScalarModelBase) and not isinstance(config, SequenceModelBase):
                    generate_classification_notebook(
                        result_notebook=reports_dir / get_ipynb_report_name(config.model_category.value),
                        config=config,
                        train_metrics=path_to_best_epoch_train,
                        val_metrics=path_to_best_epoch_val,
                        test_metrics=path_to_best_epoch_test)

                    if len(config.class_names) > 1:
                        generate_classification_multilabel_notebook(
                            result_notebook=reports_dir / get_ipynb_report_name(
                                f"{config.model_category.value}_multilabel"),
                            config=config,
                            train_metrics=path_to_best_epoch_train,
                            val_metrics=path_to_best_epoch_val,
                            test_metrics=path_to_best_epoch_test)
                else:
                    logging.info(f"Cannot create report for config of type {type(config)}.")
        except Exception as ex:
            print_exception(ex, "Failed to generated reporting notebook.")
            raise
