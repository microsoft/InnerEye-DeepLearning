#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import copy
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import stopit
import torch.multiprocessing
from azureml._restclient.constants import RunStatus
from azureml.core import Model, Run, model
from azureml.data import FileDataset
from pytorch_lightning import LightningModule, seed_everything
from torch.utils.data import DataLoader

from InnerEye.Azure import azure_util
from InnerEye.Azure.azure_config import AzureConfig, INPUT_DATA_KEY
from InnerEye.Azure.azure_runner import ENVIRONMENT_VERSION, ENV_OMPI_COMM_WORLD_RANK, get_git_tags
from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, DEFAULT_CROSS_VALIDATION_SPLIT_INDEX, \
    EFFECTIVE_RANDOM_SEED_KEY_NAME, IS_ENSEMBLE_KEY_NAME, MODEL_ID_KEY_NAME, PARENT_RUN_CONTEXT, \
    PARENT_RUN_ID_KEY_NAME, RUN_CONTEXT, RUN_RECOVERY_FROM_ID_KEY_NAME, RUN_RECOVERY_ID_KEY_NAME, \
    create_run_recovery_id, get_all_environment_files, is_offline_run_context, merge_conda_files
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import BASELINE_COMPARISONS_FOLDER, BASELINE_WILCOXON_RESULTS_FILE, \
    CROSSVAL_RESULTS_FOLDER, ENSEMBLE_SPLIT_NAME, FULL_METRICS_DATAFRAME_FILE, METRICS_AGGREGATES_FILE, \
    ModelProcessing, \
    OTHER_RUNS_SUBDIR_NAME, SCATTERPLOTS_SUBDIR_NAME, SUBJECT_METRICS_FILE_NAME, \
    change_working_directory, get_best_epoch_results_path, is_windows, logging_section, logging_to_file, \
    print_exception, remove_file_or_directory
from InnerEye.Common.fixed_paths import INNEREYE_PACKAGE_NAME, LOG_FILE_NAME, PYTHON_ENVIRONMENT_NAME
from InnerEye.ML.baselines_util import compare_folders_and_run_outputs
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.deep_learning_config import CHECKPOINT_FOLDER, DeepLearningConfig, FINAL_ENSEMBLE_MODEL_FOLDER, \
    FINAL_MODEL_FOLDER, ModelCategory, MultiprocessingStartMethod, load_checkpoint
from InnerEye.ML.lightning_base import InnerEyeContainer
from InnerEye.ML.lightning_container import InnerEyeInference, LightningContainer
from InnerEye.ML.metrics import InferenceMetrics, InferenceMetricsForSegmentation
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_inference_config import ModelInferenceConfig
from InnerEye.ML.model_testing import model_test
from InnerEye.ML.model_training import create_lightning_trainer, is_global_rank_zero, model_train
from InnerEye.ML.reports.notebook_report import generate_classification_crossval_notebook, \
    generate_classification_multilabel_notebook, generate_classification_notebook, generate_segmentation_notebook, \
    get_ipynb_report_name, reports_folder
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils.checkpoint_handling import CheckpointHandler
from InnerEye.ML.visualizers import activation_maps
from InnerEye.ML.visualizers.plot_cross_validation import \
    get_config_and_results_for_offline_runs, plot_cross_validation_from_files

ModelDeploymentHookSignature = Callable[[LightningContainer, AzureConfig, Model, ModelProcessing], Any]
PostCrossValidationHookSignature = Callable[[ModelConfigBase, Path], None]


def try_to_mount_input_dataset(dataset_index: int = 0) -> Optional[Path]:
    """
    Checks if the AzureML run context has a field for input datasets. If yes, the dataset stored there is
    returned as a Path. Returns None if no input datasets was found.

    :param dataset_index: suffix of AML dataset name, return path to INPUT_DATA_KEY_idx dataset
    """
    if hasattr(RUN_CONTEXT, "input_datasets"):
        try:
            return Path(RUN_CONTEXT.input_datasets[f"{INPUT_DATA_KEY}_{dataset_index}"])
        except KeyError:
            logging.warning(f"Run context field input_datasets has no {INPUT_DATA_KEY}_{dataset_index} entry.")
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
    azure_dataset = azure_config.get_or_create_dataset(azure_dataset_id)
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


def log_metrics(metrics: Dict[ModelExecutionMode, InferenceMetrics],
                run_context: Run) -> None:
    """
    Log metrics for each split to the provided run, or the current run context if None provided
    :param metrics: Dictionary of inference results for each split.
    :param run_context: Run for which to log the metrics to, use the current run context if None provided
    """
    for split in metrics.values():
        if isinstance(split, InferenceMetricsForSegmentation):
            split.log_metrics(run_context)


class MLRunner:

    def __init__(self,
                 model_config: Optional[DeepLearningConfig] = None,
                 container: Optional[LightningContainer] = None,
                 azure_config: Optional[AzureConfig] = None,
                 project_root: Optional[Path] = None,
                 post_cross_validation_hook: Optional[PostCrossValidationHookSignature] = None,
                 model_deployment_hook: Optional[ModelDeploymentHookSignature] = None,
                 output_subfolder: str = "") -> None:
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
        If present, it should take a LightningContainer, an AzureConfig, an AzureML Model and a ModelProcessing object
        as arguments, and return an object of any type.
        :param output_subfolder: If provided, the output folder structure will have an additional subfolder,
        when running outside AzureML.
        """
        self.model_config = model_config
        if container is None:
            assert isinstance(model_config, ModelConfigBase), \
                "When using a built-in InnerEye model, the configuration should be an instance of ModelConfigBase"
            container = InnerEyeContainer(model_config)
        self.container = container
        self.azure_config: AzureConfig = azure_config or AzureConfig()
        self.container.num_nodes = self.azure_config.num_nodes
        self.project_root: Path = project_root or fixed_paths.repository_root_directory()
        self.post_cross_validation_hook = post_cross_validation_hook
        self.model_deployment_hook = model_deployment_hook
        self.output_subfolder = output_subfolder
        self._has_setup_run = False

    def setup(self, use_mount_or_download_dataset: bool = True) -> None:
        """
        If the present object is using one of the InnerEye built-in models, create a (fake) container for it
        and call the setup method. It sets the random seeds, and then creates the actual Lightning modules.
        :param use_mount_or_download_dataset: If True, try to download or mount the dataset that is used by the model.
        If False, assume that the dataset is already available (this should only be used for unit tests).
        """
        if self._has_setup_run:
            return
        if (not self.azure_config.only_register_model) and use_mount_or_download_dataset:
            # Set local_dataset to the mounted path specified in azure_runner.py, if any, or download it if that fails
            # and config.local_dataset was not already set.
            # This must happen before container setup because that could already read datasets.
            mounted_dataset = self.mount_or_download_dataset(self.container.azure_dataset_id,
                                                             self.container.local_dataset)
            if mounted_dataset is not None:
                self.container.local_dataset = mounted_dataset

            extra_locals = []
            if self.is_offline_run and len(self.container.extra_local_dataset_paths) != 0:
                for local in self.container.extra_local_dataset_paths:
                    extra_local_dataset = self.mount_or_download_dataset(None, local)
                    assert extra_local_dataset is not None  # for mypy
                    extra_locals.append(extra_local_dataset)
            elif len(self.container.extra_azure_dataset_ids) != 0:
                for i, azure_id in enumerate(self.container.extra_azure_dataset_ids, 1):
                    extra_local_dataset = self.mount_or_download_dataset(azure_id, None, dataset_index=i)
                    assert extra_local_dataset is not None  # for mypy
                    extra_locals.append(extra_local_dataset)
            self.container.extra_local_dataset_paths = extra_locals
        # Ensure that we use fixed seeds before initializing the PyTorch models
        seed_everything(self.container.get_effective_random_seed())
        # Creating the folder structure must happen before the LightningModule is created, because the output
        # parameters of the container will be copied into the module.
        if self.output_subfolder:
            # This codepath is only executed for cross validation runs outside AzureML: The folder structure
            # uses an existing folder structure set by the caller, and just a subfolder is added.
            self.container.file_system_config = self.container.file_system_config.add_subfolder(self.output_subfolder)
        else:
            self.container.create_filesystem(self.project_root)

        # configure recovery container if provided
        self.checkpoint_handler = CheckpointHandler(container=self.container,
                                                    azure_config=self.azure_config,
                                                    project_root=self.project_root,
                                                    run_context=RUN_CONTEXT)
        self.checkpoint_handler.download_recovery_checkpoints_or_weights(only_return_path=not is_global_rank_zero())

        # A lot of the code for the built-in InnerEye models expects the output paths directly in the config files.
        if isinstance(self.container, InnerEyeContainer):
            self.container.config.local_dataset = self.container.local_dataset
            self.container.config.file_system_config = self.container.file_system_config
            self.container.config.extra_downloaded_run_id = self.container.extra_downloaded_run_id
        self.container.setup()
        self.container.create_lightning_module_and_store()
        self._has_setup_run = True

    @property
    def is_offline_run(self) -> bool:
        """
        Returns True if the present run is outside of AzureML, and False if it is inside of AzureML.
        :return:
        """
        return is_offline_run_context(RUN_CONTEXT)

    @property
    def innereye_config(self) -> DeepLearningConfig:
        """
        Gets the model configuration object for all built-in InnerEye models. Raises an exception if the present
        object trains a LightningContainer that is not a built-in InnerEye model.
        """
        if self.model_config is None or not isinstance(self.model_config, DeepLearningConfig):
            raise ValueError("This property should only be used with built-in InnerEye models, but model "
                             f"configuration is of type {type(self.model_config)}")
        return self.model_config

    @property
    def config_namespace(self) -> str:
        """
        Returns the namespace of the model configuration object, i.e. return the name of the module in which the
        model configuration object or the lightning container object is defined.
        For models defined as lightning containers, this is the namespace of the container class defining the model.
        For legacy InnerEye models, the original config is not a container object, but instead a subclass of
        ModelConfigBase. In this case, return the namespace of the original config class, not the namespace of the
        derived InnerEyeContainer.

        Examples:
        1. For the Lung config class defined in InnerEye/ML/configs/segmentation/Lung.py,
           the namespace is InnerEye.ML.configs.segmentation.Lung
        1. For the HelloContainer container class defined in InnerEye/ML/configs/other/HelloContainer.py,
           the namespace is InnerEye.ML.configs.other.HelloContainer
        """
        if isinstance(self.container, InnerEyeContainer):
            return self.innereye_config.__class__.__module__
        else:
            return self.container.__class__.__module__

    def start_logging_to_file(self) -> None:
        if self.container is None:
            self.setup()
        logging_to_file(self.container.logs_folder / LOG_FILE_NAME)

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
        assert isinstance(self.innereye_config, ScalarModelBase)

        def _spawn_run(cross_val_split_index: int) -> None:
            split_config = copy.deepcopy(self.innereye_config)
            split_config.cross_validation_split_index = cross_val_split_index
            logging.info(f"Running model train and test on cross validation split: {cross_val_split_index}")
            split_ml_runner = MLRunner(model_config=split_config,
                                       container=None,
                                       azure_config=self.azure_config,
                                       project_root=self.project_root,
                                       post_cross_validation_hook=self.post_cross_validation_hook,
                                       model_deployment_hook=self.model_deployment_hook,
                                       output_subfolder=str(cross_val_split_index))
            split_ml_runner.run()

        for i in range(self.innereye_config.number_of_cross_validation_splits):
            _spawn_run(i)

        config_and_files = get_config_and_results_for_offline_runs(self.innereye_config)
        plot_cross_validation_from_files(config_and_files, Path(config_and_files.config.outputs_directory))

    def set_run_tags_from_parent(self) -> None:
        """
        Set metadata for the run
        """
        assert PARENT_RUN_CONTEXT, "This function should only be called in a Hyperdrive run."
        run_tags_parent = PARENT_RUN_CONTEXT.get_tags()
        git_tags = get_git_tags(self.azure_config)
        tags_to_copy = [
            "tag",
            "model_name",
            "execution_mode",
            "recovered_from",
            "friendly_name",
            "build_number",
            "build_user",
            *git_tags.keys(),
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
            if self.innereye_config.is_segmentation_model:
                raise NotImplementedError("Offline cross validation is only supported for classification models.")
            self.spawn_offline_cross_val_classification_child_runs()
            return

        # Get the AzureML context in which the script is running
        if not self.is_offline_run and PARENT_RUN_CONTEXT is not None:
            logging.info("Setting tags from parent run.")
            self.set_run_tags_from_parent()

        # Set data loader start method
        self.set_multiprocessing_start_method()

        # do training and inference, unless the "only register" switch is set (which requires a run_recovery
        # to be valid).
        if not self.azure_config.only_register_model:
            # train a new model if required
            if self.azure_config.train:
                with logging_section("Model training"):
                    model_train(self.checkpoint_handler,
                                container=self.container,
                                num_nodes=self.azure_config.num_nodes)
                # log the number of epochs used for model training
                RUN_CONTEXT.log(name="Train epochs", value=self.container.num_epochs)
            elif isinstance(self.container, InnerEyeContainer):
                self.innereye_config.write_dataset_files()
                self.create_activation_maps()

        # Register the model, and then run inference as required. No models should be registered when running outside
        # AzureML.
        if not self.is_offline_run:
            if self.should_register_model():
                self.register_model(self.checkpoint_handler, ModelProcessing.DEFAULT)

        if not self.azure_config.only_register_model:
            if isinstance(self.container, InnerEyeContainer):
                # Inference for the InnerEye built-in models
                # We specify the ModelProcessing as DEFAULT here even if the run_recovery points to an ensemble run,
                # because the current run is a single one. See the documentation of ModelProcessing for more details.
                self.run_inference(self.checkpoint_handler, ModelProcessing.DEFAULT)

                if self.container.generate_report:
                    self.generate_report(ModelProcessing.DEFAULT)

                # If this is an cross validation run, and the present run is child run 0, then wait for the sibling
                # runs, build the ensemble model, and write a report for that.
                if self.container.perform_cross_validation:
                    should_wait_for_other_child_runs = (not self.is_offline_run) and \
                                                       self.container.cross_validation_split_index == 0
                    if should_wait_for_other_child_runs:
                        self.wait_for_runs_to_finish()
                        self.create_ensemble_model_and_run_inference()
            else:
                # Inference for all models that are specified via LightningContainers.
                with logging_section("Model inference"):
                    self.run_inference_for_lightning_models(self.checkpoint_handler.get_checkpoints_to_test())
                # We can't enforce that files are written to the output folder, hence change the working directory
                # manually
                with change_working_directory(self.container.outputs_folder):
                    self.container.create_report()

        if self.container.regression_test_folder:
            # Comparison with stored results for cross-validation runs only operates on child run 0. This run
            # has usually already downloaded the results for the other runs, and uploaded files to the parent
            # run context.
            logging.info("Comparing the current results against stored results")
            if self.is_normal_run_or_crossval_child_0():
                compare_folders_and_run_outputs(expected=self.container.regression_test_folder,
                                                actual=self.container.outputs_folder)
            else:
                logging.info("Skipping because this is not cross-validation child run 0.")

    def is_normal_run_or_crossval_child_0(self) -> bool:
        """
        Returns True if the present run is a non-crossvalidation run, or child run 0 of a crossvalidation run.
        """
        if self.container.perform_cross_validation:
            return self.container.cross_validation_split_index == 0
        return True

    def run_inference_for_lightning_models(self, checkpoint_paths: List[Path]) -> None:
        """
        Run inference on the test set for all models that are specified via a LightningContainer.
        :param checkpoint_paths: The path to the checkpoint that should be used for inference.
        """
        if len(checkpoint_paths) != 1:
            raise ValueError(f"This method expects exactly 1 checkpoint for inference, but got {len(checkpoint_paths)}")
        lightning_model = self.container.model
        # Run the customized inference code only if the the "inference" step has been overridden
        if isinstance(lightning_model, InnerEyeInference) and \
                type(lightning_model).inference_step != InnerEyeInference.inference_step:
            logging.info("Running inference via the InnerEyeInference.inference_step method")
            # Read the data modules before changing the working directory, in case the code relies on relative paths
            data = self.container.get_inference_data_module()
            dataloaders: List[Tuple[DataLoader, ModelExecutionMode]] = []
            if self.container.run_perform_inference(ModelProcessing.DEFAULT, ModelExecutionMode.TEST):
                dataloaders.append((data.test_dataloader(), ModelExecutionMode.TEST))  # type: ignore
            if self.container.run_perform_inference(ModelProcessing.DEFAULT, ModelExecutionMode.VAL):
                dataloaders.append((data.val_dataloader(), ModelExecutionMode.VAL))  # type: ignore
            if self.container.run_perform_inference(ModelProcessing.DEFAULT, ModelExecutionMode.TRAIN):
                dataloaders.append((data.train_dataloader(), ModelExecutionMode.TRAIN))  # type: ignore
            checkpoint = load_checkpoint(checkpoint_paths[0], use_gpu=self.container.use_gpu)
            lightning_model.load_state_dict(checkpoint['state_dict'])
            lightning_model.eval()
            with change_working_directory(self.container.outputs_folder):
                lightning_model.on_inference_start()
                for loader, split in dataloaders:
                    logging.info(f"Starting inference on {split.value} set")
                    lightning_model.on_inference_epoch_start(dataset_split=split, is_ensemble_model=False)
                    for batch_idx, item in enumerate(loader):
                        model_output = lightning_model.forward(item[0])
                        lightning_model.inference_step(item, batch_idx, model_output=model_output)
                    lightning_model.on_inference_epoch_end()
                lightning_model.on_inference_end()
        elif type(lightning_model).test_step != LightningModule.test_step:
            # Run Lightning's built-in test procedure if the `test_step` method has been overridden
            logging.info("Running inference via the LightningModule.test_step method")
            # Lightning does not cope with having two calls to .fit or .test in the same script. As a workaround for
            # now, restrict number of GPUs to 1, meaning that it will not start DDP.
            self.container.max_num_gpus = 1
            # Without this, the trainer will think it should still operate in multi-node mode, and wrongly start
            # searching for Horovod
            if ENV_OMPI_COMM_WORLD_RANK in os.environ:
                del os.environ[ENV_OMPI_COMM_WORLD_RANK]
            # From the training setup, torch still thinks that it should run in a distributed manner,
            # and would block on some GPU operations. Hence, clean up distributed training.
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            trainer, _ = create_lightning_trainer(self.container, num_nodes=1)
            # When training models that are not built-in InnerEye models, we have no guarantee that they write
            # files to the right folder. Best guess is to change the current working directory to where files should go.
            with change_working_directory(self.container.outputs_folder):
                trainer.test(self.container.model,
                             test_dataloaders=self.container.get_data_module().test_dataloader(),
                             ckpt_path=str(checkpoint_paths[0]))
        else:
            logging.warning("None of the suitable test methods is overridden. Skipping inference completely.")

    def run_inference(self, checkpoint_handler: CheckpointHandler,
                      model_proc: ModelProcessing) -> None:
        """
        Run inference on InnerEyeContainer models
        :param checkpoint_handler: Checkpoint handler object to find checkpoint paths for model initialization
        :param model_proc: whether we are running an ensemble model from within a child run with index 0. If we are,
        then outputs will be written to OTHER_RUNS/ENSEMBLE under the main outputs directory.
        """

        # run full image inference on existing or newly trained model on the training, and testing set
        self.model_inference_train_and_test(checkpoint_handler=checkpoint_handler,
                                            model_proc=model_proc)

        self.try_compare_scores_against_baselines(model_proc)

    def should_register_model(self) -> bool:
        """
        Returns True if we should register a model at all. If no training has taken place, an equivalent
        model (from the run we recovered) should already have been registered, so we should only
        do so if this run is specifically for that purpose.
        """
        return self.azure_config.train or self.azure_config.only_register_model

    def create_activation_maps(self) -> None:
        if self.innereye_config.is_segmentation_model and self.innereye_config.activation_map_layers is not None:
            logging.info("Extracting activation maps for layer")
            activation_maps.extract_activation_maps(self.innereye_config)  # type: ignore
            logging.info("Successfully extracted and saved activation maps")

    def mount_or_download_dataset(self,
                                  azure_dataset_id: Optional[str],
                                  local_dataset: Optional[Path],
                                  dataset_index: int = 0) -> Optional[Path]:
        """
        Makes the dataset that the model uses available on the executing machine. If the present training run is outside
        of AzureML, it expects that either the model has a `local_dataset` field set, in which case no action will be
        taken. If a dataset is specified in `azure_dataset_id`, it will attempt to download the dataset from Azure
        into the local repository, in the "datasets" folder.
        If the training run is inside of AzureML, the dataset that was specified at job submission time will be
        mounted or downloaded.
        :param azure_dataset_id: id of the dataset in AML workspace
        :param local_dataset: alternatively local path for this dataset
        :param index of the dataset processed
        :returns: the path of the dataset on the executing machine.
        """
        if self.is_offline_run:
            # A dataset, either local or in Azure, is required for the built-in InnerEye models. When models are
            # specified via a LightningContainer, these dataset fields are optional, because the container datasets
            # could be downloaded even from the web.
            is_dataset_required = isinstance(self.container, InnerEyeContainer)
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
        if azure_dataset_id:
            mounted = try_to_mount_input_dataset(dataset_index)
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

    def try_compare_scores_against_baselines(self, model_proc: ModelProcessing) -> None:
        """
        Attempt comparison of scores against baseline scores and scatterplot creation if possible.
        """
        if not isinstance(self.model_config, SegmentationModelBase):  # keep type checker happy
            return
        from InnerEye.ML.baselines_util import compare_scores_against_baselines
        with logging_section("Comparing scores against baselines"):
            compare_scores_against_baselines(self.model_config, self.azure_config, model_proc)

    def register_model(self,
                       checkpoint_handler: CheckpointHandler,
                       model_proc: ModelProcessing) -> Tuple[model.Model, Any]:
        """
        Registers a new model in the workspace's model registry on AzureML to be deployed further.
        The AzureML run's tags are updated to describe with information about ensemble creation and the parent run ID.
        :param checkpoint_handler: Checkpoint handler object to find checkpoint paths for model registration.
        :param model_proc: whether it's a single or ensemble model.
        :returns Tuple element 1: AML model object, or None if no model could be registered.
        Tuple element 2: The result of running the model_deployment_hook, or None if no hook was supplied.
        """
        if self.is_offline_run:
            raise ValueError("Cannot register models when InnerEye is running outside of AzureML.")

        checkpoint_paths = checkpoint_handler.get_checkpoints_to_test()
        if not checkpoint_paths:
            raise ValueError("Model registration failed: No checkpoints found")

        split_index = RUN_CONTEXT.get_tags().get(CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, None)
        if split_index == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX:
            RUN_CONTEXT.tag(IS_ENSEMBLE_KEY_NAME, str(model_proc == ModelProcessing.ENSEMBLE_CREATION))
        elif PARENT_RUN_CONTEXT is not None:
            RUN_CONTEXT.tag(PARENT_RUN_ID_KEY_NAME, str(PARENT_RUN_CONTEXT.id))

        with logging_section(f"Registering {model_proc.value} model"):
            # The files for the final model can't live in the outputs folder. If they do: when registering the model,
            # the files may not yet uploaded by hosttools, and that may (or not) cause errors. Hence, place the folder
            # for the final models outside of "outputs", and upload manually.
            model_subfolder = FINAL_MODEL_FOLDER if model_proc == ModelProcessing.DEFAULT \
                else FINAL_ENSEMBLE_MODEL_FOLDER
            # This is the path under which AzureML will know the files: Either "final_model" or "final_ensemble_model"
            artifacts_path = model_subfolder
            final_model_folder = self.container.file_system_config.run_folder / model_subfolder
            # Copy all code from project and InnerEye into the model folder, and copy over checkpoints.
            # This increases the size of the data stored for the run. The other option would be to store all checkpoints
            # right in the final model folder - however, then that would also contain any other checkpoints that the
            # model produced or downloaded for recovery, bloating the final model file.
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
                model_name=self.container.model_name,
                model_path=artifacts_path,
                tags=RUN_CONTEXT.get_tags()
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
                deployment_result = self.model_deployment_hook(
                    self.container, self.azure_config, model, model_proc)
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
                    checkpoint_relative = checkpoint.relative_to(self.container.checkpoint_folder)
                except ValueError:
                    raise ValueError(f"Checkpoint file {checkpoint} was expected to be in a subfolder of "
                                     f"{self.container.checkpoint_folder}")
                # Checkpoints go into a newly created folder "checkpoints" inside of the model folder
                relative_checkpoint_paths.append(str(Path(CHECKPOINT_FOLDER) / checkpoint_relative))
            else:
                raise ValueError(f"Expected an absolute path to a checkpoint file, but got: {checkpoint}")
        model_folder.mkdir(parents=True, exist_ok=True)
        # For reproducibility of the files used in regression tests, checkpoint paths need to be sorted.
        checkpoints_sorted = sorted(relative_checkpoint_paths)
        model_inference_config = ModelInferenceConfig(model_name=self.container.model_name,
                                                      model_configs_namespace=self.config_namespace,
                                                      checkpoint_paths=checkpoints_sorted)
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
            Dict[ModelExecutionMode, InferenceMetrics]:
        metrics: Dict[ModelExecutionMode, InferenceMetrics] = {}

        config = self.innereye_config

        for data_split in ModelExecutionMode:
            if self.container.run_perform_inference(model_proc, data_split):
                opt_metrics = model_test(config, data_split=data_split, checkpoint_handler=checkpoint_handler,
                                         model_proc=model_proc)
                if opt_metrics is not None:
                    metrics[data_split] = opt_metrics

        # log the metrics to AzureML experiment if possible. When doing ensemble runs, log to the Hyperdrive parent run,
        # so that we get the metrics of child run 0 and the ensemble separated.
        if config.is_segmentation_model and not self.is_offline_run:
            run_for_logging = PARENT_RUN_CONTEXT if model_proc.ENSEMBLE_CREATION else RUN_CONTEXT
            log_metrics(metrics=metrics, run_context=run_for_logging)  # type: ignore

        return metrics

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
            n_splits = self.innereye_config.number_of_cross_validation_splits
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

        # Register the model, and then run inference as required. No models should be registered when running outside
        # AzureML.
        if not self.is_offline_run:
            if self.should_register_model():
                self.register_model(checkpoint_handler, ModelProcessing.ENSEMBLE_CREATION)

        if not self.azure_config.only_register_model:
            self.run_inference(checkpoint_handler=checkpoint_handler,
                               model_proc=ModelProcessing.ENSEMBLE_CREATION)

        crossval_dir = self.plot_cross_validation_and_upload_results()
        if self.innereye_config.generate_report:
            self.generate_report(ModelProcessing.ENSEMBLE_CREATION)
        # CrossValResults should have been uploaded to the parent run, so we don't need it here.
        remove_file_or_directory(crossval_dir)
        # We can also remove OTHER_RUNS under the root, as it is no longer useful and only contains copies of files
        # available elsewhere. However, first we need to upload relevant parts of OTHER_RUNS/ENSEMBLE.
        other_runs_dir = self.innereye_config.outputs_folder / OTHER_RUNS_SUBDIR_NAME
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
        plot_crossval_config = crossval_config_from_model_config(self.innereye_config)
        plot_crossval_config.run_recovery_id = PARENT_RUN_CONTEXT.tags[RUN_RECOVERY_ID_KEY_NAME]
        plot_crossval_config.outputs_directory = self.innereye_config.outputs_folder
        plot_crossval_config.azure_config = self.azure_config
        cross_val_results_root = plot_cross_validation(plot_crossval_config)
        if isinstance(self.model_config, ScalarModelBase) and not isinstance(self.model_config, SequenceModelBase):
            crossval_report_name = f"{ModelCategory.Classification.value}_crossval"
            notebook_path = cross_val_results_root / get_ipynb_report_name(crossval_report_name)
            full_metrics_csv = cross_val_results_root / FULL_METRICS_DATAFRAME_FILE
            generate_classification_crossval_notebook(notebook_path, self.model_config, full_metrics_csv)
        if self.post_cross_validation_hook:
            self.post_cross_validation_hook(self.innereye_config, cross_val_results_root)
        # upload results to the parent run's outputs so that the files are visible inside the AzureML UI.
        PARENT_RUN_CONTEXT.upload_folder(name=CROSSVAL_RESULTS_FOLDER, path=str(cross_val_results_root))
        if self.innereye_config.is_scalar_model:
            try:
                aggregates = pd.read_csv(cross_val_results_root / METRICS_AGGREGATES_FILE)
                unrolled_aggregate_metrics = unroll_aggregate_metrics(aggregates)
                for m in unrolled_aggregate_metrics:
                    PARENT_RUN_CONTEXT.log(m.metric_name, m.metric_value)
            except Exception as ex:
                print_exception(ex, "Unable to log metrics to Hyperdrive parent run.", logger_fn=logging.warning)
        return cross_val_results_root

    def generate_report(self, model_proc: ModelProcessing) -> None:
        config = self.innereye_config
        if config.model_category not in [ModelCategory.Segmentation, ModelCategory.Classification]:
            logging.info(f"No reporting available for a model with category {config.model_category}")
            return
        logging.info("Saving report in HTML")
        try:
            def get_epoch_path(mode: ModelExecutionMode) -> Path:
                p = get_best_epoch_results_path(mode=mode, model_proc=model_proc)
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

                    if config.should_generate_multilabel_report():
                        generate_classification_multilabel_notebook(
                            result_notebook=reports_dir / get_ipynb_report_name(
                                f"{config.model_category.value}_multilabel"),
                            config=config,
                            train_metrics=path_to_best_epoch_train,
                            val_metrics=path_to_best_epoch_val,
                            test_metrics=path_to_best_epoch_test)
                else:
                    logging.info(f"Cannot create report for config of type {type(config)}.")

            config.generate_custom_report(report_dir=reports_dir, model_proc=model_proc)
        except Exception as ex:
            print_exception(ex, "Failed to generate reporting notebook.")
            raise
