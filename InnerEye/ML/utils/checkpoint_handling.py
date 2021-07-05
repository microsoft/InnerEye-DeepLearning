#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import uuid
from builtins import property
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests
from azureml.core import Run, Workspace, Model

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common.fixed_paths import MODEL_INFERENCE_JSON_FILE_NAME
from InnerEye.ML.common import find_recovery_checkpoint_and_epoch
from InnerEye.ML.deep_learning_config import OutputParams
from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.ML.utils.run_recovery import RunRecovery
from InnerEye.ML.model_inference_config import read_model_inference_config


MODEL_WEIGHTS_DIR_NAME = "trained_models"


class CheckpointHandler:
    """
    This class handles which checkpoints are used to initialize the model during train or test time based on the
    azure config and model config.
    """

    def __init__(self, container: LightningContainer, azure_config: AzureConfig,
                 project_root: Path, run_context: Optional[Run] = None):
        self.azure_config = azure_config
        self.container = container
        self.run_recovery: Optional[RunRecovery] = None
        self.project_root = project_root
        self.run_context = run_context
        self.trained_weights_paths: List[Path] = []
        self.has_continued_training = False

    @property
    def output_params(self) -> OutputParams:
        """
        Gets the part of the configuration that is responsible for output paths.
        """
        return self.container

    def download_checkpoints_from_hyperdrive_child_runs(self, hyperdrive_parent_run: Run) -> None:
        """
        Downloads the best checkpoints from all child runs of a Hyperdrive parent run. This is used to gather results
        for ensemble creation.
        """
        self.run_recovery = RunRecovery.download_best_checkpoints_from_child_runs(self.output_params,
                                                                                  hyperdrive_parent_run)
        # Check paths are good, just in case
        for path in self.run_recovery.checkpoints_roots:
            if not path.is_dir():
                raise NotADirectoryError(f"Does not exist or is not a directory: {path}")

    def download_recovery_checkpoints_or_weights(self, only_return_path: bool = False) -> None:
        """
        Download checkpoints from a run recovery object or from a weights url. Set the checkpoints path based on the
        run_recovery_object, weights_url or local_weights_path.
        This is called at the start of training.
        :param: only_return_path: if True, return a RunRecovery object with the path to the checkpoint without actually
        downloading the checkpoints. This is useful to avoid duplicating checkpoint download when running on multiple
        nodes. If False, return the RunRecovery object and download the checkpoint to disk.
        """
        if self.azure_config.run_recovery_id:
            run_to_recover = self.azure_config.fetch_run(self.azure_config.run_recovery_id.strip())
            self.run_recovery = RunRecovery.download_all_checkpoints_from_run(self.output_params, run_to_recover,
                                                                              only_return_path=only_return_path)

        if self.container.checkpoint_urls or self.container.local_checkpoint_paths or self.azure_config.model_id:
            if self.azure_config.model_id and (self.container.local_checkpoint_paths or self.container.checkpoint_urls):
                logging.warning("model_id will take precedence over local_checkpoint_paths or checkpoint_urls.")

            self.trained_weights_paths = self.get_local_checkpoints_path_or_download()

    def additional_training_done(self) -> None:
        """
        Lets the handler know that training was done in this run.
        """
        self.has_continued_training = True

    def get_recovery_or_checkpoint_path_train(self) -> Optional[Path]:
        """
        Decides the checkpoint path to use for the current training run. Looks for the latest checkpoint in the
        checkpoint folder. If run_recovery is provided, the checkpoints will have been downloaded to this folder
        prior to calling this function. Else, if the run gets pre-empted and automatically restarted in AML,
        the latest checkpoint will be present in this folder too.
        :return: Constructed checkpoint path to recover from.
        """
        recovery = find_recovery_checkpoint_and_epoch(self.container.checkpoint_folder)
        if recovery is not None:
            local_recovery_path, recovery_epoch = recovery
            self.container._start_epoch = recovery_epoch
            return local_recovery_path
        else:
            return None

    def get_best_checkpoints(self) -> List[Path]:
        """
        Get a list of checkpoints per epoch for testing/registration from the current training run.
        This function also checks that the checkpoint at the returned checkpoint path exists.
        """
        if not self.run_recovery and not self.has_continued_training:
            raise ValueError("Cannot recover checkpoint, no run recovery object provided and "
                             "no training has been done in this run.")

        checkpoint_paths = []
        if self.run_recovery:
            checkpoint_paths = self.run_recovery.get_best_checkpoint_paths()

            checkpoint_exists = []
            # Discard any checkpoint paths that do not exist - they will make inference/registration fail.
            # This can happen when some child runs in a hyperdrive run fail; it may still be worth running inference
            # or registering the model.
            for path in checkpoint_paths:
                if path.is_file():
                    checkpoint_exists.append(path)
                else:
                    logging.warning(f"Could not recover checkpoint path {path}")
            checkpoint_paths = checkpoint_exists

        if self.has_continued_training:
            # Checkpoint is from the current run, whether a new run or a run recovery which has been doing more
            # training, so we look for it there.
            checkpoint_from_current_run = self.output_params.get_path_to_best_checkpoint()
            if checkpoint_from_current_run.is_file():
                logging.info("Using checkpoints from current run.")
                checkpoint_paths = [checkpoint_from_current_run]
            else:
                logging.info("Training has continued, but not yet written a checkpoint. Using recovery checkpoints.")
        else:
            logging.info("Using checkpoints from run recovery")

        return checkpoint_paths

    def get_checkpoints_to_test(self) -> List[Path]:
        """
        Find the checkpoints to test. If a run recovery is provided, or if the model has been training, look for
        checkpoints corresponding to the epochs in get_test_epochs(). If there is no run recovery and the model was
        not trained in this run, then return the checkpoint from the local_weights_path.
        """

        checkpoints = []

        # If model was trained, look for the best checkpoint
        if self.run_recovery or self.has_continued_training:
            checkpoints = self.get_best_checkpoints()
        elif self.trained_weights_paths:
            # Model was not trained, check if there is a local weight path.
            logging.info(f"Using model weights from {self.trained_weights_paths} to initialize model")
            checkpoints = self.trained_weights_paths
        else:
            logging.warning("Could not find any local_weights_path, model_weights or model_id to get checkpoints from")

        return checkpoints

    @staticmethod
    def download_weights(urls: List[str], download_folder: Path) -> List[Path]:
        """
        Download a checkpoint from weights_url to the modelweights directory.
        """
        checkpoint_paths = []
        for url in urls:
            # assign the same filename as in the download url if possible, so that we can check for duplicates
            # If that fails, map to a random uuid
            file_name = os.path.basename(urlparse(url).path) or str(uuid.uuid4().hex)
            result_file = download_folder / file_name
            checkpoint_paths.append(result_file)
            # only download if hasn't already been downloaded
            if result_file.exists():
                logging.info(f"File already exists, skipping download: {result_file}")
            else:
                logging.info(f"Downloading weights from URL {url}")

                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(result_file, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)

        return checkpoint_paths

    @staticmethod
    def get_checkpoints_from_model(model_id: str, workspace: Workspace, download_path: Path) -> List[Path]:

        model_name, model_version = model_id.split(":")
        model = Model(workspace=workspace, name=model_name, version=int(model_version))
        model_path = Path(model.download(str(download_path), exist_ok=True))
        model_inference_config = read_model_inference_config(model_path / MODEL_INFERENCE_JSON_FILE_NAME)
        checkpoint_paths = [model_path / x for x in model_inference_config.checkpoint_paths]
        return checkpoint_paths

    def get_local_checkpoints_path_or_download(self) -> List[Path]:
        """
        Get the path to the local weights to use or download them and set local_weights_path
        """
        if not self.azure_config.model_id and not self.container.local_checkpoint_paths and not self.container.checkpoint_urls:
            raise ValueError("Cannot download weights - none of model_id, local_weights_path or weights_url is set in "
                             "the model config.")

        if self.container.local_checkpoint_paths:
            checkpoint_paths = self.container.local_checkpoint_paths
        else:
            download_folder = self.output_params.checkpoint_folder / MODEL_WEIGHTS_DIR_NAME
            download_folder.mkdir(exist_ok=True, parents=True)

            if self.azure_config.model_id:
                if len(self.azure_config.model_id.split(":")) != 2:
                    raise ValueError(
                        f"model_id should be in the form 'model_name:version', got {self.azure_config.model_id}")

                checkpoint_paths = CheckpointHandler.get_checkpoints_from_model(model_id=self.azure_config.model_id,
                                                                                workspace=self.azure_config.get_workspace(),
                                                                                download_path=download_folder)
            elif self.container.checkpoint_urls:
                urls = self.container.checkpoint_urls
                checkpoint_paths = CheckpointHandler.download_weights(urls=urls,
                                                                      download_folder=download_folder)

        for checkpoint_path in checkpoint_paths:
            if not checkpoint_path or not checkpoint_path.is_file():
                raise FileNotFoundError(f"Could not find the weights file at {checkpoint_path}")
        return checkpoint_paths
