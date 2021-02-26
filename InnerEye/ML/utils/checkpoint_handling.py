#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests
from azureml.core import Run, Workspace, Model

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import fixed_paths
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.utils.run_recovery import RunRecovery
from InnerEye.ML.model_inference_config import read_model_inference_config
from InnerEye.Common.fixed_paths import MODEL_INFERENCE_JSON_FILE_NAME


class CheckpointHandler:
    """
    This class handles which checkpoints are used to initialize the model during train or test time based on the
    azure config and model config.
    """

    def __init__(self, model_config: DeepLearningConfig, azure_config: AzureConfig,
                 project_root: Path, run_context: Optional[Run] = None):
        self.azure_config = azure_config
        self.model_config = model_config
        self.run_recovery: Optional[RunRecovery] = None
        self.project_root = project_root
        self.run_context = run_context
        self.local_weights_path: List[Path] = []
        self.has_continued_training = False

    def download_checkpoints_from_hyperdrive_child_runs(self, hyperdrive_parent_run: Run) -> None:
        """
        Downloads the best checkpoints from all child runs of a Hyperdrive parent runs. This is used to gather results
        for ensemble creation.
        """
        self.run_recovery = RunRecovery.download_best_checkpoints_from_child_runs(self.model_config,
                                                                                  hyperdrive_parent_run)
        # Check paths are good, just in case
        for path in self.run_recovery.checkpoints_roots:
            if not path.is_dir():
                raise NotADirectoryError(f"Does not exist or is not a directory: {path}")

    def download_recovery_checkpoints_or_inference_checkpoints(self) -> None:
        """
        Download checkpoints from a run recovery object or from a weights url. Set the checkpoints path based on the
        run_recovery_object, weights_url or local_weights_path.
        This is called at the start of training.
        """
        if self.azure_config.run_recovery_id:
            run_to_recover = self.azure_config.fetch_run(self.azure_config.run_recovery_id.strip())
            self.run_recovery = RunRecovery.download_all_checkpoints_from_run(self.model_config, run_to_recover)
        elif self.model_config.checkpoint_urls or self.model_config.local_checkpoint_paths or self.azure_config.model_id:
            self.local_weights_path = self.get_local_checkpoint_paths_or_download()

    def additional_training_done(self) -> None:
        """
        Lets the handler know that training was done in this run.
        """
        self.has_continued_training = True

    def get_recovery_path_train(self) -> Optional[Path]:
        """
        Decides the checkpoint path to use for the current training run. If a run recovery object is used, use the
        checkpoint from there, otherwise use the checkpoints from the current run.
        :return: Constructed checkpoint path to recover from.
        """

        if self.model_config.start_epoch > 0 and not self.run_recovery:
            raise ValueError("Start epoch is > 0, but no run recovery object has been provided to resume training.")

        if self.run_recovery and self.model_config.start_epoch == 0:
            raise ValueError("Run recovery set, but start epoch is 0. Please provide start epoch > 0 (for which a "
                             "checkpoint was saved in the previous run) to resume training from that run.")

        if self.run_recovery:
            checkpoints = self.run_recovery.get_recovery_checkpoint_paths()
            if len(checkpoints) > 1:
                raise ValueError(f"Recovering training of ensemble runs is not supported. Found more than one "
                                 f"checkpoint for epoch {self.model_config.start_epoch}")
            return checkpoints[0]
        else:
            return None

    def get_checkpoints_to_register(self) -> List[Path]:
        checkpoints = []
        if self.has_continued_training:
            checkpoints = self.get_best_checkpoints()
        elif self.run_recovery:
            checkpoints = self.run_recovery.get_recovery_checkpoint_paths()

        logging.warning("No checkpoints found for registration")
        return checkpoints

    def get_best_checkpoints(self) -> List[Path]:
        """
        Get a list of checkpoints per epoch for testing/registration from the current training run.
        This function also checks that the checkpoint at the returned checkpoint path exists.
        """
        if not self.has_continued_training:
            raise ValueError("Cannot find a training checkpoint, no training has been done in this run.")

        # Checkpoint is from the current run, whether a new run or a run recovery which has been doing more
        # training, so we look for it there.
        checkpoint_from_current_run = self.model_config.get_path_to_best_checkpoint()
        if checkpoint_from_current_run.is_file():
            logging.info("Using checkpoints from current run.")
            return [checkpoint_from_current_run]
        else:
            logging.warning("Training has happened, but has not written a checkpoint.")
            return []

    def get_checkpoints_to_test(self) -> List[Path]:
        """
        Find the checkpoints to test. If a run recovery is provided, or if the model has been training, look for
        checkpoints corresponding to the epochs in get_test_epochs(). If there is no run recovery and the model was
        not trained in this run, then return the checkpoint from the local_weights_path.
        """
        checkpoints = []
        # If model was trained, look for the best checkpoint
        if self.has_continued_training:
            checkpoints = self.get_best_checkpoints()
        elif self.local_weights_path:
            # Model was not trained, check if there is a local weight path.
            logging.info(f"Using model weights from {self.local_weights_path} to initialize model")
            checkpoints = self.local_weights_path
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

            logging.info(f"Downloading weights from URL {url}")

            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(result_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)

        return checkpoint_paths

    @staticmethod
    def get_checkpoints_from_model(model_id: str, workspace: Workspace, download_folder: Path) -> List[Path]:
        if len(model_id.split(":")) != 2:
            raise ValueError(f"model_id should be in the form 'model_name:verison', got {model_id}")

        model_name, version = model_id.split(":")
        model = Model(workspace=workspace, name=model_name, version=version)
        model_path = Path(model.download(str(download_folder), exist_ok=True))
        model_inference_config = read_model_inference_config(model_path / MODEL_INFERENCE_JSON_FILE_NAME)
        checkpoint_paths = [model_path / x for x in model_inference_config.checkpoint_paths]
        return checkpoint_paths

    def get_local_checkpoint_paths_or_download(self) -> List[Path]:
        """
        Get the path to the local weights to use or download them and set local_weights_path
        """
        download_folder = self.project_root / fixed_paths.MODEL_WEIGHTS_DIR_NAME
        download_folder.mkdir(exist_ok=True)

        if self.azure_config.model_id:
            checkpoint_paths = self.get_checkpoints_from_model(model_id=self.azure_config.model_id,
                                                               workspace=self.azure_config.get_workspace(),
                                                               download_folder=download_folder)
        elif self.model_config.local_checkpoint_paths:
            checkpoint_paths = self.model_config.local_checkpoint_paths
        elif self.model_config.checkpoint_urls:
            urls = self.model_config.checkpoint_urls
            checkpoint_paths = self.download_weights(urls=urls,
                                                     download_folder=download_folder)
        else:
            raise ValueError("Cannot download/modify weights - neither local_weights_path nor weights_url is set in"
                             "the model config.")

        for checkpoint_path in checkpoint_paths:
            if not checkpoint_path or not checkpoint_path.is_file():
                raise FileNotFoundError(f"Could not find the weights file at {checkpoint_path}")

        return checkpoint_paths
