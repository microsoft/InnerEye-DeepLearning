#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import requests
import os
import uuid
import torch

from pathlib import Path
from typing import List, Optional
from azureml.core import Run
from dataclasses import dataclass
from urllib.parse import urlparse

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.ML.deep_learning_config import DeepLearningConfig, WEIGHTS_FILE
from InnerEye.ML.utils.run_recovery import RunRecovery
from InnerEye.Common import fixed_paths


class CheckpointHandler:

    @dataclass
    class CheckPointPathsAndEpoch:
        epoch: int
        checkpoint_paths: Optional[List[Path]]

    def __init__(self, model_config: DeepLearningConfig, azure_config: AzureConfig,
                 project_root: Path, run_context: Optional[Run] = None):
        self.azure_config = azure_config
        self.model_config = model_config
        self.run_recovery: Optional[RunRecovery] = None
        self.project_root = project_root
        self.run_context = run_context

        self.local_weights_path: Optional[Path] = None

        self.continued_training = False

    def discover_and_download_checkpoints_from_previous_runs(self):
        if self.azure_config.run_recovery_id:
            self.run_recovery = RunRecovery.download_checkpoints_from_recovery_run(
                self.azure_config, self.model_config, self.run_context)
        else:
            self.run_recovery = None

        if self.model_config.weights_url or self.model_config.local_weights_path:
            self.local_weights_path = self.get_and_modify_local_weights()

    def additional_training_done(self) -> None:
        self.continued_training = True

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
            # run_recovery takes first precedence over local_weights_path.
            # This is to allow easy recovery of runs which have either of these parameters set in the config:
            checkpoints = self.run_recovery.get_checkpoint_paths(self.model_config.start_epoch)
            if len(checkpoints) > 1:
                raise ValueError(f"Recovering training of ensemble runs is not supported. Found more than one "
                                 f"checkpoint for epoch {self.model_config.start_epoch}")
            return checkpoints[0]
        elif self.local_weights_path:
            return self.local_weights_path
        else:
            return None

    def get_checkpoint_from_epoch(self, epoch: int) -> Optional[CheckPointPathsAndEpoch]:
        """
        Decides the checkpoint path to use for inference/registration. If a run recovery object is used, use the
        checkpoint from there. If this checkpoint does not exist, or a run recovery object is not supplied,
        use the checkpoints from the current run.
        :param config: configuration file
        :param run_recovery: Optional run recovery object
        :param epoch: Epoch to recover
        :return: Constructed checkpoint path to recover from.
        """
        if not self.run_recovery and not self.continued_training:
            raise ValueError(f"Cannot recover checkpoint for epoch {epoch}, no run recovery object provided and"
                             f"no training has been done in this run.")

        if self.run_recovery and (not self.continued_training or epoch <= self.model_config.start_epoch):
            checkpoint_paths = self.run_recovery.get_checkpoint_paths(epoch)
            logging.info(f"Using checkpoints from run recovery for epoch {epoch}")
        else:
            # Checkpoint is from the current run, whether a new run or a run recovery which has been doing more
            # training, so we look for it there.
            checkpoint_paths = [self.model_config.get_path_to_checkpoint(epoch)]
            logging.info(f"Using checkpoints from current run for epoch {epoch}.")

        checkpoint_exists = []
        # Discard any checkpoint paths that do not exist - they will make inference/registration fail.
        # This can happen when some child runs in a hyperdrive run fail; it may still be worth running inference
        # or registering the model.
        for path in checkpoint_paths:
            if path.is_file():
                checkpoint_exists.append(path)
            else:
                logging.warning(f"Could not recover checkpoint path {path}")

        if len(checkpoint_exists) > 0:
            return CheckpointHandler.CheckPointPathsAndEpoch(epoch=epoch, checkpoint_paths=checkpoint_exists)
        else:
            logging.warning(f"Could not find any checkpoints in run recovery/training checkpoints for epoch {epoch}.")
            return None

    def get_checkpoints_to_test(self) -> Optional[List[CheckPointPathsAndEpoch]]:

        test_epochs = self.model_config.epochs_to_test
        # Model was not trained, so look for checkpoints in run recovery or local weights path
        if self.run_recovery:
            checkpoints = []
            for epoch in test_epochs:
                epoch_checkpoints = self.get_checkpoint_from_epoch(epoch)
                if epoch_checkpoints:
                    checkpoints.append(epoch_checkpoints)
            return checkpoints if checkpoints else None
        elif self.local_weights_path and not self.continued_training:
            if self.local_weights_path.exists():
                logging.info(f"Using model weights at {self.local_weights_path} to initialize model")
                return [CheckpointHandler.CheckPointPathsAndEpoch(epoch=0,
                                                                  checkpoint_paths=[self.local_weights_path])]
            else:
                logging.warning(f"Local weights_path does not exist, "
                                f"cannot recover from {self.local_weights_path}")
                return None
        else:
            logging.warning(f"Could not find any run recovery object or local_weight_path to get checkpoints from")
            return None

    def download_weights(self) -> Path:
        target_folder = self.project_root / fixed_paths.MODEL_WEIGHTS_DIR_NAME
        target_folder.mkdir(exist_ok=True)

        url = self.model_config.weights_url

        # assign the same filename as in the download url if possible, so that we can check for duplicates
        # If that fails, map to a random uuid
        file_name = os.path.basename(urlparse(url).path) or str(uuid.uuid4().hex)
        result_file = target_folder / file_name

        # only download if hasn't already been downloaded
        if result_file.exists():
            logging.info(f"File already exists, skipping download: {result_file}")
            return result_file

        logging.info(f"Downloading weights from URL {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(result_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)

        return result_file

    def get_local_weights_path_or_download(self) -> Optional[Path]:
        if self.model_config.local_weights_path:
            weights_path = self.model_config.local_weights_path
        elif self.model_config.weights_url:
            weights_path = self.download_weights()
        else:
            raise ValueError("Cannot download/modify weights - neither local_weights_path nor weights_url is set in"
                             "the model config.")

        return weights_path

    def get_and_modify_local_weights(self) -> Path:
        weights_path = self.get_local_weights_path_or_download()

        if not weights_path or not weights_path.is_file():
            raise FileNotFoundError(f"Could not find the weights file at {weights_path}")

        modified_weights = self.model_config.modify_checkpoint(weights_path)
        target_file = self.model_config.outputs_folder / WEIGHTS_FILE
        torch.save(modified_weights, target_file)
        return target_file

    def should_load_optimizer_checkpoint(self):
        return self.model_config.start_epoch > 0
