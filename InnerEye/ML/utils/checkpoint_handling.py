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
import torch
from azureml.core import Run

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import fixed_paths
from InnerEye.ML.deep_learning_config import OutputParams, WEIGHTS_FILE, \
    load_checkpoint_and_modify
from InnerEye.ML.lightning_base import InnerEyeContainer
from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.ML.utils.run_recovery import RunRecovery


class CheckpointHandler:
    """
    This class handles which checkpoints are used to initialize the model during train or test time based on the
    azure config and model config.
    """

    def __init__(self, lightning_container: LightningContainer, azure_config: AzureConfig,
                 project_root: Path, run_context: Optional[Run] = None):
        self.azure_config = azure_config
        self.container = lightning_container
        self.run_recovery: Optional[RunRecovery] = None
        self.project_root = project_root
        self.run_context = run_context
        self.local_weights_path: Optional[Path] = None
        self.has_continued_training = False

    @property
    def output_params(self) -> OutputParams:
        """
        Gets the part of the configuration that is responsible for output paths.
        """
        if isinstance(self.container, InnerEyeContainer):
            return self.container.config
        else:
            return self.container.lightning_module

    @property
    def start_epoch(self) -> int:
        """
        Gets the value of the start epoch for training.
        """
        if isinstance(self.container, InnerEyeContainer):
            return self.container.config.start_epoch
        else:
            return self.container.lightning_module.start_epoch

    @property
    def use_gpu(self) -> bool:
        if isinstance(self.container, InnerEyeContainer):
            return self.container.config.use_gpu
        else:
            return self.container.lightning_module.use_gpu

    def download_checkpoints_from_hyperdrive_child_runs(self, hyperdrive_parent_run: Run) -> None:
        """
        Downloads the best checkpoints from all child runs of a Hyperdrive parent runs. This is used to gather results
        for ensemble creation.
        """
        self.run_recovery = RunRecovery.download_best_checkpoints_from_child_runs(self.output_params,
                                                                                  hyperdrive_parent_run)
        # Check paths are good, just in case
        for path in self.run_recovery.checkpoints_roots:
            if not path.is_dir():
                raise NotADirectoryError(f"Does not exist or is not a directory: {path}")

    def download_recovery_checkpoints_or_weights(self) -> None:
        """
        Download checkpoints from a run recovery object or from a weights url. Set the checkpoints path based on the
        run_recovery_object, weights_url or local_weights_path.
        This is called at the start of training.
        """
        if self.azure_config.run_recovery_id:
            run_to_recover = self.azure_config.fetch_run(self.azure_config.run_recovery_id.strip())
            self.run_recovery = RunRecovery.download_all_checkpoints_from_run(self.output_params, run_to_recover)
        else:
            self.run_recovery = None

        if self.container.weights_url or self.container.local_weights_path:
            self.local_weights_path = self.get_and_save_modified_weights()

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
        start_epoch = self.start_epoch
        if start_epoch > 0 and not self.run_recovery:
            raise ValueError("Start epoch is > 0, but no run recovery object has been provided to resume training.")

        if self.run_recovery and start_epoch == 0:
            raise ValueError("Run recovery set, but start epoch is 0. Please provide start epoch > 0 (for which a "
                             "checkpoint was saved in the previous run) to resume training from that run.")

        if self.run_recovery:
            # run_recovery takes first precedence over local_weights_path.
            # This is to allow easy recovery of runs which have either of these parameters set in the config:
            checkpoints = self.run_recovery.get_recovery_checkpoint_paths()
            if len(checkpoints) > 1:
                raise ValueError(f"Recovering training of ensemble runs is not supported. Found more than one "
                                 f"checkpoint for epoch {start_epoch}")
            return checkpoints[0]
        elif self.local_weights_path:
            return self.local_weights_path
        else:
            return None

    def get_best_checkpoint(self) -> List[Path]:
        """
        Get a list of checkpoints per epoch for testing/registration.
        1. If a run recovery object is used and no training was done in this run, use checkpoints from run recovery.
        2. If a run recovery object is used, and training was done in this run, but the start epoch is larger than
        the epoch parameter provided, use checkpoints from run recovery.
        3. If a run recovery object is used, and training was done in this run, but the start epoch is smaller than
        the epoch parameter provided, use checkpoints from the current training run.
        This function also checks that all the checkpoints at the returned checkpoint paths exist,
        and drops any that do not.
        """
        if not self.run_recovery and not self.has_continued_training:
            raise ValueError("Cannot recover checkpoint, no run recovery object provided and"
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

        # If recovery object exists, or model was trained, look for checkpoints by epoch
        if self.run_recovery or self.has_continued_training:
            checkpoints = self.get_best_checkpoint()
        elif self.local_weights_path and not self.has_continued_training:
            # No recovery object and model was not trained, check if there is a local weight path.
            if self.local_weights_path.exists():
                logging.info(f"Using model weights at {self.local_weights_path} to initialize model")
                checkpoints = [self.local_weights_path]
            else:
                logging.warning(f"local_weights_path does not exist, "
                                f"cannot recover from {self.local_weights_path}")
        else:
            logging.warning("Could not find any run recovery object or local_weights_path to get checkpoints from")

        return checkpoints

    def download_weights(self) -> Path:
        """
        Download a checkpoint from weights_url to the modelweights directory.
        """
        target_folder = self.project_root / fixed_paths.MODEL_WEIGHTS_DIR_NAME
        target_folder.mkdir(exist_ok=True)

        url = self.container.weights_url

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
        """
        Get the path to the local weights to use or download them and set local_weights_path
        """
        if self.container.local_weights_path:
            weights_path = self.container.local_weights_path
        elif self.container.weights_url:
            weights_path = self.download_weights()
        else:
            raise ValueError("Cannot download/modify weights - neither local_weights_path nor weights_url is set in"
                             "the model config.")

        return weights_path

    def get_and_save_modified_weights(self) -> Path:
        """
        Downloads the checkpoint weights if needed.
        Then passes the downloaded or local checkpoint to the modify_checkpoint function from the model_config and saves
        the modified state dict from the function in the outputs folder with the name weights.pth.
        """
        weights_path = self.get_local_weights_path_or_download()

        if not weights_path or not weights_path.is_file():
            raise FileNotFoundError(f"Could not find the weights file at {weights_path}")

        modified_weights = load_checkpoint_and_modify(weights_path, use_gpu=self.use_gpu)
        target_file = self.output_params.outputs_folder / WEIGHTS_FILE
        torch.save(modified_weights, target_file)
        return target_file

    def should_load_optimizer_checkpoint(self) -> bool:
        """
        Returns true if the optimizer should be loaded from checkpoint. Looks at the model config to determine this.
        """
        return self.start_epoch > 0
