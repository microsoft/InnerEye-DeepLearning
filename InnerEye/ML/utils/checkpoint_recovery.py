#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging

from pathlib import Path
from typing import List, Optional
from azureml.core import Run

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.utils.run_recovery import RunRecovery


class ManageRecovery:
    def __init__(self, model_config: DeepLearningConfig, azure_config: AzureConfig,
                 run_context: Optional[Run] = None):
        self.model_config = model_config
        self.run_recovery: Optional[RunRecovery]
        if azure_config.run_recovery_id:
            self.run_recovery = RunRecovery.download_checkpoints_from_recovery_run(
                azure_config, self.model_config, run_context)
        else:
            self.run_recovery = None

        self.continued_training = False

    def additional_training_done(self) -> None:
        self.continued_training = True

    def get_recovery_path_train(self) -> Optional[Path]:
        """
        Decides the checkpoint path to use for the current training run. If a run recovery object is used, use the
        checkpoint from there, otherwise use the checkpoints from the current run.
        :return: Constructed checkpoint path to recover from.
        """

        if self.run_recovery or self.model_config.local_weights_path:
            if self.model_config.start_epoch > 0 and not self.run_recovery:
                raise ValueError("Start epoch is > 0, but no run recovery object has been provided to resume training.")

            checkpoint_paths: Optional[Path]
            if self.run_recovery:
                # run_recovery takes first precedence over config.weights_url or config.local_weights_path.
                # This is to allow easy recovery of runs which have either of these parameters set in the config:
                checkpoint_paths = self.run_recovery.get_checkpoint_paths(self.model_config.start_epoch)[0]
            elif self.model_config.local_weights_path:
                # By this time, even if config.weights_url was set, model weights have been downloaded to
                # config.local_weights_path
                return self.model_config.local_weights_path
            else:
                logging.warning("No run recovery object provided to recover checkpoint from.")
                checkpoint_paths = None
            return checkpoint_paths
        else:
            return None

    def get_checkpoint_from_epoch(self, epoch: int) -> List[Path]:
        """
        Decides the checkpoint path to use for inference/registration. If a run recovery object is used, use the
        checkpoint from there. If this checkpoint does not exist, or a run recovery object is not supplied,
        use the checkpoints from the current run.
        :param config: configuration file
        :param run_recovery: Optional run recovery object
        :param epoch: Epoch to recover
        :return: Constructed checkpoint path to recover from.
        """
        if self.run_recovery and (not self.continued_training or epoch < self.model_config.start_epoch):
            checkpoint_paths = self.run_recovery.get_checkpoint_paths(epoch)
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
            return checkpoint_exists
        else:
            raise ValueError(f"Could not find any checkpoints in run recovery/training checkpoints for epoch {epoch}.")

    def get_checkpoints_to_test(self) -> List[List[Path]]:

        test_epochs = self.model_config.epochs_to_test
        if not self.continued_training:
            # Model was not trained, so look for checkpoints in run recovery or local weights path
            if self.run_recovery:
                checkpoints = []
                for epoch in test_epochs:
                    epoch_checkpoints = self.get_checkpoint_from_epoch(epoch)
                    checkpoints.append(epoch_checkpoints)
                return checkpoints
            elif self.model_config.local_weights_path:
                logging.info(f"Using model weights at {self.model_config.local_weights_path} to initialize model")
                return [[self.model_config.local_weights_path]]
            else:
                raise ValueError(f"Could not find any run recovery object or local_weight_path to get checkpoints from")

        else:
            # Model was trained, so look for checkpoints in run recovery or local checkpoints
            checkpoints = []
            for epoch in test_epochs:
                epoch_checkpoints = self.get_checkpoint_from_epoch(epoch)
                checkpoints.append(epoch_checkpoints)
            return checkpoints
