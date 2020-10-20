#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import List, Optional

from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.utils.run_recovery import RunRecovery


def get_recovery_path_train(config: DeepLearningConfig, run_recovery: Optional[RunRecovery],
                            epoch: int) -> Optional[Path]:
    """
    Decides the checkpoint path to use for the current training run. If a run recovery object is used, use the
    checkpoint from there, otherwise use the checkpoints from the current run.
    :param run_recovery: Optional run recovery object
    :param epoch: Epoch to recover
    :return: Constructed checkpoint path to recover from.
    """
    if config.start_epoch > 0 and not run_recovery:
        raise ValueError("Start epoch is > 0, but no run recovery object has been provided to resume training.")

    checkpoint_paths: Optional[Path]
    if run_recovery:
        # run_recovery takes first precedence over config.weights_url or config.local_weights_path.
        # This is to allow easy recovery of runs which have either of these parameters set in the config:
        checkpoint_paths = run_recovery.get_checkpoint_paths(epoch)[0]
    elif config.local_weights_path:
        # By this time, even if config.weights_url was set, model weights have been downloaded to
        # config.local_weights_path
        return config.local_weights_path
    else:
        logging.warning("No run recovery object provided to recover checkpoint from.")
        checkpoint_paths = None
    return checkpoint_paths


def get_recovery_path_test(config: DeepLearningConfig, run_recovery: Optional[RunRecovery],
                           epoch: int) -> Optional[List[Path]]:
    """
    Decides the checkpoint path to use for inference/registration. If a run recovery object is used, use the
    checkpoint from there. If this checkpoint does not exist, or a run recovery object is not supplied,
    use the checkpoints from the current run.
    :param config: configuration file
    :param run_recovery: Optional run recovery object
    :param epoch: Epoch to recover
    :return: Constructed checkpoint path to recover from.
    """
    if run_recovery:
        checkpoint_paths = run_recovery.get_checkpoint_paths(epoch)
        checkpoint_exists = []
        # Discard any checkpoint paths that do not exist - they will make inference/registration fail.
        # This can happen when some child runs fail; it may still be worth running inference
        # or registering the model.
        for path in checkpoint_paths:
            if path.is_file():
                checkpoint_exists.append(path)
            else:
                logging.warning(f"Could not recover checkpoint path {path}")

        if len(checkpoint_exists) > 0:
            return checkpoint_exists

    logging.warning(f"Using checkpoints from current run, "
                    f"could not find any run recovery checkpoints for epoch {epoch}")
    # We found the checkpoint(s) in the run being recovered. If we didn't, it's probably because the epoch
    # is from the current run, which has been doing more training, so we look for it there.
    checkpoint_path = config.get_path_to_checkpoint(epoch)
    if checkpoint_path.is_file():
        return [checkpoint_path]

    logging.warning(f"Could not find checkpoint at path {checkpoint_path}")

    # last place to check is in config.local_weights_path
    if config.local_weights_path:
        logging.info(f"Using model weights at {config.local_weights_path} to initialize model")
        return [config.local_weights_path]

    return None
