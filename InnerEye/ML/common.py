#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import abc
import logging
import re
from datetime import datetime
from enum import Enum, unique
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DATASET_CSV_FILE_NAME = "dataset.csv"
CHECKPOINT_SUFFIX = ".ckpt"

RECOVERY_CHECKPOINT_FILE_NAME = "recovery"
RECOVERY_CHECKPOINT_FILE_NAME_WITH_SUFFIX = RECOVERY_CHECKPOINT_FILE_NAME + CHECKPOINT_SUFFIX

BEST_CHECKPOINT_FILE_NAME = "best_checkpoint"
BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX = BEST_CHECKPOINT_FILE_NAME + CHECKPOINT_SUFFIX

# This is a constant that must match a filename defined in pytorch_lightning.ModelCheckpoint, but we don't want
# to import that here.
LAST_CHECKPOINT_FILE_NAME = "last"
LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX = LAST_CHECKPOINT_FILE_NAME + CHECKPOINT_SUFFIX


@unique
class ModelExecutionMode(Enum):
    """
    Model execution mode
    """
    TRAIN = "Train"
    TEST = "Test"
    VAL = "Val"


STORED_CSV_FILE_NAMES = \
    {
        ModelExecutionMode.TRAIN: "train_dataset.csv",
        ModelExecutionMode.TEST: "test_dataset.csv",
        ModelExecutionMode.VAL: "val_dataset.csv"
    }


class OneHotEncoderBase(abc.ABC):
    """Abstract class for a one hot encoder object"""

    @abc.abstractmethod
    def encode(self, x: Dict[str, List[str]]) -> Any:
        """Encode dict mapping features to values and returns encoded vector."""
        raise NotImplementedError("encode must be implemented by sub classes")

    @abc.abstractmethod
    def get_supported_dataset_column_names(self) -> List[str]:
        """Gets the names of the columns that this encoder supports"""
        raise NotImplementedError("get_columns must be implemented by sub classes")

    @abc.abstractmethod
    def get_feature_length(self, column: str) -> int:
        """Gets the expected feature lengths for one hot encoded features using this encoder"""
        raise NotImplementedError("get_feature_length must be implemented by sub classes")


def get_recovery_checkpoint_path(path: Path) -> Path:
    """
    Returns the path to the last recovery checkpoint in the given folder or the provided filename. Raises a
    FileNotFoundError if no
    recovery checkpoint file is present.
    :param path: Path to checkpoint folder
    """
    recovery_ckpt_and_epoch = find_recovery_checkpoint_and_epoch(path)
    if recovery_ckpt_and_epoch is not None:
        return recovery_ckpt_and_epoch[0]
    files = list(path.glob("*"))
    raise FileNotFoundError(f"No checkpoint files found in {path}. Existing files: {' '.join(p.name for p in files)}")


def get_best_checkpoint_path(path: Path) -> Path:
    """
    Given a path and checkpoint, formats a path based on the checkpoint file name format.
    :param path to checkpoint folder
    """
    return path / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX


def find_all_recovery_checkpoints(path: Path) -> Optional[List[Path]]:
    """
    Extracts all file starting with RECOVERY_CHECKPOINT_FILE_NAME in path
    :param path:
    :return:
    """
    all_recovery_files = [f for f in path.glob(RECOVERY_CHECKPOINT_FILE_NAME + "*")]
    if len(all_recovery_files) == 0:
        return None
    return all_recovery_files


PathAndEpoch = Tuple[Path, int]


def extract_latest_checkpoint_and_epoch(available_files: List[Path]) -> PathAndEpoch:
    """
     Checkpoints are saved as recovery_epoch={epoch}.ckpt, find the latest ckpt and epoch number.
    :param available_files: all available checkpoints
    :return: path the checkpoint from latest epoch and epoch number
    """
    recovery_epochs = [int(re.findall(r"[\d]+", f.stem)[0]) for f in available_files]
    idx_max_epoch = int(np.argmax(recovery_epochs))
    return available_files[idx_max_epoch], recovery_epochs[idx_max_epoch]


def find_recovery_checkpoint_and_epoch(path: Path) -> Optional[PathAndEpoch]:
    """
    Looks at all the recovery files, extracts the epoch number for all of them and returns the most recent (latest
    epoch)
    checkpoint path along with the corresponding epoch number. If no recovery checkpoint are found, return None.
    :param path: The folder to start searching in.
    :return: None if there is no file matching the search pattern, or a Tuple with Path object and integer pointing to
    recovery checkpoint path and recovery epoch.
    """
    available_checkpoints = find_all_recovery_checkpoints(path)
    if available_checkpoints is not None:
        return extract_latest_checkpoint_and_epoch(available_checkpoints)
    return None


def create_best_checkpoint(path: Path) -> Path:
    """
    Creates the best checkpoint file. "Best" is at the moment defined as being the checkpoint whose name matches
    LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX or the only available checkpoint, but it could be based on some defined
    policy.
    The best checkpoint will be renamed to `best_checkpoint.ckpt`.
    :param path: The folder that contains all checkpoint files.
    """
    candidate_checkpoint: Optional[Path] = None
    checkpoint_files = list(path.glob('*.ckpt'))
    if (path / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX) in checkpoint_files:
        candidate_checkpoint = path / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    elif len(checkpoint_files) == 1:
        candidate_checkpoint = checkpoint_files[0]
    else:
        raise FileNotFoundError(
            f"Checkpoint file {LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX} not found in ",
            f"{str(' '.join(p.name for p in checkpoint_files))}, and there were ",
            f"{len(checkpoint_files)} so the policy of falling back to the only checkpoint could not work.")
    assert candidate_checkpoint  # mypy
    logging.info(
        f"Using {candidate_checkpoint.name} as best checkpoint. Renaming it to {BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX}")
    best = path / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    candidate_checkpoint.rename(best)
    return best


def create_unique_timestamp_id() -> str:
    """
    Creates a unique string using the current time in UTC, up to seconds precision, with characters that
    are suitable for use in filenames. For example, on 31 Dec 2019 at 11:59:59pm UTC, the result would be
    2019-12-31T235959Z.
    """
    unique_id = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    return unique_id
