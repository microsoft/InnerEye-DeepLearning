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
from typing import Any, Dict, List, Optional

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
    recovery_checkpoint = find_latest_recovery_checkpoint(path)
    if recovery_checkpoint.is_file():
        return recovery_checkpoint
    files = list(path.glob("*"))
    raise FileNotFoundError(f"No checkpoint files found in {path}. Existing files: {' '.join(p.name for p in files)}")


def get_best_checkpoint_path(path: Path) -> Path:
    """
    Given a path and checkpoint, formats a path based on the checkpoint file name format.
    :param path to checkpoint folder
    """
    return path / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX


def find_latest_recovery_checkpoint(path: Path) -> Optional[Path]:
    """
    Looks at all the recovery files, extracts the epoch number for all of them. Returns the most recent (latest epoch)
    checkpoint path. If no recovery checkpoint are found, return None.
    :param path: The folder to start searching in.
    :return: None if there is no file matching the search pattern, or a Path object that has the latest file matching
    the pattern.
    """
    filenames = [f for f in path.glob(RECOVERY_CHECKPOINT_FILE_NAME + "*")]
    if filenames == 0:
        return None
    # Checkpoints are saved as recovery_epoch={epoch}.ckpt, find the latest ckpt.
    recovery_epochs = [int(re.findall(r"[\d]+", f.stem)[0]) for f in filenames]
    idx_max_epoch = int(np.argmax(recovery_epochs))
    return filenames[idx_max_epoch]


def create_best_checkpoint(path: Path) -> Path:
    """
    Creates the best checkpoint file. "Best" is at the moment defined as being the last checkpoint, but could be
    based on some defined policy.
    The best checkpoint will be renamed to `best_checkpoint.ckpt`.
    :param path: The folder that contains all checkpoint files.
    """
    logging.info(f"Files in checkpoint folder: {' '.join(p.name for p in path.glob('*'))}")
    last_ckpt = path / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    all_files = f"Existing files: {' '.join(p.name for p in path.glob('*'))}"
    if not last_ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint file {LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX} not found. {all_files}")
    logging.info(f"Using {LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX} as the best checkpoint: Renaming to "
                 f"{BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX}")
    best = path / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    last_ckpt.rename(best)
    return best

def create_unique_timestamp_id() -> str:
    """
    Creates a unique string using the current time in UTC, up to seconds precision, with characters that
    are suitable for use in filenames. For example, on 31 Dec 2019 at 11:59:59pm UTC, the result would be
    2019-12-31T235959Z.
    """
    unique_id = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    return unique_id
