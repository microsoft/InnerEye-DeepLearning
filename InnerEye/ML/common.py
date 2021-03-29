#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import abc
import logging
from datetime import datetime
from enum import Enum, unique
from pathlib import Path
from typing import Any, Dict, List, Optional

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


# TODO antonsc: This should be renamed to DatasetSplit or something alike.
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


def create_recovery_checkpoint_path(path: Path) -> Path:
    """
    Returns the file name of a recovery checkpoint in the given folder. Raises a FileNotFoundError if no
    recovery checkpoint file is present.
    :param path: Path to checkpoint folder
    """
    # Recovery checkpoints are written alternately as recovery.ckpt and recovery-v0.ckpt.
    best_checkpoint1 = path / f"{RECOVERY_CHECKPOINT_FILE_NAME_WITH_SUFFIX}"
    best_checkpoint2 = path / f"{RECOVERY_CHECKPOINT_FILE_NAME}-v0{CHECKPOINT_SUFFIX}"
    for p in [best_checkpoint1, best_checkpoint2]:
        if p.is_file():
            return p
    files = list(path.glob("*"))
    raise FileNotFoundError(f"No checkpoint files found in {path}. Existing files: {' '.join(p.name for p in files)}")


def get_best_checkpoint_path(path: Path) -> Path:
    """
    Given a path and checkpoint, formats a path based on the checkpoint file name format.
    :param path to checkpoint folder
    """
    return path / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX


def keep_latest(path: Path, search_pattern: str) -> Optional[Path]:
    """
    Looks at all files that match the given pattern via "glob", and deletes all of them apart from the most most
    recent file. The surviving file is returned. If there is no single file that matches the search pattern, then
    return None.
    :param path: The folder to start searching in.
    :param search_pattern: The glob pattern that specifies the files that should be searched.
    :return: None if there is no file matching the search pattern, or a Path object that has the latest file matching
    the pattern.
    """
    files_and_mod_time = [(f, f.stat().st_mtime) for f in path.glob(search_pattern)]
    files_and_mod_time.sort(key=lambda f: f[1], reverse=True)
    for (f, _) in files_and_mod_time[1:]:
        logging.info(f"Removing file: {f}")
        f.unlink()
    if files_and_mod_time:
        return files_and_mod_time[0][0]
    return None


def keep_best_checkpoint(path: Path) -> Path:
    """
    Clean up all checkpoints that are found in the given folder, and keep only the "best" one. "Best" is at the moment
    defined as being the last checkpoint, but could be based on some defined policy. The best checkpoint will be
    renamed to `best_checkpoint.ckpt`. All other files checkpoint files
    but the best will be removed (or an existing checkpoint renamed to be the best checkpoint).
    :param path: The folder that contains all checkpoint files.
    """
    last_ckpt = path / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    all_files = f"Existing files: {' '.join(p.name for p in path.glob('*'))}"
    if not last_ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint file {LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX} not found. {all_files}")
    logging.info(f"Using {LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX} as the best checkpoint: Renaming to "
                 f"{BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX}")
    best = path / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    last_ckpt.rename(best)
    return best


def cleanup_checkpoint_folder(path: Path) -> None:
    """
    Removes surplus files from the checkpoint folder, and unifies the names of the files that are kept:
    1) Keep only the most recent recovery checkpoint file
    2) Chooses the best checkpoint file according to keep_best_checkpoint, and rename it to
    BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    :param path: The folder containing all model checkpoints.
    """
    logging.info(f"Files in checkpoint folder: {' '.join(p.name for p in path.glob('*'))}")
    recovery = keep_latest(path, RECOVERY_CHECKPOINT_FILE_NAME + "*")
    if recovery:
        recovery.rename(path / RECOVERY_CHECKPOINT_FILE_NAME_WITH_SUFFIX)
    keep_best_checkpoint(path)


def create_unique_timestamp_id() -> str:
    """
    Creates a unique string using the current time in UTC, up to seconds precision, with characters that
    are suitable for use in filenames. For example, on 31 Dec 2019 at 11:59:59pm UTC, the result would be
    2019-12-31T235959Z.
    """
    unique_id = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    return unique_id
