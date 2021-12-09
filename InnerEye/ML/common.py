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
import tempfile

from azureml.core import Run

from InnerEye.Azure.azure_util import RUN_CONTEXT
from InnerEye.Common.fixed_paths import DEFAULT_AML_UPLOAD_DIR
from InnerEye.ML.deep_learning_config import CHECKPOINT_FOLDER
from health_azure import download_files_from_run_id, is_running_in_azure_ml
from health_azure.utils import get_run_file_names

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
    if available_checkpoints is None and is_running_in_azure_ml():
        logging.info("No recovery checkpoints available in the checkpoint folder. Trying to find checkpoints in "
                     "AzureML from previous runs of this job.")
        # Download checkpoints from AzureML, then try to find recovery checkpoints among those.
        temp_folder = download_checkpoints_to_temp_folder(RUN_CONTEXT)
        available_checkpoints = find_all_recovery_checkpoints(temp_folder)
    if available_checkpoints is not None:
        return extract_latest_checkpoint_and_epoch(available_checkpoints)
    return None


def download_checkpoints_to_temp_folder(run: Run) -> Path:
    """
    Downloads all files with the outputs/checkpoints prefix of the given run to a temporary folder.
    In distributed training, the download only happens once per node.

    :return: The path to which the files were downloaded.
    """
    # Downloads should go to a temporary folder because downloading the files to the checkpoint folder might
    # cause artifact conflicts later.
    temp_folder = Path(tempfile.mkdtemp())
    checkpoint_prefix = f"{DEFAULT_AML_UPLOAD_DIR}/{CHECKPOINT_FOLDER}/"
    existing_checkpoints = get_run_file_names(run, prefix=checkpoint_prefix)
    logging.info(f"Number of checkpoints available in AzureML: {len(existing_checkpoints)}")
    if len(existing_checkpoints) > 0:
        try:
            download_files_from_run_id(run_id=run.id,
                                       output_folder=temp_folder,
                                       prefix=checkpoint_prefix)
        except Exception as ex:
            logging.warning(f"Unable to download checkpoints from AzureML. Error: {str(ex)}")
    return temp_folder


def create_best_checkpoint(path: Path) -> Path:
    """
    Creates the best checkpoint file. "Best" is at the moment defined as being the last checkpoint, but could be
    based on some defined policy.
    The best checkpoint will be renamed to `best_checkpoint.ckpt`.
    :param path: The folder that contains all checkpoint files.
    """
    logging.debug(f"Files in checkpoint folder: {' '.join(p.name for p in path.glob('*'))}")
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
