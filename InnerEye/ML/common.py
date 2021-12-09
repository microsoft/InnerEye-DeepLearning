#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import abc
from datetime import datetime
from enum import Enum, unique
from typing import Any, Dict, List

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

FINAL_MODEL_FOLDER = "final_model"
FINAL_ENSEMBLE_MODEL_FOLDER = "final_ensemble_model"
CHECKPOINT_FOLDER = "checkpoints"
VISUALIZATION_FOLDER = "visualizations"
EXTRA_RUN_SUBFOLDER = "extra_run_id"
ARGS_TXT = "args.txt"


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


def create_unique_timestamp_id() -> str:
    """
    Creates a unique string using the current time in UTC, up to seconds precision, with characters that
    are suitable for use in filenames. For example, on 31 Dec 2019 at 11:59:59pm UTC, the result would be
    2019-12-31T235959Z.
    """
    unique_id = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    return unique_id
