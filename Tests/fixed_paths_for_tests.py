#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Optional

from InnerEye.Common.type_annotations import PathOrString


def tests_root_directory(path: Optional[PathOrString] = None) -> Path:
    """
    Gets the full path to the root directory that holds the tests.
    If a relative path is provided then concatenate it with the absolute path
    to the repository root.

    :return: The full path to the repository's root directory, with symlinks resolved if any.
    """
    root = Path(os.path.realpath(__file__)).parent
    return root / path if path else root


def full_ml_test_data_path(path: str = "") -> Path:
    """
    Takes a relative path inside of the Tests/ML/test_data folder, and returns its
    full absolute path.

    :param path: A path relative to the ML/tests/test_data
    :return: The full absolute path of the argument.
    """
    return _full_test_data_path("ML", path)

def full_reconstruction_test_data_path(path: str = "") -> Path:
    """
    Takes a relative path inside of the Tests/Reconstruction/test_data folder, and returns its
    full absolute path.

    :param path: A path relative to the ML/tests/test_data
    :return: The full absolute path of the argument.
    """
    return _full_test_data_path("Reconstruction", path)

def full_azure_test_data_path(path: str = "") -> Path:
    """
    Takes a relative path inside of the Azure/tests/test_data folder, and returns its
    full absolute path.

    :param path: A path relative to the Tests/Azure/test_data
    :return: The full absolute path of the argument.
    """
    return _full_test_data_path("Azure", path)


def _full_test_data_path(prefix: str, suffix: str) -> Path:
    root = tests_root_directory()
    return root / prefix / "test_data" / suffix


RELATIVE_TEST_OUTPUTS_PATH = "test_outputs"
TEST_OUTPUTS_PATH = tests_root_directory().parent / RELATIVE_TEST_OUTPUTS_PATH
