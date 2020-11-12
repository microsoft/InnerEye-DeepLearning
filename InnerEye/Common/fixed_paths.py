#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Optional

from InnerEye.Common.type_annotations import PathOrString


def repository_root_directory(path: Optional[PathOrString] = None) -> Path:
    """
    Gets the full path to the root directory that holds the present repository.
    :param path: if provided, a relative path to append to the absolute path to the repository root.
    :return: The full path to the repository's root directory, with symlinks resolved if any.
    """
    current = Path(__file__)
    root = current.parent.parent.parent
    if path:
        return root / path
    else:
        return root


INNEREYE_PACKAGE_NAME = "InnerEye"
# Child paths to include in a registered model that live outside InnerEye/.
ENVIRONMENT_YAML_FILE_NAME = "environment.yml"

DEFAULT_AML_UPLOAD_DIR = "outputs"
DEFAULT_RESULT_IMAGE_NAME = "segmentation.nii.gz"
DEFAULT_AML_LOGS_DIR = "azureml-logs"

DEFAULT_LOGS_DIR_NAME = "logs"
DEFAULT_MODEL_SUMMARIES_DIR_PATH = Path(DEFAULT_LOGS_DIR_NAME) / "model_summaries"
# The folder at the project root directory that holds datasets for local execution.
DATASETS_DIR_NAME = "datasets"

# Points to a folder at the project root directory that holds model weights downloaded from URLs.
MODEL_WEIGHTS_DIR_NAME = "modelweights"

ML_RELATIVE_SOURCE_PATH = os.path.join("ML")
ML_RELATIVE_RUNNER_PATH = os.path.join(ML_RELATIVE_SOURCE_PATH, "runner.py")
ML_FULL_SOURCE_FOLDER_PATH = str(repository_root_directory() / ML_RELATIVE_SOURCE_PATH)

VISUALIZATION_NOTEBOOK_PATH = os.path.join("ML", "visualizers", "gradcam_visualization.ipynb")

# A file that contains secrets. This is expected to live in the root folder of the repository or project.
PROJECT_SECRETS_FILE = "InnerEyeTestVariables.txt"
# A file with additional settings that should not be added to source control.
# This file is expected to live in the root folder of the repository or project.
PRIVATE_SETTINGS_FILE = "InnerEyePrivateSettings.yml"

# Names of secrets stored as environment variables or in the PROJECT_SECRETS_FILE:
# Secret for the Service Principal
SERVICE_PRINCIPAL_KEY = "APPLICATION_KEY"
# The access key for the Azure storage account that holds the datasets.
DATASETS_ACCOUNT_KEY = "DATASETS_ACCOUNT_KEY"

INNEREYE_PACKAGE_ROOT = repository_root_directory(INNEREYE_PACKAGE_NAME)
SETTINGS_YAML_FILE_NAME = "settings.yml"
SETTINGS_YAML_FILE = INNEREYE_PACKAGE_ROOT / SETTINGS_YAML_FILE_NAME

MODEL_INFERENCE_JSON_FILE_NAME = 'model_inference_config.json'
AZURE_RUNNER_ENVIRONMENT_YAML_FILE_NAME = "azure_runner.yml"
AZURE_RUNNER_ENVIRONMENT_YAML = repository_root_directory(AZURE_RUNNER_ENVIRONMENT_YAML_FILE_NAME)


def get_environment_yaml_file() -> Path:
    """
    Returns the path where the environment.yml file is located. This can be inside of the InnerEye package, or in
    the repository root when working with the code as a submodule.
    The function throws an exception if the file is not found at either of the two possible locations.
    :return: The full path to the environment files.
    """
    # The environment file is copied into the package folder in setup.py.
    env = INNEREYE_PACKAGE_ROOT / ENVIRONMENT_YAML_FILE_NAME
    if not env.exists():
        env = repository_root_directory(ENVIRONMENT_YAML_FILE_NAME)
        if not env.exists():
            raise ValueError(f"File {ENVIRONMENT_YAML_FILE_NAME} was not found not found in the package folder "
                             f"{INNEREYE_PACKAGE_ROOT}, and not in the repository root {repository_root_directory()}.")
    return env
