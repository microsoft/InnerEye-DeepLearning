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
    current = os.path.dirname(os.path.realpath(__file__))
    root = Path(os.path.realpath(os.path.join(current, "..", "..")))
    if path:
        return root / path
    else:
        return root


INNEREYE_PACKAGE_NAME = "InnerEye"
# Child paths to include in a registered model that live outside InnerEye/.
ENVIRONMENT_YAML_FILE_NAME = "environment.yml"

DEFAULT_AML_UPLOAD_DIR = "outputs"
DEFAULT_AML_LOGS_DIR = "azureml-logs"

DEFAULT_LOGS_DIR_NAME = "logs"
DATASETS_DIR_NAME = "datasets"
DATASETS_ACCOUNT_NAME = "innereyepublicdatasets"
# Inside of the AzureML workspace, a Datastore has to be created manually. That Datastore
# points to a container inside of a storage account.
AZUREML_DATASTORE_NAME = "innereyedatasets"

ML_RELATIVE_SOURCE_PATH = os.path.join("ML")
ML_RELATIVE_RUNNER_PATH = os.path.join(ML_RELATIVE_SOURCE_PATH, "runner.py")
ML_FULL_SOURCE_FOLDER_PATH = str(repository_root_directory() / ML_RELATIVE_SOURCE_PATH)

VISUALIZATION_NOTEBOOK_PATH = os.path.join("ML", "visualizers", "gradcam_visualization.ipynb")

# A file that contains secrets. This is expected to live in the repository root.
PROJECT_SECRETS_FILE = "InnerEyeTestVariables.txt"

INNEREYE_PACKAGE_ROOT = repository_root_directory(INNEREYE_PACKAGE_NAME)
TRAIN_YAML_FILE_NAME = "train_variables.yml"
TRAIN_YAML_FILE = INNEREYE_PACKAGE_ROOT / TRAIN_YAML_FILE_NAME

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
