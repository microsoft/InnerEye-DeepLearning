#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import sys
from pathlib import Path
from typing import Tuple

import json
import param
from attr import dataclass
from azureml.core import Environment, Model, Workspace

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import fixed_paths

print(f"Starting runner at {sys.argv[0]}")
innereye_root = Path(__file__).absolute().parent.parent.parent
if (innereye_root / "InnerEye").is_dir():
    innereye_root_str = str(innereye_root)
    if innereye_root_str not in sys.path:
        print(f"Adding InnerEye folder to sys.path: {innereye_root_str}")
        sys.path.insert(0, innereye_root_str)

from InnerEye.Common.generic_parsing import GenericConfig  # noqa: E402

# The property in the model registry that holds the name of the Python environment
PYTHON_ENVIRONMENT_NAME = "python_environment_name"
MODEL_PATH = "MODEL"
ENVIRONMENT_PATH = "ENVIRONMENT"
MODEL_JSON = "model.json"


@dataclass
class MoveModelConfig(GenericConfig):
    workspace_name: str = param.String(default="workspace_name",
                                       doc="AzureML workspace name")
    subscription_id: str = param.String(default="subscription_id",
                                        doc="AzureML subscription id")
    resource_group: str = param.String(default="resource_group",
                                       doc="AzureML resource group")
    model_id: str = param.String(default="model_id",
                                 doc="AzureML model_id")
    path: str = param.String(default="path",
                             doc="Path to import or export model")
    action: str = param.String(default="action",
                               doc="Import or export model from workspace. E.g. import or export")


def get_paths(path: Path, model_id: str) -> Tuple[Path, Path]:
    """
    Gets paths and creates folders if necessary
    :param path: Base path
    :param model_id: The model ID
    :return: model_path, environment_path
    """
    model_id_path = Path(path) / model_id.replace(":", "_")
    model_id_path.mkdir(parents=True, exist_ok=True)
    model_path = model_id_path / MODEL_PATH
    model_path.mkdir(parents=True, exist_ok=True)
    env_path = model_id_path / ENVIRONMENT_PATH
    env_path.mkdir(parents=True, exist_ok=True)
    return model_path, env_path


def download_model(ws: Workspace, config: MoveModelConfig) -> Model:
    """
    Downloads an InnerEye model from an AzureML workspace
    :param ws: The AzureML workspace
    :param config: move config
    :return: the exported Model
    """
    model = Model(ws, id=config.model_id)
    model_path, environment_path = get_paths(Path(config.path), config.model_id)
    with open(model_path / MODEL_JSON, 'w') as f:
        json.dump(model.serialize(), f)
    model.download(target_dir=str(model_path))
    env_name = model.tags.get(PYTHON_ENVIRONMENT_NAME)
    environment = ws.environments.get(env_name)
    environment.save_to_directory(str(environment_path), overwrite=True)
    return model


def upload_model(ws: Workspace, config: MoveModelConfig) -> Model:
    """
    Uploads an InnerEye model to an AzureML workspace
    :param ws: The AzureML workspace
    :param config: move config
    :return: imported Model
    """
    model_path, environment_path = get_paths(Path(config.path), config.model_id)
    with open(model_path / MODEL_JSON, 'r') as f:
        model_dict = json.load(f)

    new_model = Model.register(ws, model_path=str(model_path / "final_model"), model_name=model_dict['name'],
                               tags=model_dict['tags'], properties=model_dict['properties'],
                               description=model_dict['description'])
    env = Environment.load_from_directory(str(environment_path))
    env.register(workspace=ws)
    print(f"Environment {env.name} registered")
    return new_model


def get_workspace(config: MoveModelConfig) -> Workspace:
    """
    Get workspace based on command line input config
    :param config: MoveModelConfig
    :return: an Azure ML workspace
    """
    return Workspace.get(name=config.workspace_name, subscription_id=config.subscription_id,
                         resource_group=config.resource_group)


def main() -> None:
    config = MoveModelConfig.parse_args()
    ws = get_workspace(config)
    move(config, ws)


def move(ws: Workspace, config: MoveModelConfig) -> Model:
    """
    Moves a model: downloads or uploads the model depending on the configs
    :param config: the move model config
    :param ws: The Azure ML workspace
    :return: the import or exported model
    """
    if config.action == "export":
        return download_model(ws, config)
    elif config.action == "import":
        return upload_model(ws, config)
    else:
        raise ValueError(f'Invalid action {config.action}, allowed values: import or export')


if __name__ == "__main__":
    main()
