#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import json
from attr import dataclass
from azureml.core import Environment, Model, Workspace

from InnerEye.ML.deep_learning_config import FINAL_MODEL_FOLDER, FINAL_ENSEMBLE_MODEL_FOLDER

PYTHON_ENVIRONMENT_NAME = "python_environment_name"
MODEL_PATH = "MODEL"
ENVIRONMENT_PATH = "ENVIRONMENT"
MODEL_JSON = "model.json"


@dataclass
class MoveModelConfig:
    model_id: str
    path: str
    action: str
    workspace_name: str = ""
    subscription_id: str = ""
    resource_group: str = ""

    def get_paths(self) -> Tuple[Path, Path]:
        """
        Gets paths and creates folders if necessary
        :param path: Base path
        :param model_id: The model ID
        :return: model_path, environment_path
        """
        model_id_path = Path(self.path) / self.model_id.replace(":", "_")
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
    model_path, environment_path = config.get_paths()
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
    model_path, environment_path = config.get_paths()
    with open(model_path / MODEL_JSON, 'r') as f:
        model_dict = json.load(f)

    # Find the folder containing the final model.
    final_model_path = model_path / FINAL_MODEL_FOLDER
    full_model_path = final_model_path if final_model_path.exists() else model_path / FINAL_ENSEMBLE_MODEL_FOLDER

    new_model = Model.register(ws, model_path=str(full_model_path), model_name=model_dict['name'],
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
    parser = ArgumentParser()
    parser.add_argument("-a", "--action", type=str, required=True,
                        help="Action (download or upload)")
    parser.add_argument("-w", "--workspace_name", type=str, required=True,
                        help="Azure ML workspace name")
    parser.add_argument("-s", "--subscription_id", type=str, required=True,
                        help="AzureML subscription id")
    parser.add_argument("-r", "--resource_group", type=str, required=True,
                        help="AzureML resource group")
    parser.add_argument("-p", "--path", type=str, required=True,
                        help="The path to download or upload model")
    parser.add_argument("-m", "--model_id", type=str, required=True,
                        help="The AzureML model ID")

    args = parser.parse_args()
    config = MoveModelConfig(workspace_name=args.workspace_name, subscription_id=args.subscription_id,
                             resource_group=args.resource_group,
                             path=args.path, action=args.action, model_id=args.model_id)
    ws = get_workspace(config)
    move(ws, config)


def move(ws: Workspace, config: MoveModelConfig) -> Model:
    """
    Moves a model: downloads or uploads the model depending on the configs
    :param config: the move model config
    :param ws: The Azure ML workspace
    :return: the download or upload model
    """
    if config.action == "download":
        return download_model(ws, config)
    elif config.action == "upload":
        return upload_model(ws, config)
    else:
        raise ValueError(f'Invalid action {config.action}, allowed values: import or export')


if __name__ == "__main__":
    main()
