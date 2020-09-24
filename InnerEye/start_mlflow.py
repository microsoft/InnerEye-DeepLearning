#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
import sys
from pathlib import Path

import param
from mlflow.server import _run_server
from mlflow.server.handlers import initialize_backend_stores

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import print_exception
from InnerEye.Common.generic_parsing import GenericConfig


class MLFlowSettings(GenericConfig):
    settings: Path = param.ClassSelector(class_=Path, default=fixed_paths.SETTINGS_YAML_FILE,
                                         doc="File containing subscription details, typically your settings.yml")
    port: int = param.Integer(default=8008, doc="The port on which the MLFlow server will operate.")


def aml_ui(backend_store_uri, default_artifact_root, port, host):
    try:
        initialize_backend_stores(backend_store_uri, default_artifact_root)
    except Exception as ex:
        print_exception(ex, "Unable to initialize back-end")
        sys.exit(1)
    try:
        _run_server(backend_store_uri, default_artifact_root, host, port, None, 1)
    except Exception as ex:
        print_exception(ex, "Unable to start MLFlow server")
        sys.exit(1)


def start_ui(project_root: Path) -> None:
    server_settings = MLFlowSettings.parse_args()
    azure_config = AzureConfig.from_yaml(server_settings.settings, project_root=project_root)
    ws = azure_config.get_workspace()
    uri = ws.get_mlflow_tracking_uri()
    # Workaround for a bug in MLFlow: Without this, no artifacts will be displayed.
    os.environ["MLFLOW_TRACKING_URI"] = uri
    # Host argument needs to be set to 0.0.0.0 to have the server accessible from outside
    aml_ui(backend_store_uri=uri, default_artifact_root=uri, port=server_settings.port, host="0.0.0.0")


if __name__ == "__main__":
    start_ui(project_root=fixed_paths.repository_root_directory())
