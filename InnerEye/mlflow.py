#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
import sys
from pathlib import Path

from azureml.core import Workspace
from mlflow.server import _run_server
from mlflow.server.handlers import initialize_backend_stores

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import print_exception


def get_workspace(settings_file: Path) -> Workspace:
    azure_config = AzureConfig.from_yaml(settings_file)
    ws = azure_config.get_workspace()
    print(ws.get_details())
    return ws


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


def start_ui(settings_file: Path):
    ws = get_workspace(settings_file)
    uri = ws.get_mlflow_tracking_uri()
    # Workaround for a bug in MLFlow: Without this, no artifacts will be displayed.
    os.environ["MLFLOW_TRACKING_URI"] = uri
    # Host argument needs to be set to 0.0.0.0 to have the server access from outside
    aml_ui(backend_store_uri=uri, default_artifact_root=uri, port=80, host="0.0.0.0")


if __name__ == "__main__":
    start_ui(settings_file=fixed_paths.SETTINGS_YAML_FILE)
