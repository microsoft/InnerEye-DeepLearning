#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import fixed_paths
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Scripts.move_model import MoveModelConfig, PYTHON_ENVIRONMENT_NAME, move

MODEL_ID = "PassThroughModel:1"


def test_download_and_upload(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that downloads and uploads a model to a workspace
    """
    azure_config = AzureConfig.from_yaml(yaml_file_path=fixed_paths.SETTINGS_YAML_FILE,
                                         project_root=fixed_paths.repository_root_directory())
    ws = azure_config.get_workspace()
    config_download = MoveModelConfig(model_id=MODEL_ID, path=str(test_output_dirs.root_dir), action="export")
    move(ws, config_download)
    assert (test_output_dirs.root_dir / MODEL_ID.replace(":", "_")).is_dir()
    config_upload = MoveModelConfig(model_id=MODEL_ID, path=str(test_output_dirs.root_dir), action="import")
    model = move(ws, config_upload)
    assert model is not None
    assert PYTHON_ENVIRONMENT_NAME in model.tags
    assert model.description != ""
