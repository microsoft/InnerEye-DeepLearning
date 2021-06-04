#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pytest

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import fixed_paths
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Scripts.move_model import MoveModelConfig, PYTHON_ENVIRONMENT_NAME, move

MODEL_ID = "PassThroughModel:1"
ENSEMBLE_MODEL_ID = "BasicModel2Epochs:8351"


@pytest.mark.parametrize("model_id", [MODEL_ID, ENSEMBLE_MODEL_ID])
def test_download_and_upload(model_id: str, test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that downloads and uploads a model to a workspace
    """
    azure_config = AzureConfig.from_yaml(yaml_file_path=fixed_paths.SETTINGS_YAML_FILE,
                                         project_root=fixed_paths.repository_root_directory())
    ws = azure_config.get_workspace()
    config_download = MoveModelConfig(model_id=model_id, path=str(test_output_dirs.root_dir), action="download")
    move(ws, config_download)
    assert (test_output_dirs.root_dir / model_id.replace(":", "_")).is_dir()
    config_upload = MoveModelConfig(model_id=model_id, path=str(test_output_dirs.root_dir), action="upload")
    model = move(ws, config_upload)
    assert model is not None
    assert PYTHON_ENVIRONMENT_NAME in model.tags
    assert model.description != ""
