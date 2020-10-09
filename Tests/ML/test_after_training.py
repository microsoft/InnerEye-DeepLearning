#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import pytest
from azureml.core import Model

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_util import MODEL_ID_KEY_NAME, fetch_run
from InnerEye.Common import fixed_paths
from InnerEye.Common.output_directories import TestOutputDirectories


@pytest.mark.after_training
def test_model_file_structure(test_output_dirs: TestOutputDirectories) -> None:
    """
    Downloads the model that was built in the most recent run, and checks if its file structure is as expected.
    """
    most_recent_run = "refs_pull_270_merge:refs_pull_270_merge_1602000978_66813f6e"
    # assert RUN_RECOVERY_FILE.is_file()
    azure_config = AzureConfig.from_yaml(fixed_paths.SETTINGS_YAML_FILE,
                                         project_root=fixed_paths.repository_root_directory())
    workspace = azure_config.get_workspace()
    run = fetch_run(workspace, most_recent_run)
    model_id = run.get_tags().get(MODEL_ID_KEY_NAME, None)
    assert model_id, "No model_id tag on the run"
    model = Model(workspace=workspace, id=model_id)
    downloaded_folder = Path(model.download(test_output_dirs.root_dir))
    expected_files = \
        [
            "model_inference_config.json",
            "score.py",
            "python_wrapper.py",
            "run_scoring.py",
            "InnerEye/ML/config.py",
            "InnerEye/ML/metrics.py",
            "InnerEye/ML/runner.py",
        ]
    missing = []
    for f in expected_files:
        full_path = downloaded_folder / f
        if not full_path.is_file():
            missing.append(f)
    if missing:
        print("Missing files:")
        for m in missing:
            print(m)
        pytest.fail(f"{len(missing)} files in the registered model")
