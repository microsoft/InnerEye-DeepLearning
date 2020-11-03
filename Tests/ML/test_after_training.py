#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import pytest
from azureml.core import Model

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_runner import RUN_RECOVERY_FILE
from InnerEye.Azure.azure_util import MODEL_ID_KEY_NAME, fetch_run
from InnerEye.Common import fixed_paths
from InnerEye.Common.output_directories import OutputFolderForTests
from Tests.ML.util import is_running_on_azure


@pytest.mark.after_training
def test_model_file_structure(test_output_dirs: OutputFolderForTests) -> None:
    """
    Downloads the model that was built in the most recent run, and checks if its file structure is as expected.
    """
    run_recovery_file = Path(RUN_RECOVERY_FILE)
    if is_running_on_azure():
        assert run_recovery_file.is_file(), "When running in cloud builds, this should pick up the ID of a previous " \
                                            "training run"
        print("Reading run information from file.")
        most_recent_run = run_recovery_file.read_text().strip()
    else:
        # This is usually executed when starting the test on a local box, where cwd is set to the directory
        # containing this file
        print("Using hardcoded run ID.")
        most_recent_run = "refs_pull_270_merge:refs_pull_270_merge_1602000978_66813f6e"
    azure_config = AzureConfig.from_yaml(fixed_paths.SETTINGS_YAML_FILE,
                                         project_root=fixed_paths.repository_root_directory())
    workspace = azure_config.get_workspace()
    run = fetch_run(workspace, most_recent_run)
    model_id = run.get_tags().get(MODEL_ID_KEY_NAME, None)
    assert model_id, "No model_id tag on the run"
    model = Model(workspace=workspace, id=model_id)
    downloaded_folder = Path(model.download(str(test_output_dirs.root_dir)))
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
    print("Downloaded model contains these files:")
    for actual_file in downloaded_folder.rglob("*"):
        print("  " + str(actual_file.relative_to(downloaded_folder)))
    missing = []
    for expected_file in expected_files:
        full_path = downloaded_folder / expected_file
        if not full_path.is_file():
            missing.append(expected_file)
    if missing:
        print("Missing files:")
        for m in missing:
            print(m)
        pytest.fail(f"{len(missing)} files in the registered model are missing: {missing[:5]}")
