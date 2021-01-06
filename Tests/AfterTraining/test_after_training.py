#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
IMPORTANT: This file should ideally not import anything from the InnerEye.ML namespace.
This can avoid creating a full InnerEye Conda environment in the test suite.

All of the tests in this file rely on previous InnerEye runs that submit an AzureML job. They pick
up the most recently run AzureML job from most_recent_run.txt
"""

import os
from pathlib import Path

import pytest
from azureml.core import Model

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_runner import RUN_RECOVERY_FILE
from InnerEye.Azure.azure_util import MODEL_ID_KEY_NAME, fetch_run, get_comparison_baseline_paths, \
    is_offline_run_context, is_running_on_azure_agent, to_azure_friendly_string
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import get_epoch_results_path
from InnerEye.Common.fixed_paths import DEFAULT_RESULT_IMAGE_NAME
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.Scripts import submit_for_inference
from Tests import fixed_paths_for_tests


def get_most_recent_run() -> str:
    """
    Gets the string name of the most recently executed AzureML run. This is picked up from the `most_recent_run.txt`
    file when running on the cloud. For execution on the local dev box, a fixed path is returned.
    :return:
    """
    run_recovery_file = Path(RUN_RECOVERY_FILE)
    if is_running_on_azure_agent():
        assert run_recovery_file.is_file(), "When running in cloud builds, this should pick up the ID of a previous " \
                                            "training run"
        print("Reading run information from file.")
        return run_recovery_file.read_text().strip()
    else:
        # When executing on the local box, we usually don't have any recent runs. Use a hardcoded run ID here.
        # Consequently, local execution of tests that use this run may fail, while executing in the cloud passes.
        # In this case, modify the run here to something more recent.
        print("Using hardcoded run ID.")
        # This is Run 89 in refs_pull_276_merge
        return "refs_pull_276_merge:refs_pull_276_merge_1605139760_18b251e1"


def get_most_recent_model() -> Model:
    most_recent_run = get_most_recent_run()
    azure_config = AzureConfig.from_yaml(fixed_paths.SETTINGS_YAML_FILE,
                                         project_root=fixed_paths.repository_root_directory())
    workspace = azure_config.get_workspace()
    run = fetch_run(workspace, most_recent_run)
    tags = run.get_tags()
    model_id = tags.get(MODEL_ID_KEY_NAME, None)
    assert model_id, f"No model_id tag was found on run {most_recent_run}"
    return Model(workspace=workspace, id=model_id)


@pytest.mark.after_training
def test_model_file_structure(test_output_dirs: OutputFolderForTests) -> None:
    """
    Downloads the model that was built in the most recent run, and checks if its file structure is as expected.
    """
    model = get_most_recent_model()
    downloaded_folder = Path(model.download(str(test_output_dirs.root_dir)))
    print(f"Model was downloaded to folder {downloaded_folder}")
    expected_files = \
        [
            *fixed_paths.SCRIPTS_AT_ROOT,
            fixed_paths.MODEL_INFERENCE_JSON_FILE_NAME,
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


@pytest.mark.after_training
def test_get_comparison_data(test_output_dirs: OutputFolderForTests) -> None:
    """
    Check that metrics.csv and dataset.csv are created after the second epoch, if running on Azure.
    """
    most_recent_run = get_most_recent_run()
    azure_config = AzureConfig.from_yaml(fixed_paths.SETTINGS_YAML_FILE,
                                         project_root=fixed_paths.repository_root_directory())
    workspace = azure_config.get_workspace()
    run = fetch_run(workspace, most_recent_run)
    blob_path = get_epoch_results_path(2, ModelExecutionMode.TEST)
    (comparison_dataset_path, comparison_metrics_path) = get_comparison_baseline_paths(test_output_dirs.root_dir,
                                                                                       blob_path, run,
                                                                                       DATASET_CSV_FILE_NAME)
    assert comparison_dataset_path is not None
    assert comparison_metrics_path is not None


@pytest.mark.inference
def test_submit_for_inference(test_output_dirs: OutputFolderForTests) -> None:
    """
    Execute the submit_for_inference script on the model that was recently trained. This starts an AzureML job,
    and downloads the segmentation. Then check if the segmentation was actually produced.
    :return:
    """
    model = get_most_recent_model()
    image_file = fixed_paths_for_tests.full_ml_test_data_path() / "train_and_test_data" / "id1_channel1.nii.gz"
    assert image_file.exists(), f"Image file not found: {image_file}"
    settings_file = fixed_paths.SETTINGS_YAML_FILE
    assert settings_file.exists(), f"Settings file not found: {settings_file}"
    azure_config = AzureConfig.from_yaml(settings_file, project_root=fixed_paths.repository_root_directory())
    # Read the name of the branch from environment, so that the inference experiment is also listed alongside
    # all other AzureML runs that belong to the current PR.
    build_branch = os.environ.get("BUILD_BRANCH", None)
    experiment_name = to_azure_friendly_string(build_branch) if build_branch else "model_inference"
    azure_config.get_git_information()
    args = ["--image_file", str(image_file),
            "--model_id", model.id,
            "--settings", str(settings_file),
            "--download_folder", str(test_output_dirs.root_dir),
            "--cluster", "training-nc12",
            "--experiment", experiment_name]
    seg_path = test_output_dirs.root_dir / DEFAULT_RESULT_IMAGE_NAME
    assert not seg_path.exists(), f"Result file {seg_path} should not yet exist"
    submit_for_inference.main(args, project_root=fixed_paths.repository_root_directory())
    assert seg_path.exists(), f"Result file {seg_path} was not created"
