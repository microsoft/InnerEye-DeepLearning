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
from InnerEye.Common.fixed_paths import DEFAULT_RESULT_IMAGE_NAME, RUN_SCORING_SCRIPT, SCORE_SCRIPT
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Scripts import submit_for_inference
from Tests import fixed_paths_for_tests
from Tests.ML.util import is_running_on_azure


def get_most_recent_run() -> str:
    """
    Gets the string name of the most recently executed AzureML run. This is picked up from the `most_recent_run.txt`
    file when running on the cloud. For execution on the local dev box, a fixed path is returned.
    :return:
    """
    run_recovery_file = Path(RUN_RECOVERY_FILE)
    if is_running_on_azure():
        assert run_recovery_file.is_file(), "When running in cloud builds, this should pick up the ID of a previous " \
                                            "training run"
        print("Reading run information from file.")
        return run_recovery_file.read_text().strip()
    else:
        # When executing on the local box, we usually don't have any recent runs. Use a hardcoded run ID here.
        # Consequently, local execution of tests that use this run may fail, while executing in the cloud passes.
        # In this case, modify the run here to something more recent.
        print("Using hardcoded run ID.")
        return "refs_pull_276_merge:refs_pull_276_merge_1604614910_f7e2e0cc"


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


@pytest.mark.after_training
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
    args = ["--image_file", str(image_file),
            "--model_id", model.id,
            "--settings", str(settings_file),
            "--download_folder", str(test_output_dirs.root_dir),
            "--cluster", "training-nc12"]
    seg_path = test_output_dirs.root_dir / DEFAULT_RESULT_IMAGE_NAME
    assert not seg_path.exists(), f"Result file {seg_path} should not yet exist"
    submit_for_inference.main(args, project_root=fixed_paths.repository_root_directory())
    assert seg_path.exists(), f"Result file {seg_path} was not created"


def test_run_scoring_exists() -> None:
    full_scoring_file = fixed_paths.repository_root_directory() / RUN_SCORING_SCRIPT
    assert full_scoring_file.exists(), f"{RUN_SCORING_SCRIPT} must exist at repository root."
    assert full_scoring_file.exists(), f"{SCORE_SCRIPT} must exist at repository root."
