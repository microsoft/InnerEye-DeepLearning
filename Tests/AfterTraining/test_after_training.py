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
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from azureml.core import Model, Run

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_runner import RUN_RECOVERY_FILE
from InnerEye.Azure.azure_util import MODEL_ID_KEY_NAME, fetch_run, get_comparison_baseline_paths, \
    is_running_on_azure_agent, to_azure_friendly_string
from InnerEye.Common import common_util, fixed_paths, fixed_paths_for_tests
from InnerEye.Common.common_util import get_epoch_results_path
from InnerEye.Common.fixed_paths import DEFAULT_RESULT_IMAGE_NAME
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.ML.deep_learning_config import CHECKPOINT_FOLDER
from InnerEye.ML.utils.image_util import get_unit_image_header
from InnerEye.Scripts import submit_for_inference
from Tests.ML.util import assert_nifti_content, get_default_workspace, get_nifti_shape
from TestsOutsidePackage.test_register_model import SubprocessConfig

FALLBACK_ENSEMBLE_RUN = "refs_pull_323_merge:HD_92be13fd-a69f-49b2-8d84-620f35d61711"
FALLBACK_SINGLE_RUN = "refs_pull_323_merge:refs_pull_323_merge_1611612512_ac434012"


def get_most_recent_run_id(fallback_run_id_for_local_execution: str = FALLBACK_SINGLE_RUN) -> str:
    """
    Gets the string name of the most recently executed AzureML run. This is picked up from the `most_recent_run.txt`
    file when running on the cloud. For execution on the local dev box, use a hardcoded run ID.
    Consequently, local execution of tests that use this run may fail, while executing in the cloud passes.
    In this case, modify the run here to something more recent.
    :param fallback_run_id_for_local_execution: A hardcoded AzureML run ID that is used when executing this code
    on a local box, outside of Azure build agents.
    :return:
    """
    run_recovery_file = Path(RUN_RECOVERY_FILE)
    if is_running_on_azure_agent():
        assert run_recovery_file.is_file(), "When running in cloud builds, this should pick up the ID of a previous " \
                                            "training run"
        run_id = run_recovery_file.read_text().strip()
        print(f"Read this run ID from file: {run_id}")
        return run_id
    else:
        assert fallback_run_id_for_local_execution, "When running on local box, a hardcoded run ID must be given."
        print(f"Using this hardcoded run ID: {fallback_run_id_for_local_execution}")
        return fallback_run_id_for_local_execution


def get_most_recent_run(fallback_run_id_for_local_execution: str = FALLBACK_SINGLE_RUN) -> Run:
    """
    Gets the name of the most recently executed AzureML run, instantiates that Run object and returns it.
    :param fallback_run_id_for_local_execution: A hardcoded AzureML run ID that is used when executing this code
    on a local box, outside of Azure build agents.
    """
    return fetch_run(
        workspace=get_default_workspace(),
        run_recovery_id=get_most_recent_run_id(fallback_run_id_for_local_execution=fallback_run_id_for_local_execution)
    )


def get_most_recent_model(fallback_run_id_for_local_execution: str = FALLBACK_SINGLE_RUN) -> Model:
    """
    Gets the string name of the most recently executed AzureML run, extracts which model that run had registered,
    and return the instantiated model object.
    :param fallback_run_id_for_local_execution: A hardcoded AzureML run ID that is used when executing this code
    on a local box, outside of Azure build agents.
    """
    most_recent_run = get_most_recent_run_id(fallback_run_id_for_local_execution=fallback_run_id_for_local_execution)
    azure_config = AzureConfig.from_yaml(fixed_paths.SETTINGS_YAML_FILE,
                                         project_root=fixed_paths.repository_root_directory())
    workspace = azure_config.get_workspace()
    run = fetch_run(workspace, most_recent_run)
    tags = run.get_tags()
    model_id = tags.get(MODEL_ID_KEY_NAME, None)
    assert model_id, f"No model_id tag was found on run {most_recent_run}"
    return Model(workspace=workspace, id=model_id)


@pytest.mark.after_training_single_run
def test_model_file_structure(test_output_dirs: OutputFolderForTests) -> None:
    """
    Downloads the model that was built in the most recent run, and checks if its file structure is as expected.
    """
    model = get_most_recent_model(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)
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


@pytest.mark.after_training_single_run
def test_get_comparison_data(test_output_dirs: OutputFolderForTests) -> None:
    """
    Check that metrics.csv and dataset.csv are created after the second epoch, if running on Azure.
    """
    run = get_most_recent_run()
    blob_path = get_epoch_results_path(ModelExecutionMode.TEST)
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
    model = get_most_recent_model(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)
    image_file = fixed_paths_for_tests.full_ml_test_data_path() / "train_and_test_data" / "id1_channel1.nii.gz"
    assert image_file.exists(), f"Image file not found: {image_file}"
    settings_file = fixed_paths.SETTINGS_YAML_FILE
    assert settings_file.exists(), f"Settings file not found: {settings_file}"
    # Read the name of the branch from environment, so that the inference experiment is also listed alongside
    # all other AzureML runs that belong to the current PR.
    build_branch = os.environ.get("BUILD_BRANCH", None)
    experiment_name = to_azure_friendly_string(build_branch) if build_branch else "model_inference"
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


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on Windows")
@pytest.mark.after_training_ensemble_run
def test_register_and_score_model(test_output_dirs: OutputFolderForTests) -> None:
    """
    End-to-end test which ensures the scoring pipeline is functioning as expected when used on a recently created
    model. This test is run after training an ensemble run in AzureML. It starts "submit_for_inference" via
    Popen.
    """
    azureml_model = get_most_recent_model(fallback_run_id_for_local_execution=FALLBACK_ENSEMBLE_RUN)
    assert azureml_model is not None
    # download the registered model and test that we can run the score pipeline on it
    model_root = Path(azureml_model.download(str(test_output_dirs.root_dir)))
    # The model needs to contain score.py at the root, the (merged) environment definition,
    # and the inference config.
    expected_files = [
        *fixed_paths.SCRIPTS_AT_ROOT,
        fixed_paths.ENVIRONMENT_YAML_FILE_NAME,
        fixed_paths.MODEL_INFERENCE_JSON_FILE_NAME,
        "InnerEye/ML/runner.py",
    ]
    for expected_file in expected_files:
        assert (model_root / expected_file).is_file(), f"File {expected_file} missing"
    checkpoint_folder = model_root / CHECKPOINT_FOLDER
    assert checkpoint_folder.is_dir()
    checkpoints = list(checkpoint_folder.rglob("*"))
    assert len(checkpoints) >= 1, "There must be at least 1 checkpoint"

    # create a dummy datastore to store the image data
    test_datastore = test_output_dirs.root_dir / "test_datastore"
    # move test data into the data folder to simulate an actual run
    train_and_test_data_dir = full_ml_test_data_path("train_and_test_data")
    img_files = ["id1_channel1.nii.gz", "id1_channel2.nii.gz"]
    data_root = test_datastore / fixed_paths.DEFAULT_DATA_FOLDER
    data_root.mkdir(parents=True)
    for f in img_files:
        shutil.copy(str(train_and_test_data_dir / f), str(data_root))

    # run score pipeline as a separate process
    python_executable = sys.executable
    [return_code1, stdout1] = SubprocessConfig(process=python_executable,
                                               args=["--version"]).spawn_and_monitor_subprocess()
    assert return_code1 == 0
    print(f"Executing Python version {stdout1[0]}")
    return_code, stdout2 = SubprocessConfig(process=python_executable, args=[
        str(model_root / fixed_paths.SCORE_SCRIPT),
        f"--data_folder={str(data_root)}",
        f"--image_files={img_files[0]},{img_files[1]}",
        "--use_gpu=False"
    ]).spawn_and_monitor_subprocess()

    # check that the process completed as expected
    assert return_code == 0, f"Subprocess failed with return code {return_code}. Stdout: {os.linesep.join(stdout2)}"
    expected_segmentation_path = Path(model_root) / DEFAULT_RESULT_IMAGE_NAME
    assert expected_segmentation_path.exists(), f"Result file not found: {expected_segmentation_path}"

    # sanity check the resulting segmentation
    expected_shape = get_nifti_shape(train_and_test_data_dir / img_files[0])
    image_header = get_unit_image_header()
    assert_nifti_content(str(expected_segmentation_path), expected_shape, image_header, [3], np.ubyte)
