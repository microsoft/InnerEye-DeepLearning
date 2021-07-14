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
from typing import List
from unittest import mock

import numpy as np
import pytest
from azureml._restclient.constants import RunStatus
from azureml.core import Model, Run

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_runner import RUN_RECOVERY_FILE
from InnerEye.Azure.azure_util import MODEL_ID_KEY_NAME, download_run_outputs_by_prefix, \
    get_comparison_baseline_paths, \
    is_running_on_azure_agent, to_azure_friendly_string
from InnerEye.Common import common_util, fixed_paths, fixed_paths_for_tests
from InnerEye.Common.common_util import BEST_EPOCH_FOLDER_NAME, CROSSVAL_RESULTS_FOLDER, ENSEMBLE_SPLIT_NAME, \
    get_best_epoch_results_path
from InnerEye.Common.fixed_paths import (DEFAULT_AML_LOGS_DIR, DEFAULT_RESULT_IMAGE_NAME, DEFAULT_RESULT_ZIP_DICOM_NAME,
                                         PYTHON_ENVIRONMENT_NAME, repository_root_directory)
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Common.spawn_subprocess import spawn_and_monitor_subprocess
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.ML.configs.segmentation.BasicModel2Epochs import BasicModel2Epochs
from InnerEye.ML.configs.other.HelloContainer import HelloContainer
from InnerEye.ML.deep_learning_config import CHECKPOINT_FOLDER, ModelCategory
from InnerEye.ML.model_inference_config import read_model_inference_config
from InnerEye.ML.model_testing import THUMBNAILS_FOLDER
from InnerEye.ML.reports.notebook_report import get_html_report_name
from InnerEye.ML.runner import main
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.utils.config_loader import ModelConfigLoader
from InnerEye.ML.utils.image_util import get_unit_image_header
from InnerEye.ML.utils.io_util import zip_random_dicom_series
from InnerEye.Scripts import submit_for_inference
from Tests.ML.util import assert_nifti_content, get_default_azure_config, get_nifti_shape, get_default_workspace

FALLBACK_SINGLE_RUN = "refs_pull_498_merge:refs_pull_498_merge_1624292750_743430ab"
FALLBACK_ENSEMBLE_RUN = "refs_pull_498_merge:HD_4bf4efc3-182a-4596-8f93-76f128418142"
FALLBACK_2NODE_RUN = "refs_pull_498_merge:refs_pull_498_merge_1624292776_52b2f7e1"
FALLBACK_CV_GLAUCOMA = "refs_pull_498_merge:HD_cefb6e59-3929-43aa-8fc8-821b9a062219"
FALLBACK_HELLO_CONTAINER_RUN = "refs_pull_498_merge:refs_pull_498_merge_1624292748_45756bf8"


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
    run_recovery_id = get_most_recent_run_id(fallback_run_id_for_local_execution=fallback_run_id_for_local_execution)
    return get_default_azure_config().fetch_run(run_recovery_id=run_recovery_id)


def get_most_recent_model_id(fallback_run_id_for_local_execution: str = FALLBACK_SINGLE_RUN) -> str:
    """
    Gets the string name of the most recently executed AzureML run, extracts which model that run had registered,
    and return the model id.
    :param fallback_run_id_for_local_execution: A hardcoded AzureML run ID that is used when executing this code
    on a local box, outside of Azure build agents.
    """
    most_recent_run = get_most_recent_run_id(fallback_run_id_for_local_execution=fallback_run_id_for_local_execution)
    azure_config = AzureConfig.from_yaml(fixed_paths.SETTINGS_YAML_FILE,
                                         project_root=fixed_paths.repository_root_directory())
    run = azure_config.fetch_run(most_recent_run)
    tags = run.get_tags()
    model_id = tags.get(MODEL_ID_KEY_NAME, None)
    assert model_id, f"No model_id tag was found on run {most_recent_run}"
    return model_id


def get_most_recent_model(fallback_run_id_for_local_execution: str = FALLBACK_SINGLE_RUN) -> Model:
    """
    Gets the string name of the most recently executed AzureML run, extracts which model that run had registered,
    and return the instantiated model object.
    :param fallback_run_id_for_local_execution: A hardcoded AzureML run ID that is used when executing this code
    on a local box, outside of Azure build agents.
    """
    model_id = get_most_recent_model_id(fallback_run_id_for_local_execution=fallback_run_id_for_local_execution)
    return Model(workspace=get_default_workspace(), id=model_id)


def get_experiment_name_from_environment() -> str:
    """
    Reads the name of the present branch from environment variable "BUILD_BRANCH". This must be set in the YML file.
    If the variable is not found, return an empty string.
    With this setup, all AML runs that belong to a given pull request are listed at the same place.
    """
    env_branch = "BUILD_BRANCH"
    build_branch = os.environ.get(env_branch, None)
    if not build_branch:
        if is_running_on_azure_agent():
            raise ValueError(f"Environment variable {env_branch} should be set when running on Azure agents.")
        return ""
    return to_azure_friendly_string(build_branch) or ""


@pytest.mark.after_training_single_run
@pytest.mark.after_training_ensemble_run
@pytest.mark.after_training_glaucoma_cv_run
@pytest.mark.after_training_hello_container
def test_registered_model_file_structure_and_instantiate(test_output_dirs: OutputFolderForTests) -> None:
    """
    Downloads the model that was built in the most recent run, and checks if its file structure is as expected.
    """
    fallback_run_id_for_local_execution = FALLBACK_SINGLE_RUN
    model = get_most_recent_model(fallback_run_id_for_local_execution=fallback_run_id_for_local_execution)
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

    model_inference_config = read_model_inference_config(downloaded_folder / fixed_paths.MODEL_INFERENCE_JSON_FILE_NAME)
    tags = get_most_recent_run(fallback_run_id_for_local_execution=fallback_run_id_for_local_execution).get_tags()
    model_name = tags["model_name"]
    assert model_inference_config.model_name == model_name
    assert model_inference_config.model_configs_namespace.startswith("InnerEye.ML.configs.")
    assert model_inference_config.model_configs_namespace.endswith(model_name)
    loader = ModelConfigLoader(model_configs_namespace=model_inference_config.model_configs_namespace)
    model_config = loader.create_model_config_from_name(model_name=model_inference_config.model_name)
    assert type(model_config).__name__ == model_inference_config.model_name


@pytest.mark.after_training_single_run
def test_get_comparison_data(test_output_dirs: OutputFolderForTests) -> None:
    """
    Check that metrics.csv and dataset.csv are created after the second epoch, if running on Azure.
    """
    run = get_most_recent_run()
    blob_path = get_best_epoch_results_path(ModelExecutionMode.TEST)
    (comparison_dataset_path, comparison_metrics_path) = get_comparison_baseline_paths(test_output_dirs.root_dir,
                                                                                       blob_path, run,
                                                                                       DATASET_CSV_FILE_NAME)
    assert comparison_dataset_path is not None
    assert comparison_metrics_path is not None


@pytest.mark.after_training_single_run
def test_check_dataset_mountpoint(test_output_dirs: OutputFolderForTests) -> None:
    """
    Check that the dataset mountpoint has been used correctly. The PR build submits the BasicModel2Epochs with
    dataset mounting, using a fixed mount path that is given in the model.
    """
    run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)
    files = run.get_file_names()
    driver_log = f"{DEFAULT_AML_LOGS_DIR}/70_driver_log.txt"
    assert driver_log in files
    downloaded = test_output_dirs.root_dir / "70_driver_log.txt"
    run.download_file(driver_log, output_file_path=str(downloaded))
    logs = downloaded.read_text()
    expected_mountpoint = BasicModel2Epochs().dataset_mountpoint
    assert f"local_dataset                           : {expected_mountpoint}" in logs


@pytest.mark.inference
@pytest.mark.parametrize("use_dicom", [False, True])
def test_submit_for_inference(use_dicom: bool, test_output_dirs: OutputFolderForTests) -> None:
    """
    Execute the submit_for_inference script on the model that was recently trained. This starts an AzureML job,
    and downloads the segmentation. Then check if the segmentation was actually produced.

    :param use_dicom: True to test DICOM in/out, False otherwise.
    :param test_output_dirs: Test output directories.
    """
    model = get_most_recent_model(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)
    assert PYTHON_ENVIRONMENT_NAME in model.tags, "Environment name not present in model properties"
    if use_dicom:
        size = (64, 64, 64)
        spacing = (1., 1., 2.5)
        image_file = test_output_dirs.root_dir / "temp_pack_dicom_series" / "dicom_series.zip"
        scratch_folder = test_output_dirs.root_dir / "temp_dicom_series"
        zip_random_dicom_series(size, spacing, image_file, scratch_folder)
    else:
        image_file = fixed_paths_for_tests.full_ml_test_data_path() / "train_and_test_data" / "id1_channel1.nii.gz"
    assert image_file.exists(), f"Image file not found: {image_file}"
    settings_file = fixed_paths.SETTINGS_YAML_FILE
    assert settings_file.exists(), f"Settings file not found: {settings_file}"
    args = ["--image_file", str(image_file),
            "--model_id", model.id,
            "--settings", str(settings_file),
            "--download_folder", str(test_output_dirs.root_dir),
            "--cluster", "training-nc12",
            "--experiment", get_experiment_name_from_environment() or "model_inference",
            "--use_dicom", str(use_dicom)]
    download_file = DEFAULT_RESULT_ZIP_DICOM_NAME if use_dicom else DEFAULT_RESULT_IMAGE_NAME
    seg_path = test_output_dirs.root_dir / download_file
    assert not seg_path.exists(), f"Result file {seg_path} should not yet exist"
    submit_for_inference.main(args, project_root=fixed_paths.repository_root_directory())
    assert seg_path.exists(), f"Result file {seg_path} was not created"


def _check_presence_cross_val_metrics_file(split: str, mode: ModelExecutionMode, available_files: List[str]) -> bool:
    return f"{CROSSVAL_RESULTS_FOLDER}/{split}/{mode.value}/metrics.csv" in available_files


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on Windows")
@pytest.mark.after_training_glaucoma_cv_run
def test_expected_cv_files_classification(test_output_dirs: OutputFolderForTests) -> None:
    run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_CV_GLAUCOMA)
    assert run is not None
    available_files = run.get_file_names()
    for split in ["0", "1"]:
        for mode in [ModelExecutionMode.TEST, ModelExecutionMode.TRAIN, ModelExecutionMode.VAL]:
            assert _check_presence_cross_val_metrics_file(split, mode, available_files)
    # We should not have any ensemble metrics in CV folder
    for mode in [ModelExecutionMode.TEST, ModelExecutionMode.TRAIN, ModelExecutionMode.VAL]:
        assert not _check_presence_cross_val_metrics_file(ENSEMBLE_SPLIT_NAME, mode, available_files)
    crossval_report_name = f"{ModelCategory.Classification.value}_crossval"
    crossval_report_file = f"{CROSSVAL_RESULTS_FOLDER}/{get_html_report_name(crossval_report_name)}"
    assert crossval_report_file in available_files


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on Windows")
@pytest.mark.after_training_ensemble_run
def test_expected_cv_files_segmentation() -> None:
    run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_ENSEMBLE_RUN)
    assert run is not None
    available_files = run.get_file_names()
    for split in ["0", "1"]:
        for mode in [ModelExecutionMode.TEST, ModelExecutionMode.VAL]:
            assert _check_presence_cross_val_metrics_file(split, mode, available_files)
    # For ensemble we should have the test metrics only
    assert _check_presence_cross_val_metrics_file(ENSEMBLE_SPLIT_NAME, ModelExecutionMode.TEST, available_files)
    assert not _check_presence_cross_val_metrics_file(ENSEMBLE_SPLIT_NAME, ModelExecutionMode.VAL, available_files)


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on Windows")
@pytest.mark.after_training_ensemble_run
def test_register_and_score_model(test_output_dirs: OutputFolderForTests) -> None:
    """
    End-to-end test which ensures the scoring pipeline is functioning as expected when used on a recently created
    model. This test is run after training an ensemble run in AzureML. It starts "submit_for_inference" via
    Popen. The inference run here is on a 2-channel model, whereas test_submit_for_inference works with a 1-channel
    model.
    """
    azureml_model = get_most_recent_model(fallback_run_id_for_local_execution=FALLBACK_ENSEMBLE_RUN)
    assert azureml_model is not None
    assert PYTHON_ENVIRONMENT_NAME in azureml_model.tags, "Environment name not present in model properties"
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
    [return_code1, stdout1] = spawn_and_monitor_subprocess(process=python_executable,
                                                           args=["--version"])
    assert return_code1 == 0
    print(f"Executing Python version {stdout1[0]}")
    return_code, stdout2 = spawn_and_monitor_subprocess(process=python_executable, args=[
        str(model_root / fixed_paths.SCORE_SCRIPT),
        f"--data_folder={str(data_root)}",
        f"--image_files={img_files[0]},{img_files[1]}",
        "--use_gpu=False"])

    # check that the process completed as expected
    assert return_code == 0, f"Subprocess failed with return code {return_code}. Stdout: {os.linesep.join(stdout2)}"
    expected_segmentation_path = Path(model_root) / DEFAULT_RESULT_IMAGE_NAME
    assert expected_segmentation_path.exists(), f"Result file not found: {expected_segmentation_path}"

    # sanity check the resulting segmentation
    expected_shape = get_nifti_shape(train_and_test_data_dir / img_files[0])
    image_header = get_unit_image_header()
    assert_nifti_content(str(expected_segmentation_path), expected_shape, image_header, [3], np.ubyte)


@pytest.mark.after_training_2node
def test_training_2nodes(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if a job running on 2 nodes trains correctly.
    """
    run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_2NODE_RUN)
    assert run.status == RunStatus.COMPLETED
    files = run.get_file_names()
    # There are two nodes, so there should be one log file per node.
    log0_path = "azureml-logs/70_driver_log_0.txt"
    log1_path = "azureml-logs/70_driver_log_1.txt"
    assert log0_path in files, "Node rank 0 log file is missing"
    assert log1_path in files, "Node rank 1 log file is missing"
    # Download both log files and check their contents
    log0 = test_output_dirs.root_dir / log0_path
    log1 = test_output_dirs.root_dir / log1_path
    run.download_file(log0_path, output_file_path=str(log0))
    run.download_file(log1_path, output_file_path=str(log1))
    log0_txt = log0.read_text()
    log1_txt = log1.read_text()
    # Only the node at rank 0 should be done certain startup activities, like visualizing crops.
    # Running inference similarly should only run on one node.
    for in_log0_only in ["Visualizing the effect of sampling random crops for training",
                         "STARTING: Registering default model",
                         "STARTING: Running default model on test set"]:
        assert in_log0_only in log0_txt
        assert in_log0_only not in log1_txt
    training_indicator = "STARTING: Model training"
    assert training_indicator in log0_txt
    assert training_indicator in log1_txt
    # Check diagnostic messages that show if DDP was set up correctly. This could fail if Lightning
    # changes its diagnostic outputs.
    assert "initializing ddp: GLOBAL_RANK: 0, MEMBER: 1/4" in log0_txt
    assert "initializing ddp: GLOBAL_RANK: 1, MEMBER: 2/4" in log0_txt
    assert "initializing ddp: GLOBAL_RANK: 2, MEMBER: 3/4" in log1_txt
    assert "initializing ddp: GLOBAL_RANK: 3, MEMBER: 4/4" in log1_txt


@pytest.mark.after_training_2node
@pytest.mark.skip("Test times out for unknown reasons.")
def test_recovery_on_2_nodes(test_output_dirs: OutputFolderForTests) -> None:
    args_list = ["--model", "BasicModel2EpochsMoreData",
                 "--azureml", "True",
                 "--num_nodes", "2",
                 "--run_recovery_id",
                 str(get_most_recent_run_id(fallback_run_id_for_local_execution=FALLBACK_2NODE_RUN)),
                 "--num_epochs", "4",
                 "--wait_for_completion", "True",
                 "--cluster", "training-nc12",
                 "--experiment", get_experiment_name_from_environment() or "recovery_on_2_nodes",
                 "--tag", "recovery_on_2_nodes"
                 ]
    script = str(repository_root_directory() / "InnerEye" / "ML" / "runner.py")
    with mock.patch("sys.argv", [script] + args_list):
        main()
    run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_2NODE_RUN)
    assert run.status == RunStatus.COMPLETED
    files = run.get_file_names()
    # There are two nodes, so there should be one log file per node.
    log0_path = "azureml-logs/70_driver_log_0.txt"
    log1_path = "azureml-logs/70_driver_log_1.txt"
    assert log0_path in files, "Node rank 0 log file is missing"
    assert log1_path in files, "Node rank 1 log file is missing"
    # Download both log files and check their contents
    log0 = test_output_dirs.root_dir / log0_path
    log1 = test_output_dirs.root_dir / log1_path
    run.download_file(log0_path, output_file_path=str(log0))
    run.download_file(log1_path, output_file_path=str(log1))
    log0_txt = log0.read_text()
    log1_txt = log1.read_text()
    assert "Downloading multiple files from run" in log0_txt
    assert "Downloading multiple files from run" not in log1_txt
    assert "Loading checkpoint that was created at (epoch = 2, global_step = 2)" in log0_txt
    assert "Loading checkpoint that was created at (epoch = 2, global_step = 2)" in log1_txt


@pytest.mark.after_training_single_run
def test_download_outputs(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if downloading multiple files works as expected
    """
    run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)
    prefix = Path(BEST_EPOCH_FOLDER_NAME) / ModelExecutionMode.TEST.value / THUMBNAILS_FOLDER
    download_run_outputs_by_prefix(prefix, test_output_dirs.root_dir, run=run)
    expected_files = ["005_lung_l_slice_053.png", "005_lung_r_slice_037.png", "005_spinalcord_slice_088.png"]
    for file in expected_files:
        expected = test_output_dirs.root_dir / file
        assert expected.is_file(), f"File missing: {file}"
    # Check that no more than the expected files were downloaded
    all_files = [f for f in test_output_dirs.root_dir.rglob("*") if f.is_file()]
    assert len(all_files) == len(expected_files)


@pytest.mark.after_training_single_run
def test_download_outputs_skipped(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if downloading multiple skips files where the prefix string is not a folder.
    """
    run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)
    # There is a file outputs/Train/metrics.csv, but for that file the prefix is not the full folder, hence should
    # not be downloaded.
    prefix = Path("Tra")
    download_run_outputs_by_prefix(prefix, test_output_dirs.root_dir, run=run)
    all_files = list(test_output_dirs.root_dir.rglob("*"))
    assert len(all_files) == 0


@pytest.mark.after_training_hello_container
def test_model_inference_on_single_run(test_output_dirs: OutputFolderForTests) -> None:
    fallback_run_id_for_local_execution = FALLBACK_HELLO_CONTAINER_RUN

    files_to_check = ["test_mse.txt", "test_mae.txt"]

    training_run = get_most_recent_run(fallback_run_id_for_local_execution=fallback_run_id_for_local_execution)
    all_training_files = training_run.get_file_names()
    for file in files_to_check:
        assert f"outputs/{file}" in all_training_files, f"{file} is missing"
    training_folder = test_output_dirs.root_dir / "training"
    training_folder.mkdir()
    training_files = [training_folder / file for file in files_to_check]
    for file, download_path in zip(files_to_check, training_files):
        training_run.download_file(f"outputs/{file}", output_file_path=str(download_path))

    container = HelloContainer()
    container.set_output_to(test_output_dirs.root_dir)
    container.model_id = get_most_recent_model_id(fallback_run_id_for_local_execution=fallback_run_id_for_local_execution)
    azure_config = get_default_azure_config()
    azure_config.train = False
    ml_runner = MLRunner(container=container, azure_config=azure_config, project_root=test_output_dirs.root_dir)
    ml_runner.setup()
    ml_runner.start_logging_to_file()
    ml_runner.run()

    inference_files = [container.outputs_folder / file for file in files_to_check]
    for inference_file in inference_files:
        assert inference_file.exists(), f"{inference_file} is missing"

    for training_file, inference_file in zip(training_files, inference_files):
        training_lines = training_file.read_text().splitlines()
        inference_lines = inference_file.read_text().splitlines()
        # We expect all the files we are reading to have a single float value
        assert len(training_lines) == 1
        train_value = float(training_lines[0].strip())
        assert len(inference_lines) == 1
        inference_value = float(inference_lines[0].strip())
        assert inference_value == pytest.approx(train_value, 1e-6)
