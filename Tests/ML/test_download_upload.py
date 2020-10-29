#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import pytest
import os
import torch

from pathlib import Path
from urllib.parse import urlparse

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_util import fetch_child_runs, fetch_run, get_results_blob_path
from InnerEye.Common import common_util, fixed_paths
from InnerEye.Common.common_util import logging_section, logging_to_stdout
from InnerEye.Common.output_directories import TestOutputDirectories
from InnerEye.ML import run_ml
from InnerEye.ML.common import CHECKPOINT_FILE_SUFFIX, DATASET_CSV_FILE_NAME
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.utils.blobxfer_util import download_blobs
from InnerEye.ML.utils.run_recovery import RunRecovery
from InnerEye.ML.deep_learning_config import WEIGHTS_FILE
from Tests.Common.test_util import DEFAULT_ENSEMBLE_RUN_RECOVERY_ID, DEFAULT_RUN_RECOVERY_ID
from Tests.ML.util import get_default_azure_config
from Tests.ML.configs.DummyModel import DummyModel

EXTERNAL_WEIGHTS_URL_EXAMPLE = "https://download.pytorch.org/models/resnet18-5c106cde.pth"

logging_to_stdout(logging.DEBUG)

@pytest.fixture
def runner_config() -> AzureConfig:
    """
    Gets an Azure config that masks out the storage account for datasets, to avoid accidental overwriting.
    :return:
    """
    config = get_default_azure_config()
    config.model = ""
    config.train = False
    config.datasets_container = ""
    return config


@pytest.mark.parametrize("is_ensemble", [True, False])
def test_download_checkpoints(test_output_dirs: TestOutputDirectories, is_ensemble: bool,
                              runner_config: AzureConfig) -> None:
    output_dir = Path(test_output_dirs.root_dir)
    assert get_results_blob_path("some_run_id") == "azureml/ExperimentRun/dcid.some_run_id"
    # Any recent run ID from a PR build will do. Use a PR build because the checkpoint files are small there.
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(output_dir)

    runner_config.run_recovery_id = DEFAULT_ENSEMBLE_RUN_RECOVERY_ID if is_ensemble else DEFAULT_RUN_RECOVERY_ID
    run_recovery = RunRecovery.download_checkpoints_from_recovery_run(runner_config, config)
    run_to_recover = fetch_run(workspace=runner_config.get_workspace(), run_recovery_id=runner_config.run_recovery_id)
    expected_checkpoint_file = "1" + CHECKPOINT_FILE_SUFFIX
    if is_ensemble:
        child_runs = fetch_child_runs(run_to_recover)
        expected_files = [Path(config.checkpoint_folder) / run_to_recover.id
                          / str(x.get_tags()['cross_validation_split_index']) / expected_checkpoint_file
                          for x in child_runs]
    else:
        expected_files = [Path(config.checkpoint_folder) / run_to_recover.id / expected_checkpoint_file]

    checkpoint_paths = run_recovery.get_checkpoint_paths(1)
    if is_ensemble:
        assert len(run_recovery.checkpoints_roots) == len(expected_files)
        assert all([(x in [y.parent for y in expected_files]) for x in run_recovery.checkpoints_roots])
        assert len(checkpoint_paths) == len(expected_files)
        assert all([x in expected_files for x in checkpoint_paths])
    else:
        assert len(checkpoint_paths) == 1
        assert checkpoint_paths[0] == expected_files[0]

    assert all([expected_file.exists() for expected_file in expected_files])


@pytest.mark.skipif(common_util.is_windows(), reason="Has issues on the windows build")
def test_download_checkpoints_hyperdrive_run(test_output_dirs: TestOutputDirectories,
                                             runner_config: AzureConfig) -> None:
    output_dir = Path(test_output_dirs.root_dir)
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(output_dir)
    runner_config.run_recovery_id = DEFAULT_ENSEMBLE_RUN_RECOVERY_ID
    child_runs = fetch_child_runs(run=fetch_run(runner_config.get_workspace(), DEFAULT_ENSEMBLE_RUN_RECOVERY_ID))
    # recover child runs separately also to test hyperdrive child run recovery functionality
    expected_checkpoint_file = "1" + CHECKPOINT_FILE_SUFFIX
    for child in child_runs:
        expected_files = [Path(config.checkpoint_folder) / child.id / expected_checkpoint_file]
        run_recovery = RunRecovery.download_checkpoints_from_recovery_run(runner_config, config, child)
        assert all([x in expected_files for x in run_recovery.get_checkpoint_paths(epoch=1)])
        assert all([expected_file.exists() for expected_file in expected_files])


def test_download_azureml_dataset(test_output_dirs: TestOutputDirectories) -> None:
    dataset_name = "test-dataset"
    config = ModelConfigBase(should_validate=False)
    azure_config = get_default_azure_config()
    runner = MLRunner(config, azure_config)
    runner.project_root = Path(test_output_dirs.root_dir)

    # If the model has neither local_dataset or azure_dataset_id, mount_or_download_dataset should fail.
    with pytest.raises(ValueError):
        runner.mount_or_download_dataset()

    # Pointing the model to a dataset folder that does not exist should raise an Exception
    fake_folder = runner.project_root / "foo"
    runner.model_config.local_dataset = fake_folder
    with pytest.raises(FileNotFoundError):
        runner.mount_or_download_dataset()

    # If the local dataset folder exists, mount_or_download_dataset should not do anything.
    fake_folder.mkdir()
    local_dataset = runner.mount_or_download_dataset()
    assert local_dataset == fake_folder

    # Pointing the model to a dataset in Azure should trigger a download
    runner.model_config.local_dataset = None
    runner.model_config.azure_dataset_id = dataset_name
    with logging_section("Starting download"):
        result_path = runner.mount_or_download_dataset()
    # Download goes into <project_root> / "datasets" / "test_dataset"
    expected_path = runner.project_root / fixed_paths.DATASETS_DIR_NAME / dataset_name
    assert result_path == expected_path
    assert result_path.is_dir()
    dataset_csv = Path(result_path) / DATASET_CSV_FILE_NAME
    assert dataset_csv.is_file()
    # Check that each individual file in the dataset is present
    for folder in [1, *range(10, 20)]:
        sub_folder = result_path / str(folder)
        sub_folder.is_dir()
        for file in ["ct", "esophagus", "heart", "lung_l", "lung_r", "spinalcord"]:
            f = (sub_folder / file).with_suffix(".nii.gz")
            assert f.is_file()


def test_download_dataset_via_blobxfer(test_output_dirs: TestOutputDirectories) -> None:
    azure_config = get_default_azure_config()
    result_path = run_ml.download_dataset_via_blobxfer(dataset_id="test-dataset",
                                                       azure_config=azure_config,
                                                       target_folder=Path(test_output_dirs.root_dir))
    assert result_path
    assert result_path.is_dir()
    dataset_csv = Path(result_path) / DATASET_CSV_FILE_NAME
    assert dataset_csv.exists()


@pytest.mark.parametrize("is_file", [True, False])
def test_download_blobxfer(test_output_dirs: TestOutputDirectories, is_file: bool, runner_config: AzureConfig) -> None:
    """
    Test for a bug in early versions of download_blobs: download is happening via prefixes, but because of
    stripping leading directory names, blobs got overwritten.
    """
    root = Path(test_output_dirs.root_dir)
    account_key = runner_config.get_dataset_storage_account_key()
    assert account_key is not None
    # Expected test data in Azure blobs:
    # folder1/folder1.txt with content "folder1.txt"
    # folder1_with_suffix/folder2.txt with content "folder2.txt"
    # folder1_with_suffix/folder1.txt with content "this comes from folder2"
    # with bug present, folder1_with_suffix/folder1.txt will overwrite folder1/folder1.txt
    blobs_root_path = "data-for-testsuite/folder1"
    if is_file:
        blobs_root_path += "/folder1.txt"
    download_blobs(runner_config.datasets_storage_account, account_key, blobs_root_path, root, is_file)

    folder1 = root / "folder1.txt"
    assert folder1.exists()
    if not is_file:
        otherfile = root / "otherfile.txt"
        folder2 = root / "folder2.txt"
        assert folder1.read_text().strip() == "folder1.txt"
        assert otherfile.exists()
        assert otherfile.read_text().strip() == "folder1.txt"
        assert not folder2.exists()


def test_download_model_weights(test_output_dirs: TestOutputDirectories) -> None:

    # Download a sample ResNet model from a URL given in the Pytorch docs
    # The downloaded model does not match the architecture, which is okay since we are only testing the download here.

    model_config = DummyModel(weights_url=EXTERNAL_WEIGHTS_URL_EXAMPLE)
    azure_config = get_default_azure_config()
    runner = MLRunner(model_config, azure_config)
    runner.project_root = Path(test_output_dirs.root_dir)

    result_path = runner.download_weights()
    assert result_path.is_file()


def test_get_local_weights_path_or_download(test_output_dirs: TestOutputDirectories) -> None:
    config = ModelConfigBase(should_validate=False)
    azure_config = get_default_azure_config()
    runner = MLRunner(config, azure_config)
    runner.project_root = Path(test_output_dirs.root_dir)

    # If the model has neither local_weights_path or weights_url set, should fail.
    with pytest.raises(ValueError):
        runner.get_local_weights_path_or_download()

    # If local_weights_path folder exists, get_local_weights_path_or_download should not do anything.
    local_weights_path = runner.project_root / "exist.pth"
    local_weights_path.touch()
    runner.model_config.local_weights_path = local_weights_path
    returned_weights_path = runner.get_local_weights_path_or_download()
    assert local_weights_path == returned_weights_path

    # Pointing the model to a URL should trigger a download
    runner.model_config.local_weights_path = None
    runner.model_config.weights_url = EXTERNAL_WEIGHTS_URL_EXAMPLE
    downloaded_weights = runner.get_local_weights_path_or_download()
    # Download goes into <project_root> / "modelweights" / "resnet18-5c106cde.pth"
    expected_path = runner.project_root / fixed_paths.MODEL_WEIGHTS_DIR_NAME / \
                    os.path.basename(urlparse(EXTERNAL_WEIGHTS_URL_EXAMPLE).path)
    assert downloaded_weights
    assert downloaded_weights.is_file()
    assert expected_path == downloaded_weights

    # try again, should not re-download
    modified_time = downloaded_weights.stat().st_mtime
    downloaded_weights_new = runner.get_local_weights_path_or_download()
    assert downloaded_weights_new
    assert downloaded_weights_new.stat().st_mtime == modified_time


def test_get_and_modify_local_weights(test_output_dirs: TestOutputDirectories) -> None:

    config = ModelConfigBase(should_validate=False)
    azure_config = get_default_azure_config()
    runner = MLRunner(config, azure_config)
    runner.project_root = Path(test_output_dirs.root_dir)
    runner.model_config.set_output_to(test_output_dirs.root_dir)
    runner.model_config.outputs_folder.mkdir()

    # If the model has neither local_weights_path or weights_url set, should fail.
    with pytest.raises(ValueError):
        runner.get_and_modify_local_weights()

    # Pointing the model to a local_weights_path that does not exist will raise an error.
    runner.model_config.local_weights_path = runner.project_root / "non_exist"
    with pytest.raises(FileNotFoundError):
        runner.get_and_modify_local_weights()

    # Test that weights are properly modified when a local_weights_path is set

    # set a method to modify weights:
    ModelConfigBase.modify_checkpoint = lambda self, path_to_checkpoint: {"modified": "local",  # type: ignore
                                                                          "path": path_to_checkpoint}
    # Set the local_weights_path to an empty file, which will be passed to modify_checkpoint
    local_weights_path = runner.project_root / "exist.pth"
    local_weights_path.touch()
    runner.model_config.local_weights_path = local_weights_path
    weights_path = runner.get_and_modify_local_weights()
    expected_path = runner.model_config.outputs_folder / WEIGHTS_FILE
    # read from weights_path and check that the dict has been written
    assert weights_path.is_file()
    assert expected_path == weights_path
    read = torch.load(str(weights_path))
    assert read.keys() == {"modified"} and read["modified"] == "local"
    assert read.keys() == {"path"} and read["path"] == local_weights_path
    # clean up
    weights_path.unlink()

    # Test that weights are properly modified when weights_url is set

    # set a different method to modify weights, to avoid using old files from other tests:
    ModelConfigBase.modify_checkpoint = lambda self, path_to_checkpoint: {"modified": "url",  # type: ignore
                                                                          "path": path_to_checkpoint}
    # Set the weights_url to the sample pytorch URL, which will be passed to modify_checkpoint
    runner.model_config.weights_url = EXTERNAL_WEIGHTS_URL_EXAMPLE
    weights_path = runner.get_and_modify_local_weights()
    expected_path = runner.model_config.outputs_folder / WEIGHTS_FILE
    # read from weights_path and check that the dict has been written
    assert weights_path.is_file()
    assert expected_path == weights_path
    read = torch.load(str(weights_path))
    assert read.keys() == {"modified"} and read["modified"] == "url"
    assert read.keys() == {"path"} and read["path"] == runner.project_root / fixed_paths.MODEL_WEIGHTS_DIR_NAME / \
                                                       os.path.basename(urlparse(EXTERNAL_WEIGHTS_URL_EXAMPLE).path)

