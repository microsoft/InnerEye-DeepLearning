#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path

import pytest

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_util import fetch_child_runs, fetch_run, get_results_blob_path
from InnerEye.Common import common_util
from InnerEye.Common.common_util import logging_to_stdout
from InnerEye.Common.output_directories import TestOutputDirectories
from InnerEye.ML.common import CHECKPOINT_FILE_SUFFIX, DATASET_CSV_FILE_NAME
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.utils.blobxfer_util import download_blobs
from InnerEye.ML.utils.ml_util import RunRecovery
from Tests.Common.test_util import DEFAULT_ENSEMBLE_RUN_RECOVERY_ID, DEFAULT_RUN_RECOVERY_ID
from Tests.ML.util import get_default_azure_config

logging_to_stdout(logging.DEBUG)


@pytest.fixture
def runner_config() -> AzureConfig:
    """
    Gets an Azure config that masks out the storage account for datasets, to avoid accidental overwriting.
    :return:
    """
    config = get_default_azure_config()
    config.model = ""
    config.is_train = False
    config.datasets_storage_account = ""
    config.datasets_container = ""
    return config


@pytest.mark.parametrize("is_ensemble", [True, False])
def test_download_checkpoints(test_output_dirs: TestOutputDirectories, is_ensemble: bool,
                              runner_config: AzureConfig) -> None:
    output_dir = Path(test_output_dirs.root_dir)
    assert get_results_blob_path("some_run_id") == "azureml/ExperimentRun/dcid.some_run_id"
    # Any recent run ID from a PR build will do. Use a PR build because the checkpoint files are small there.
    config = SegmentationModelBase(should_validate=False)
    config.set_output_to(output_dir)

    runner_config.run_recovery_id = DEFAULT_ENSEMBLE_RUN_RECOVERY_ID if is_ensemble else DEFAULT_RUN_RECOVERY_ID
    run_recovery = RunRecovery.download_checkpoints(runner_config, config)
    run_to_recover = fetch_run(workspace=runner_config.get_workspace(), run_recovery_id=runner_config.run_recovery_id)
    expected_checkpoint_file = "1" + CHECKPOINT_FILE_SUFFIX
    if is_ensemble:
        child_runs = fetch_child_runs(run_to_recover)
        expected_files = [Path(config.checkpoint_folder) / run_to_recover.id
                          / str(x.number) / expected_checkpoint_file for x in child_runs]
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
    config = SegmentationModelBase(should_validate=False)
    config.set_output_to(output_dir)
    runner_config.run_recovery_id = DEFAULT_ENSEMBLE_RUN_RECOVERY_ID
    child_runs = fetch_child_runs(run=fetch_run(runner_config.get_workspace(), DEFAULT_ENSEMBLE_RUN_RECOVERY_ID))
    # recover child runs separately also to test hyperdrive child run recovery functionality
    expected_checkpoint_file = "1" + CHECKPOINT_FILE_SUFFIX
    for child in child_runs:
        expected_files = [Path(config.checkpoint_folder) / child.id / expected_checkpoint_file]
        run_recovery = RunRecovery.download_checkpoints(runner_config, config, child)
        assert all([x in expected_files for x in run_recovery.get_checkpoint_paths(epoch=1)])
        assert all([expected_file.exists() for expected_file in expected_files])


def test_download_dataset(test_output_dirs: TestOutputDirectories) -> None:
    config = SegmentationModelBase(azure_dataset_id="test-dataset", should_validate=False)
    azure_config = get_default_azure_config()
    result_path = MLRunner(config, azure_config).download_dataset(None, dataset_path=Path(test_output_dirs.root_dir))
    assert result_path
    dataset_csv = Path(result_path) / DATASET_CSV_FILE_NAME
    assert dataset_csv.exists()


@pytest.mark.parametrize("is_file", [True, False])
def test_download_blobs(test_output_dirs: TestOutputDirectories, is_file: bool, runner_config: AzureConfig) -> None:
    """
    Test for a bug in early versions of download_blobs: download is happening via prefixes, but because of
    stripping leading directory names, blobs got overwritten.
    """
    root = Path(test_output_dirs.root_dir)
    account_key = runner_config.get_storage_account_key()
    assert account_key is not None
    # Expected test data in Azure blobs:
    # folder1/folder1.txt with content "folder1.txt"
    # folder1_with_suffix/folder2.txt with content "folder2.txt"
    # folder1_with_suffix/folder1.txt with content "this comes from folder2"
    # with bug present, folder1_with_suffix/folder1.txt will overwrite folder1/folder1.txt
    blobs_root_path = "data-for-testsuite/folder1"
    if is_file:
        blobs_root_path += "/folder1.txt"
    download_blobs(runner_config.storage_account, account_key, blobs_root_path, root, is_file)

    folder1 = root / "folder1.txt"
    assert folder1.exists()
    if not is_file:
        otherfile = root / "otherfile.txt"
        folder2 = root / "folder2.txt"
        assert folder1.read_text().strip() == "folder1.txt"
        assert otherfile.exists()
        assert otherfile.read_text().strip() == "folder1.txt"
        assert not folder2.exists()
