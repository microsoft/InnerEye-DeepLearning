#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import List

import pytest

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_util import fetch_child_runs, fetch_run, get_results_blob_path
from InnerEye.Common import common_util, fixed_paths
from InnerEye.Common.common_util import OTHER_RUNS_SUBDIR_NAME, logging_section, logging_to_stdout
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, create_checkpoint_path, BEST_CHECKPOINT_FILE_NAME, CHECKPOINT_SUFFIX
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.utils.run_recovery import RunRecovery
from Tests.ML.util import get_default_azure_config
from Tests.AfterTraining.test_after_training import get_most_recent_run

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
    return config


def test_download_checkpoints_invalid_run(test_output_dirs: OutputFolderForTests,
                              runner_config: AzureConfig) -> None:
    assert get_results_blob_path("some_run_id") == "azureml/ExperimentRun/dcid.some_run_id"


def check_single_checkpoint(downloaded_checkpoints: List[Path], expected_checkpoint: Path):
    assert len(downloaded_checkpoints) == 1
    assert downloaded_checkpoints[0] == expected_checkpoint
    assert expected_checkpoint.exists()


def check_multiple_checkpoints(downloaded_checkpoints: List[Path], expected_checkpoints: List[Path]):
    assert len(downloaded_checkpoints) == len(expected_checkpoints)
    assert all([x in expected_checkpoints for x in downloaded_checkpoints])
    assert all([expected_file.exists() for expected_file in expected_checkpoints])


@pytest.mark.after_training_single_run
def test_download_checkpoints_single_run(test_output_dirs: OutputFolderForTests,
                                         runner_config: AzureConfig) -> None:

    output_dir = test_output_dirs.root_dir
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(output_dir)

    runner_config.run_recovery_id = get_most_recent_run()
    run_recovery = RunRecovery.download_checkpoints_from_recovery_run(runner_config, config)
    run_to_recover = fetch_run(workspace=runner_config.get_workspace(), run_recovery_id=runner_config.run_recovery_id)
    checkpoint_root = config.checkpoint_folder / run_to_recover.id

    expected_checkpoint_epoch_1 = create_checkpoint_path(path=checkpoint_root, epoch=1)
    downloaded_checkpoint_path_epoch_1 = run_recovery.get_checkpoint_paths(1)
    check_single_checkpoint(downloaded_checkpoint_path_epoch_1, expected_checkpoint_epoch_1)

    expected_checkpoint_best_epoch = checkpoint_root / f"{BEST_CHECKPOINT_FILE_NAME}-v0{CHECKPOINT_SUFFIX}"
    downloaded_checkpoint_path_best_epoch = run_recovery.get_best_checkpoint_paths()
    check_single_checkpoint(downloaded_checkpoint_path_best_epoch, expected_checkpoint_best_epoch)


@pytest.mark.after_training_ensemble_run
def test_download_checkpoints_ensemble_run(test_output_dirs: OutputFolderForTests,
                                           runner_config: AzureConfig) -> None:

    output_dir = test_output_dirs.root_dir
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(output_dir)

    runner_config.run_recovery_id = get_most_recent_run()
    run_recovery = RunRecovery.download_checkpoints_from_recovery_run(runner_config, config)
    run_to_recover = fetch_run(workspace=runner_config.get_workspace(), run_recovery_id=runner_config.run_recovery_id)
    child_runs = fetch_child_runs(run_to_recover)
    checkpoint_root_other_runs = config.checkpoint_folder / OTHER_RUNS_SUBDIR_NAME
    expected_checkpoint_roots = [checkpoint_root_other_runs / str(x.get_tags()['cross_validation_split_index'])
                                 for x in child_runs]

    assert len(run_recovery.checkpoints_roots) == len(expected_checkpoint_roots)
    assert all([x in expected_checkpoint_roots for x in run_recovery.checkpoints_roots])

    expected_checkpoints_epoch_1 = [create_checkpoint_path(path=root, epoch=1)
                                    for root in expected_checkpoint_roots]
    checkpoint_paths_epoch_1 = run_recovery.get_checkpoint_paths(1)
    check_multiple_checkpoints(checkpoint_paths_epoch_1, expected_checkpoints_epoch_1)

    expected_checkpoint_best_epoch = [root / f"{BEST_CHECKPOINT_FILE_NAME}-v0{CHECKPOINT_SUFFIX}"
                                      for root in expected_checkpoint_roots]
    checkpoint_paths_best_epoch = run_recovery.get_best_checkpoint_paths()
    check_multiple_checkpoints(checkpoint_paths_best_epoch, expected_checkpoint_best_epoch)


@pytest.mark.after_training_ensemble_run
@pytest.mark.skipif(common_util.is_windows(), reason="Has issues on the windows build")
def test_download_checkpoints_hyperdrive_run(test_output_dirs: OutputFolderForTests,
                                             runner_config: AzureConfig) -> None:
    output_dir = test_output_dirs.root_dir
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(output_dir)
    run_recovery_id = get_most_recent_run()
    runner_config.run_recovery_id = run_recovery_id
    child_runs = fetch_child_runs(run=fetch_run(runner_config.get_workspace(), run_recovery_id))
    # recover child runs separately also to test hyperdrive child run recovery functionality
    for child in child_runs:
        expected_file = create_checkpoint_path(path=config.checkpoint_folder / child.id, epoch=1)
        run_recovery = RunRecovery.download_checkpoints_from_recovery_run(runner_config, config, child)
        checkpoint_paths = run_recovery.get_checkpoint_paths(epoch=1)
        check_single_checkpoint(checkpoint_paths, expected_file)


def test_download_azureml_dataset(test_output_dirs: OutputFolderForTests) -> None:
    dataset_name = "test-dataset"
    config = ModelConfigBase(should_validate=False)
    azure_config = get_default_azure_config()
    runner = MLRunner(config, azure_config)
    runner.project_root = test_output_dirs.root_dir

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
