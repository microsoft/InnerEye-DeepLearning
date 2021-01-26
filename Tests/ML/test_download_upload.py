#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import List

import pytest

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_util import get_results_blob_path
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import OTHER_RUNS_SUBDIR_NAME, logging_section, logging_to_stdout
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import DATASET_CSV_FILE_NAME
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.utils.run_recovery import RunRecovery
from Tests.AfterTraining.test_after_training import FALLBACK_ENSEMBLE_RUN, FALLBACK_SINGLE_RUN, get_most_recent_run
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
    config.train = False
    return config


def test_download_checkpoints_invalid_run(test_output_dirs: OutputFolderForTests,
                                          runner_config: AzureConfig) -> None:
    assert get_results_blob_path("some_run_id") == "azureml/ExperimentRun/dcid.some_run_id"


def check_single_checkpoint(downloaded_checkpoints: List[Path]) -> None:
    assert len(downloaded_checkpoints) == 1
    assert downloaded_checkpoints[0].is_file()


@pytest.mark.after_training_single_run
def test_download_recovery_single_run(test_output_dirs: OutputFolderForTests,
                                      runner_config: AzureConfig) -> None:
    output_dir = test_output_dirs.root_dir
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(output_dir)
    run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)
    run_recovery = RunRecovery.download_all_checkpoints_from_run(config, run)

    # This fails if there is no recovery checkpoint
    check_single_checkpoint(run_recovery.get_recovery_checkpoint_paths())
    check_single_checkpoint(run_recovery.get_best_checkpoint_paths())


@pytest.mark.after_training_ensemble_run
def test_download_best_checkpoints_ensemble_run(test_output_dirs: OutputFolderForTests,
                                                runner_config: AzureConfig) -> None:
    output_dir = test_output_dirs.root_dir
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(output_dir)

    run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_ENSEMBLE_RUN)
    run_recovery = RunRecovery.download_best_checkpoints_from_child_runs(config, run)
    other_runs_folder = config.checkpoint_folder / OTHER_RUNS_SUBDIR_NAME
    assert other_runs_folder.is_dir()
    for child in ["0", "1"]:
        assert (other_runs_folder / child).is_dir(), "Child run folder does not exist"
    for checkpoint in run_recovery.get_best_checkpoint_paths():
        assert checkpoint.is_file(), f"File {checkpoint} does not exist"


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
