#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, List, Optional
from unittest import mock

import pytest

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import OTHER_RUNS_SUBDIR_NAME, logging_section, logging_to_stdout
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import DATASET_CSV_FILE_NAME
from InnerEye.ML.lightning_base import InnerEyeContainer
from InnerEye.ML.lightning_container import LightningContainer, LightningWithInference
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.utils.run_recovery import RunRecovery
from Tests.AfterTraining.test_after_training import FALLBACK_ENSEMBLE_RUN, FALLBACK_SINGLE_RUN, get_most_recent_run
from Tests.ML.configs.DummyModel import DummyModel
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
    config = DummyModel()
    config.local_dataset = None
    config.azure_dataset_id = ""
    azure_config = get_default_azure_config()
    runner = MLRunner(config, azure_config)
    runner.setup()
    runner.project_root = test_output_dirs.root_dir

    # If the model has neither local_dataset or azure_dataset_id, mount_or_download_dataset should fail.
    with pytest.raises(ValueError):
        runner.mount_or_download_dataset()

    # Pointing the model to a dataset folder that does not exist should raise an Exception
    fake_folder = runner.project_root / "foo"
    runner.container.local_dataset = fake_folder
    with pytest.raises(FileNotFoundError):
        runner.mount_or_download_dataset()

    # If the local dataset folder exists, mount_or_download_dataset should not do anything.
    fake_folder.mkdir()
    local_dataset = runner.mount_or_download_dataset()
    assert local_dataset == fake_folder

    # Pointing the model to a dataset in Azure should trigger a download
    runner.container.local_dataset = None
    runner.container.azure_dataset_id = dataset_name
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


def _test_mount_for_lightning_container(test_output_dirs: OutputFolderForTests,
                                        is_offline_run: bool,
                                        local_dataset: Optional[Path],
                                        azure_dataset: str,
                                        is_lightning_model: bool):
    if is_lightning_model:
        lightning_container = LightningContainer()
        lightning_container.azure_dataset_id = azure_dataset
        lightning_container.local_dataset = local_dataset
        config = LightningWithInference()
    else:
        config = DummyModel()
        config.azure_dataset_id = azure_dataset
        config.local_dataset = local_dataset
        lightning_container = InnerEyeContainer(config)
    with mock.patch("InnerEye.ML.run_ml.MLRunner.is_offline_run", is_offline_run):
        with mock.patch("InnerEye.ML.run_ml.download_dataset", return_value="download"):
            with mock.patch("InnerEye.ML.run_ml.try_to_mount_input_dataset", return_value="mount"):
                runner = MLRunner(config, container=lightning_container,
                                  azure_config=None, project_root=test_output_dirs.root_dir)
                runner.setup()
                return runner.mount_or_download_dataset()


@pytest.mark.parametrize(("is_offline_run", "is_lightning_model", "expected_error"),
                         [
                             # A built-in InnerEye model must have either local dataset or azure dataset provided.
                             (True, False, "The model must contain either local_dataset or azure_dataset_id"),
                             # ... but this is OK for Lightning container models. A Lightning container could simply
                             # download its data from the web before training.
                             (True, True, ""),
                             # A built-in InnerEye model must have an azure dataset provided when running in AzureML
                             (False, False, "The model must contain azure_dataset_id for running on AML"),
                             # ... but this is OK for Lightning container models.
                             (False, True, "")])
def test_mount_failing(test_output_dirs: OutputFolderForTests,
                       is_offline_run: bool,
                       is_lightning_model: bool,
                       expected_error: str):
    """
    Test cases when MLRunner.mount_or_download_dataset raises an exception.
    """
    def run() -> Any:
        return _test_mount_for_lightning_container(test_output_dirs=test_output_dirs,
                                                   is_offline_run=is_offline_run,
                                                   local_dataset=None,
                                                   azure_dataset="",
                                                   is_lightning_model=is_lightning_model)

    if expected_error:
        with pytest.raises(ValueError) as ex:
            run()
        assert expected_error in str(ex)
    else:
        assert run() is None


def test_mount_or_download(test_output_dirs: OutputFolderForTests) -> None:
    """
    Tests the different combinations of local and Azure datasets, with Innereye built-in and container models.
    """
    root = test_output_dirs.root_dir
    for is_lightning_model in [True, False]:
        # With runs outside of AzureML, an AML dataset should get downloaded.
        assert _test_mount_for_lightning_container(test_output_dirs=test_output_dirs,
                                                   is_offline_run=True,
                                                   local_dataset=None,
                                                   azure_dataset="foo",
                                                   is_lightning_model=is_lightning_model) == "download"
        # With runs in AzureML, an AML dataset should get mounted.
        assert _test_mount_for_lightning_container(test_output_dirs=test_output_dirs,
                                                   is_offline_run=False,
                                                   local_dataset=None,
                                                   azure_dataset="foo",
                                                   is_lightning_model=is_lightning_model) == "mount"
        # With runs outside of AzureML, a local dataset should be used as-is. Azure dataset ID is ignored here.
        assert _test_mount_for_lightning_container(test_output_dirs=test_output_dirs,
                                                   is_offline_run=True,
                                                   local_dataset=root,
                                                   azure_dataset="",
                                                   is_lightning_model=is_lightning_model) == root
