#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import shutil
from pathlib import Path
from typing import Any, List, Optional
from unittest import mock

import pytest

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import OTHER_RUNS_SUBDIR_NAME, logging_section, logging_to_stdout
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import DATASET_CSV_FILE_NAME
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.utils.run_recovery import RunRecovery
from Tests.AfterTraining.test_after_training import FALLBACK_ENSEMBLE_RUN, FALLBACK_SINGLE_RUN, get_most_recent_run
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.configs.lightning_test_containers import DummyContainerWithDatasets
from Tests.ML.util import get_default_azure_config
from health_azure import AzureRunInfo

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
