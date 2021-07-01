#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import time
from unittest import mock

import pytest
from azureml.train.hyperdrive.runconfig import HyperDriveConfig

from InnerEye.Common import common_util, fixed_paths
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, ModelExecutionMode
from InnerEye.ML.metrics import InferenceMetricsForSegmentation
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.runner import Runner
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import get_default_checkpoint_handler
from Tests.ML.utils.test_model_util import create_model_and_store_checkpoint


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
@pytest.mark.parametrize("perform_cross_validation", [True, False])
@pytest.mark.parametrize("perform_training_set_inference", [True, False])
def test_model_inference_train_and_test(test_output_dirs: OutputFolderForTests,
                                        perform_cross_validation: bool,
                                        perform_training_set_inference: bool) -> None:
    config = DummyModel()
    config.number_of_cross_validation_splits = 2 if perform_cross_validation else 0
    config.perform_training_set_inference = perform_training_set_inference
    # Plotting crashes with random TCL errors on Windows, disable that for Windows PR builds.
    config.is_plotting_enabled = common_util.is_linux()

    config.set_output_to(test_output_dirs.root_dir)
    config.local_dataset = full_ml_test_data_path()

    checkpoint_path = config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    create_model_and_store_checkpoint(config, checkpoint_path)
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)
    checkpoint_handler.additional_training_done()
    result, _, _ = MLRunner(config).model_inference_train_and_test(checkpoint_handler=checkpoint_handler)
    if result is None:
        raise ValueError("Error result cannot be None")
    assert isinstance(result, InferenceMetricsForSegmentation)
    epoch_folder_name = common_util.BEST_EPOCH_FOLDER_NAME
    for folder in [ModelExecutionMode.TRAIN.value, ModelExecutionMode.VAL.value, ModelExecutionMode.TEST.value]:
        results_folder = config.outputs_folder / epoch_folder_name / folder
        folder_exists = results_folder.is_dir()
        if folder in [ModelExecutionMode.TRAIN.value, ModelExecutionMode.VAL.value]:
            if perform_training_set_inference:
                assert folder_exists
        else:
            assert folder_exists


def test_logging_to_file(test_output_dirs: OutputFolderForTests) -> None:
    # Log file should go to a new, non-existent folder, 2 levels deep
    file_path = test_output_dirs.root_dir / "subdir1" / "subdir2" / "logfile.txt"
    assert common_util.logging_to_file_handler is None
    common_util.logging_to_file(file_path)
    assert common_util.logging_to_file_handler is not None
    log_line = "foo bar"
    logging.getLogger().setLevel(logging.INFO)
    logging.info(log_line)
    common_util.disable_logging_to_file()
    should_not_be_present = "This should not be present in logs"
    logging.info(should_not_be_present)
    assert common_util.logging_to_file_handler is None
    # Wait for a bit, tests sometimes fail with the file not existing yet
    time.sleep(2)
    assert file_path.exists()
    assert log_line in file_path.read_text()
    assert should_not_be_present not in file_path.read_text()


def test_cross_validation_for_LightingContainer_models_is_supported() -> None:
    '''
    Prior to https://github.com/microsoft/InnerEye-DeepLearning/pull/483 we raised an exception in
    runner.run when cross validation was attempted on a lightning container. This test checks that
    we do not raise the exception anymore, and instead pass on a cross validation hyperdrive config
    to azure_runner's submit_to_azureml method.
    '''
    args_list = ["--model=HelloContainer", "--number_of_cross_validation_splits=5", "--azureml=True"]
    with mock.patch("sys.argv", [""] + args_list):
        runner = Runner(project_root=fixed_paths.repository_root_directory(), yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)
        with mock.patch("InnerEye.Azure.azure_runner.create_and_submit_experiment", return_value=None) as create_and_submit_experiment_patch:
            runner.run()
            assert runner.lightning_container.model_name == 'HelloContainer'
            assert runner.lightning_container.number_of_cross_validation_splits == 5
            args, _ = create_and_submit_experiment_patch.call_args
            script_run_config = args[1]
            assert isinstance(script_run_config, HyperDriveConfig)
