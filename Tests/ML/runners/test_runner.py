#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import time
from unittest import mock
from unittest.mock import Mock

import pytest
from azureml.train.hyperdrive.runconfig import HyperDriveConfig

from InnerEye.Common import common_util, fixed_paths
from InnerEye.Common.common_util import ModelProcessing, get_best_epoch_results_path
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
@pytest.mark.parametrize("perform_validation_set_inference", [True, False])
@pytest.mark.parametrize("perform_test_set_inference", [True, False])
def test_model_inference_train_and_test(test_output_dirs: OutputFolderForTests,
                                        perform_cross_validation: bool,
                                        perform_training_set_inference: bool,
                                        perform_validation_set_inference: bool,
                                        perform_test_set_inference: bool) -> None:
    run_model_inference_train_and_test(test_output_dirs,
                                       perform_cross_validation,
                                       perform_training_set_inference,
                                       perform_validation_set_inference,
                                       perform_test_set_inference,
                                       False,
                                       False,
                                       False,
                                       ModelProcessing.DEFAULT)


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
@pytest.mark.parametrize("perform_training_set_inference", [True, False])
@pytest.mark.parametrize("perform_validation_set_inference", [True, False])
@pytest.mark.parametrize("perform_test_set_inference", [True, False])
def test_ensemble_model_inference_train_and_test(test_output_dirs: OutputFolderForTests,
                                                 perform_training_set_inference: bool,
                                                 perform_validation_set_inference: bool,
                                                 perform_test_set_inference: bool) -> None:
    run_model_inference_train_and_test(test_output_dirs,
                                       True,
                                       perform_training_set_inference,
                                       perform_validation_set_inference,
                                       perform_test_set_inference,
                                       False,
                                       False,
                                       False,
                                       ModelProcessing.ENSEMBLE_CREATION)


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
@pytest.mark.parametrize("perform_ensemble_child_training_set_inference", [True, False])
@pytest.mark.parametrize("perform_ensemble_child_validation_set_inference", [True, False])
@pytest.mark.parametrize("perform_ensemble_child_test_set_inference", [True, False])
def test_ensemble_child_model_inference_train_and_test(test_output_dirs: OutputFolderForTests,
                                                       perform_ensemble_child_training_set_inference: bool,
                                                       perform_ensemble_child_validation_set_inference: bool,
                                                       perform_ensemble_child_test_set_inference: bool) -> None:
    run_model_inference_train_and_test(test_output_dirs,
                                       True,
                                       False,
                                       False,
                                       False,
                                       perform_ensemble_child_training_set_inference,
                                       perform_ensemble_child_validation_set_inference,
                                       perform_ensemble_child_test_set_inference,
                                       ModelProcessing.DEFAULT)


def run_model_inference_train_and_test(test_output_dirs: OutputFolderForTests,
                                       perform_cross_validation: bool,
                                       perform_training_set_inference: bool,
                                       perform_validation_set_inference: bool,
                                       perform_test_set_inference: bool,
                                       perform_ensemble_child_training_set_inference: bool,
                                       perform_ensemble_child_validation_set_inference: bool,
                                       perform_ensemble_child_test_set_inference: bool,
                                       model_proc: ModelProcessing) -> None:
    config = DummyModel()
    config.crop_size = (29, 29, 29)
    config.number_of_cross_validation_splits = 2 if perform_cross_validation else 0
    config.perform_training_set_inference = perform_training_set_inference
    config.perform_validation_set_inference = perform_validation_set_inference
    config.perform_test_set_inference = perform_test_set_inference
    config.perform_ensemble_child_training_set_inference = perform_ensemble_child_training_set_inference
    config.perform_ensemble_child_validation_set_inference = perform_ensemble_child_validation_set_inference
    config.perform_ensemble_child_test_set_inference = perform_ensemble_child_test_set_inference
    # Plotting crashes with random TCL errors on Windows, disable that for Windows PR builds.
    config.is_plotting_enabled = common_util.is_linux()

    config.set_output_to(test_output_dirs.root_dir)
    config.local_dataset = full_ml_test_data_path()

    checkpoint_path = config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    create_model_and_store_checkpoint(config, checkpoint_path)
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)
    checkpoint_handler.additional_training_done()

    with mock.patch("InnerEye.ML.model_testing.PARENT_RUN_CONTEXT", Mock()) as m:
        metrics = MLRunner(config).model_inference_train_and_test(
            checkpoint_handler=checkpoint_handler,
            model_proc=model_proc)

    if perform_cross_validation and model_proc == ModelProcessing.DEFAULT:
        named_metrics = \
            {
                ModelExecutionMode.TRAIN: perform_ensemble_child_training_set_inference,
                ModelExecutionMode.TEST: perform_ensemble_child_test_set_inference,
                ModelExecutionMode.VAL: perform_ensemble_child_validation_set_inference
            }
    else:
        named_metrics = \
            {
                ModelExecutionMode.TRAIN: perform_training_set_inference,
                ModelExecutionMode.TEST: perform_test_set_inference,
                ModelExecutionMode.VAL: perform_validation_set_inference
            }

    error = ''
    expected_upload_folder_count = 0
    for mode, flag in named_metrics.items():
        if mode in metrics:
            metric = metrics[mode]
            assert isinstance(metric, InferenceMetricsForSegmentation)
        if mode in metrics and not flag:
            error = error + f"Error: {mode.value} cannot be not None."
        elif mode not in metrics and flag:
            error = error + f"Error: {mode.value} cannot be None."
        results_folder = config.outputs_folder / get_best_epoch_results_path(mode, model_proc)
        folder_exists = results_folder.is_dir()
        assert folder_exists == flag
        if flag and model_proc == ModelProcessing.ENSEMBLE_CREATION:
            expected_upload_folder_count = expected_upload_folder_count + 1
            expected_name = get_best_epoch_results_path(mode, ModelProcessing.DEFAULT)
            m.upload_folder.assert_any_call(name=str(expected_name), path=str(results_folder))
    if len(error):
        raise ValueError(error)

    assert m.upload_folder.call_count == expected_upload_folder_count


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
