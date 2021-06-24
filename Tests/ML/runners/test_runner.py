#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import time

import pytest

from InnerEye.Common import common_util
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, ModelExecutionMode
from InnerEye.ML.metrics import InferenceMetricsForSegmentation
from InnerEye.ML.run_ml import MLRunner
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import get_default_checkpoint_handler
from Tests.ML.utils.test_model_util import create_model_and_store_checkpoint


@pytest.mark.parametrize("perform_cross_validation", [True, False])
@pytest.mark.parametrize("perform_training_set_inference", [True, False])
@pytest.mark.parametrize("perform_validation_set_inference", [True, False])
@pytest.mark.parametrize("perform_test_set_inference", [True, False])
def test_model_inference_train_and_test(test_output_dirs: OutputFolderForTests,
                                        perform_cross_validation: bool,
                                        perform_training_set_inference: bool,
                                        perform_validation_set_inference: bool,
                                        perform_test_set_inference: bool) -> None:
    config = DummyModel()
    config.number_of_cross_validation_splits = 2 if perform_cross_validation else 0
    config.perform_training_set_inference = perform_training_set_inference
    config.perform_validation_set_inference = perform_validation_set_inference
    config.perform_test_set_inference = perform_test_set_inference
    # Plotting crashes with random TCL errors on Windows, disable that for Windows PR builds.
    config.is_plotting_enabled = common_util.is_linux()

    config.set_output_to(test_output_dirs.root_dir)
    config.local_dataset = full_ml_test_data_path()

    checkpoint_path = config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    create_model_and_store_checkpoint(config, checkpoint_path)
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)
    checkpoint_handler.additional_training_done()
    test_metrics, val_metrics, train_metrics = MLRunner(config).model_inference_train_and_test(
        checkpoint_handler=checkpoint_handler)
    named_metrics = [(ModelExecutionMode.TEST.value, test_metrics, perform_test_set_inference),
                     (ModelExecutionMode.VAL.value, val_metrics, perform_validation_set_inference),
                     (ModelExecutionMode.TRAIN.value, train_metrics, perform_training_set_inference)]
    error = ''
    for name, metric, flag in named_metrics:
        if perform_cross_validation:
            if metric is not None:
                error = error + f"Error: {name} cannot be not None."
        else:
            if metric is not None and not flag:
                error = error + f"Error: {name} cannot be not None."
            elif metric is None and flag:
                error = error + f"Error: {name} cannot be None."
    if len(error):
        raise ValueError(error)

    if test_metrics is not None:
        assert isinstance(test_metrics, InferenceMetricsForSegmentation)
    if val_metrics is not None:
        assert isinstance(val_metrics, InferenceMetricsForSegmentation)
    if train_metrics is not None:
        assert isinstance(train_metrics, InferenceMetricsForSegmentation)

    epoch_folder_name = common_util.BEST_EPOCH_FOLDER_NAME
    for folder, _, flag in named_metrics:
        results_folder = config.outputs_folder / epoch_folder_name / folder
        folder_exists = results_folder.is_dir()
        if perform_cross_validation:
            assert not folder_exists
        else:
            assert folder_exists == flag


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
