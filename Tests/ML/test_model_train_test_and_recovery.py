#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import shutil
import pytest
import os

from InnerEye.Common.metrics_dict import MetricType
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import ModelExecutionMode, create_checkpoint_path
from InnerEye.ML.configs.classification.DummyClassification import DummyClassification
from InnerEye.ML.model_training import model_train
from InnerEye.ML.model_testing import model_test
from InnerEye.ML.utils.run_recovery import RunRecovery
from InnerEye.ML.metrics import InferenceMetricsForClassification

from Tests.ML.util import get_default_checkpoint_handler


@pytest.mark.parametrize("mean_teacher_model", [True, False])
def test_recover_testing_from_run_recovery(mean_teacher_model: bool,
                                            test_output_dirs: OutputFolderForTests) -> None:
    """
    Checks that inference results are the same whether from a checkpoint in the same run, from a run recovery or from a
    local_weights_path param.
    """
    # Train for 4 epochs
    config = DummyClassification()
    if mean_teacher_model:
        config.mean_teacher_alpha = 0.999
    config.set_output_to(test_output_dirs.root_dir / "original")
    os.makedirs(str(config.outputs_folder))
    config.save_start_epoch = 2
    config.save_step_epochs = 2

    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)
    train_results = model_train(config, checkpoint_handler=checkpoint_handler)
    assert len(train_results.learning_rates_per_epoch) == config.num_epochs

    # Run inference on this
    test_results = model_test(config=config, data_split=ModelExecutionMode.TEST, checkpoint_handler=checkpoint_handler)
    assert isinstance(test_results, InferenceMetricsForClassification)
    assert list(test_results.epochs.keys()) == [config.num_epochs]

    # Mimic using a run recovery and see if it is the same
    config_run_recovery = DummyClassification()
    if mean_teacher_model:
        config_run_recovery.mean_teacher_alpha = 0.999
    config_run_recovery.set_output_to(test_output_dirs.root_dir / "run_recovery")
    os.makedirs(str(config_run_recovery.outputs_folder))

    checkpoint_handler_run_recovery = get_default_checkpoint_handler(model_config=config_run_recovery,
                                                                     project_root=test_output_dirs.root_dir)
    # make it seem like run recovery objects have been downloaded
    checkpoint_root = config_run_recovery.checkpoint_folder / "recovered"
    shutil.copytree(str(config.checkpoint_folder), str(checkpoint_root))
    checkpoint_handler_run_recovery.run_recovery = RunRecovery([checkpoint_root])
    test_results_run_recovery = model_test(config_run_recovery, data_split=ModelExecutionMode.TEST,
                                           checkpoint_handler=checkpoint_handler_run_recovery)
    assert isinstance(test_results_run_recovery, InferenceMetricsForClassification)
    assert list(test_results_run_recovery.epochs.keys()) == [config.num_epochs]
    assert test_results.epochs[config.num_epochs].values()[MetricType.CROSS_ENTROPY.value] == \
           test_results_run_recovery.epochs[config.num_epochs].values()[MetricType.CROSS_ENTROPY.value]

    # Run inference with the local checkpoints
    config_local_weights = DummyClassification()
    if mean_teacher_model:
        config_local_weights.mean_teacher_alpha = 0.999
    config_local_weights.set_output_to(test_output_dirs.root_dir / "local_weights_path")
    os.makedirs(str(config_local_weights.outputs_folder))

    local_weights_path = test_output_dirs.root_dir / "local_weights_file.pth"
    shutil.copyfile(str(create_checkpoint_path(config.checkpoint_folder, epoch=config.num_epochs)),
                    local_weights_path)
    config_local_weights.local_weights_path = local_weights_path

    checkpoint_handler_local_weights = get_default_checkpoint_handler(model_config=config_local_weights,
                                                                      project_root=test_output_dirs.root_dir)
    checkpoint_handler_local_weights.discover_and_download_checkpoints_from_previous_runs()
    test_results_local_weights = model_test(config_local_weights, data_split=ModelExecutionMode.TEST,
                                            checkpoint_handler=checkpoint_handler_local_weights)
    assert isinstance(test_results_local_weights, InferenceMetricsForClassification)
    assert list(test_results_local_weights.epochs.keys()) == [0]
    assert test_results.epochs[config.num_epochs].values()[MetricType.CROSS_ENTROPY.value] == \
           test_results_local_weights.epochs[0].values()[MetricType.CROSS_ENTROPY.value]
