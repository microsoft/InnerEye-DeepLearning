#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
import shutil

import pytest

from InnerEye.Common.metrics_constants import MetricType
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, ModelExecutionMode
from InnerEye.ML.configs.classification.DummyClassification import DummyClassification
from InnerEye.ML.metrics import InferenceMetricsForClassification
from InnerEye.ML.model_testing import model_test
from InnerEye.ML.utils.run_recovery import RunRecovery
from Tests.ML.util import get_default_checkpoint_handler, model_train_unittest


# @pytest.mark.parametrize("mean_teacher_model", [True, False])
@pytest.mark.parametrize("mean_teacher_model", [False])
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
    config.recovery_checkpoint_save_interval = 2

    train_results, checkpoint_handler = model_train_unittest(config, dirs=test_output_dirs)
    assert len(train_results.train_results_per_epoch()) == config.num_epochs

    # Run inference on this
    test_results = model_test(config=config, data_split=ModelExecutionMode.TEST,
                              checkpoint_paths=checkpoint_handler.get_checkpoints_to_test())
    assert isinstance(test_results, InferenceMetricsForClassification)

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
                                           checkpoint_paths=checkpoint_handler_run_recovery.get_checkpoints_to_test())
    assert isinstance(test_results_run_recovery, InferenceMetricsForClassification)
    assert test_results.metrics.values()[MetricType.CROSS_ENTROPY.value] == \
           test_results_run_recovery.metrics.values()[MetricType.CROSS_ENTROPY.value]

    # Run inference with the local checkpoints
    config_local_weights = DummyClassification()
    if mean_teacher_model:
        config_local_weights.mean_teacher_alpha = 0.999
    config_local_weights.set_output_to(test_output_dirs.root_dir / "local_weights_path")
    os.makedirs(str(config_local_weights.outputs_folder))

    local_weights_path = test_output_dirs.root_dir / "local_weights_file.pth"
    shutil.copyfile(str(config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX),
                    local_weights_path)
    config_local_weights.local_checkpoint_paths = local_weights_path

    checkpoint_handler_local_weights = get_default_checkpoint_handler(model_config=config_local_weights,
                                                                      project_root=test_output_dirs.root_dir)
    checkpoint_handler_local_weights.download_recovery_checkpoints_or_weights()
    test_results_local_weights = model_test(config_local_weights, data_split=ModelExecutionMode.TEST,
                                            checkpoint_paths=checkpoint_handler_local_weights.get_checkpoints_to_test())
    assert isinstance(test_results_local_weights, InferenceMetricsForClassification)
    assert test_results.metrics.values()[MetricType.CROSS_ENTROPY.value] == \
           test_results_local_weights.metrics.values()[MetricType.CROSS_ENTROPY.value]
