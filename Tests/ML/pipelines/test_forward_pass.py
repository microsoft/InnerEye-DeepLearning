#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pytest
from pathlib import Path

from InnerEye.Common import common_util
from InnerEye.ML.deep_learning_config import DeepLearningConfig, TrainerParams
from InnerEye.ML.lightning_container import LightningContainer
from Tests.ML.util import machine_has_gpu


@pytest.mark.skipif(common_util.is_windows(), reason="Has issues on windows build")
@pytest.mark.cpu_and_gpu
@pytest.mark.parametrize("config", [DeepLearningConfig(local_dataset=Path("foo")),
                                    LightningContainer(local_dataset=Path("foo"))])
def test_use_gpu_flag(config: TrainerParams) -> None:
    """
    Test that the use_gpu flag is set correctly on both InnerEye configs and containers.
    This checks for a bug in an earlier version where it was off for containers only.
    """
    # With the default settings, the use_gpu flag should return True exactly when a GPU is
    # actually present.
    assert config.use_gpu == machine_has_gpu
    if machine_has_gpu:
        # If there is a GPU present, only setting max_num_gpus to 0 should make the use_gpu flag False.
        config.max_num_gpus = -1
        assert config.use_gpu
        config.max_num_gpus = 1
        assert config.use_gpu
        config.max_num_gpus = 0
        assert config.use_gpu is False
    else:
        # If there is no GPU at all, changing max_num_gpus should not matter
        for p in [-1, 1, 0]:
            config.max_num_gpus = p
            assert config.use_gpu is False

# @pytest.mark.azureml
# def test_mean_teacher_model(test_output_dirs: OutputFolderForTests) -> None:
#    """
#    Test training and weight updates of the mean teacher model computation.
#    """
#
#    def _get_parameters_of_model(model: DeviceAwareModule) -> Any:
#        """
#        Returns the iterator of model parameters
#        """
#        if isinstance(model, DataParallelModel):
#            return model.module.parameters()
#        else:
#            return model.parameters()
#
#    config = DummyClassification()
#    config.set_output_to(test_output_dirs.root_dir)
#    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
#                                                        project_root=test_output_dirs.root_dir)
#
#    config.num_epochs = 1
#    # Set train batch size to be arbitrary big to ensure we have only one training step
#    # i.e. one mean teacher update.
#    config.train_batch_size = 100
#    # Train without mean teacher
#    model_train(config, checkpoint_handler=checkpoint_handler)
#
#    # Retrieve the weight after one epoch
#    model_and_info = ModelAndInfo(config=config, model_execution_mode=ModelExecutionMode.TEST,
#                                  checkpoint_path=config.get_path_to_checkpoint(epoch=1))
#    model_and_info.try_create_model_and_load_from_checkpoint()
#    model = model_and_info.model
#    model_weight = next(_get_parameters_of_model(model))
#
#    # Get the starting weight of the mean teacher model
#    ml_util.set_random_seed(config.get_effective_random_seed())
#
#    model_and_info_mean_teacher = ModelAndInfo(config=config,
#                                               model_execution_mode=ModelExecutionMode.TEST,
#                                               checkpoint_path=None)
#    model_and_info_mean_teacher.try_create_model_and_load_from_checkpoint()
#
#    model_and_info_mean_teacher.try_create_mean_teacher_model_and_load_from_checkpoint()
#    mean_teach_model = model_and_info_mean_teacher.mean_teacher_model
#    assert mean_teach_model is not None  # for mypy
#    initial_weight_mean_teacher_model = next(_get_parameters_of_model(mean_teach_model))
#
#    # Now train with mean teacher and check the update of the weight
#    alpha = 0.999
#    config.mean_teacher_alpha = alpha
#    model_train(config, checkpoint_handler=checkpoint_handler)
#
#    # Retrieve weight of mean teacher model saved in the checkpoint
#    model_and_info_mean_teacher = ModelAndInfo(config=config, model_execution_mode=ModelExecutionMode.TEST,
#                                               checkpoint_path=config.get_path_to_checkpoint(1))
#    model_and_info_mean_teacher.try_create_mean_teacher_model_and_load_from_checkpoint()
#    mean_teacher_model = model_and_info_mean_teacher.mean_teacher_model
#    assert mean_teacher_model is not None  # for mypy
#    result_weight = next(_get_parameters_of_model(mean_teacher_model))
#    # Retrieve the associated student weight
#    model_and_info_mean_teacher.try_create_model_and_load_from_checkpoint()
#    student_model = model_and_info_mean_teacher.model
#    student_model_weight = next(_get_parameters_of_model(student_model))
#
#    # Assert that the student weight corresponds to the weight of a simple training without mean teacher
#    # computation
#    assert student_model_weight.allclose(model_weight)
#
#    # Check the update of the parameters
#    assert torch.all(alpha * initial_weight_mean_teacher_model + (1 - alpha) * student_model_weight == result_weight)
