#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, no_type_check

import pytest
import torch

from InnerEye.Common import common_util
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.configs.classification.DummyClassification import DummyClassification
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.model_training import model_train
from InnerEye.ML.models.architectures.base_model import BaseSegmentationModel, CropSizeConstraints
from InnerEye.ML.models.parallel.data_parallel import DataParallelModel
from InnerEye.ML.utils import ml_util
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from Tests.ML.util import get_default_checkpoint_handler, machine_has_gpu


class SimpleModel(BaseSegmentationModel):
    def __init__(self, input_channels: int, channels: list, n_classes: int, kernel_size: int,
                 crop_size_constraints: CropSizeConstraints = None):
        super().__init__(input_channels=input_channels, name="SimpleModel", crop_size_constraints=crop_size_constraints)
        self.channels = channels
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.model = torch.nn.Sequential(
            torch.nn.Conv3d(input_channels, channels[0], kernel_size=self.kernel_size),
            torch.nn.ConvTranspose3d(channels[0], n_classes, kernel_size=self.kernel_size)
        )

    def forward(self, x: Any) -> Any:  # type: ignore
        x = self.model(x)
        return x


@pytest.mark.skipif(common_util.is_windows(), reason="Has issues on windows build")
@pytest.mark.cpu_and_gpu
@pytest.mark.parametrize("use_gpu_override", [False, True])
def test_use_gpu_flag(use_gpu_override: bool) -> None:
    config = DeepLearningConfig(should_validate=False)
    # On a model that does not have a use_gpu_override, the use_gpu flag should return True exactly when a GPU is
    # actually present.
    assert config.use_gpu == machine_has_gpu
    if machine_has_gpu:
        # If a GPU is present, the use_gpu flag should exactly return whatever the override says
        # (we can run in CPU mode even on a GPU)
        config.use_gpu = use_gpu_override
        assert config.use_gpu == use_gpu_override
    else:
        if use_gpu_override:
            # We are on a machine without a GPU, but the override says we should use the GPU: fail.
            with pytest.raises(ValueError) as ex:
                config.use_gpu = use_gpu_override
            assert "use_gpu to True if there is not CUDA capable GPU present" in str(ex)
        else:
            config.use_gpu = use_gpu_override
            assert config.use_gpu == use_gpu_override


# @pytest.mark.azureml
# TODO antonsc: re-enable test and type checking once we have mean teacher in place again
@no_type_check
def disabled_test_mean_teacher_model(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test training and weight updates of the mean teacher model computation.
    """

    def _get_parameters_of_model(model: DeviceAwareModule) -> Any:
        """
        Returns the iterator of model parameters
        """
        if isinstance(model, DataParallelModel):
            return model.module.parameters()
        else:
            return model.parameters()

    config = DummyClassification()
    config.set_output_to(test_output_dirs.root_dir)
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)

    config.num_epochs = 1
    # Set train batch size to be arbitrary big to ensure we have only one training step
    # i.e. one mean teacher update.
    config.train_batch_size = 100
    # Train without mean teacher
    model_train(config, checkpoint_handler=checkpoint_handler)

    # Retrieve the weight after one epoch
    model_and_info = ModelAndInfo(config=config, model_execution_mode=ModelExecutionMode.TEST,
                                  checkpoint_path=config.get_path_to_checkpoint(epoch=1))
    model_and_info.try_create_model_and_load_from_checkpoint()
    model = model_and_info.model
    model_weight = next(_get_parameters_of_model(model))

    # Get the starting weight of the mean teacher model
    ml_util.set_random_seed(config.get_effective_random_seed())

    model_and_info_mean_teacher = ModelAndInfo(config=config,
                                               model_execution_mode=ModelExecutionMode.TEST,
                                               checkpoint_path=None)
    model_and_info_mean_teacher.try_create_model_and_load_from_checkpoint()

    model_and_info_mean_teacher.try_create_mean_teacher_model_and_load_from_checkpoint()
    mean_teach_model = model_and_info_mean_teacher.mean_teacher_model
    assert mean_teach_model is not None  # for mypy
    initial_weight_mean_teacher_model = next(_get_parameters_of_model(mean_teach_model))

    # Now train with mean teacher and check the update of the weight
    alpha = 0.999
    config.mean_teacher_alpha = alpha
    model_train(config, checkpoint_handler=checkpoint_handler)

    # Retrieve weight of mean teacher model saved in the checkpoint
    model_and_info_mean_teacher = ModelAndInfo(config=config, model_execution_mode=ModelExecutionMode.TEST,
                                               checkpoint_path=config.get_path_to_checkpoint(1))
    model_and_info_mean_teacher.try_create_mean_teacher_model_and_load_from_checkpoint()
    mean_teacher_model = model_and_info_mean_teacher.mean_teacher_model
    assert mean_teacher_model is not None  # for mypy
    result_weight = next(_get_parameters_of_model(mean_teacher_model))
    # Retrieve the associated student weight
    model_and_info_mean_teacher.try_create_model_and_load_from_checkpoint()
    student_model = model_and_info_mean_teacher.model
    student_model_weight = next(_get_parameters_of_model(student_model))

    # Assert that the student weight corresponds to the weight of a simple training without mean teacher
    # computation
    assert student_model_weight.allclose(model_weight)

    # Check the update of the parameters
    assert torch.all(alpha * initial_weight_mean_teacher_model + (1 - alpha) * student_model_weight == result_weight)
