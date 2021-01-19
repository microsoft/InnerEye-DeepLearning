#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pytest
import torch
import os

from torch.optim import Optimizer
from torch import nn
from typing import Callable

from InnerEye.ML.utils.model_util import ModelAndInfo
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.Common.common_util import ModelExecutionMode
from InnerEye.ML.models.architectures.base_model import BaseModel
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.models.parallel.data_parallel import DataParallelModel
from Tests.ML.configs.ClassificationModelForTesting import ClassificationModelForTesting
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import no_gpu_available
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path


@pytest.mark.parametrize("config, checkpoint_path",
                         [(DummyModel(), "checkpoints/1_checkpoint.pth.tar"),
                          (ClassificationModelForTesting(),
                           "classification_data_generated_random/checkpoints/1_checkpoint.pth.tar")])
def test_try_create_model_and_load_from_checkpoint(config: ModelConfigBase, checkpoint_path: str) -> None:
    # no checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=ModelExecutionMode.TEST,
                                  checkpoint_path=None)

    with pytest.raises(ValueError):
        model_and_info.model

    model_loaded = model_and_info.try_create_model_and_load_from_checkpoint()
    assert model_loaded
    if isinstance(config, SegmentationModelBase):
        assert isinstance(model_and_info.model, BaseModel)
    else:
        assert isinstance(model_and_info.model, DeviceAwareModule)

    # Invalid checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=ModelExecutionMode.TEST,
                                  checkpoint_path=full_ml_test_data_path("non_exist.pth.tar"))
    model_loaded = model_and_info.try_create_model_and_load_from_checkpoint()
    assert not model_loaded
    # Current code assumes that even if this function returns False, the model itself was created, only the checkpoint
    # loading failed.
    if isinstance(config, SegmentationModelBase):
        assert isinstance(model_and_info.model, BaseModel)
    else:
        assert isinstance(model_and_info.model, DeviceAwareModule)

    # Valid checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=ModelExecutionMode.TEST,
                                  checkpoint_path=full_ml_test_data_path(checkpoint_path))
    model_loaded = model_and_info.try_create_model_and_load_from_checkpoint()
    assert model_loaded
    if isinstance(config, SegmentationModelBase):
        assert isinstance(model_and_info.model, BaseModel)
    else:
        assert isinstance(model_and_info.model, DeviceAwareModule)
    assert model_and_info.checkpoint_epoch == 1


@pytest.mark.gpu
@pytest.mark.skipif(no_gpu_available, reason="Testing shift to DataParallelModel requires a GPU")
@pytest.mark.parametrize("model_execution_mode", [ModelExecutionMode.TRAIN, ModelExecutionMode.TEST])
@pytest.mark.parametrize("config, checkpoint_path",
                         [(DummyModel(), "checkpoints/1_checkpoint.pth.tar"),
                          (ClassificationModelForTesting(),
                           "classification_data_generated_random/checkpoints/1_checkpoint.pth.tar")])
def test_try_create_model_load_from_checkpoint_and_adjust(config: ModelConfigBase, checkpoint_path: str,
                                                          model_execution_mode: ModelExecutionMode) -> None:
    config.use_gpu = True

    # no checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=model_execution_mode,
                                  checkpoint_path=None)

    with pytest.raises(ValueError):
        model_and_info.model

    model_loaded = model_and_info.try_create_model_load_from_checkpoint_and_adjust()
    assert model_loaded
    assert isinstance(model_and_info.model, DataParallelModel)

    # Invalid checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=model_execution_mode,
                                  checkpoint_path=full_ml_test_data_path("non_exist.pth.tar"))
    model_loaded = model_and_info.try_create_model_load_from_checkpoint_and_adjust()
    assert not model_loaded
    # Current code assumes that even if this function returns False, the model itself was created, only the checkpoint
    # loading failed.
    assert isinstance(model_and_info.model, DataParallelModel)

    # Valid checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=model_execution_mode,
                                  checkpoint_path=full_ml_test_data_path(checkpoint_path))
    model_loaded = model_and_info.try_create_model_load_from_checkpoint_and_adjust()
    assert model_loaded
    assert isinstance(model_and_info.model, DataParallelModel)
    assert model_and_info.checkpoint_epoch == 1


@pytest.mark.parametrize("config, checkpoint_path",
                         [(DummyModel(), "checkpoints/1_checkpoint.pth.tar"),
                          (ClassificationModelForTesting(),
                           "classification_data_generated_random/checkpoints/1_checkpoint.pth.tar")])
def test_try_create_optimizer_and_load_from_checkpoint(config: ModelConfigBase, checkpoint_path: str) -> None:
    # no checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=ModelExecutionMode.TEST,
                                  checkpoint_path=None)

    with pytest.raises(ValueError):
        model_and_info.optimizer

    model_loaded = model_and_info.try_create_model_and_load_from_checkpoint()
    assert model_loaded
    optimizer_loaded = model_and_info.try_create_optimizer_and_load_from_checkpoint()
    assert optimizer_loaded
    assert isinstance(model_and_info.optimizer, Optimizer)

    # Invalid checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=ModelExecutionMode.TEST,
                                  checkpoint_path=full_ml_test_data_path("non_exist.pth.tar"))
    model_loaded = model_and_info.try_create_model_and_load_from_checkpoint()
    assert not model_loaded
    # Current code assumes that even if this function returns False, the model itself was created, only the checkpoint
    # loading failed.
    optimizer_loaded = model_and_info.try_create_optimizer_and_load_from_checkpoint()
    assert not optimizer_loaded
    # Current code assumes that even if this function returns False,
    # the optimizer itself was created, only the checkpoint loading failed.
    assert isinstance(model_and_info.optimizer, Optimizer)

    # Valid checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=ModelExecutionMode.TEST,
                                  checkpoint_path=full_ml_test_data_path(checkpoint_path))
    model_loaded = model_and_info.try_create_model_and_load_from_checkpoint()
    assert model_loaded
    assert model_and_info.checkpoint_epoch == 1
    optimizer_loaded = model_and_info.try_create_optimizer_and_load_from_checkpoint()
    assert optimizer_loaded
    assert isinstance(model_and_info.optimizer, Optimizer)
    assert model_and_info.checkpoint_epoch == 1


@pytest.mark.parametrize("config, checkpoint_path",
                         [(ClassificationModelForTesting(),
                           "classification_data_generated_random/checkpoints/1_checkpoint.pth.tar")])
def test_try_create_mean_teacher_model_and_load_from_checkpoint(config: ModelConfigBase, checkpoint_path: str) -> None:
    config.mean_teacher_alpha = 0.999

    # no checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=ModelExecutionMode.TEST,
                                  checkpoint_path=None)

    with pytest.raises(ValueError):
        model_and_info.mean_teacher_model

    model_loaded = model_and_info.try_create_mean_teacher_model_and_load_from_checkpoint()
    assert model_loaded
    if isinstance(config, SegmentationModelBase):
        assert isinstance(model_and_info.mean_teacher_model, BaseModel)
    else:
        assert isinstance(model_and_info.mean_teacher_model, DeviceAwareModule)

    # Invalid checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=ModelExecutionMode.TEST,
                                  checkpoint_path=full_ml_test_data_path("non_exist.pth.tar"))
    model_loaded = model_and_info.try_create_mean_teacher_model_and_load_from_checkpoint()
    assert not model_loaded
    # Current code assumes that even if this function returns False, the model itself was created, only the checkpoint
    # loading failed.
    if isinstance(config, SegmentationModelBase):
        assert isinstance(model_and_info.mean_teacher_model, BaseModel)
    else:
        assert isinstance(model_and_info.mean_teacher_model, DeviceAwareModule)

    # Valid checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=ModelExecutionMode.TEST,
                                  checkpoint_path=full_ml_test_data_path(checkpoint_path))
    model_loaded = model_and_info.try_create_mean_teacher_model_and_load_from_checkpoint()
    assert model_loaded
    if isinstance(config, SegmentationModelBase):
        assert isinstance(model_and_info.mean_teacher_model, BaseModel)
    else:
        assert isinstance(model_and_info.mean_teacher_model, DeviceAwareModule)
    assert model_and_info.checkpoint_epoch == 1


@pytest.mark.parametrize("config",
                         [DummyModel(), ClassificationModelForTesting()])
def test_save_checkpoint(config: ModelConfigBase) -> None:
    """
    Test that checkpoints are saved correctly
    """

    config.mean_teacher_alpha = 0.999

    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=ModelExecutionMode.TEST,
                                  checkpoint_path=None)
    model_and_info.try_create_model_and_load_from_checkpoint()
    model_and_info.try_create_mean_teacher_model_and_load_from_checkpoint()
    model_and_info.try_create_optimizer_and_load_from_checkpoint()

    def get_constant_init_function(constant: float) -> Callable:
        def init(layer: nn.Module) -> None:
            if type(layer) == nn.Conv3d:
                layer.weight.data.fill_(constant)  # type: ignore
        return init

    assert model_and_info.mean_teacher_model is not None  # for mypy

    model_and_info.model.apply(get_constant_init_function(1.0))
    model_and_info.mean_teacher_model.apply(get_constant_init_function(2.0))

    epoch = 3

    checkpoint_path = config.get_path_to_checkpoint(epoch=epoch)
    checkpoint_dir = checkpoint_path.parent
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_and_info.save_checkpoint(epoch=epoch)

    model_and_info_restored = ModelAndInfo(config,
                                           model_execution_mode=ModelExecutionMode.TEST,
                                           checkpoint_path=config.get_path_to_checkpoint(epoch=epoch))
    model_and_info_restored.try_create_model_load_from_checkpoint_and_adjust()
    model_and_info_restored.try_create_mean_teacher_model_load_from_checkpoint_and_adjust()

    assert model_and_info_restored.mean_teacher_model is not None  # for mypy

    for module in model_and_info_restored.model.modules():
        if type(module) == nn.Conv3d:
            assert torch.equal(module.weight.detach(), torch.full_like(module.weight.detach(), 1.0))  # type: ignore

    for module in model_and_info_restored.mean_teacher_model.modules():
        if type(module) == nn.Conv3d:
            assert torch.equal(module.weight.detach(), torch.full_like(module.weight.detach(), 2.0))  # type: ignore
