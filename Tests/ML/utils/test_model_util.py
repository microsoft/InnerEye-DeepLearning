#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pytest

from pathlib import Path
from torch.optim import Optimizer

from InnerEye.ML.utils.model_util import ModelAndInfo
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.Common.common_util import ModelExecutionMode
from InnerEye.ML.models.architectures.base_model import BaseModel
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.config import SegmentationModelBase
from Tests.ML.configs.ClassificationModelForTesting import ClassificationModelForTesting
from Tests.ML.configs.DummyModel import DummyModel
from Tests.fixed_paths_for_tests import full_ml_test_data_path


@pytest.mark.parametrize("config, checkpoint_path",
                         [(DummyModel(), "checkpoints/1_checkpoint.pth.tar"),
                          (ClassificationModelForTesting(),
                           "classification_data_generated_random/checkpoints/1_checkpoint.pth.tar")])
def test_try_create_model_and_load_from_checkpoint(config: ModelConfigBase, checkpoint_path: str) -> None:
    # no checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=ModelExecutionMode.TEST,
                                  is_mean_teacher=False,
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
                                  is_mean_teacher=False,
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
                                  is_mean_teacher=False,
                                  checkpoint_path=full_ml_test_data_path(checkpoint_path))
    model_loaded = model_and_info.try_create_model_and_load_from_checkpoint()
    assert model_loaded
    if isinstance(config, SegmentationModelBase):
        assert isinstance(model_and_info.model, BaseModel)
    else:
        assert isinstance(model_and_info.model, DeviceAwareModule)
    assert model_and_info.checkpoint_epoch == 1


@pytest.mark.parametrize("config, checkpoint_path",
                         [(DummyModel(), "checkpoints/1_checkpoint.pth.tar"),
                          (ClassificationModelForTesting(),
                           "classification_data_generated_random/checkpoints/1_checkpoint.pth.tar")])
def test_try_create_optimizer_and_load_from_checkpoint(config: ModelConfigBase, checkpoint_path: str) -> None:
    # no checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=ModelExecutionMode.TEST,
                                  is_mean_teacher=False,
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
                                  is_mean_teacher=False,
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
                                  is_mean_teacher=False,
                                  checkpoint_path=full_ml_test_data_path(checkpoint_path))
    model_loaded = model_and_info.try_create_model_and_load_from_checkpoint()
    assert model_loaded
    assert model_and_info.checkpoint_epoch == 1
    optimizer_loaded = model_and_info.try_create_optimizer_and_load_from_checkpoint()
    assert optimizer_loaded
    assert isinstance(model_and_info.optimizer, Optimizer)
    assert model_and_info.checkpoint_epoch == 1
