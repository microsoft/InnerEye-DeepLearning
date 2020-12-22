#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pytest
import torch

from InnerEye.Common.common_util import ModelExecutionMode
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.lightning_models import create_lightning_model, create_model_from_lightning_checkpoint
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_training import create_lightning_trainer
from InnerEye.ML.models.architectures.base_model import BaseSegmentationModel
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.ML.utils.model_util import ModelAndInfo
from Tests.ML.configs.ClassificationModelForTesting import ClassificationModelForTesting
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import machine_has_gpu
from Tests.fixed_paths_for_tests import full_ml_test_data_path


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
        assert isinstance(model_and_info.model, BaseSegmentationModel)
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
        assert isinstance(model_and_info.model, BaseSegmentationModel)
    else:
        assert isinstance(model_and_info.model, DeviceAwareModule)

    # Valid checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=ModelExecutionMode.TEST,
                                  checkpoint_path=full_ml_test_data_path(checkpoint_path))
    model_loaded = model_and_info.try_create_model_and_load_from_checkpoint()
    assert model_loaded
    if isinstance(config, SegmentationModelBase):
        assert isinstance(model_and_info.model, BaseSegmentationModel)
    else:
        assert isinstance(model_and_info.model, DeviceAwareModule)
    assert model_and_info.checkpoint_epoch == 1


@pytest.mark.cpu_and_gpu
@pytest.mark.parametrize("config", [DummyModel(), ClassificationModelForTesting()])
def test_try_create_model_load_from_checkpoint_and_adjust(test_output_dirs: OutputFolderForTests,
                                                          config: ModelConfigBase) -> None:
    # Force loading of the model onto CPU, even though we write it from GPU.
    config.use_gpu = False
    trainer, _ = create_lightning_trainer(config)
    epoch = 42
    trainer.current_epoch = epoch
    checkpoint_path = test_output_dirs.root_dir / "checkpoint.ckpt"
    model = create_lightning_model(config)
    if machine_has_gpu:
        model = model.cuda()
    trainer.model = model
    # In PL, it is the Trainer's responsibility to save the model. Checkpoint handling refers back to the trainer
    # to get a save_func. Mimicking that here.
    trainer.save_checkpoint(checkpoint_path, weights_only=True)
    loaded_model = create_model_from_lightning_checkpoint(config, checkpoint_path)
    assert loaded_model is not None
    first_param = next(model.parameters())
    assert first_param.device == torch.device("cpu"), "Model should have been mapped to CPU."


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
        assert isinstance(model_and_info.mean_teacher_model, BaseSegmentationModel)
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
        assert isinstance(model_and_info.mean_teacher_model, BaseSegmentationModel)
    else:
        assert isinstance(model_and_info.mean_teacher_model, DeviceAwareModule)

    # Valid checkpoint path provided
    model_and_info = ModelAndInfo(config,
                                  model_execution_mode=ModelExecutionMode.TEST,
                                  checkpoint_path=full_ml_test_data_path(checkpoint_path))
    model_loaded = model_and_info.try_create_mean_teacher_model_and_load_from_checkpoint()
    assert model_loaded
    if isinstance(config, SegmentationModelBase):
        assert isinstance(model_and_info.mean_teacher_model, BaseSegmentationModel)
    else:
        assert isinstance(model_and_info.mean_teacher_model, DeviceAwareModule)
    assert model_and_info.checkpoint_epoch == 1
