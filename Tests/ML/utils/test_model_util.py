#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import pytest
import torch

from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.lightning_models import create_lightning_model, create_model_from_lightning_checkpoint
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_training import create_lightning_trainer
from Tests.ML.configs.ClassificationModelForTesting import ClassificationModelForTesting
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import machine_has_gpu


def create_model_and_store(config: ModelConfigBase, checkpoint_path: Path) -> None:
    """
    Creates a Lightning model for the given model configuration, and stores it as a checkpoint file.
    If a GPU is available, the model is moved to the GPU before storing.
    :param config: The model configuration.
    :param checkpoint_path: The path and filename of the checkpoint file.
    """
    trainer, _ = create_lightning_trainer(config)
    model = create_lightning_model(config)
    if machine_has_gpu:
        model = model.cuda()
    trainer.model = model
    # In PL, it is the Trainer's responsibility to save the model. Checkpoint handling refers back to the trainer
    # to get a save_func. Mimicking that here.
    trainer.save_checkpoint(checkpoint_path, weights_only=True)


@pytest.mark.cpu_and_gpu
@pytest.mark.parametrize("config", [DummyModel(), ClassificationModelForTesting()])
def test_create_model_from_lightning_checkpoint(test_output_dirs: OutputFolderForTests,
                                                          config: ModelConfigBase) -> None:
    # Force loading of the model onto CPU, even though we write it from GPU.
    config.use_gpu = False

    # Check that loading from an invalid checkpoint raises an error
    with pytest.raises(FileNotFoundError):
        checkpoint_path = test_output_dirs.root_dir / "nonexist.ckpt"
        create_model_from_lightning_checkpoint(config, checkpoint_path)

    checkpoint_path = test_output_dirs.root_dir / "checkpoint.ckpt"
    create_model_and_store(config, checkpoint_path=checkpoint_path)

    loaded_model = create_model_from_lightning_checkpoint(config, checkpoint_path)
    assert loaded_model is not None
    first_param = next(loaded_model.parameters())
    # TODO check loading happens properly by comparing values
    assert first_param.device == torch.device("cpu"), "Model should have been mapped to CPU."
