#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any

import pytest
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import (AUTOSAVE_CHECKPOINT_CANDIDATES, LAST_CHECKPOINT_FILE_NAME,
                                LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, RECOVERY_CHECKPOINT_FILE_NAME)
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.lightning_base import InnerEyeContainer
from InnerEye.ML.lightning_helpers import load_from_checkpoint_and_adjust_for_inference
from InnerEye.ML.lightning_models import create_lightning_model
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_training import create_lightning_trainer
from InnerEye.ML.utils.checkpoint_handling import (cleanup_checkpoints, find_recovery_checkpoint)
from Tests.ML.configs.ClassificationModelForTesting import ClassificationModelForTesting
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import machine_has_gpu

FIXED_EPOCH = 42
FIXED_GLOBAL_STEP = 4242


def create_model_and_store_checkpoint(config: ModelConfigBase, checkpoint_path: Path,
                                      weights_only: bool = True) -> None:
    """
    Creates a Lightning model for the given model configuration, and stores it as a checkpoint file.
    If a GPU is available, the model is moved to the GPU before storing.
    The trainer properties `current_epoch` and `global_step` are set to fixed non-default values.
    :param config: The model configuration.
    :param checkpoint_path: The path and filename of the checkpoint file.
    """
    container = InnerEyeContainer(config)
    trainer, _ = create_lightning_trainer(container)
    model = create_lightning_model(config)
    if machine_has_gpu:
        model = model.cuda()  # type: ignore
    trainer.model = model
    # Before saving, the values for epoch and step are incremented. Save them here in such a way that we can assert
    # easily later. We can't mock that because otherwise the mock object would be written to disk (that fails)
    trainer.fit_loop.current_epoch = FIXED_EPOCH - 1  # type: ignore
    trainer.fit_loop.global_step = FIXED_GLOBAL_STEP - 1  # type: ignore
    # In PL, it is the Trainer's responsibility to save the model. Checkpoint handling refers back to the trainer
    # to get a save_func. Mimicking that here.
    trainer.save_checkpoint(checkpoint_path, weights_only=weights_only)


@pytest.mark.cpu_and_gpu
@pytest.mark.parametrize("config_cls", [DummyModel, ClassificationModelForTesting])
@pytest.mark.parametrize("use_gpu", [True, False] if machine_has_gpu else [False])
def test_create_model_from_lightning_checkpoint(test_output_dirs: OutputFolderForTests,
                                                config_cls: Any,
                                                use_gpu: bool) -> None:
    config = config_cls()
    config.use_model_parallel = True
    config.max_num_gpus = -1 if use_gpu else 0
    # Check that loading from an invalid checkpoint raises an error
    with pytest.raises(FileNotFoundError):
        checkpoint_path = test_output_dirs.root_dir / "nonexist.ckpt"
        load_from_checkpoint_and_adjust_for_inference(config, checkpoint_path)

    checkpoint_path = test_output_dirs.root_dir / "checkpoint.ckpt"
    create_model_and_store_checkpoint(config, checkpoint_path=checkpoint_path)

    if isinstance(config, SegmentationModelBase):
        assert config._test_output_size is None
        assert config._train_output_size is None
    # method to get all devices of a model
    loaded_model = load_from_checkpoint_and_adjust_for_inference(config, checkpoint_path)
    # Information about epoch and global step must be present in the message that on_checkpoint_load writes
    assert str(FIXED_EPOCH) in loaded_model.checkpoint_loading_message
    assert str(FIXED_GLOBAL_STEP) in loaded_model.checkpoint_loading_message
    assert loaded_model is not None
    if isinstance(config, SegmentationModelBase):
        assert config._test_output_size is not None
        assert config._train_output_size is not None
        if use_gpu:
            # Check that a model summary for segmentation models was created, which is necessary for model partitioning.
            assert loaded_model.model.summary is not None
    devices = set(p.device for p in loaded_model.parameters())
    if use_gpu:
        assert torch.device("cuda", 0) in devices
    else:
        assert len(devices) == 1
        assert torch.device("cpu") in devices, "Model should have been mapped to CPU."


def test_checkpoint_path() -> None:
    assert LAST_CHECKPOINT_FILE_NAME == ModelCheckpoint.CHECKPOINT_NAME_LAST


def test_recovery_checkpoints_fails(test_output_dirs: OutputFolderForTests) -> None:
    """
    Using old recovering checkpoints is not supported, and should raise an error.
    """
    checkpoint_folder = test_output_dirs.root_dir
    assert find_recovery_checkpoint(checkpoint_folder) is None
    (checkpoint_folder / RECOVERY_CHECKPOINT_FILE_NAME).touch()
    with pytest.raises(ValueError) as ex:
        find_recovery_checkpoint(checkpoint_folder)
    assert "The legacy recovery checkpoint setup is no longer supported." in str(ex)


def test_find_all_recovery_checkpoints(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if the search for recovery checkpoints respects the correct order of files
    """
    checkpoint_folder = test_output_dirs.root_dir
    # If the checkpoint folder only contains a single checkpoint file of whatever kind, return that.
    single_files = [*AUTOSAVE_CHECKPOINT_CANDIDATES, LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX]
    for i, file in enumerate(single_files):
        subfolder = checkpoint_folder / str(i)
        subfolder.mkdir()
        (subfolder / file).touch()
        result = find_recovery_checkpoint(subfolder)
        assert result is not None
        assert result.name == file

    # If both "autosave" and "best_checkpoint" are present, return autosave
    both = checkpoint_folder / "both"
    both.mkdir()
    for file in single_files:
        (both / file).touch()
    result_both = find_recovery_checkpoint(both)
    assert result_both is not None
    assert result_both.name == AUTOSAVE_CHECKPOINT_CANDIDATES[0]


def test_keep_best_checkpoint(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if the logic to keep only one checkpoint file works as expected.
    """
    folder = test_output_dirs.root_dir
    with pytest.raises(FileNotFoundError) as ex:
        cleanup_checkpoints(folder)
    assert "Checkpoint file" in str(ex)
    # Create a folder with a "last" and "autosave" checkpoint, as they come out of the trainer loop.
    last = folder / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    last.touch()
    for autosave in AUTOSAVE_CHECKPOINT_CANDIDATES:
        (folder / autosave).touch()
    assert len(list(folder.glob("*"))) > 1
    cleanup_checkpoints(folder)
    # All code outside the trainer loop assumes that there is a checkpoint with this name. The constant actually
    # matches "last.ckpt", but the constant is kept to reduce code changes from the legacy behaviour.
    expected = folder / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    assert expected.is_file()
    # The autosave checkpoint should be deleted after training, only the single best checkpoint should remain
    assert len(list(folder.glob("*"))) == 1
