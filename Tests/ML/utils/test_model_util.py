#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, LAST_CHECKPOINT_FILE_NAME, \
    LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, RECOVERY_CHECKPOINT_FILE_NAME, RECOVERY_CHECKPOINT_FILE_NAME_WITH_SUFFIX, \
    cleanup_checkpoint_folder, keep_best_checkpoint, keep_latest
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.lightning_helpers import create_lightning_model, load_from_checkpoint_and_adjust_for_inference
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_training import create_lightning_trainer
from Tests.ML.configs.ClassificationModelForTesting import ClassificationModelForTesting
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import machine_has_gpu

FIXED_EPOCH = 42
FIXED_GLOBAL_STEP = 4242


def create_model_and_store_checkpoint(config: ModelConfigBase, checkpoint_path: Path) -> None:
    """
    Creates a Lightning model for the given model configuration, and stores it as a checkpoint file.
    If a GPU is available, the model is moved to the GPU before storing.
    The trainer properties `current_epoch` and `global_step` are set to fixed non-default values.
    :param config: The model configuration.
    :param checkpoint_path: The path and filename of the checkpoint file.
    """
    trainer, _ = create_lightning_trainer(config)
    model = create_lightning_model(config)
    if machine_has_gpu:
        model = model.cuda()  # type: ignore
    trainer.model = model
    # Before saving, the values for epoch and step are incremented. Save them here in such a way that we can assert
    # easily later.
    trainer.current_epoch = FIXED_EPOCH - 1
    trainer.global_step = FIXED_GLOBAL_STEP - 1
    # In PL, it is the Trainer's responsibility to save the model. Checkpoint handling refers back to the trainer
    # to get a save_func. Mimicking that here.
    trainer.save_checkpoint(checkpoint_path, weights_only=True)


@pytest.mark.cpu_and_gpu
@pytest.mark.parametrize("config_cls", [DummyModel, ClassificationModelForTesting])
@pytest.mark.parametrize("use_gpu", [True, False] if machine_has_gpu else [False])
def test_create_model_from_lightning_checkpoint(test_output_dirs: OutputFolderForTests,
                                                config_cls: Any,
                                                use_gpu: bool) -> None:
    config = config_cls()
    config.use_model_parallel = True
    config.use_gpu = use_gpu
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


def test_keep_latest(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if the logic to keep only the most recently modified file works.
    """
    folder = test_output_dirs.root_dir
    prefix = "foo"
    pattern = prefix + "*"
    file1 = folder / (prefix + ".txt")
    file2 = folder / (prefix + "2.txt")
    # No file present yet
    assert keep_latest(folder, pattern) is None
    # Single file present: This should be returned.
    file1.touch()
    # Without sleeping, the test can fail in Azure build agents
    time.sleep(0.1)
    latest = keep_latest(folder, pattern)
    assert latest == file1
    assert latest.is_file()
    # Two files present: keep file2, file1 should be deleted
    file2.touch()
    time.sleep(0.1)
    latest = keep_latest(folder, pattern)
    assert latest == file2
    assert latest.is_file()
    assert not file1.is_file()
    # Add file1 again: Now this one should be the most recent one
    file1.touch()
    time.sleep(0.1)
    latest = keep_latest(folder, pattern)
    assert latest == file1
    assert latest.is_file()
    assert not file2.is_file()


def test_keep_best_checkpoint(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if the logic to keep only one checkpoint file works as expected.
    """
    folder = test_output_dirs.root_dir
    with pytest.raises(FileNotFoundError) as ex:
        keep_best_checkpoint(folder)
    assert "Checkpoint file" in str(ex)
    last = folder / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    last.touch()
    actual = keep_best_checkpoint(folder)
    assert not last.is_file(), "Checkpoint file should have been renamed"
    expected = folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    assert actual == expected
    assert actual.is_file()


def test_cleanup_checkpoints1(test_output_dirs: OutputFolderForTests) -> None:
    folder = test_output_dirs.root_dir
    with pytest.raises(FileNotFoundError) as ex:
        cleanup_checkpoint_folder(folder)
    assert "Checkpoint file" in str(ex)
    # Single checkpoint file, nothing else: This file should be rename to best_checkpoint
    last = folder / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    last.touch()
    cleanup_checkpoint_folder(folder)
    assert len(list(folder.glob("*"))) == 1
    assert (folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX).is_file()


def test_cleanup_checkpoints2(test_output_dirs: OutputFolderForTests) -> None:
    # Single checkpoint file and two recovery checkpoints: Should keep the last and rename it.
    folder = test_output_dirs.root_dir
    last = folder / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    last.touch()
    (folder / f"{RECOVERY_CHECKPOINT_FILE_NAME}-v0").touch()
    (folder / f"{RECOVERY_CHECKPOINT_FILE_NAME}-v1").touch()
    cleanup_checkpoint_folder(folder)
    assert len(list(folder.glob("*"))) == 2
    assert (folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX).is_file()
    assert (folder / RECOVERY_CHECKPOINT_FILE_NAME_WITH_SUFFIX).is_file()
