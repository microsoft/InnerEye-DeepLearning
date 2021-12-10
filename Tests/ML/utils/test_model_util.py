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
    LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, RECOVERY_CHECKPOINT_FILE_NAME
from InnerEye.ML.utils.checkpoint_handling import create_best_checkpoint, extract_latest_checkpoint_and_epoch, \
    find_all_recovery_checkpoints, \
    find_recovery_checkpoint_and_epoch
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.lightning_base import InnerEyeContainer
from InnerEye.ML.lightning_helpers import load_from_checkpoint_and_adjust_for_inference
from InnerEye.ML.lightning_models import create_lightning_model
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_training import create_lightning_trainer
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
    # easily later.
    trainer.current_epoch = FIXED_EPOCH - 1  # type: ignore
    trainer.global_step = FIXED_GLOBAL_STEP - 1  # type: ignore
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


def test_find_all_recovery_checkpoints(test_output_dirs: OutputFolderForTests) -> None:
    checkpoint_folder = test_output_dirs.root_dir
    # No recovery yet available
    (checkpoint_folder / "epoch=2.ckpt").touch()
    assert find_all_recovery_checkpoints(checkpoint_folder) is None
    # Add recovery file to fake folder
    file_list = ["recovery_epoch=1.ckpt", "recovery.ckpt"]
    for f in file_list:
        (checkpoint_folder / f).touch()
    found_file_names = set([f.stem for f in find_all_recovery_checkpoints(checkpoint_folder)])  # type: ignore
    assert len(found_file_names.difference(found_file_names)) == 0


def test_find_latest_checkpoint_and_epoch() -> None:
    file_list = [Path("epoch=1.ckpt"), Path("epoch=3.ckpt"), Path("epoch=2.ckpt")]
    assert Path("epoch=3.ckpt"), 3 == extract_latest_checkpoint_and_epoch(file_list)
    invalid_file_list = [Path("epoch.ckpt"), Path("epoch=3.ckpt"), Path("epoch=2.ckpt")]
    with pytest.raises(IndexError):
        extract_latest_checkpoint_and_epoch(invalid_file_list)


def test_find_recovery_checkpoint(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if the logic to keep only the most recently modified file works.
    """
    folder = test_output_dirs.root_dir
    prefix = RECOVERY_CHECKPOINT_FILE_NAME
    file1 = folder / (prefix + "epoch=1.txt")
    file2 = folder / (prefix + "epoch=2.txt")
    # No file present yet
    assert find_recovery_checkpoint_and_epoch(folder) is None
    # Single file present: This should be returned.
    file1.touch()
    # Without sleeping, the test can fail in Azure build agents
    time.sleep(0.1)
    recovery = find_recovery_checkpoint_and_epoch(folder)
    assert recovery is not None
    latest_checkpoint, latest_epoch = recovery
    assert latest_checkpoint == file1
    assert latest_epoch == 1
    assert latest_checkpoint.is_file()
    # Two files present: keep file2 should be returned
    file2.touch()
    time.sleep(0.1)
    recovery = find_recovery_checkpoint_and_epoch(folder)
    assert recovery is not None
    latest_checkpoint, latest_epoch = recovery
    assert latest_checkpoint == file2
    assert latest_checkpoint.is_file()
    assert latest_epoch == 2
    # Add file1 again: file should should still be returned as it has the
    # highest epoch number
    file1.touch()
    time.sleep(0.1)
    recovery = find_recovery_checkpoint_and_epoch(folder)
    assert recovery is not None
    latest_checkpoint, latest_epoch = recovery
    assert latest_checkpoint == file2
    assert latest_checkpoint.is_file()
    assert latest_epoch == 2


def test_keep_best_checkpoint(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if the logic to keep only one checkpoint file works as expected.
    """
    folder = test_output_dirs.root_dir
    with pytest.raises(FileNotFoundError) as ex:
        create_best_checkpoint(folder)
    assert "Checkpoint file" in str(ex)
    last = folder / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    last.touch()
    actual = create_best_checkpoint(folder)
    assert not last.is_file(), "Checkpoint file should have been renamed"
    expected = folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    assert actual == expected
    assert actual.is_file()


def test_cleanup_checkpoints1(test_output_dirs: OutputFolderForTests) -> None:
    folder = test_output_dirs.root_dir
    with pytest.raises(FileNotFoundError) as ex:
        create_best_checkpoint(folder)
    assert "Checkpoint file" in str(ex)
    # Single checkpoint file, nothing else: This file should be rename to best_checkpoint
    last = folder / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    last.touch()
    create_best_checkpoint(folder)
    assert len(list(folder.glob("*"))) == 1
    assert (folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX).is_file()


def test_cleanup_checkpoints2(test_output_dirs: OutputFolderForTests) -> None:
    # Single checkpoint file and two recovery checkpoints: Should keep the last and rename it.
    folder = test_output_dirs.root_dir
    last = folder / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    last.touch()
    (folder / f"{RECOVERY_CHECKPOINT_FILE_NAME}-epoch=3").touch()
    (folder / f"{RECOVERY_CHECKPOINT_FILE_NAME}-epoch=6").touch()
    # Before cleanup: last.ckpt, recovery-epoch=6.ckpt, recovery-epoch=3.ckpt
    create_best_checkpoint(folder)
    # After cleanup: best.ckpt, recovery-epoch=6.ckpt, recovery-epoch=3.ckpt
    assert len(list(folder.glob("*"))) == 3
    assert (folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX).is_file()
    assert (folder / f"{RECOVERY_CHECKPOINT_FILE_NAME}-epoch=6").is_file()
    assert (folder / f"{RECOVERY_CHECKPOINT_FILE_NAME}-epoch=3").is_file()
