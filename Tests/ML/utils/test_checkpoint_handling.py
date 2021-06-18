#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

import pytest
import torch

from InnerEye.Common.common_util import OTHER_RUNS_SUBDIR_NAME
from InnerEye.Common.fixed_paths import MODEL_WEIGHTS_DIR_NAME
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, get_recovery_checkpoint_path
from InnerEye.ML.deep_learning_config import WEIGHTS_FILE
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.lightning_base import InnerEyeContainer
from Tests.AfterTraining.test_after_training import FALLBACK_ENSEMBLE_RUN, FALLBACK_SINGLE_RUN, get_most_recent_run, \
    get_most_recent_run_id
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import get_default_checkpoint_handler

EXTERNAL_WEIGHTS_URL_EXAMPLE = "https://download.pytorch.org/models/resnet18-5c106cde.pth"


def create_checkpoint_file(file: Path) -> None:
    """
    Creates a very simple torch checkpoint file. The only requirement is that it can safely pass torch.load.
    :param file: The path of the checkpoint file that should be written.
    """
    weights = {'state_dict': {'foo': torch.ones((2, 2))}}
    torch.save(weights, str(file))
    loaded = torch.load(str(file))
    assert loaded, "Unable to read the checkpoint file that was just created"


def test_use_local_weights_file(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()

    # No checkpoint handling options set.
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)

    checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert not checkpoint_handler.run_recovery
    assert not checkpoint_handler.local_weights_path

    # weights from local_weights_path and weights_url will be modified if needed and stored at this location
    expected_path = checkpoint_handler.output_params.outputs_folder / WEIGHTS_FILE

    # Set a weights_path
    checkpoint_handler.azure_config.run_recovery_id = ""
    checkpoint_handler.container.weights_url = EXTERNAL_WEIGHTS_URL_EXAMPLE
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert checkpoint_handler.local_weights_path == expected_path
    assert checkpoint_handler.local_weights_path.is_file()

    # set a local_weights_path
    checkpoint_handler.container.weights_url = ""
    local_weights_path = test_output_dirs.root_dir / "exist.pth"
    create_checkpoint_file(local_weights_path)
    checkpoint_handler.container.local_weights_path = local_weights_path
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert checkpoint_handler.local_weights_path == expected_path


@pytest.mark.after_training_single_run
def test_download_checkpoints_from_single_run(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)

    # No checkpoint handling options set.
    with mock.patch.object(InnerEyeContainer, "validate"):
        checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                            project_root=test_output_dirs.root_dir)
    run_recovery_id = get_most_recent_run_id(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)

    # Set a run recovery object - non ensemble
    checkpoint_handler.azure_config.run_recovery_id = run_recovery_id
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert checkpoint_handler.run_recovery

    expected_checkpoint_root = config.checkpoint_folder
    expected_paths = [get_recovery_checkpoint_path(path=expected_checkpoint_root),
                      expected_checkpoint_root / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX]
    assert checkpoint_handler.run_recovery.checkpoints_roots == [expected_checkpoint_root]
    for path in expected_paths:
        assert path.is_file()


@pytest.mark.after_training_ensemble_run
def test_download_checkpoints_from_ensemble_run(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    with mock.patch.object(InnerEyeContainer, "validate"):
        checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                            project_root=test_output_dirs.root_dir)

    run_recovery_id = get_most_recent_run_id(fallback_run_id_for_local_execution=FALLBACK_ENSEMBLE_RUN)
    checkpoint_handler.azure_config.run_recovery_id = run_recovery_id
    with pytest.raises(ValueError) as ex:
        checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert "has child runs" in str(ex)


def test_get_recovery_path_train(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()
    with mock.patch.object(InnerEyeContainer, "validate"):
        checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                            project_root=test_output_dirs.root_dir)

    assert checkpoint_handler.get_recovery_path_train() is None

    # weights from local_weights_path and weights_url will be modified if needed and stored at this location
    expected_path = checkpoint_handler.output_params.outputs_folder / WEIGHTS_FILE

    # Set a weights_url to get checkpoint from
    checkpoint_handler.azure_config.run_recovery_id = ""
    checkpoint_handler.container.weights_url = EXTERNAL_WEIGHTS_URL_EXAMPLE
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert checkpoint_handler.local_weights_path == expected_path
    assert checkpoint_handler.get_recovery_path_train() == expected_path

    # Set a local_weights_path to get checkpoint from
    checkpoint_handler.container.weights_url = ""
    local_weights_path = test_output_dirs.root_dir / "exist.pth"
    create_checkpoint_file(local_weights_path)
    checkpoint_handler.container.local_weights_path = local_weights_path
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert checkpoint_handler.local_weights_path == expected_path
    assert checkpoint_handler.get_recovery_path_train() == expected_path


@pytest.mark.after_training_single_run
def test_get_recovery_path_train_single_run(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    with mock.patch.object(InnerEyeContainer, "validate"):
        checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                            project_root=test_output_dirs.root_dir)

    run_recovery_id = get_most_recent_run_id(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)

    checkpoint_handler.azure_config.run_recovery_id = run_recovery_id
    checkpoint_handler.download_recovery_checkpoints_or_weights()

    # Run recovery with start epoch provided should succeed
    expected_path = get_recovery_checkpoint_path(path=config.checkpoint_folder)
    assert checkpoint_handler.get_recovery_path_train() == expected_path


@pytest.mark.after_training_single_run
def test_get_best_checkpoint_single_run(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    with mock.patch.object(InnerEyeContainer, "validate"):
        checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                            project_root=test_output_dirs.root_dir)

    # We have not set a run_recovery, nor have we trained, so this should fail to get a checkpoint
    with pytest.raises(ValueError) as ex:
        checkpoint_handler.get_best_checkpoint()
        assert "no run recovery object provided and no training has been done in this run" in ex.value.args[0]

    run_recovery_id = get_most_recent_run_id(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)

    # We have set a run_recovery_id now, so this should work: Should download all checkpoints that are available
    # in the run, into a subfolder of the checkpoint folder
    checkpoint_handler.azure_config.run_recovery_id = run_recovery_id
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    expected_checkpoint = config.checkpoint_folder / f"{BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX}"
    checkpoint_paths = checkpoint_handler.get_best_checkpoint()
    assert checkpoint_paths
    assert len(checkpoint_paths) == 1
    assert expected_checkpoint == checkpoint_paths[0]

    # From now on, the checkpoint handler will think that the run was started from epoch 1. We should pick up
    # the best checkpoint from the current run, or from the run recovery if the best checkpoint is there
    # and so no checkpoints have been written in the resumed run.
    checkpoint_handler.additional_training_done()
    # go back to non ensemble run recovery
    checkpoint_handler.azure_config.run_recovery_id = run_recovery_id
    checkpoint_handler.download_recovery_checkpoints_or_weights()

    # There is no checkpoint in the current run - use the one from run_recovery
    checkpoint_paths = checkpoint_handler.get_best_checkpoint()
    expected_checkpoint = config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    assert checkpoint_paths
    assert len(checkpoint_paths) == 1
    assert checkpoint_paths[0] == expected_checkpoint

    # Copy over checkpoints to make it look like training has happened and a better checkpoint written
    expected_checkpoint = config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    expected_checkpoint.touch()
    checkpoint_paths = checkpoint_handler.get_best_checkpoint()
    assert checkpoint_paths
    assert len(checkpoint_paths) == 1
    assert expected_checkpoint == checkpoint_paths[0]


@pytest.mark.after_training_ensemble_run
def test_get_all_checkpoints_from_ensemble_run(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()
    with mock.patch.object(InnerEyeContainer, "validate"):
        manage_recovery = get_default_checkpoint_handler(model_config=config,
                                                         project_root=test_output_dirs.root_dir)
    hyperdrive_run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_ENSEMBLE_RUN)
    manage_recovery.download_checkpoints_from_hyperdrive_child_runs(hyperdrive_run)
    expected_checkpoints = [config.checkpoint_folder / OTHER_RUNS_SUBDIR_NAME / str(i)
                            / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX for i in range(2)]
    checkpoint_paths = manage_recovery.get_best_checkpoint()
    assert checkpoint_paths
    assert len(checkpoint_paths) == 2
    assert set(expected_checkpoints) == set(checkpoint_paths)


def test_get_checkpoints_to_test(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()
    with mock.patch.object(InnerEyeContainer, "validate"):
        manage_recovery = get_default_checkpoint_handler(model_config=config,
                                                         project_root=test_output_dirs.root_dir)

    # Set a local_weights_path to get checkpoint from. Model has not trained and no run recovery provided,
    # so the local weights should be used ignoring any epochs to test
    local_weights_path = test_output_dirs.root_dir / "exist.pth"
    create_checkpoint_file(local_weights_path)
    manage_recovery.container.local_weights_path = local_weights_path
    manage_recovery.download_recovery_checkpoints_or_weights()
    checkpoint_and_paths = manage_recovery.get_checkpoints_to_test()
    assert checkpoint_and_paths
    assert len(checkpoint_and_paths) == 1
    assert checkpoint_and_paths[0] == manage_recovery.output_params.outputs_folder / WEIGHTS_FILE

    manage_recovery.additional_training_done()
    manage_recovery.container.checkpoint_folder.mkdir()

    # Copy checkpoint to make it seem like training has happened
    expected_checkpoint = config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    expected_checkpoint.touch()
    checkpoint_and_paths = manage_recovery.get_checkpoints_to_test()

    assert checkpoint_and_paths
    assert len(checkpoint_and_paths) == 1
    assert checkpoint_and_paths[0] == expected_checkpoint


@pytest.mark.after_training_single_run
def test_get_checkpoints_to_test_single_run(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()
    with mock.patch.object(InnerEyeContainer, "validate"):
        manage_recovery = get_default_checkpoint_handler(model_config=config,
                                                         project_root=test_output_dirs.root_dir)

    run_recovery_id = get_most_recent_run_id(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)

    # Now set a run recovery object and set the start epoch to 1, so we get one epoch from
    # run recovery and one from the training checkpoints
    manage_recovery.azure_config.run_recovery_id = run_recovery_id

    manage_recovery.additional_training_done()
    manage_recovery.download_recovery_checkpoints_or_weights()

    checkpoint_and_paths = manage_recovery.get_checkpoints_to_test()

    assert checkpoint_and_paths
    assert len(checkpoint_and_paths) == 1
    assert checkpoint_and_paths[0] == config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX

    # Copy checkpoint to make it seem like training has happened
    expected_checkpoint = config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    expected_checkpoint.touch()
    checkpoint_and_paths = manage_recovery.get_checkpoints_to_test()

    assert checkpoint_and_paths
    assert len(checkpoint_and_paths) == 1
    assert checkpoint_and_paths[0] == expected_checkpoint


def test_download_model_weights(test_output_dirs: OutputFolderForTests) -> None:
    # Download a sample ResNet model from a URL given in the Pytorch docs
    # The downloaded model does not match the architecture, which is okay since we are only testing the download here.

    model_config = DummyModel(weights_url=EXTERNAL_WEIGHTS_URL_EXAMPLE)
    manage_recovery = get_default_checkpoint_handler(model_config=model_config,
                                                     project_root=test_output_dirs.root_dir)
    result_path = manage_recovery.download_weights()
    assert result_path.is_file()


def test_get_local_weights_path_or_download(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    with mock.patch.object(InnerEyeContainer, "validate"):
        manage_recovery = get_default_checkpoint_handler(model_config=config,
                                                         project_root=test_output_dirs.root_dir)

    # If the model has neither local_weights_path or weights_url set, should fail.
    with pytest.raises(ValueError) as ex:
        manage_recovery.get_local_weights_path_or_download()
        assert "neither local_weights_path nor weights_url is set in the model config" in ex.value.args[0]

    # If local_weights_path folder exists, get_local_weights_path_or_download should not do anything.
    local_weights_path = manage_recovery.project_root / "exist.pth"
    create_checkpoint_file(local_weights_path)
    manage_recovery.container.local_weights_path = local_weights_path
    returned_weights_path = manage_recovery.get_local_weights_path_or_download()
    assert local_weights_path == returned_weights_path

    # Pointing the model to a URL should trigger a download
    manage_recovery.container.local_weights_path = None
    manage_recovery.container.weights_url = EXTERNAL_WEIGHTS_URL_EXAMPLE
    downloaded_weights = manage_recovery.get_local_weights_path_or_download()
    # Download goes into <project_root> / "modelweights" / "resnet18-5c106cde.pth"
    expected_path = manage_recovery.project_root / MODEL_WEIGHTS_DIR_NAME / \
                    os.path.basename(urlparse(EXTERNAL_WEIGHTS_URL_EXAMPLE).path)
    assert downloaded_weights
    assert downloaded_weights.is_file()
    assert expected_path == downloaded_weights

    # try again, should not re-download
    modified_time = downloaded_weights.stat().st_mtime
    downloaded_weights_new = manage_recovery.get_local_weights_path_or_download()
    assert downloaded_weights_new
    assert downloaded_weights_new.stat().st_mtime == modified_time


def test_get_and_modify_local_weights(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()

    with mock.patch.object(InnerEyeContainer, "validate"):
        manage_recovery = get_default_checkpoint_handler(model_config=config,
                                                         project_root=test_output_dirs.root_dir)

    # If the model has neither local_weights_path or weights_url set, should fail.
    with pytest.raises(ValueError) as ex:
        manage_recovery.get_and_save_modified_weights()
        assert "neither local_weights_path nor weights_url is set in the model config" in ex.value.args[0]

    # Pointing the model to a local_weights_path that does not exist will raise an error.
    manage_recovery.container.local_weights_path = manage_recovery.project_root / "non_exist"
    with pytest.raises(FileNotFoundError) as file_ex:
        manage_recovery.get_and_save_modified_weights()
        assert "Could not find the weights file" in file_ex.value.args[0]

    # Test that weights are properly modified when a local_weights_path is set

    # set a method to modify weights:
    with mock.patch.object(ModelConfigBase,
                           'load_checkpoint_and_modify',
                           lambda self, path_to_checkpoint: {"modified": "local",  # type: ignore
                                                             "path": path_to_checkpoint}):
        # Set the local_weights_path to an empty file, which will be passed to modify_checkpoint
        local_weights_path = manage_recovery.project_root / "exist.pth"
        create_checkpoint_file(local_weights_path)
        manage_recovery.container.local_weights_path = local_weights_path
        weights_path = manage_recovery.get_and_save_modified_weights()
        expected_path = manage_recovery.output_params.outputs_folder / WEIGHTS_FILE
        # read from weights_path and check that the dict has been written
        assert weights_path.is_file()
        assert expected_path == weights_path
        read = torch.load(str(weights_path))
        assert read.keys() == {"modified", "path"}
        assert read["modified"] == "local"
        assert read["path"] == local_weights_path
        # clean up
        weights_path.unlink()

    # Test that weights are properly modified when weights_url is set

    # set a different method to modify weights, to avoid using old files from other tests:
    with mock.patch.object(ModelConfigBase,
                           'load_checkpoint_and_modify',
                           lambda self, path_to_checkpoint: {"modified": "url", "path": path_to_checkpoint}):
        # Set the weights_url to the sample pytorch URL, which will be passed to modify_checkpoint
        manage_recovery.container.local_weights_path = None
        manage_recovery.container.weights_url = EXTERNAL_WEIGHTS_URL_EXAMPLE
        weights_path = manage_recovery.get_and_save_modified_weights()
        expected_path = manage_recovery.output_params.outputs_folder / WEIGHTS_FILE
        # read from weights_path and check that the dict has been written
        assert weights_path.is_file()
        assert expected_path == weights_path
        read = torch.load(str(weights_path))
        assert read.keys() == {"modified", "path"}
        assert read["modified"] == "url"
        assert read["path"] == manage_recovery.project_root / MODEL_WEIGHTS_DIR_NAME / \
                os.path.basename(urlparse(EXTERNAL_WEIGHTS_URL_EXAMPLE).path)
