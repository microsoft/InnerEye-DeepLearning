#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
import torch
import pytest
import shutil

from urllib.parse import urlparse

from InnerEye.ML.deep_learning_config import WEIGHTS_FILE
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Common.fixed_paths import MODEL_WEIGHTS_DIR_NAME
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.common import create_checkpoint_path

from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import get_default_checkpoint_handler
from Tests.Common.test_util import DEFAULT_RUN_RECOVERY_ID, DEFAULT_ENSEMBLE_RUN_RECOVERY_ID
from Tests.fixed_paths_for_tests import full_ml_test_data_path


EXTERNAL_WEIGHTS_URL_EXAMPLE = "https://download.pytorch.org/models/resnet18-5c106cde.pth"


def test_discover_and_download_checkpoints_from_previous_runs(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()

    # No checkpoint handling options set.
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)

    checkpoint_handler.discover_and_download_checkpoints_from_previous_runs()
    assert not checkpoint_handler.run_recovery
    assert not checkpoint_handler.local_weights_path

    # Set a run recovery object - non ensemble
    checkpoint_handler.azure_config.run_recovery_id = DEFAULT_RUN_RECOVERY_ID
    checkpoint_handler.discover_and_download_checkpoints_from_previous_runs()

    expected_checkpoint_root = config.checkpoint_folder / DEFAULT_RUN_RECOVERY_ID.split(":")[1]
    expected_paths = [create_checkpoint_path(path=expected_checkpoint_root,
                                             epoch=epoch) for epoch in [1, 2, 3, 4, 20]]
    assert checkpoint_handler.run_recovery
    assert checkpoint_handler.run_recovery.checkpoints_roots == [expected_checkpoint_root]
    for path in expected_paths:
        assert path.is_file()

    # Set a run recovery object - ensemble
    checkpoint_handler.azure_config.run_recovery_id = DEFAULT_ENSEMBLE_RUN_RECOVERY_ID
    checkpoint_handler.discover_and_download_checkpoints_from_previous_runs()

    expected_checkpoint_roots = [config.checkpoint_folder / DEFAULT_ENSEMBLE_RUN_RECOVERY_ID.split(":")[1]
                                 / str(i) for i in range(3)]
    expected_path_lists = [[create_checkpoint_path(path=expected_checkpoint_root,
                                              epoch=epoch) for epoch in [1, 2]]
                      for expected_checkpoint_root in expected_checkpoint_roots]
    assert set(checkpoint_handler.run_recovery.checkpoints_roots) == set(expected_checkpoint_roots)
    for path_list in expected_path_lists:
        for path in path_list:
            assert path.is_file()

    # weights from local_weights_path and weights_url will be modified if needed and stored at this location
    expected_path = checkpoint_handler.model_config.outputs_folder / WEIGHTS_FILE

    # Set a weights_path
    checkpoint_handler.azure_config.run_recovery_id = ""
    config.weights_url = EXTERNAL_WEIGHTS_URL_EXAMPLE
    checkpoint_handler.discover_and_download_checkpoints_from_previous_runs()
    assert checkpoint_handler.local_weights_path == expected_path
    assert checkpoint_handler.local_weights_path.is_file()

    # set a local_weights_path
    config.weights_url = ""
    local_weights_path = test_output_dirs.root_dir / "exist.pth"
    stored_checkpoint = create_checkpoint_path(path=full_ml_test_data_path("checkpoints"), epoch=1)
    shutil.copyfile(str(stored_checkpoint), local_weights_path)
    config.local_weights_path = local_weights_path
    checkpoint_handler.discover_and_download_checkpoints_from_previous_runs()
    assert checkpoint_handler.local_weights_path == expected_path


def test_get_recovery_path_train(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)

    assert checkpoint_handler.get_recovery_path_train() is None

    checkpoint_handler.azure_config.run_recovery_id = DEFAULT_RUN_RECOVERY_ID
    checkpoint_handler.discover_and_download_checkpoints_from_previous_runs()

    # We have not set a start_epoch but we are trying to use run_recovery, this should fail
    with pytest.raises(ValueError) as ex:
        checkpoint_handler.get_recovery_path_train()
        assert "Run recovery set, but start epoch is 0" in ex.value.args[0]

    # Run recovery with start epoch provided should succeed
    config.start_epoch = 20
    expected_path = create_checkpoint_path(path=config.checkpoint_folder / DEFAULT_RUN_RECOVERY_ID.split(":")[1],
                                           epoch=config.start_epoch)
    assert checkpoint_handler.get_recovery_path_train() == expected_path

    # set an ensemble run as recovery - not supported
    checkpoint_handler.azure_config.run_recovery_id = DEFAULT_ENSEMBLE_RUN_RECOVERY_ID
    checkpoint_handler.discover_and_download_checkpoints_from_previous_runs()
    with pytest.raises(ValueError) as ex:
        checkpoint_handler.get_recovery_path_train()
        assert "Found more than one checkpoint for epoch" in ex.value.args[0]

    # weights from local_weights_path and weights_url will be modified if needed and stored at this location
    expected_path = checkpoint_handler.model_config.outputs_folder / WEIGHTS_FILE

    # Set a weights_url to get checkpoint from
    checkpoint_handler.azure_config.run_recovery_id = ""
    config.weights_url = EXTERNAL_WEIGHTS_URL_EXAMPLE
    checkpoint_handler.discover_and_download_checkpoints_from_previous_runs()
    assert checkpoint_handler.local_weights_path == expected_path
    config.start_epoch = 0
    assert checkpoint_handler.get_recovery_path_train() == expected_path
    # Can't resume training from an external checkpoint
    config.start_epoch = 20
    with pytest.raises(ValueError) as ex:
        checkpoint_handler.get_recovery_path_train()
        assert ex.value.args == "Start epoch is > 0, but no run recovery object has been provided to resume training."

    # Set a local_weights_path to get checkpoint from
    config.weights_url = ""
    local_weights_path = test_output_dirs.root_dir / "exist.pth"
    stored_checkpoint = create_checkpoint_path(full_ml_test_data_path("checkpoints"), epoch=1)
    shutil.copyfile(str(stored_checkpoint), local_weights_path)
    config.local_weights_path = local_weights_path
    checkpoint_handler.discover_and_download_checkpoints_from_previous_runs()
    assert checkpoint_handler.local_weights_path == expected_path
    config.start_epoch = 0
    assert checkpoint_handler.get_recovery_path_train() == expected_path
    # Can't resume training from an external checkpoint
    config.start_epoch = 20
    with pytest.raises(ValueError) as ex:
        checkpoint_handler.get_recovery_path_train()
        assert ex.value.args == "Start epoch is > 0, but no run recovery object has been provided to resume training."


def test_get_checkpoint_from_epoch(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()
    manage_recovery = get_default_checkpoint_handler(model_config=config,
                                                     project_root=test_output_dirs.root_dir)

    # We have not set a run_recovery, nor have we trained, so this should fail to get a checkpoint
    with pytest.raises(ValueError) as ex:
        manage_recovery.get_checkpoint_from_epoch(1)
        assert "no run recovery object provided and no training has been done in this run" in ex.value.args[0]

    # We have set a run_recovery_id now, so this should work
    manage_recovery.azure_config.run_recovery_id = DEFAULT_RUN_RECOVERY_ID
    manage_recovery.discover_and_download_checkpoints_from_previous_runs()
    expected_checkpoint = create_checkpoint_path(path=config.checkpoint_folder
                                                      / DEFAULT_RUN_RECOVERY_ID.split(":")[1], epoch=1)
    checkpoint = manage_recovery.get_checkpoint_from_epoch(1)
    assert checkpoint
    assert len(checkpoint.checkpoint_paths) == 1
    assert expected_checkpoint == checkpoint.checkpoint_paths[0]
    assert checkpoint.epoch == 1

    # ensemble run recovery
    manage_recovery.azure_config.run_recovery_id = DEFAULT_ENSEMBLE_RUN_RECOVERY_ID
    manage_recovery.discover_and_download_checkpoints_from_previous_runs()
    expected_checkpoints = [create_checkpoint_path(path=config.checkpoint_folder
                                                       / DEFAULT_ENSEMBLE_RUN_RECOVERY_ID.split(":")[1] / str(i), epoch=1)
                            for i in range(3)]
    checkpoint = manage_recovery.get_checkpoint_from_epoch(1)
    assert checkpoint
    assert len(checkpoint.checkpoint_paths) == 3
    assert set(expected_checkpoints) == set(checkpoint.checkpoint_paths)
    assert checkpoint.epoch == 1

    # From now on, the checkpoint handler will think that the run was started from epoch 1, i.e. we should use the
    # run recovery checkpoint for epoch 1 and the training run checkpoint for epoch 2
    manage_recovery.additional_training_done()
    # go back to non ensemble run recovery
    manage_recovery.azure_config.run_recovery_id = DEFAULT_RUN_RECOVERY_ID
    manage_recovery.discover_and_download_checkpoints_from_previous_runs()

    config.start_epoch = 1
    # We haven't actually done a training run ,so the checkpoint for epoch 2 is missing - and we should not use the one
    # from run recovery
    assert manage_recovery.get_checkpoint_from_epoch(2) is None

    # Should work for epoch 1
    checkpoint = manage_recovery.get_checkpoint_from_epoch(1)
    expected_checkpoint = create_checkpoint_path(path=config.checkpoint_folder
                                                      / DEFAULT_RUN_RECOVERY_ID.split(":")[1], epoch=1)
    assert checkpoint
    assert len(checkpoint.checkpoint_paths) == 1
    assert checkpoint.checkpoint_paths[0] == expected_checkpoint
    assert checkpoint.epoch == 1

    # Copy over checkpoints to make it look like training has happened
    stored_checkpoint = create_checkpoint_path(path=full_ml_test_data_path("checkpoints"), epoch=1)
    expected_checkpoint = create_checkpoint_path(path=config.checkpoint_folder, epoch=2)
    shutil.copyfile(str(stored_checkpoint), str(expected_checkpoint))

    # Should now work for epoch 2
    checkpoint = manage_recovery.get_checkpoint_from_epoch(2)
    assert checkpoint
    assert len(checkpoint.checkpoint_paths) == 1
    assert expected_checkpoint == checkpoint.checkpoint_paths[0]
    assert checkpoint.epoch == 2


def test_get_checkpoints_to_test(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()
    manage_recovery = get_default_checkpoint_handler(model_config=config,
                                                     project_root=test_output_dirs.root_dir)

    # Set a local_weights_path to get checkpoint from. Model has not trained and no run recovery provided,
    # so the local weights should be used ignoring any epochs to test
    config.epochs_to_test = [1, 2]
    local_weights_path = test_output_dirs.root_dir / "exist.pth"
    stored_checkpoint = create_checkpoint_path(full_ml_test_data_path("checkpoints"), epoch=1)
    shutil.copyfile(str(stored_checkpoint), local_weights_path)
    config.local_weights_path = local_weights_path
    manage_recovery.discover_and_download_checkpoints_from_previous_runs()
    checkpoint_and_paths = manage_recovery.get_checkpoints_to_test()
    assert checkpoint_and_paths
    assert len(checkpoint_and_paths) == 1
    assert checkpoint_and_paths[0].epoch == 0
    assert checkpoint_and_paths[0].checkpoint_paths == [manage_recovery.model_config.outputs_folder / WEIGHTS_FILE]

    # Now set a run recovery object and set the start epoch to 1, so we get one epoch from
    # run recovery and one from the training checkpoints
    manage_recovery.azure_config.run_recovery_id = DEFAULT_RUN_RECOVERY_ID
    config.start_epoch = 1
    manage_recovery.additional_training_done()
    manage_recovery.discover_and_download_checkpoints_from_previous_runs()
    # Copy checkpoint to make it seem like training has happened
    stored_checkpoint = create_checkpoint_path(path=full_ml_test_data_path("checkpoints"), epoch=1)
    expected_checkpoint = create_checkpoint_path(path=config.checkpoint_folder, epoch=2)
    shutil.copyfile(str(stored_checkpoint), str(expected_checkpoint))

    checkpoint_and_paths = manage_recovery.get_checkpoints_to_test()

    assert checkpoint_and_paths
    assert len(checkpoint_and_paths) == 2
    assert checkpoint_and_paths[0].epoch == 1
    assert checkpoint_and_paths[0].checkpoint_paths == [create_checkpoint_path(path=config.checkpoint_folder
                                                        / DEFAULT_RUN_RECOVERY_ID.split(":")[1], epoch=1)]
    assert checkpoint_and_paths[1].epoch == 2
    assert checkpoint_and_paths[1].checkpoint_paths == [create_checkpoint_path(path=config.checkpoint_folder,
                                                                               epoch=2)]

    # This epoch does not exist
    config.epochs_to_test = [3]
    checkpoint_and_paths = manage_recovery.get_checkpoints_to_test()
    assert checkpoint_and_paths is None


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
    manage_recovery = get_default_checkpoint_handler(model_config=config,
                                                     project_root=test_output_dirs.root_dir)

    # If the model has neither local_weights_path or weights_url set, should fail.
    with pytest.raises(ValueError) as ex:
        manage_recovery.get_local_weights_path_or_download()
        assert "neither local_weights_path nor weights_url is set in the model config" in ex.value.args[0]

    # If local_weights_path folder exists, get_local_weights_path_or_download should not do anything.
    local_weights_path = manage_recovery.project_root / "exist.pth"
    local_weights_path.touch()
    manage_recovery.model_config.local_weights_path = local_weights_path
    returned_weights_path = manage_recovery.get_local_weights_path_or_download()
    assert local_weights_path == returned_weights_path

    # Pointing the model to a URL should trigger a download
    config.local_weights_path = None
    config.weights_url = EXTERNAL_WEIGHTS_URL_EXAMPLE
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

    manage_recovery = get_default_checkpoint_handler(model_config=config,
                                                     project_root=test_output_dirs.root_dir)

    # If the model has neither local_weights_path or weights_url set, should fail.
    with pytest.raises(ValueError) as ex:
        manage_recovery.get_and_save_modified_weights()
        assert "neither local_weights_path nor weights_url is set in the model config" in ex.value.args[0]

    # Pointing the model to a local_weights_path that does not exist will raise an error.
    config.local_weights_path = manage_recovery.project_root / "non_exist"
    with pytest.raises(FileNotFoundError) as file_ex:
        manage_recovery.get_and_save_modified_weights()
        assert "Could not find the weights file" in file_ex.value.args[0]

    # Test that weights are properly modified when a local_weights_path is set

    # set a method to modify weights:
    ModelConfigBase.load_checkpoint_and_modify = lambda self, path_to_checkpoint: {"modified": "local",  # type: ignore
                                                                          "path": path_to_checkpoint}
    # Set the local_weights_path to an empty file, which will be passed to modify_checkpoint
    local_weights_path = manage_recovery.project_root / "exist.pth"
    local_weights_path.touch()
    config.local_weights_path = local_weights_path
    weights_path = manage_recovery.get_and_save_modified_weights()
    expected_path = manage_recovery.model_config.outputs_folder / WEIGHTS_FILE
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
    ModelConfigBase.load_checkpoint_and_modify = lambda self, path_to_checkpoint: {"modified": "url",  # type: ignore
                                                                          "path": path_to_checkpoint}
    # Set the weights_url to the sample pytorch URL, which will be passed to modify_checkpoint
    config.local_weights_path = None
    config.weights_url = EXTERNAL_WEIGHTS_URL_EXAMPLE
    weights_path = manage_recovery.get_and_save_modified_weights()
    expected_path = manage_recovery.model_config.outputs_folder / WEIGHTS_FILE
    # read from weights_path and check that the dict has been written
    assert weights_path.is_file()
    assert expected_path == weights_path
    read = torch.load(str(weights_path))
    assert read.keys() == {"modified", "path"}
    assert read["modified"] == "url"
    assert read["path"] == manage_recovery.project_root / MODEL_WEIGHTS_DIR_NAME / \
                                                       os.path.basename(urlparse(EXTERNAL_WEIGHTS_URL_EXAMPLE).path)
