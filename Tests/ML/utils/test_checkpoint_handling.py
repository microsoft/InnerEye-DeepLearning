#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
import torch

from InnerEye.Azure.azure_util import MODEL_ID_KEY_NAME
from InnerEye.Common.common_util import OTHER_RUNS_SUBDIR_NAME
from InnerEye.Common.fixed_paths import MODEL_INFERENCE_JSON_FILE_NAME
from InnerEye.ML.utils.checkpoint_handling import MODEL_WEIGHTS_DIR_NAME
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, get_recovery_checkpoint_path
from InnerEye.ML.deep_learning_config import FINAL_MODEL_FOLDER, FINAL_ENSEMBLE_MODEL_FOLDER
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_inference_config import read_model_inference_config
from InnerEye.ML.utils.checkpoint_handling import CheckpointHandler
from Tests.AfterTraining.test_after_training import FALLBACK_ENSEMBLE_RUN, FALLBACK_SINGLE_RUN, get_most_recent_run, \
    get_most_recent_run_id
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


def test_use_checkpoint_paths_or_urls(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()
    config.checkpoint_folder.mkdir()

    # No checkpoint handling options set.
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)

    checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert not checkpoint_handler.run_recovery
    assert not checkpoint_handler.trained_weights_paths

    # weights from local_weights_path and weights_url will be modified if needed and stored at this location

    # Set a weights_path
    checkpoint_handler.azure_config.run_recovery_id = ""
    checkpoint_handler.container.checkpoint_urls = [EXTERNAL_WEIGHTS_URL_EXAMPLE]
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    expected_download_path = checkpoint_handler.output_params.checkpoint_folder / MODEL_WEIGHTS_DIR_NAME /\
                             os.path.basename(urlparse(EXTERNAL_WEIGHTS_URL_EXAMPLE).path)
    assert checkpoint_handler.trained_weights_paths[0] == expected_download_path
    assert checkpoint_handler.trained_weights_paths[0].is_file()

    # set a local_weights_path
    checkpoint_handler.container.checkpoint_urls = []
    local_weights_path = test_output_dirs.root_dir / "exist.pth"
    create_checkpoint_file(local_weights_path)
    checkpoint_handler.container.local_checkpoint_paths = [local_weights_path]
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert checkpoint_handler.trained_weights_paths[0] == local_weights_path
    assert checkpoint_handler.trained_weights_paths[0].is_file()


@pytest.mark.after_training_single_run
def test_download_recovery_checkpoints_from_single_run(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)

    # No checkpoint handling options set.
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
def test_download_recovery_checkpoints_from_ensemble_run(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)

    run_recovery_id = get_most_recent_run_id(fallback_run_id_for_local_execution=FALLBACK_ENSEMBLE_RUN)
    checkpoint_handler.azure_config.run_recovery_id = run_recovery_id
    with pytest.raises(ValueError) as ex:
        checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert "has child runs" in str(ex)


@pytest.mark.after_training_single_run
def test_download_model_from_single_run(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()
    config.checkpoint_folder.mkdir()

    # No checkpoint handling options set.
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)
    run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)
    model_id = run.get_tags().get(MODEL_ID_KEY_NAME, None)
    assert model_id is not None

    # Set a run recovery object - non ensemble
    checkpoint_handler.azure_config.model_id = model_id
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert checkpoint_handler.trained_weights_paths

    expected_model_root = config.checkpoint_folder / MODEL_WEIGHTS_DIR_NAME / FINAL_MODEL_FOLDER
    model_inference_config = read_model_inference_config(expected_model_root / MODEL_INFERENCE_JSON_FILE_NAME)
    expected_paths = [expected_model_root / x for x in model_inference_config.checkpoint_paths]

    assert len(expected_paths) == 1  # A registered model for a non-ensemble run should contain only one checkpoint
    assert len(checkpoint_handler.trained_weights_paths) == 1
    assert expected_paths[0] == checkpoint_handler.trained_weights_paths[0]
    assert expected_paths[0].is_file()


@pytest.mark.after_training_ensemble_run
def test_download_model_from_ensemble_run(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()
    config.checkpoint_folder.mkdir()

    # No checkpoint handling options set.
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)
    run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_ENSEMBLE_RUN)
    model_id = run.get_tags().get(MODEL_ID_KEY_NAME, None)
    assert model_id is not None

    # Set a run recovery object - non ensemble
    checkpoint_handler.azure_config.model_id = model_id
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert checkpoint_handler.trained_weights_paths

    expected_model_root = config.checkpoint_folder / MODEL_WEIGHTS_DIR_NAME / FINAL_ENSEMBLE_MODEL_FOLDER
    model_inference_config = read_model_inference_config(expected_model_root / MODEL_INFERENCE_JSON_FILE_NAME)
    expected_paths = [expected_model_root / x for x in model_inference_config.checkpoint_paths]

    assert len(checkpoint_handler.trained_weights_paths) == len(expected_paths)
    assert set(checkpoint_handler.trained_weights_paths) == set(expected_paths)
    for path in expected_paths:
        assert path.is_file()


def test_get_recovery_path_train(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)

    assert checkpoint_handler.get_recovery_or_checkpoint_path_train() is None


@pytest.mark.after_training_single_run
def test_get_recovery_path_train_single_run(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)

    run_recovery_id = get_most_recent_run_id(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)

    checkpoint_handler.azure_config.run_recovery_id = run_recovery_id
    checkpoint_handler.download_recovery_checkpoints_or_weights()

    # Run recovery with start epoch provided should succeed
    expected_path = get_recovery_checkpoint_path(path=config.checkpoint_folder)
    assert checkpoint_handler.get_recovery_or_checkpoint_path_train() == expected_path


@pytest.mark.after_training_single_run
def test_get_best_checkpoint_single_run(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)

    # We have not set a run_recovery, nor have we trained, so this should fail to get a checkpoint
    with pytest.raises(ValueError) as ex:
        checkpoint_handler.get_best_checkpoints()
        assert "no run recovery object provided and no training has been done in this run" in ex.value.args[0]

    run_recovery_id = get_most_recent_run_id(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)

    # We have set a run_recovery_id now, so this should work: Should download all checkpoints that are available
    # in the run, into a subfolder of the checkpoint folder
    checkpoint_handler.azure_config.run_recovery_id = run_recovery_id
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    expected_checkpoint = config.checkpoint_folder / f"{BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX}"
    checkpoint_paths = checkpoint_handler.get_best_checkpoints()
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
    checkpoint_paths = checkpoint_handler.get_best_checkpoints()
    expected_checkpoint = config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    assert checkpoint_paths
    assert len(checkpoint_paths) == 1
    assert checkpoint_paths[0] == expected_checkpoint

    # Copy over checkpoints to make it look like training has happened and a better checkpoint written
    expected_checkpoint = config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    expected_checkpoint.touch()
    checkpoint_paths = checkpoint_handler.get_best_checkpoints()
    assert checkpoint_paths
    assert len(checkpoint_paths) == 1
    assert expected_checkpoint == checkpoint_paths[0]


@pytest.mark.after_training_ensemble_run
def test_download_checkpoints_from_hyperdrive_child_runs(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)
    hyperdrive_run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_ENSEMBLE_RUN)
    checkpoint_handler.download_checkpoints_from_hyperdrive_child_runs(hyperdrive_run)
    expected_checkpoints = [config.checkpoint_folder / OTHER_RUNS_SUBDIR_NAME / str(i)
                            / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX for i in range(2)]
    checkpoint_paths = checkpoint_handler.get_best_checkpoints()
    assert checkpoint_paths
    assert len(checkpoint_paths) == 2
    assert set(expected_checkpoints) == set(checkpoint_paths)


def test_get_checkpoints_to_test(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                     project_root=test_output_dirs.root_dir)

    # Set a local_checkpoint_paths to get checkpoint from. Model has not trained and no run recovery provided,
    # so the local weights should be used ignoring any epochs to test
    local_weights_path = test_output_dirs.root_dir / "exist.pth"
    create_checkpoint_file(local_weights_path)
    checkpoint_handler.container.local_checkpoint_paths = local_weights_path
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    checkpoint_and_paths = checkpoint_handler.get_checkpoints_to_test()
    assert checkpoint_and_paths
    assert len(checkpoint_and_paths) == 1
    assert checkpoint_and_paths[0] == local_weights_path

    checkpoint_handler.additional_training_done()
    checkpoint_handler.container.checkpoint_folder.mkdir()

    # Copy checkpoint to make it seem like training has happened
    expected_checkpoint = config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    expected_checkpoint.touch()
    checkpoint_and_paths = checkpoint_handler.get_checkpoints_to_test()

    assert checkpoint_and_paths
    assert len(checkpoint_and_paths) == 1
    assert checkpoint_and_paths[0] == expected_checkpoint


@pytest.mark.after_training_single_run
def test_get_checkpoints_to_test_single_run(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.outputs_folder.mkdir()
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                     project_root=test_output_dirs.root_dir)

    run_recovery_id = get_most_recent_run_id(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)

    # Now set a run recovery object and set the start epoch to 1, so we get one epoch from
    # run recovery and one from the training checkpoints
    checkpoint_handler.azure_config.run_recovery_id = run_recovery_id

    checkpoint_handler.additional_training_done()
    checkpoint_handler.download_recovery_checkpoints_or_weights()

    checkpoint_and_paths = checkpoint_handler.get_checkpoints_to_test()

    assert checkpoint_and_paths
    assert len(checkpoint_and_paths) == 1
    assert checkpoint_and_paths[0] == config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX

    # Copy checkpoint to make it seem like training has happened
    expected_checkpoint = config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    expected_checkpoint.touch()
    checkpoint_and_paths = checkpoint_handler.get_checkpoints_to_test()

    assert checkpoint_and_paths
    assert len(checkpoint_and_paths) == 1
    assert checkpoint_and_paths[0] == expected_checkpoint


def test_download_model_weights(test_output_dirs: OutputFolderForTests) -> None:
    # Download a sample ResNet model from a URL given in the Pytorch docs
    result_path = CheckpointHandler.download_weights(urls=[EXTERNAL_WEIGHTS_URL_EXAMPLE],
                                                     download_folder=test_output_dirs.root_dir)
    assert len(result_path) == 1
    assert result_path[0] == test_output_dirs.root_dir / os.path.basename(urlparse(EXTERNAL_WEIGHTS_URL_EXAMPLE).path)
    assert result_path[0].is_file()

    # Downloading twice should have no effect
    result_path = CheckpointHandler.download_weights(urls=[EXTERNAL_WEIGHTS_URL_EXAMPLE, EXTERNAL_WEIGHTS_URL_EXAMPLE],
                                                     download_folder=test_output_dirs.root_dir)
    assert len(result_path) == 1
    assert result_path[0] == test_output_dirs.root_dir / os.path.basename(urlparse(EXTERNAL_WEIGHTS_URL_EXAMPLE).path)
    assert result_path[0].is_file()


def test_get_local_weights_path_or_download(test_output_dirs: OutputFolderForTests) -> None:
    config = ModelConfigBase(should_validate=False)
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                     project_root=test_output_dirs.root_dir)

    # If the model has neither local_weights_path or weights_url set, should fail.
    with pytest.raises(ValueError) as ex:
        checkpoint_handler.get_local_checkpoints_path_or_download()
        assert "neither local_weights_path nor weights_url is set in the model config" in ex.value.args[0]

    # If local_weights_path folder exists, get_local_weights_path_or_download should not do anything.
    local_weights_path = checkpoint_handler.project_root / "exist.pth"
    create_checkpoint_file(local_weights_path)
    checkpoint_handler.container.local_checkpoint_paths = local_weights_path
    returned_weights_path = checkpoint_handler.get_local_checkpoints_path_or_download()
    assert local_weights_path == returned_weights_path

    # Pointing the model to a URL should trigger a download
    checkpoint_handler.container.local_checkpoint_paths = None
    checkpoint_handler.container.checkpoint_urls = EXTERNAL_WEIGHTS_URL_EXAMPLE
    downloaded_weights = checkpoint_handler.get_local_checkpoints_path_or_download()
    # Download goes into <project_root> / "modelweights" / "resnet18-5c106cde.pth"
    expected_path = checkpoint_handler.output_params.checkpoint_folder / MODEL_WEIGHTS_DIR_NAME / \
                    os.path.basename(urlparse(EXTERNAL_WEIGHTS_URL_EXAMPLE).path)
    assert len(downloaded_weights) == 1
    assert downloaded_weights[0].is_file()
    assert expected_path == downloaded_weights[0]

    # try again, should not re-download
    modified_time = downloaded_weights.stat().st_mtime
    downloaded_weights_new = checkpoint_handler.get_local_checkpoints_path_or_download()
    assert len(downloaded_weights_new) == 1
    assert downloaded_weights_new[0].stat().st_mtime == modified_time
