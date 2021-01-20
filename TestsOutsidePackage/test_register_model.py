#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest import mock

import numpy as np
import param
import pytest
from azureml.core import Model

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_util import MODEL_ID_KEY_NAME, fetch_run
from InnerEye.Common import common_util, fixed_paths
from InnerEye.Common.common_util import ModelProcessing
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Common.spawn_subprocess import spawn_and_monitor_subprocess
from InnerEye.ML.common import CHECKPOINT_SUFFIX, create_checkpoint_path
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.deep_learning_config import CHECKPOINT_FOLDER
from InnerEye.ML.model_inference_config import ModelInferenceConfig
from InnerEye.ML.model_testing import DEFAULT_RESULT_IMAGE_NAME
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.utils.image_util import get_unit_image_header
from Tests.AfterTraining.test_after_training import get_most_recent_run
from Tests.ML.util import assert_nifti_content, get_default_azure_config, get_model_loader, get_nifti_shape


class SubprocessConfig(GenericConfig):
    """
    Config class to store settings for sub-process spawning
    """
    process: str = param.String(None, doc="Path to the process to spawn")
    args: List[str] = param.List(instantiate=True, doc="List of arguments to pass to the spawned process")
    env: Dict[str, str] = param.Dict(instantiate=True, doc="Dictionary of environment variables "
                                                           "to override for this process")

    def spawn_and_monitor_subprocess(self) -> Tuple[int, List[str]]:
        return spawn_and_monitor_subprocess(process=self.process, args=self.args, env=self.env)


def create_checkpoints(model_config: SegmentationModelBase, is_ensemble: bool) -> Tuple[List[Path], List[Path]]:
    """
    Copies 1 or 2 checkpoint files from the stored test data into the model's checkpoint folder, and returns
    the absolute paths of those files.
    :param model_config: The model configuration, where a correct output folder must be set.
    :param is_ensemble: If true, 2 checkpoints (simulating an ensemble run) will be created. If false, only a
    single checkpoint will be created.
    """
    # To simulate ensemble models, there are two checkpoints, one in the root dir and one in a folder
    stored_checkpoints = full_ml_test_data_path('checkpoints')
    checkpoints = list(stored_checkpoints.rglob(f"*{CHECKPOINT_SUFFIX}")) if is_ensemble \
        else list(stored_checkpoints.glob(f"*{CHECKPOINT_SUFFIX}"))
    assert len(checkpoints) == (2 if is_ensemble else 1)
    checkpoints_relative = [checkpoint.relative_to(stored_checkpoints) for checkpoint in checkpoints]
    checkpoints_absolute = []
    # copy_child_paths_to_folder expects all checkpoints to live inside the model's checkpoint folder
    for (file_abs, file_relative) in list(zip(checkpoints, checkpoints_relative)):
        destination_abs = model_config.checkpoint_folder / file_relative
        destination_abs.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(str(file_abs), str(destination_abs))
        checkpoints_absolute.append(destination_abs)
    return checkpoints_absolute, checkpoints_relative


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on Windows")
@pytest.mark.after_training_ensemble_run
def test_register_and_score_model(test_output_dirs: OutputFolderForTests) -> None:
    """
    End-to-end test which ensures the scoring pipeline is functioning as expected when used on a recently created
    model. This test is run after training an ensemble run in AzureML.
    """
    run_id = get_most_recent_run()
    azure_config = get_default_azure_config()
    workspace = azure_config.get_workspace()
    azureml_run = fetch_run(workspace, run_id)
    model_id = azureml_run.get_tags().get(MODEL_ID_KEY_NAME, None)
    assert model_id is not None, f"Run {run_id} does not have a tag for registered model"
    azureml_model = Model(workspace=workspace, id=model_id)
    assert azureml_model is not None
    # download the registered model and test that we can run the score pipeline on it
    model_root = Path(azureml_model.download(str(test_output_dirs.root_dir)))
    # The model needs to contain score.py at the root, the (merged) environment definition,
    # and the inference config.
    expected_files = [
        *fixed_paths.SCRIPTS_AT_ROOT,
        fixed_paths.ENVIRONMENT_YAML_FILE_NAME,
        fixed_paths.MODEL_INFERENCE_JSON_FILE_NAME,
        "InnerEye/ML/runner.py",
    ]
    for expected_file in expected_files:
        assert (model_root / expected_file).is_file(), f"File {expected_file} missing"
    checkpoint_folder = model_root / CHECKPOINT_FOLDER
    assert checkpoint_folder.is_dir()
    checkpoints = list(checkpoint_folder.rglob("*"))
    assert len(checkpoints) >= 1, "There must be at least 1 checkpoint"

    # create a dummy datastore to store the image data
    test_datastore = test_output_dirs.root_dir / "test_datastore"
    # move test data into the data folder to simulate an actual run
    train_and_test_data_dir = full_ml_test_data_path("train_and_test_data")
    img_files = ["id1_channel1.nii.gz", "id1_channel2.nii.gz"]
    data_root = test_datastore / fixed_paths.DEFAULT_DATA_FOLDER
    data_root.mkdir(parents=True)
    for f in img_files:
        shutil.copy(str(train_and_test_data_dir / f), str(data_root))

    # run score pipeline as a separate process
    python_executable = sys.executable
    [return_code1, stdout1] = SubprocessConfig(process=python_executable,
                                               args=["--version"]).spawn_and_monitor_subprocess()
    assert return_code1 == 0
    print(f"Executing Python version {stdout1[0]}")
    return_code, stdout2 = SubprocessConfig(process=python_executable, args=[
        str(model_root / fixed_paths.SCORE_SCRIPT),
        f"--data_folder={str(data_root)}",
        f"--image_files={img_files[0]},{img_files[1]}",
        "--use_gpu=False"
    ]).spawn_and_monitor_subprocess()

    # check that the process completed as expected
    assert return_code == 0, f"Subprocess failed with return code {return_code}. Stdout: {os.linesep.join(stdout2)}"
    expected_segmentation_path = Path(model_root) / DEFAULT_RESULT_IMAGE_NAME
    assert expected_segmentation_path.exists(), f"Result file not found: {expected_segmentation_path}"

    # sanity check the resulting segmentation
    expected_shape = get_nifti_shape(train_and_test_data_dir / img_files[0])
    image_header = get_unit_image_header()
    assert_nifti_content(str(expected_segmentation_path), expected_shape, image_header, [3], np.ubyte)


def test_register_model_skip() -> None:
    """
    If the AzureML workspace can't be read, model registration should be skipped and return None.
    """
    checkpoint_paths = [create_checkpoint_path(full_ml_test_data_path('checkpoints'), epoch=0)]
    config = get_model_loader().create_model_config_from_name("Lung")
    ml_runner = MLRunner(config, None)
    raises = mock.Mock()
    raises.side_effect = Exception
    with mock.patch.object(AzureConfig, 'get_workspace', raises):
        model, deployment_result = ml_runner.register_segmentation_model(
            model_description="",
            checkpoint_paths=checkpoint_paths,
            model_proc=ModelProcessing.DEFAULT
        )
    assert model is None
    assert deployment_result is None


@pytest.mark.parametrize("is_ensemble", [True, False])
@pytest.mark.parametrize("extra_code_directory", ["TestsOutsidePackage", ""])
def test_copy_child_paths_to_folder(is_ensemble: bool,
                                    extra_code_directory: str,
                                    test_output_dirs: OutputFolderForTests) -> None:
    azure_config = AzureConfig(extra_code_directory=extra_code_directory)
    fake_model = SegmentationModelBase(should_validate=False)
    fake_model.set_output_to(test_output_dirs.root_dir)
    # To simulate ensemble models, there are two checkpoints, one in the root dir and one in a folder
    checkpoints_absolute, checkpoints_relative = create_checkpoints(fake_model, is_ensemble)
    # Simulate a project root: We can't derive that from the repository root because that might point
    # into Python's package folder
    project_root = Path(__file__).parent.parent
    ml_runner = MLRunner(model_config=fake_model, azure_config=azure_config, project_root=project_root)
    model_folder = test_output_dirs.root_dir / "final"
    ml_runner.copy_child_paths_to_folder(model_folder=model_folder,
                                         checkpoint_paths=checkpoints_absolute)
    expected_files = [
        fixed_paths.ENVIRONMENT_YAML_FILE_NAME,
        fixed_paths.MODEL_INFERENCE_JSON_FILE_NAME,
        "InnerEye/ML/runner.py",
        "InnerEye/ML/model_testing.py",
        "InnerEye/Common/fixed_paths.py",
        "InnerEye/Common/common_util.py",
    ]
    for r in checkpoints_relative:
        expected_files.append(f"{CHECKPOINT_FOLDER}/{r}")
    for expected_file in expected_files:
        assert (model_folder / expected_file).is_file(), f"File missing: {expected_file}"
    trm = model_folder / "TestsOutsidePackage/test_register_model.py"
    if extra_code_directory:
        assert trm.is_file()
    else:
        assert not trm.is_file()


def test_model_inference_config() -> None:
    # check if normal path works
    normal_path = "/".join(list(map(str, range(1, 91))))
    assert len(normal_path) == 260
    ModelInferenceConfig(model_name="Test", checkpoint_paths=[normal_path], structure_names=["organ1", "tumour2"],
                         colours=[(255, 0, 0), (255, 0, 0)],
                         fill_holes=[True, False])
    # check if long path fails ie: > 260
    long_path = normal_path + "/"
    assert len(long_path) == 261
    with pytest.raises(ValueError):
        ModelInferenceConfig(model_name="Test", checkpoint_paths=[long_path], structure_names=["organ1", "tumour2"],
                             colours=[(255, 0, 0), (255, 0, 0)],
                             fill_holes=[True, False])
