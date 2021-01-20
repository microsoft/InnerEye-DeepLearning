#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import param
import pytest

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import fixed_paths
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Common.spawn_subprocess import spawn_and_monitor_subprocess
from InnerEye.ML.common import CHECKPOINT_SUFFIX
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.deep_learning_config import CHECKPOINT_FOLDER
from InnerEye.ML.model_inference_config import ModelInferenceConfig
from InnerEye.ML.run_ml import MLRunner


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
