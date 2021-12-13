#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import List, Tuple

import pytest

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import fixed_paths
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.common import CHECKPOINT_FOLDER
from InnerEye.ML.model_inference_config import ModelInferenceConfig
from InnerEye.ML.run_ml import MLRunner


def create_checkpoints(model_config: SegmentationModelBase, is_ensemble: bool) -> Tuple[List[Path], List[Path]]:
    """
    Creates 1 or 2 empty checkpoint files in the model's checkpoint folder, and returns
    the paths of those files, both absolute paths and paths relative to the checkpoint folder.
    :param model_config: The model configuration, where a correct output folder must be set.
    :param is_ensemble: If true, 2 checkpoints (simulating an ensemble run) will be created. If false, only a
    single checkpoint will be created.
    :return: Tuple[absolute checkpoint paths, relative checkpoint paths]
    """
    # To simulate ensemble models, there are two checkpoints, one in the root dir and one in a folder
    folder = model_config.checkpoint_folder
    checkpoints_absolute = [folder / "foo.ckpt"]
    if is_ensemble:
        checkpoints_absolute.append(folder / "other" / "foo2.ckpt")
    for checkpoint in checkpoints_absolute:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()
    checkpoints_relative = [checkpoint.relative_to(folder) for checkpoint in checkpoints_absolute]
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
    hi_ml_submodules = [p for p, _ in fixed_paths.get_hi_ml_submodule_relative_paths()]
    has_submodule = any(folder.is_dir() for folder in hi_ml_submodules)
    ml_runner.copy_child_paths_to_folder(model_folder=model_folder, checkpoint_paths=checkpoints_absolute)
    expected_files = [
        fixed_paths.ENVIRONMENT_YAML_FILE_NAME,
        fixed_paths.MODEL_INFERENCE_JSON_FILE_NAME,
        "InnerEye/ML/runner.py",
        "InnerEye/ML/model_testing.py",
        "InnerEye/Common/fixed_paths.py",
        "InnerEye/Common/common_util.py",
    ]
    if has_submodule:
        expected_files.extend(["hi-ml/hi-ml/src/health_ml/__init__.py",
                               "hi-ml/hi-ml-azure/src/health_azure/__init__.py"])
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
    ModelInferenceConfig(model_name="Test", checkpoint_paths=[normal_path])
    # check if long path fails ie: > 260
    long_path = normal_path + "/"
    assert len(long_path) == 261
    with pytest.raises(ValueError):
        ModelInferenceConfig(model_name="Test", checkpoint_paths=[long_path])
