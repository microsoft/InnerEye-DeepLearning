#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import numpy as np
import param
import pytest
from azureml.core import Run

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import common_util, fixed_paths
from InnerEye.Common.common_util import ModelProcessing
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_inference_config import ModelInferenceConfig
from InnerEye.ML.model_testing import DEFAULT_RESULT_IMAGE_NAME
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.utils.image_util import get_unit_image_header
from Tests.ML.util import assert_nifti_content, get_default_azure_config, get_default_workspace, get_model_loader, \
    get_nifti_shape
from Tests.fixed_paths_for_tests import RELATIVE_TEST_OUTPUTS_PATH, full_ml_test_data_path, tests_root_directory
from run_scoring import spawn_and_monitor_subprocess
from score import DEFAULT_DATA_FOLDER

checkpoint_paths = [full_ml_test_data_path('checkpoints') / '1_checkpoint.pth.tar']


class SubprocessConfig(GenericConfig):
    """
    Config class to store settings for sub-process spawning
    """
    process: str = param.String(None, doc="Path to the process to spawn")
    args: List[str] = param.List(instantiate=True, doc="List of arguments to pass to the spawned process")
    env: Dict[str, str] = param.Dict(instantiate=True, doc="Dictionary of environment variables "
                                                           "to override for this process")

    def spawn_and_monitor_subprocess(self) -> int:
        return spawn_and_monitor_subprocess(process=self.process, args=self.args, env=self.env)


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on Windows")
@pytest.mark.parametrize("is_ensemble", [True, False])
# We currently don't support registered geonormalized models, so dataset_expected_spacing_xyz = (1.0, 1.0, 3.0)
# is excluded.
@pytest.mark.parametrize("dataset_expected_spacing_xyz", [None])
@pytest.mark.parametrize("model_outside_package", [True, False])
def test_register_and_score_model(is_ensemble: bool,
                                  dataset_expected_spacing_xyz: Any,
                                  model_outside_package: bool,
                                  test_output_dirs: OutputFolderForTests) -> None:
    """
    End-to-end test which ensures the scoring pipeline is functioning as expected by performing the following:
    1) Registering a pre-trained model to AML
    2) Checking that a model zip from the registered model can be created successfully
    3) Calling the scoring pipeline to check inference can be run from the published model successfully
    """
    ws = get_default_workspace()
    # Get an existing config as template
    loader = get_model_loader("Tests.ML.configs" if model_outside_package else None)
    config: SegmentationModelBase = loader.create_model_config_from_name(
        model_name="BasicModel2EpochsOutsidePackage" if model_outside_package else "BasicModel2Epochs"
    )
    config.dataset_expected_spacing_xyz = dataset_expected_spacing_xyz
    config.set_output_to(test_output_dirs.root_dir)
    # copy checkpoints into the outputs (simulating a run)
    stored_checkpoints = full_ml_test_data_path(os.path.join("train_and_test_data", "checkpoints"))
    shutil.copytree(str(stored_checkpoints), str(config.checkpoint_folder))
    paths = [config.checkpoint_folder / "1_checkpoint.pth.tar"]
    checkpoints = paths * 2 if is_ensemble else paths
    model = None
    model_path = None
    # Mocking to get the source from the current directory
    # the score.py and python_wrapper.py cannot be moved inside the InnerEye package, which will be the
    # only code running (if these tests are run on the package).
    with mock.patch('InnerEye.Common.fixed_paths.repository_root_directory',
                    return_value=tests_root_directory().parent):
        try:
            tags = {"model_name": config.model_name}
            azure_config = get_default_azure_config()
            if model_outside_package:
                azure_config.extra_code_directory = "Tests"  # contains DummyModel
            deployment_hook = lambda cfg, azure_cfg, mdl, is_ens: (Path(cfg.model_name), azure_cfg.docker_shm_size)
            ml_runner = MLRunner(config, azure_config, model_deployment_hook=deployment_hook)
            model, deployment_path, deployment_details = ml_runner.register_segmentation_model(
                workspace=ws,
                tags=tags,
                best_epoch=0,
                best_epoch_dice=0,
                checkpoint_paths=checkpoints,
                model_proc=ModelProcessing.DEFAULT)
            assert model is not None
            model_path = Path(model.get_model_path(model.name, model.version, ws))
            assert (model_path / fixed_paths.ENVIRONMENT_YAML_FILE_NAME).exists()
            assert (model_path / Path("InnerEye/ML/runner.py")).exists()
            assert deployment_path == Path(config.model_name)
            assert deployment_details == azure_config.docker_shm_size

            # move test data into the data folder to simulate an actual run
            train_and_test_data_dir = full_ml_test_data_path("train_and_test_data")

            img_channel_1_name = "id1_channel1.nii.gz"
            img_channel_1_path = train_and_test_data_dir / img_channel_1_name
            img_channel_2_name = "id1_channel2.nii.gz"
            img_channel_2_path = train_and_test_data_dir / img_channel_2_name

            # download the registered model and test that we can run the score pipeline on it
            model_root = Path(model.download(str(test_output_dirs.root_dir)))
            # create a dummy datastore to store model checkpoints and image data
            # this simulates the code shapshot being executed in a real run
            test_datastore = test_output_dirs.root_dir / "test_datastore"
            shutil.move(
                str(model_root / "test_outputs"),
                str(test_datastore / RELATIVE_TEST_OUTPUTS_PATH)
            )
            data_root = test_datastore / DEFAULT_DATA_FOLDER
            os.makedirs(data_root)
            shutil.copy(str(img_channel_1_path), data_root)
            shutil.copy(str(img_channel_2_path), data_root)

            # run score pipeline as a separate process using the python_wrapper.py code to simulate a real run
            return_code = SubprocessConfig(process="python", args=[
                str(model_root / "python_wrapper.py"),
                "--spawnprocess=python",
                str(model_root / "score.py"),
                f"--data-folder={str(test_datastore)}",
                f"--test_image_channels={img_channel_1_name},{img_channel_2_name}",
                "--use_gpu=False"
            ]).spawn_and_monitor_subprocess()

            # check that the process completed as expected
            assert return_code == 0
            expected_segmentation_path = Path(model_root) / DEFAULT_RESULT_IMAGE_NAME
            assert expected_segmentation_path.exists()

            # sanity check the resulting segmentation
            expected_shape = get_nifti_shape(img_channel_1_path)
            image_header = get_unit_image_header()
            assert_nifti_content(str(expected_segmentation_path), expected_shape, image_header, [0], np.ubyte)

        finally:
            # delete the registered model, and any downloaded artifacts
            shutil.rmtree(test_output_dirs.root_dir)
            if model and model_path:
                model.delete()
                shutil.rmtree(model_path)


def test_register_model_invalid() -> None:
    ws = get_default_workspace()
    config = get_model_loader().create_model_config_from_name("Lung")
    with pytest.raises(Exception):
        ml_runner = MLRunner(config, None)
        ml_runner.register_segmentation_model(
            run=Run.get_context(),
            workspace=ws,
            best_epoch=0,
            best_epoch_dice=0,
            checkpoint_paths=checkpoint_paths,
            model_proc=ModelProcessing.DEFAULT
        )
    with pytest.raises(Exception):
        ml_runner = MLRunner(config, get_default_azure_config())
        ml_runner.register_segmentation_model(
            best_epoch=0,
            best_epoch_dice=0,
            checkpoint_paths=checkpoint_paths,
            model_proc=ModelProcessing.DEFAULT
        )


@pytest.mark.parametrize("is_ensemble", [True, False])
@pytest.mark.parametrize("extra_code_directory", ["TestsOutsidePackage", ""])
def test_get_child_paths(is_ensemble: bool, extra_code_directory: str) -> None:
    checkpoints = checkpoint_paths * 2 if is_ensemble else checkpoint_paths
    path_to_root = tests_root_directory().parent
    azure_config = AzureConfig(extra_code_directory=extra_code_directory)
    fake_model = ModelConfigBase(azure_dataset_id="fake_dataset_id")
    ml_runner = MLRunner(model_config=fake_model, azure_config=azure_config, project_root=path_to_root)
    child_paths = ml_runner.get_child_paths(checkpoints)
    assert fixed_paths.ENVIRONMENT_YAML_FILE_NAME in child_paths
    assert fixed_paths.MODEL_INFERENCE_JSON_FILE_NAME in child_paths
    assert str(Path("InnerEye/ML/runner.py")) in child_paths
    assert str(Path("InnerEye/ML/model_testing.py")) in child_paths
    assert str(Path("InnerEye/Common/fixed_paths.py")) in child_paths
    assert str(Path("InnerEye/Common/common_util.py")) in child_paths
    trm = str(Path("TestsOutsidePackage/test_register_model.py"))
    if extra_code_directory:
        assert trm in child_paths
    else:
        assert trm not in child_paths
    assert all([x.relative_to(path_to_root) for x in checkpoints])


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
