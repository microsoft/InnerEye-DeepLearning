#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from io import StringIO
from unittest import mock

import pandas as pd

from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.deep_learning_config import ARGS_TXT, DatasetParams, EssentialParams
from InnerEye.ML.lightning_base import InnerEyeContainer
from InnerEye.ML.lightning_container import LightningContainer, LightningWithInference
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.run_ml import MLRunner
from Tests.ML.configs.lightning_test_containers import DummyContainerWithModel
from Tests.ML.util import default_runner


def test_run_container_in_situ(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if we can get the config loader to load a Lightning container model, and then train locally.
    """
    runner = default_runner()
    local_dataset = test_output_dirs.root_dir / "dataset"
    local_dataset.mkdir()
    args = ["", "--model=DummyContainerWithModel", "--model_configs_namespace=Tests.ML.configs",
            f"--output_to={test_output_dirs.root_dir}", f"--local_dataset={local_dataset}"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = runner.run()
    assert actual_run is None
    assert isinstance(runner.lightning_container, DummyContainerWithModel)
    # Test if the outputs folder is relative to the folder that we specified via the commandline
    runner.lightning_container.outputs_folder.relative_to(test_output_dirs.root_dir)
    results = runner.lightning_container.outputs_folder
    # Test that the setup method has been called
    assert (runner.lightning_container.local_dataset / "setup.txt").is_file()
    # Test if all the files that are written during inference exist. Data for all 3 splits must be processed
    assert (results / "on_inference_start.txt").is_file()
    assert (results / "on_inference_end.txt").is_file()
    for mode in ModelExecutionMode:
        assert (results / f"on_inference_start_{mode.value}.txt").is_file()
        assert (results / f"on_inference_end_{mode.value}.txt").is_file()
        step_results = results / f"inference_step_{mode.value}.txt"
        assert step_results.is_file()
        # We should have one line per data item, and there are around 6 of them
        result_lines = [line for line in step_results.read_text().splitlines() if line.strip()]
        assert len(result_lines) >= 5
    metrics_per_split = pd.read_csv(results / "metrics_per_split.csv")
    # Training should have reduced the MSE to pretty much zero.
    expected = pd.read_csv(StringIO("""Split,MSE
Test,1e-7
Val,1e-7
Train,1e-7"""))
    pd.testing.assert_frame_equal(metrics_per_split, expected, check_less_precise=True)
    # Test if we have an args file that lists all parameters
    args_file = (results / ARGS_TXT).read_text()
    assert "Container:" in args_file
    assert "adam_betas" in args_file


def test_innereye_container_init() -> None:
    """
    Test if the constructor of the InnerEye container copies attributes as expected.
    """
    # The constructor should copy all fields that belong to either EssentialParams or DatasetParams from the
    # config object to the container.
    for (attrib, type_) in [("weights_url", EssentialParams), ("azure_dataset_id", DatasetParams)]:
        config = ModelConfigBase()
        assert hasattr(type_, attrib)
        assert hasattr(config, attrib)
        setattr(config, attrib, "foo")
        container = InnerEyeContainer(config)
        assert getattr(container, attrib) == "foo"


def test_create_fastmri_container() -> None:
    """
    Test if we can create a model that uses the fastMRI submodule. This is effectively just testing module imports,
    and if the submodule is created correctly.
    """
    from InnerEye.ML.configs.other.fastmri_varnet import VarNetWithInference, FastMriDemo
    FastMriDemo()
    VarNetWithInference()


def test_run_fastmri_container(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if we can get run the fastMRI model end-to-end.
    """
    runner = default_runner()
    dataset_dir = test_output_dirs.root_dir / "dataset"
    dataset_dir.mkdir(parents=True)
    args = ["", "--model=FastMriDemo",
            f"--output_to={test_output_dirs.root_dir}",
            f"--local_dataset={dataset_dir}"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = runner.run()
    assert actual_run is None
    from InnerEye.ML.configs.other.fastmri_varnet import FastMriDemo
    assert isinstance(runner.lightning_container, FastMriDemo)


def test_model_name_is_set(test_output_dirs: OutputFolderForTests) -> None:
    container = DummyContainerWithModel()
    container.local_dataset = test_output_dirs.root_dir
    runner = MLRunner(model_config=None, container=container)
    runner.setup()
    expected_name = "DummyContainerWithModel"
    assert runner.container._model_name == expected_name
    assert runner.container.model.output_params._model_name == expected_name
    assert expected_name in str(runner.container.model.outputs_folder)


class DummyContainerWithFields(LightningContainer):

    def __init__(self):
        super().__init__()
        self.perform_training_set_inference = True
        self.num_epochs = 123456
        self.l_rate = 1e-2

    def create_model(self) -> LightningWithInference:
        return LightningWithInference()


def test_container_to_str() -> None:
    """
    Test how a string representation of a container looks like.
    """
    c = DummyContainerWithFields()
    # Set any other field that is not done via the params library
    c.foo = "bar"
    s = str(c)
    print(s)
    assert "foo" in s
    assert "bar" in s
    assert "param" not in s
    assert "initialized" not in s
    assert "123456" in s
