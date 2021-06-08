#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from io import StringIO
from pathlib import Path
from typing import List
from unittest import mock

import pandas as pd
import pytest
from pytorch_lightning import LightningModule
from azureml.core import ScriptRunConfig
from azureml.train.hyperdrive.runconfig import HyperDriveConfig

from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.deep_learning_config import ARGS_TXT, DatasetParams, WorkflowParams
from InnerEye.ML.lightning_base import InnerEyeContainer
from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.run_ml import MLRunner
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.configs.lightning_test_containers import DummyContainerWithAzureDataset, DummyContainerWithHooks, DummyContainerWithModel, \
    DummyContainerWithPlainLightning
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
    assert runner.lightning_container.local_dataset is not None
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
    # Report generation must run
    assert (results / "create_report.txt").is_file()


def test_run_container_with_plain_lightning_in_situ(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if we can train a plain Lightning model, without any additional methods defined, end-to-end.
    """
    runner = default_runner()
    local_dataset = test_output_dirs.root_dir / "dataset"
    local_dataset.mkdir()
    args = ["", "--model=DummyContainerWithPlainLightning", "--model_configs_namespace=Tests.ML.configs",
            f"--output_to={test_output_dirs.root_dir}", f"--local_dataset={local_dataset}"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = runner.run()
    assert actual_run is None
    assert isinstance(runner.lightning_container, DummyContainerWithPlainLightning)
    # Test if the outputs folder is relative to the folder that we specified via the commandline
    runner.lightning_container.outputs_folder.relative_to(test_output_dirs.root_dir)
    results = runner.lightning_container.outputs_folder
    # Test if all the files that are written during inference exist.
    assert not (results / "on_inference_start.txt").is_file()
    assert (results / "test_step.txt").is_file()


def test_innereye_container_init() -> None:
    """
    Test if the constructor of the InnerEye container copies attributes as expected.
    """
    # The constructor should copy all fields that belong to either WorkflowParams or DatasetParams from the
    # config object to the container.
    for (attrib, type_) in [("weights_url", WorkflowParams), ("azure_dataset_id", DatasetParams)]:
        config = ModelConfigBase()
        assert hasattr(type_, attrib)
        assert hasattr(config, attrib)
        setattr(config, attrib, "foo")
        container = InnerEyeContainer(config)
        assert getattr(container, attrib) == "foo"


def test_copied_properties() -> None:
    config = ModelConfigBase()
    # This field lives in DatasetParams
    config.azure_dataset_id = "foo"
    # This field lives in WorkflowParams
    config.number_of_cross_validation_splits = 5
    assert config.perform_cross_validation
    container = InnerEyeContainer(config)
    assert container.azure_dataset_id == "foo"
    assert container.perform_cross_validation


def test_create_fastmri_container() -> None:
    """
    Test if we can create a model that uses the fastMRI submodule. This is effectively just testing module imports,
    and if the submodule is created correctly.
    """
    from InnerEye.ML.configs.other.fastmri_varnet import VarNetWithImageLogging
    from Tests.ML.configs.fastmri_random import FastMriOnRandomData
    FastMriOnRandomData()
    VarNetWithImageLogging()


@pytest.mark.gpu
def test_run_fastmri_container(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if we can get run the fastMRI model end-to-end. This takes about 2min on a CPU machine, hence only run
    in AzureML
    """
    runner = default_runner()
    dataset_dir = test_output_dirs.root_dir / "dataset"
    dataset_dir.mkdir(parents=True)
    args = ["", "--model=FastMriOnRandomData",
            f"--output_to={test_output_dirs.root_dir}",
            "--model_configs_namespace=Tests.ML.configs"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = runner.run()
    assert actual_run is None
    from Tests.ML.configs.fastmri_random import FastMriOnRandomData
    assert isinstance(runner.lightning_container, FastMriOnRandomData)


def test_model_name_is_set(test_output_dirs: OutputFolderForTests) -> None:
    container = DummyContainerWithModel()
    container.local_dataset = test_output_dirs.root_dir
    runner = MLRunner(model_config=None, container=container)
    runner.setup()
    expected_name = "DummyContainerWithModel"
    assert runner.container._model_name == expected_name
    assert expected_name in str(runner.container.outputs_folder)


def test_model_name_for_innereye_container() -> None:
    """
    Test if the InnerEye container picks up the name of the model correctly. The name will impact the output folder
    structure that is created.
    """
    expected_name = "DummyModel"
    model = DummyModel()
    assert model.model_name == expected_name
    container = InnerEyeContainer(model)
    assert container.model_name == expected_name


class DummyContainerWithFields(LightningContainer):

    def __init__(self) -> None:
        super().__init__()
        self.perform_training_set_inference = True
        self.num_epochs = 123456
        self.l_rate = 1e-2

    def create_model(self) -> LightningModule:
        return LightningModule()


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
    assert "123456" in s
    # These two are internal variables of the params library, and should be skipped.
    # The extra spaces are on purpose, because there are fields in the container that contain the string "param".
    # The output of __str__ is a big table in the format "    field_name        : field_value"
    assert " param                " not in s
    assert " initialized          " not in s


def test_file_system_with_subfolders(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if a subfolder can be created within the output folder structure, for use with cross validation.
    """
    model = DummyModel()
    model.set_output_to(test_output_dirs.root_dir)
    container = InnerEyeContainer(model)
    # File system should be copied from model config to container
    assert container.file_system_config == model.file_system_config
    runner = MLRunner(model_config=model)
    runner.setup()
    assert str(runner.container.outputs_folder).endswith(model.model_name)
    output_subfolder = "foo"
    expected_folder = runner.container.outputs_folder / output_subfolder
    runner = MLRunner(model_config=model, output_subfolder=output_subfolder)
    runner.setup()
    assert runner.container.outputs_folder == expected_folder


def test_optim_params1(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if the optimizer parameters are read correctly for InnerEye configs.
    """
    model = DummyModel()
    model.set_output_to(test_output_dirs.root_dir)
    runner = MLRunner(model_config=model)
    runner.setup()
    lightning_model = runner.container.model
    optim, _ = lightning_model.configure_optimizers()
    assert optim[0].param_groups[0]["lr"] == 1e-3


def test_optim_params2(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if the optimizer parameters are read correctly for containers.
    """
    container = DummyContainerWithModel()
    container.local_dataset = test_output_dirs.root_dir
    runner = MLRunner(model_config=None, container=container)
    runner.setup()
    lightning_model = runner.container.model
    optim, _ = lightning_model.configure_optimizers()
    expected_lr = 1e-1
    assert container.l_rate == expected_lr
    assert optim[0].param_groups[0]["lr"] == expected_lr


def test_extra_directory_available(test_output_dirs: OutputFolderForTests) -> None:
    def _create_container(extra_local_dataset_paths: List[Path] = [],
                          extra_azure_dataset_ids: List[str] = []) -> LightningContainer:
        container = DummyContainerWithModel()
        container.local_dataset = test_output_dirs.root_dir
        container.extra_local_dataset_paths = extra_local_dataset_paths
        container.extra_azure_dataset_ids = extra_azure_dataset_ids
        runner = MLRunner(model_config=None, container=container)
        runner.setup()
        return runner.container

    extra_local_dataset_paths = [test_output_dirs.root_dir, test_output_dirs.root_dir]
    container = _create_container(extra_local_dataset_paths)
    assert container.extra_local_dataset_paths == [test_output_dirs.root_dir, test_output_dirs.root_dir]

    # Check default behavior (no extra datasets provided)
    container = _create_container()
    assert container.extra_local_dataset_paths == []


def test_container_hooks(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if the hooks before training are called at the right place and in the right order.
    """
    container = DummyContainerWithHooks()
    container.local_dataset = test_output_dirs.root_dir
    runner = MLRunner(model_config=None, container=container)
    runner.setup()
    runner.run()
    # The hooks in DummyContainerWithHooks itself check that the hooks are called in the right order. Here,
    # only check that they have all been called.
    for file in ["global_rank_zero.txt", "local_rank_zero.txt", "all_ranks.txt"]:
        assert (runner.container.outputs_folder / file).is_file(), f"Missing file: {file}"

@pytest.mark.parametrize("number_of_cross_validation_splits", [0, 2])
def test_get_hyperdrive_config(number_of_cross_validation_splits: int,
                               test_output_dirs: OutputFolderForTests) -> None:
    """
    Testing that the hyperdrive config returned for the lightnig container is right for submitting
    to AzureML.

    Note that because the function get_hyperdrive_config now lives in the super class WorkflowParams,
    it is also tested for other aspects of functionality by a test of the same name in
    Tests.ML.test_model_config_base.
    """
    container = DummyContainerWithAzureDataset()
    container.number_of_cross_validation_splits = number_of_cross_validation_splits
    run_config = ScriptRunConfig(
        source_directory=str(test_output_dirs.root_dir),
        script=str(Path("something.py")),
        arguments=["foo"],
        compute_target="EnormousCluster")
    if number_of_cross_validation_splits == 0:
        with pytest.raises(NotImplementedError) as not_implemented_error:
            container.get_hyperdrive_config(run_config=run_config)
        assert 'Parameter search is not implemented' in str(not_implemented_error.value)
        # The error should be thrown by 
        #     InnerEye.ML.lightning_container.LightningContainer.get_parameter_search_hyperdrive_config
        # since number_of_cross_validation_splits == 0 implies a parameter search hyperdrive config and
        # not a cross validation one.
    else:
        hd_config = container.get_hyperdrive_config(run_config=run_config)
        assert isinstance(hd_config, HyperDriveConfig)
