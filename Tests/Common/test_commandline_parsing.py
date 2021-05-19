#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from unittest import mock

import pytest

from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import logging_to_stdout
from InnerEye.Common.fixed_paths import DEFAULT_AML_UPLOAD_DIR, DEFAULT_LOGS_DIR_NAME
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.config import PhotometricNormalizationMethod, SegmentationModelBase
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.runner import Runner
from Tests.ML.configs.DummyModel import DummyModel


@pytest.mark.parametrize("is_container", [True, False])
@pytest.mark.parametrize("is_offline_run", [True, False])
@pytest.mark.parametrize("set_output_to", [True, False])
def test_create_ml_runner_args(is_container: bool,
                               test_output_dirs: OutputFolderForTests,
                               is_offline_run: bool,
                               set_output_to: bool) -> None:
    """Test round trip parsing of commandline arguments:
    From arguments to the Azure runner to the arguments of the ML runner, checking that
    whatever is passed on can be correctly parsed. It also checks that the output files go into the right place
    in local runs and in AzureML."""
    logging_to_stdout()
    model_name = "DummyContainerWithPlainLightning" if is_container else "DummyModel"
    if is_container:
        dataset_folder = Path("download")
    else:
        local_dataset = DummyModel().local_dataset
        assert local_dataset is not None
        dataset_folder = local_dataset
    outputs_folder = test_output_dirs.root_dir
    project_root = fixed_paths.repository_root_directory()
    model_configs_namespace = "Tests.ML.configs"

    args_list = [f"--model={model_name}", "--train=True", "--l_rate=100.0",
                 "--subscription_id", "Test1", "--tenant_id=Test2",
                 "--application_id", "Test3", "--azureml_datastore", "Test5"]

    # toggle the output_to flag off only for online runs
    if set_output_to or is_offline_run:
        args_list.append(f"--output_to={outputs_folder}")
    if not is_container:
        args_list.append("--norm_method=Simple Norm")

    args_list.append(f"--model_configs_namespace={model_configs_namespace}")

    with mock.patch("sys.argv", [""] + args_list):
        with mock.patch("InnerEye.ML.deep_learning_config.is_offline_run_context", return_value=is_offline_run):
            with mock.patch("InnerEye.ML.run_ml.MLRunner.run", return_value=None):
                with mock.patch("InnerEye.ML.run_ml.MLRunner.mount_or_download_dataset", return_value=dataset_folder):
                    runner = Runner(project_root=project_root, yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)
                    runner.parse_and_load_model()
                    # Only when calling config.create_filesystem we expect to see the correct paths, and this happens
                    # inside run_in_situ
                    runner.run_in_situ()
                    azure_config = runner.azure_config
                    container_or_legacy_config = runner.lightning_container if is_container else runner.model_config
    assert azure_config.model == model_name
    assert container_or_legacy_config is not None
    if not is_container:
        assert isinstance(container_or_legacy_config, DeepLearningConfig)
        assert container_or_legacy_config.norm_method == PhotometricNormalizationMethod.SimpleNorm
    if set_output_to or is_offline_run:
        # The actual output folder must be a subfolder of the folder given on the commandline. The folder will contain
        # a timestamp, that will start with the year number, hence will start with 20...
        assert str(container_or_legacy_config.outputs_folder).startswith(str(outputs_folder / "20"))
        assert container_or_legacy_config.logs_folder == \
               (container_or_legacy_config.outputs_folder / DEFAULT_LOGS_DIR_NAME)
    else:
        # For runs inside AzureML, the output folder is the project root (the root of the folders that are
        # included in the snapshot). The "outputs_to" argument will be ignored.
        assert container_or_legacy_config.outputs_folder == (project_root / DEFAULT_AML_UPLOAD_DIR)
        assert container_or_legacy_config.logs_folder == (project_root / DEFAULT_LOGS_DIR_NAME)

    assert not hasattr(container_or_legacy_config, "azureml_datastore")


def test_overridable_properties() -> None:
    """
    Test to make sure all valid types can be parsed by the config parser
    """
    overridable = ["--num_dataload_workers=100",
                   "--local_dataset=hello_world",
                   "--norm_method=Simple Norm",
                   "--l_rate=100.0",
                   "--test_crop_size=1,2,3",
                   "--output_range=-100.0,100.0"]
    parser = SegmentationModelBase.create_argparser()
    args = vars(parser.parse_args(overridable))
    assert args["num_dataload_workers"] == 100
    assert str(args["local_dataset"]) == "hello_world"
    assert args["norm_method"] == PhotometricNormalizationMethod.SimpleNorm
    assert args["test_crop_size"] == (1, 2, 3)
    assert args["output_range"] == (-100.0, 100.0)


def test_non_overridable_properties() -> None:
    """
    Test to make sure properties that are private or marked as constant/readonly/NON OVERRIDABLE are not
    configurable through the commandline.
    """
    non_overridable = ["--" + k + "=" + str(v.default) for k, v in SegmentationModelBase.params().items()
                       if k not in SegmentationModelBase.get_overridable_parameters().keys()]
    parser = SegmentationModelBase.create_argparser()
    # try to parse the non overridable arguments
    _, unknown = parser.parse_known_args(non_overridable)
    assert all([x in unknown for x in non_overridable])


def test_read_yaml_file_into_args(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if the arguments for specifying the YAML config file with storage account, etc
    are correctly wired up.
    """
    empty_yaml = test_output_dirs.root_dir / "nothing.yaml"
    empty_yaml.write_text("variables:\n")
    with mock.patch("sys.argv", ["", "--model=Lung"]):
        # Default behaviour: tenant_id should be picked up from YAML
        runner1 = Runner(project_root=fixed_paths.repository_root_directory(),
                         yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)
        runner1.parse_and_load_model()
        assert len(runner1.azure_config.application_id) > 0
        assert len(runner1.azure_config.resource_group) > 0
        # When specifying a dummy YAML file that does not contain any settings, no information in AzureConfig should
        # be set. Some settings are read from a private settings file, most notably application ID, which should
        # be present on people's local dev boxes. Hence, only assert on `resource_group` here.
        runner2 = Runner(project_root=fixed_paths.repository_root_directory(),
                         yaml_config_file=empty_yaml)
        runner2.parse_and_load_model()
        assert runner2.azure_config.resource_group == ""


def test_parsing_with_custom_yaml(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if additional model or Azure config settings can be read correctly from YAML files.
    """
    yaml_file = test_output_dirs.root_dir / "custom.yml"
    yaml_file.write_text("""variables:
  tenant_id: 'foo'
  l_rate: 1e-4
  random_seed: 1
""")
    # Arguments partly to be set in AzureConfig, and partly in model config.
    args = ["",
            "--tenant_id=bar",
            "--model", "Lung",
            "--num_epochs", "42",
            "--random_seed", "2"]
    with mock.patch("sys.argv", args):
        runner = Runner(project_root=fixed_paths.repository_root_directory(),
                        yaml_config_file=yaml_file)
        loader_result = runner.parse_and_load_model()
    assert runner.azure_config is not None
    assert runner.model_config is not None
    # This is only present in yaml
    # This is present in yaml and command line, and the latter should be used.
    assert runner.azure_config.tenant_id == "bar"
    # Settings in model config: l_rate is only in yaml
    assert runner.model_config.l_rate == 1e-4
    # Settings in model config: num_epochs is only on commandline
    assert runner.model_config.num_epochs == 42
    # Settings in model config: random_seed is both in yaml and command line, the latter should be used
    assert runner.model_config.get_effective_random_seed() == 2
    assert loader_result.overrides == {"num_epochs": 42, "random_seed": 2}
