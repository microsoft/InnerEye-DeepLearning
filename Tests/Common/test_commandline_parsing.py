#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from unittest import mock

import pytest

from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import logging_to_stdout
from InnerEye.Common.fixed_paths import DEFAULT_AML_UPLOAD_DIR, DEFAULT_LOGS_DIR_NAME
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.config import PhotometricNormalizationMethod, SegmentationModelBase
from InnerEye.ML.runner import Runner


@pytest.mark.parametrize("is_default_namespace", [True, False])
@pytest.mark.parametrize("is_offline_run", [True, False])
def test_create_ml_runner_args(is_default_namespace: bool,
                               test_output_dirs: OutputFolderForTests,
                               is_offline_run: bool) -> None:
    """Test round trip parsing of commandline arguments:
    From arguments to the Azure runner to the arguments of the ML runner, checking that
    whatever is passed on can be correctly parsed."""
    logging_to_stdout()
    model_name = "Lung"
    outputs_folder = test_output_dirs.root_dir
    project_root = fixed_paths.repository_root_directory()
    if is_default_namespace:
        model_configs_namespace = None
    else:
        model_configs_namespace = "Tests.ML.configs"
        model_name = "DummyModel"

    args_list = [f"--model={model_name}", "--train=True", "--l_rate=100.0",
                 "--norm_method=Simple Norm", "--subscription_id", "Test1", "--tenant_id=Test2",
                 "--application_id", "Test3", "--datasets_storage_account=Test4", "--datasets_container", "Test5",
                 "--pytest_mark", "gpu", f"--output_to={outputs_folder}"]
    if not is_default_namespace:
        args_list.append(f"--model_configs_namespace={model_configs_namespace}")

    with mock.patch("sys.argv", [""] + args_list):
        with mock.patch("InnerEye.ML.deep_learning_config.is_offline_run_context", return_value=is_offline_run):
            runner = Runner(project_root=project_root, yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)
            runner.parse_and_load_model()
            azure_config = runner.azure_config
            model_config = runner.model_config
    assert azure_config.datasets_storage_account == "Test4"
    assert azure_config.model == model_name
    assert model_config.l_rate == 100.0
    assert model_config.norm_method == PhotometricNormalizationMethod.SimpleNorm
    if is_offline_run:
        # The actual output folder must be a subfolder of the folder given on the commandline. The folder will contain
        # a timestamp, that will start with the year number, hence will start with 20...
        assert str(model_config.outputs_folder).startswith(str(outputs_folder / "20"))
        assert model_config.logs_folder == (model_config.outputs_folder / DEFAULT_LOGS_DIR_NAME)
    else:
        # For runs inside AzureML, the output folder is the project root (the root of the folders that are
        # included in the snapshot). The "outputs_to" argument will be ignored.
        assert model_config.outputs_folder == (project_root / DEFAULT_AML_UPLOAD_DIR)
        assert model_config.logs_folder == (project_root / DEFAULT_LOGS_DIR_NAME)

    assert not hasattr(model_config, "datasets_storage_account")
    assert azure_config.pytest_mark == "gpu"


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
  datasets_storage_account: 'account'
  start_epoch: 7
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
    assert loader_result is not None
    assert runner.azure_config is not None
    # This is only present in yaml
    assert runner.azure_config.datasets_storage_account == "account"
    # This is present in yaml and command line, and the latter should be used.
    assert runner.azure_config.tenant_id == "bar"
    # Settings in model config: start_epoch is only in yaml
    assert runner.model_config.start_epoch == 7
    # Settings in model config: num_epochs is only on commandline
    assert runner.model_config.num_epochs == 42
    # Settings in model config: random_seed is both in yaml and command line, the latter should be used
    assert runner.model_config.get_effective_random_seed() == 2
    assert loader_result.overrides == {"num_epochs": 42, "random_seed": 2}
