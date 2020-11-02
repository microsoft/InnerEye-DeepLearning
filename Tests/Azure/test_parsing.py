#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from unittest import mock

import pytest

from InnerEye.Azure.azure_config import AZURECONFIG_SUBMIT_TO_AZUREML, AzureConfig, SourceConfig
from InnerEye.Azure.azure_runner import create_runner_parser, parse_args_and_add_yaml_variables, \
    run_duration_string_to_seconds
from InnerEye.Azure.parser_util import _is_empty_or_empty_string_list, item_to_script_param
from InnerEye.Common import fixed_paths
from InnerEye.ML.config import SegmentationModelBase


@pytest.mark.parametrize("items", ["", [], [""], [[]]])
def test_is_empty_list(items: Any) -> None:
    """
    Check empty items identified correctly.
    """
    assert _is_empty_or_empty_string_list(items)


@pytest.mark.parametrize("items", [[1], ["a"], ["", ""]])
def test_is_not_empty_list(items: Any) -> None:
    """
    Check non-empty items identified correctly.
    """
    assert not _is_empty_or_empty_string_list(items)


class MyTestEnum(Enum):
    Value1 = "a"


def test_item_to_script_param() -> None:
    """
    Check that items are converted to strings representing script parameters correctly.
    """
    assert "foo" == item_to_script_param("foo")
    assert "\"foo bar\"" == item_to_script_param("foo bar")
    assert "1,2" == item_to_script_param([1, 2])
    assert "1" == item_to_script_param([1])
    assert item_to_script_param(None) is None
    assert item_to_script_param([]) is None
    assert item_to_script_param([""]) is None
    assert item_to_script_param(MyTestEnum.Value1) == "a"


def test_create_runner_parser_check_fails_unknown() -> None:
    """
    Test parsing of commandline arguments: Check if unknown arguments fail the parsing
    """
    azure_parser = create_runner_parser(SegmentationModelBase)
    valid_args = ["--model=Lung"]
    invalid_args = ["--unknown=1"]
    # Ensure that the valid arguments that we provide later do actually parse correctly
    with mock.patch("sys.argv", [""] + valid_args):
        result = parse_args_and_add_yaml_variables(azure_parser, fail_on_unknown_args=True)
    assert "model" in result.args
    assert result.args["model"] == "Lung"
    # Supply both valid and invalid arguments, and we expect a failure because of the invalid ones:
    with mock.patch("sys.argv", [""] + valid_args + invalid_args):
        with pytest.raises(Exception) as e_info:
            parse_args_and_add_yaml_variables(azure_parser, fail_on_unknown_args=True)
    assert str(e_info.value) == 'Unknown arguments: [\'--unknown=1\']'
    # Supply both valid and invalid arguments, and we expect that the invalid ones are silently ignored:
    with mock.patch("sys.argv", [""] + valid_args + invalid_args):
        result = parse_args_and_add_yaml_variables(azure_parser, fail_on_unknown_args=False)
    assert "model" in result.args
    assert result.args["model"] == "Lung"
    assert invalid_args[0] in result.unknown


@pytest.mark.parametrize("with_config", [True, False])
def test_create_runner_parser(with_config: bool) -> None:
    """
    Test parsing of commandline arguments:
    From arguments to the runner, can we reconstruct arguments for AzureConfig and for the model config?
    Check that default and non-default arguments are set correctly and recognized as default/non-default.
    """
    azure_parser = create_runner_parser(SegmentationModelBase if with_config else None)
    args_list = ["--model=Lung", "--train=False", "--l_rate=100.0",
                 "--unknown=1", "--subscription_id", "Test1", "--tenant_id=Test2",
                 "--application_id", "Test3", "--datasets_storage_account=Test4",
                 "--log_level=INFO",
                 # Normally we don't use extra index URLs in InnerEye, hence this won't be set in YAML.
                 "--pip_extra_index_url=foo"]
    with mock.patch("sys.argv", [""] + args_list):
        parser_result = parse_args_and_add_yaml_variables(azure_parser, yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)
    azure_config = AzureConfig(**parser_result.args)

    # These values have been set on the commandline, to values that are not the parser defaults.
    non_default_args = {
        "datasets_storage_account": "Test4",
        "train": False,
        "model": "Lung",
        "subscription_id": "Test1",
        "application_id": "Test3",
    }
    for prop, value in non_default_args.items():
        assert prop in parser_result.args, f"Property {prop} missing in args"
        assert parser_result.args[prop] == value, f"Property {prop} does not have the expected value"
        assert getattr(azure_config, prop) == value, f"Property {prop} not in object"
        assert parser_result.overrides[prop] == value, \
            f"Property {prop} has a non-default value, and should be recognized as such."

    # log_level is set on the commandline, to a value that is equal to the default. It should be recognized as an
    # override.
    log_level = "log_level"
    assert log_level in parser_result.args
    assert parser_result.args[log_level] == "INFO"
    assert log_level in parser_result.overrides
    assert parser_result.overrides[log_level] == "INFO"

    # These next variables should have been read from YAML. They should be in the args dictionary and in the object,
    # but not in the list overrides
    from_yaml = {
        "workspace_name": "InnerEye-DeepLearning",
        "datasets_container": "datasets",
    }
    for prop, value in from_yaml.items():
        assert prop in parser_result.args, f"Property {prop} missing in args"
        assert parser_result.args[prop] == value, f"Property {prop} does not have the expected value"
        assert getattr(azure_config, prop) == value, f"Property {prop} not in object"
        assert prop not in parser_result.overrides, f"Property {prop} should not be listed as having a " \
                                                    f"non-default value"

    assert "unknown" not in parser_result.args
    l_rate = "l_rate"
    if with_config:
        assert l_rate in parser_result.args
        assert parser_result.args[l_rate] == 100.0
        assert parser_result.unknown == ["--unknown=1"]
    else:
        assert l_rate not in parser_result.args
        assert parser_result.unknown == ["--l_rate=100.0", "--unknown=1"]


def test_azureml_submit_constant() -> None:
    """
    Make sure the config has the 'submit to azureml' key.
    """
    azure_config = AzureConfig()
    assert hasattr(azure_config, AZURECONFIG_SUBMIT_TO_AZUREML)


def test_source_config_set_params() -> None:
    """
    Check that commandline arguments are set correctly when submitting the script to AzureML.
    In particular, the azureml flag should be omitted, irrespective of how the argument is written.
    """
    s = SourceConfig(root_folder=Path(""), entry_script=Path("something.py"), conda_dependencies_files=[])

    def assert_has_params(expected_args: str) -> None:
        assert s.script_params is not None
        # Arguments are in the keys of the dictionary only, and should have been added in the right order
        assert " ".join(s.script_params.keys()) == expected_args

    with mock.patch("sys.argv", ["", "some", "--param", "1", f"--{AZURECONFIG_SUBMIT_TO_AZUREML}=True", "more"]):
        s.set_script_params_except_submit_flag()
    assert_has_params("some --param 1 more")
    with mock.patch("sys.argv", ["", "some", "--param", "1", f"--{AZURECONFIG_SUBMIT_TO_AZUREML}", "False", "more"]):
        s.set_script_params_except_submit_flag()
    assert_has_params("some --param 1 more")
    # Arguments where azureml is just the prefix should not be removed.
    with mock.patch("sys.argv", ["", "some", f"--{AZURECONFIG_SUBMIT_TO_AZUREML}foo", "False", "more"]):
        s.set_script_params_except_submit_flag()
    assert_has_params(f"some --{AZURECONFIG_SUBMIT_TO_AZUREML}foo False more")


@pytest.mark.parametrize(["s", "expected"],
                         [
                             ("1s", 1),
                             ("0.5m", 30),
                             ("1.5h", 90 * 60),
                             ("1.0d", 24 * 3600),
                             ("", None),
                         ])
def test_run_duration(s: str, expected: Optional[float]) -> None:
    actual = run_duration_string_to_seconds(s)
    assert actual == expected
    if expected:
        assert isinstance(actual, int)


def test_run_duration_fails() -> None:
    with pytest.raises(Exception):
        run_duration_string_to_seconds("17b")
