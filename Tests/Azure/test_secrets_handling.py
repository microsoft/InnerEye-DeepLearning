#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os

import pytest

from InnerEye.Azure import secrets_handling
from InnerEye.Azure.secrets_handling import SecretsHandling, read_variables_from_yaml
from InnerEye.Common import fixed_paths
from Tests.fixed_paths_for_tests import full_azure_test_data_path


def test_environ_get_set() -> None:
    """
    Test that environment variables are set correctly.
    """
    variables = {"foo_env": "foo_value", "bar_env": "bar_value"}
    for name in variables.keys():
        if name in os.environ:
            del os.environ[name]
    secrets_handling.set_environment_variables(variables)
    for name, value in variables.items():
        name = name.upper()
        assert name in os.environ
        assert os.environ[name] == value


def test_get_secrets() -> None:
    """
    Test that secrets can always be retrieved correctly from the environment.
    When running on the local dev box, the secrets would be read from a secrets file in the repository root directory
    and be written to the environment, retrieved later.
    When running in Azure, the secrets would be set as environment variables directly
    in the build definition.
    """
    print("Environment variables:")
    for env_variable, value in os.environ.items():
        print("{}: {}".format(env_variable, value))
    secrets_handler = SecretsHandling(project_root=fixed_paths.repository_root_directory())
    secrets = secrets_handler.get_secrets_from_environment_or_file()
    for name in secrets_handling.SECRETS_IN_ENVIRONMENT:
        assert name in secrets, "No value found for {}".format(name)
        assert secrets[name] is not None, "Value for {} is empty".format(name)
        # Variable names should automatically be converted to uppercase when using get_secret:
        assert secrets_handler.get_secret_from_environment(name=name.lower()) is not None
    no_such_variable = "no_such_variable"
    with pytest.raises(ValueError):
        secrets_handler.get_secret_from_environment(name=no_such_variable)
    assert secrets_handler.get_secret_from_environment(name=no_such_variable, allow_missing=True) is None


def test_all_secrets_is_upper() -> None:
    """
    Tests that all secret keys in SECRETS_IN_ENVIRONMENT are uppercase strings.
    """
    for name in secrets_handling.SECRETS_IN_ENVIRONMENT:
        assert name == name.upper(), "Secret '{}' should have a only uppercase value".format(name)


def test_read_variables_from_yaml() -> None:
    """
    Test that variables are read from a yaml file correctly.
    """
    # this will return a dictionary of all variables in the yaml file
    yaml_path = full_azure_test_data_path('settings.yml')
    vars_dict = secrets_handling.read_variables_from_yaml(yaml_path)
    assert vars_dict == {'some_key': 'some_val'}
    # YAML file missing "variables" key should raise key error
    fail_yaml_path = full_azure_test_data_path('settings_with_missing_section.yml')
    with pytest.raises(KeyError):
        secrets_handling.read_variables_from_yaml(fail_yaml_path)


def test_parse_yaml() -> None:
    assert os.path.isfile(fixed_paths.SETTINGS_YAML_FILE)
    variables = read_variables_from_yaml(fixed_paths.SETTINGS_YAML_FILE)
    # Check that there are at least two of the variables that we know of
    tenant_id = "tenant_id"
    assert tenant_id in variables
    assert variables[tenant_id] == "72f988bf-86f1-41af-91ab-2d7cd011db47"
    assert "datasets_container" in variables
