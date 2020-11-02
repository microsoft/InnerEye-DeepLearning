#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from typing import Dict

import pytest

from InnerEye.Azure import secrets_handling
from InnerEye.Azure.secrets_handling import SecretsHandling
from InnerEye.Common import fixed_paths
from InnerEye.Common.output_directories import OutputFolderForTests

# A list of all secrets that are stored in environment variables or local secrets files.
SECRETS_IN_ENVIRONMENT = [fixed_paths.SERVICE_PRINCIPAL_KEY]


def set_environment_variables(variables: Dict[str, str]) -> None:
    """
    Creates an environment variable for each entry in the given dictionary. The dictionary key is the variable
    name, it will be converted to uppercase before setting.
    :param variables: The variable names and their associated values that should be set.
    """
    for name, value in variables.items():
        os.environ[name.upper()] = value


def test_environ_get_set() -> None:
    """
    Test that environment variables are set correctly.
    """
    variables = {"foo_env": "foo_value", "bar_env": "bar_value"}
    for name in variables.keys():
        if name in os.environ:
            del os.environ[name]
    set_environment_variables(variables)
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
    secrets = secrets_handler.get_secrets_from_environment_or_file(SECRETS_IN_ENVIRONMENT)
    for name in SECRETS_IN_ENVIRONMENT:
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
    for name in SECRETS_IN_ENVIRONMENT:
        assert name == name.upper(), "Secret '{}' should have a only uppercase value".format(name)


def test_read_variables_from_yaml(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that variables are read from a yaml file correctly.
    """
    root = test_output_dirs.root_dir
    # this will return a dictionary of all variables in the yaml file
    yaml_path = root / "foo.yml"
    yaml_path.write_text("""variables:
  some_key: 'some_val'
  key2: 'val2'""")
    vars_dict = secrets_handling.read_all_settings(yaml_path)
    assert vars_dict == {"some_key": "some_val", "key2": "val2"}
    # YAML file missing "variables" key should raise key error
    fail_yaml_path = root / "error.yml"
    fail_yaml_path.write_text("""some_key: 'some_val'""")
    with pytest.raises(KeyError):
        secrets_handling.read_all_settings(fail_yaml_path)
    # Write a private settings file, and check if that is merged correctly.
    # Keys in the private settings file should have higher priority than those in the normal file.
    private_file = root / secrets_handling.PRIVATE_SETTINGS_FILE
    private_file.write_text("""variables:
  some_key: 'private_value'
  key42: 42
""")
    vars_with_private = secrets_handling.read_all_settings(yaml_path, project_root=root)
    assert vars_with_private == \
           {
               "some_key": "private_value",
               "key42": 42,
               "key2": "val2"
           }
    # Providing no files should return an empty dictionary
    vars_from_no_file = secrets_handling.read_all_settings()
    assert vars_from_no_file == {}
    # Provide only a project root with a private file:
    private_only = secrets_handling.read_all_settings(project_root=root)
    assert private_only == \
           {
               "some_key": "private_value",
               "key42": 42,
           }
    # Invalid file name should raise an exception
    does_not_exist = "does_not_exist"
    with pytest.raises(FileNotFoundError) as ex:
        secrets_handling.read_all_settings(project_settings_file=root / does_not_exist)
    assert does_not_exist in str(ex)


def test_parse_yaml() -> None:
    assert os.path.isfile(fixed_paths.SETTINGS_YAML_FILE)
    variables = secrets_handling.read_settings_yaml_file(fixed_paths.SETTINGS_YAML_FILE)
    # Check that there are at least two of the variables that we know of
    tenant_id = "tenant_id"
    assert tenant_id in variables
    assert variables[tenant_id] == "72f988bf-86f1-41af-91ab-2d7cd011db47"
    assert "datasets_container" in variables
