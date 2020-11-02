#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Any, Dict, Optional, cast, List

import yaml

from InnerEye.Common import fixed_paths

# The names of various keys that are required for accessing Azure.
# The keys are expected in either environment variables, or in a secrets file that
# lives in the repository root.
# All values must be in upper case. On Windows, os.environ is case insensitive, on Linux it
# is case sensitive.

# The application key to access the subscription via ServicePrincipal authentication.
from InnerEye.Common.fixed_paths import PRIVATE_SETTINGS_FILE


class SecretsHandling:
    """
    Contains method to read secrets from environment variables and/or files on disk.
    """

    def __init__(self, project_root: Path) -> None:
        """
        Creates a new instance of the class.
        :param project_root: The root folder of the project that starts the InnerEye run.
        """
        self.project_root = project_root

    def read_secrets_from_file(self, secrets_to_read: List[str]) -> Optional[Dict[str, str]]:
        """
        Reads the secrets from file in YAML format, and returns the contents as a dictionary. The YAML file is expected
        in the project root directory.
        :param secrets_to_read: The list of secret names to read from the YAML file. These will be converted to
        uppercase.
        :return: A dictionary with secrets, or None if the file does not exist.
        """
        secrets_file = self.project_root / fixed_paths.PROJECT_SECRETS_FILE
        if not secrets_file.is_file():
            return None
        all_keys_upper = set([name.upper() for name in secrets_to_read])
        d: Dict[str, str] = {}
        for line in secrets_file.read_text().splitlines():
            parts = line.strip().split("=", 1)
            key = parts[0].strip().upper()
            if key in all_keys_upper:
                d[key] = parts[1].strip()
        return d

    def get_secrets_from_environment_or_file(self, secrets_to_read: List[str]) -> Dict[str, Optional[str]]:
        """
        Attempts to read secrets from the project secret file. If there is no secrets file, it returns all secrets
        in secrets_to_read read from environment variables. When reading from environment, if an expected
        secret is not found, its value will be None.
        :param secrets_to_read: The list of secret names to read from the YAML file. These will be converted to
        uppercase.
        """
        # Read all secrets from a local file if present, and sets the matching environment variables.
        # If no secrets file is present, no environment variable is modified or created.
        secrets_from_file = self.read_secrets_from_file(secrets_to_read=secrets_to_read)
        return secrets_from_file or {name: os.environ.get(name.upper(), None)  # type: ignore
                                     for name in secrets_to_read}

    def get_secret_from_environment(self, name: str, allow_missing: bool = False) -> Optional[str]:
        """
        Gets a password or key from the secrets file or environment variables.
        :param name: The name of the environment variable to read. It will be converted to uppercase.
        :param allow_missing: If true, the function returns None if there is no entry of the given name in
        any of the places searched. If false, missing entries will raise a ValueError.
        :return: Value of the secret. None, if there is no value and allow_missing is True.
        """

        def throw_or_return_none(message: str) -> Optional[str]:
            if allow_missing:
                return None
            else:
                raise ValueError(message)

        name = name.upper()
        secrets = self.get_secrets_from_environment_or_file(secrets_to_read=[name])
        if name not in secrets:
            return throw_or_return_none(f"There is no secret named '{name}' available.")
        value = secrets[name]
        if value is None or len(value) == 0:
            return throw_or_return_none(f"There is no value stored for the secret named '{name}'")
        return value


def read_all_settings(project_settings_file: Optional[Path] = None,
                      project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Reads settings from files in YAML format, and returns the union of settings found. The first settings file
    to read is `project_settings_file`. The second settings file is 'InnerEyePrivateSettings.yml' expected in
     the `project_root` folder. Settings in the private settings file
    override those in the project settings. Both settings files are expected in YAML format, with an entry called
    'variables'.
    :param project_settings_file: The first YAML settings file to read.
    :param project_root: The folder that can contain a 'InnerEyePrivateSettings.yml' file.
    :return: A dictionary mapping from string to variable value. The dictionary key is the union of variable names
    found in the two settings files.
    """
    private_settings_file = None
    if project_root and project_root.is_dir():
        private_settings_file = project_root / PRIVATE_SETTINGS_FILE
    return read_settings_and_merge(project_settings_file, private_settings_file)


def read_settings_and_merge(project_settings_file: Optional[Path] = None,
                            private_settings_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Reads settings from files in YAML format, and returns the union of settings found. First, the project settings
    file is read into a dictionary, then the private settings file is read. Settings in the private settings file
    override those in the project settings. Both settings files are expected in YAML format, with an entry called
    'variables'.
    :param project_settings_file: The first YAML settings file to read.
    :param private_settings_file: The second YAML settings file to read. Settings in this file has higher priority.
    :return: A dictionary mapping from string to variable value. The dictionary key is the union of variable names
    found in the two settings files.
    """
    result = dict()
    if project_settings_file:
        if not project_settings_file.is_file():
            raise FileNotFoundError(f"Settings file does not exist: {project_settings_file}")
        result = read_settings_yaml_file(yaml_file=project_settings_file)
    if private_settings_file and private_settings_file.is_file():
        dict2 = read_settings_yaml_file(yaml_file=private_settings_file)
        for key, value in dict2.items():
            result[key] = value
    return result


def read_settings_yaml_file(yaml_file: Path) -> Dict[str, Any]:
    """
    Reads a YAML file, that is expected to contain an entry 'variables'. Returns the dictionary for the 'variables'
    section of the file.
    :param yaml_file: The yaml file to read.
    :return: A dictionary with the variables from the yaml file.
    """
    if yaml_file is None:
        return dict()
    yaml_contents = yaml.load(yaml_file.open('r'), Loader=yaml.Loader)
    v = "variables"
    if v in yaml_contents:
        if yaml_contents[v]:
            return cast(Dict[str, Any], yaml_contents[v])
        # If the file only contains the "variable:" prefix, but nothing below, then yaml_contents becomes None
        return dict()
    else:
        raise KeyError(f"The Yaml file must contain a section '{v}', but that was not found in {yaml_file}")
