#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Any, Dict, Optional, cast

import yaml

from InnerEye.Common import fixed_paths

# The names of various keys that are required for accessing Azure.
# The keys are expected in either environment variables, or in a secrets file that
# lives in the repository root.
# All values must be in upper case. On Windows, os.environ is case insensitive, on Linux it
# is case sensitive.
# The application key to access the subscription via ServicePrincipal authentication.
APPLICATION_KEY = "APPLICATION_KEY"

# A list of all secrets that are stored in environment variables or local secrets files.
SECRETS_IN_ENVIRONMENT = [APPLICATION_KEY]


def set_environment_variables(variables: Dict[str, str]) -> None:
    """
    Creates an environment variable for each entry in the given dictionary. The dictionary key is the variable
    name, it will be converted to uppercase before setting.
    :param variables: The variable names and their associated values that should be set.
    """
    for name, value in variables.items():
        os.environ[name.upper()] = value


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

    def read_secrets_from_file(self) -> Optional[Dict[str, str]]:
        """
        Reads the secrets from a file, and returns the relevant secrets as a dictionary.
        Searches for the secrets file in the project root.
        :return: A dictionary with secrets.
        """
        secrets_file = self.project_root / fixed_paths.PROJECT_SECRETS_FILE
        if not secrets_file.is_file():
            return None
        all_keys_upper = set([name.upper() for name in SECRETS_IN_ENVIRONMENT])
        d: Dict[str, str] = {}
        for line in secrets_file.read_text().splitlines():
            parts = line.strip().split("=", 1)
            key = parts[0].strip().upper()
            if key in all_keys_upper:
                d[key] = parts[1].strip()
        return d

    def get_secrets_from_environment_or_file(self) -> Dict[str, Optional[str]]:
        """
        Attempts to read secrets from the project secret file. If there is no secrets file, it returns all secrets
        in SECRETS_IN_ENVIRONMENT read from environment variables. When reading from environment, if an expected
        secret is not found, its value will be None.
        """
        # Read all secrets from a local file if present, and sets the matching environment variables.
        # If no secrets file is present, no environment variable is modified or created.
        secrets_from_file = self.read_secrets_from_file()
        return secrets_from_file or {name: os.environ.get(name.upper(), None)  # type: ignore
                                     for name in SECRETS_IN_ENVIRONMENT}

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
        secrets = self.get_secrets_from_environment_or_file()
        if name not in secrets:
            return throw_or_return_none(f"There is no secret named '{name}' available.")
        value = secrets[name]
        if value is None or len(value) == 0:
            return throw_or_return_none(f"There is no value stored for the secret named '{name}'")
        return value


def read_variables_from_yaml(yaml_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Reads a YAML file, that is expected to contain an entry 'variables'. Returns the dictionary for the 'variables'
    section of the file. If the file name is not given, an empty dictionary will be returned.
    :param yaml_file: The yaml file to read.
    :return: A dictionary with the variables from the yaml file.
    """
    if yaml_file is None:
        return dict()
    yaml_contents = yaml.load(yaml_file.open('r'), Loader=yaml.Loader)
    v = "variables"
    if v in yaml_contents:
        return cast(Dict[str, Any], yaml_contents[v])
    else:
        raise KeyError(f"The Yaml file was expected to contain a section '{v}'")
