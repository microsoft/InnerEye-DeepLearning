#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import sys
from pathlib import Path

from azureml.core import Keyvault

from InnerEye.Azure.azure_config import AzureConfig

if __name__ == "__main__":
    """
    Script to set the values in the AzureML workspace keyvault.
    argument [0]: Path to yaml file with workspace settings
    argument [1]: key for storage account
    argument [2]: key for datasets storage account
    """
    yaml_path = sys.argv[1]
    azure_config = AzureConfig.from_yaml(Path(yaml_path))
    workspace = azure_config.get_workspace()
    keyvault = Keyvault(workspace)
    keyvault.set_secret(azure_config.storage_account_secret_name, sys.argv[2])
    keyvault.set_secret(azure_config.datasets_storage_account_secret_name, sys.argv[3])
