#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_runner import create_dataset_consumptions
from Tests.ML.util import get_default_azure_config


def test_validate() -> None:
    with pytest.raises(ValueError) as ex:
        AzureConfig(only_register_model=True)
    assert ex.value.args[0] == "If only_register_model is set, must also provide a valid run_recovery_id"


def test_dataset_consumption1() -> None:
    """
    Test that an empty dataset ID will not produce any dataset consumption.
    """
    azure_config = get_default_azure_config()
    assert len(create_dataset_consumptions(azure_config, [""], [""])) == 0


def test_dataset_consumption2() -> None:
    """
    Test that given mount point with empty dataset ID raises an error
    """
    azure_config = get_default_azure_config()
    with pytest.raises(ValueError) as ex:
        create_dataset_consumptions(azure_config, [""], ["foo"])
    assert "but a mount point has been provided" in str(ex)


def test_dataset_consumption3() -> None:
    """
    Test that a matching number of mount points is created.
    """
    azure_config = get_default_azure_config()
    assert len(create_dataset_consumptions(azure_config, ["test-dataset", "test-dataset"], [])) == 2


def test_dataset_consumption4() -> None:
    """
    Test error handling for number of mount points.
    """
    azure_config = get_default_azure_config()
    with pytest.raises(ValueError) as ex:
        create_dataset_consumptions(azure_config, ["test-dataset", "test-dataset"], ["foo"])
    assert "must equal the number of Azure dataset IDs" in str(ex)


def test_dataset_consumption5() -> None:
    """
    Test error handling for empty dataset IDs.
    """
    azure_config = get_default_azure_config()
    with pytest.raises(ValueError) as ex:
        azure_config.get_or_create_dataset("")
    assert "No dataset ID provided" in str(ex)
