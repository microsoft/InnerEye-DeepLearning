#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest

from InnerEye.Azure.azure_config import AzureConfig


def test_validate() -> None:
    with pytest.raises(ValueError) as ex:
        AzureConfig(only_register_model=True)
    assert ex.value.args[0] == "If only_register_model is set, must also provide a valid run_recovery_id"
