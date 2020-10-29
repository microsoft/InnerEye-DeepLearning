#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest

from InnerEye.Azure.azure_config import AzureConfig


def test_validate() -> None:
    with pytest.raises(ValueError):
        AzureConfig(register_model_only_for_epoch=True)
