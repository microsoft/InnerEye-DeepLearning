#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest

from InnerEye.Azure.azure_runner import RUN_RECOVERY_FILE


@pytest.mark.after_training
def test_model_file_structure() -> None:
    """
    Downloads the model that was built in the most recent run, and checks if its file structure is as expected.
    """
    assert RUN_RECOVERY_FILE.is_file()
