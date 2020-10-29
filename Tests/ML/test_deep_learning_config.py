#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pytest

from pathlib import Path

from InnerEye.ML.deep_learning_config import DeepLearningConfig


def test_validate() -> None:
    # DeepLearningConfig cannot be initialized with neither of these these parameters set
    with pytest.raises(ValueError):
        DeepLearningConfig(local_dataset=None, azure_dataset_id="")
    # The following should be okay
    DeepLearningConfig(local_dataset=Path("foo"))
    DeepLearningConfig(azure_dataset_id="bar")

    # DeepLearningConfig cannot be initialized with both these parameters set
    with pytest.raises(ValueError):
        DeepLearningConfig(local_dataset=Path("foo"),
                           local_weights_path=Path("foo"), weights_url="bar")
    # The following should be okay
    DeepLearningConfig(local_dataset=Path("foo"), local_weights_path=Path("foo"))
    DeepLearningConfig(local_dataset=Path("foo"), weights_url="bar")
