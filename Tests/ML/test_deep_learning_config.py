#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pytest

from pathlib import Path

from InnerEye.ML.deep_learning_config import DeepLearningConfig


def test_validate_dataset_params() -> None:
    # DatasetParams cannot be initialized with neither of these these parameters set
    with pytest.raises(ValueError) as ex:
        DeepLearningConfig(local_dataset=None, azure_dataset_id="")
    assert ex.value.args[0] == "Either of local_dataset or azure_dataset_id must be set."

    # The following should be okay
    DeepLearningConfig(local_dataset=Path("foo"))
    DeepLearningConfig(azure_dataset_id="bar")

    with pytest.raises(ValueError) as ex:
        DeepLearningConfig(local_dataset=None,
                           azure_dataset_id="foo",
                           extra_local_dataset_paths=[Path("bar")])
    assert ex.value.args[0] == "Please specify a local dataset in local_dataset " \
                               "before adding datasets to extra_local_dataset_paths."

    config = DeepLearningConfig(local_dataset=Path("foo"),
                                azure_dataset_id="",
                                extra_azure_dataset_ids=[])
    assert not config.all_azure_dataset_ids()

    with pytest.raises(ValueError) as ex:
        DeepLearningConfig(local_dataset=Path("foo"),
                           azure_dataset_id="",
                           extra_azure_dataset_ids=["bar"])
    assert ex.value.args[0] == "Please specify an azure dataset in azure_dataset_id " \
                               "before adding datasets to extra_azure_dataset_ids."

    config = DeepLearningConfig(azure_dataset_id="foo",
                                extra_azure_dataset_ids=[])
    assert len(config.all_azure_dataset_ids()) == 1

    config = DeepLearningConfig(azure_dataset_id="foo",
                                extra_azure_dataset_ids=["bar"])
    assert len(config.all_azure_dataset_ids()) == 2

    config = DeepLearningConfig(azure_dataset_id="foo",
                                dataset_mountpoint="",
                                extra_dataset_mountpoints=[])
    assert not config.all_dataset_mountpoints()

    with pytest.raises(ValueError) as ex:
        DeepLearningConfig(azure_dataset_id="foo",
                           dataset_mountpoint="",
                           extra_dataset_mountpoints=["bar"])
    assert ex.value.args[0] == "Please specify a mountpoint for your first dataset using dataset_mountpoint " \
                               "before adding mountpoints to extra_dataset_mountpoints."

    config = DeepLearningConfig(azure_dataset_id="foo",
                                dataset_mountpoint="foo",
                                extra_dataset_mountpoints=[])
    assert len(config.all_dataset_mountpoints()) == 1

    config = DeepLearningConfig(azure_dataset_id="foo",
                                dataset_mountpoint="foo",
                                extra_dataset_mountpoints=["bar"])
    assert len(config.all_dataset_mountpoints()) == 2


def test_validate_deep_learning_config() -> None:

    # DeepLearningConfig cannot be initialized with both these parameters set
    with pytest.raises(ValueError) as ex:
        DeepLearningConfig(local_dataset=Path("foo"),
                           local_weights_path=Path("foo"), weights_url="bar")
    assert ex.value.args[0] == "Cannot specify both local_weights_path and weights_url."

    # The following should be okay
    DeepLearningConfig(local_dataset=Path("foo"), local_weights_path=Path("foo"))
    DeepLearningConfig(local_dataset=Path("foo"), weights_url="bar")
