#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pytest

from pathlib import Path

from InnerEye.ML.deep_learning_config import DatasetParams, WorkflowParams


def test_validate_dataset_params() -> None:
    # DatasetParams cannot be initialized with neither of these these parameters set
    with pytest.raises(ValueError) as ex:
        DatasetParams(local_dataset=None, azure_dataset_id="").validate()
    assert ex.value.args[0] == "Either of local_dataset or azure_dataset_id must be set."

    # The following should be okay
    DatasetParams(local_dataset=Path("foo")).validate()
    DatasetParams(azure_dataset_id="bar").validate()

    config = DatasetParams(local_dataset=Path("foo"),
                           azure_dataset_id="",
                           extra_azure_dataset_ids=[])
    config.validate()
    assert not config.all_azure_dataset_ids()

    config = DatasetParams(azure_dataset_id="foo",
                           extra_azure_dataset_ids=[])
    config.validate()
    assert len(config.all_azure_dataset_ids()) == 1

    config = DatasetParams(local_dataset=Path("foo"),
                           azure_dataset_id="",
                           extra_azure_dataset_ids=["bar"])
    config.validate()
    assert len(config.all_azure_dataset_ids()) == 1

    config = DatasetParams(azure_dataset_id="foo",
                           extra_azure_dataset_ids=["bar"])
    config.validate()
    assert len(config.all_azure_dataset_ids()) == 2

    config = DatasetParams(azure_dataset_id="foo",
                           dataset_mountpoint="",
                           extra_dataset_mountpoints=[])
    config.validate()
    assert not config.all_dataset_mountpoints()

    config = DatasetParams(azure_dataset_id="foo",
                           dataset_mountpoint="foo",
                           extra_dataset_mountpoints=[])
    config.validate()
    assert len(config.all_dataset_mountpoints()) == 1

    config = DatasetParams(azure_dataset_id="foo",
                           dataset_mountpoint="",
                           extra_dataset_mountpoints=["bar"])
    config.validate()
    assert len(config.all_dataset_mountpoints()) == 1

    config = DatasetParams(azure_dataset_id="foo",
                           extra_azure_dataset_ids=["bar"],
                           dataset_mountpoint="foo",
                           extra_dataset_mountpoints=["bar"])
    config.validate()
    assert len(config.all_dataset_mountpoints()) == 2

    with pytest.raises(ValueError) as ex:
        DatasetParams(azure_dataset_id="foo",
                      dataset_mountpoint="foo",
                      extra_dataset_mountpoints=["bar"]).validate()
    assert "Expected the number of azure datasets to equal the number of mountpoints" in ex.value.args[0]


def test_validate_workflow_params() -> None:

    # DeepLearningConfig cannot be initialized with more than one of these parameters set
    with pytest.raises(ValueError) as ex:
        WorkflowParams(local_dataset=Path("foo"),
                       local_weights_path=[Path("foo")],
                       weights_url=["bar"]).validate()
    assert ex.value.args[0] == "Cannot specify more than one of local_weights_path, weights_url or model_id."

    with pytest.raises(ValueError) as ex:
        WorkflowParams(local_dataset=Path("foo"),
                       local_weights_path=[Path("foo")],
                       model_id="foo:1").validate()
    assert ex.value.args[0] == "Cannot specify more than one of local_weights_path, weights_url or model_id."

    with pytest.raises(ValueError) as ex:
        WorkflowParams(local_dataset=Path("foo"),
                       weights_url=["foo"],
                       model_id="foo:1").validate()
    assert ex.value.args[0] == "Cannot specify more than one of local_weights_path, weights_url or model_id."

    with pytest.raises(ValueError) as ex:
        WorkflowParams(local_dataset=Path("foo"),
                       local_weights_path=[Path("foo")],
                       weights_url=["foo"],
                       model_id="foo:1").validate()
    assert ex.value.args[0] == "Cannot specify more than one of local_weights_path, weights_url or model_id."

    with pytest.raises(ValueError) as ex:
        WorkflowParams(local_dataset=Path("foo"),
                       model_id="foo").validate()
    assert "model_id should be in the form 'model_name:version'" in ex.value.args[0]

    # The following should be okay
    WorkflowParams(local_dataset=Path("foo"), local_weights_path=[Path("foo")]).validate()
    WorkflowParams(local_dataset=Path("foo"), weights_url=["foo"]).validate()
    WorkflowParams(local_dataset=Path("foo"), model_id="foo:1").validate()
