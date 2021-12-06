#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path, PosixPath
from typing import List

import pytest

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_runner import create_dataset_configs
from InnerEye.ML.deep_learning_config import DatasetParams
from Tests.ML.util import get_default_azure_config


def test_validate() -> None:
    with pytest.raises(ValueError) as ex:
        AzureConfig(only_register_model=True)
    assert ex.value.args[0] == "If only_register_model is set, must also provide a valid run_recovery_id"


def test_dataset_consumption1() -> None:
    """
    Creating datasets, case 1: Azure datasets given, no local folders or mount points
    :return:
    """
    azure_config = get_default_azure_config()
    datasets = create_dataset_configs(azure_config,
                                      all_azure_dataset_ids=["1", "2"],
                                      all_dataset_mountpoints=[],
                                      all_local_datasets=[])
    assert len(datasets) == 2
    assert datasets[0].name == "1"
    assert datasets[1].name == "2"
    for i in range(2):
        assert datasets[i].local_folder is None
        assert datasets[i].target_folder is None

    # Two error cases: number of mount points or number of local datasets does not match
    with pytest.raises(ValueError) as ex:
        create_dataset_configs(azure_config,
                               all_azure_dataset_ids=["1", "2"],
                               all_dataset_mountpoints=["mp"],
                               all_local_datasets=[])
        assert "Invalid dataset setup" in str(ex)
    with pytest.raises(ValueError) as ex:
        create_dataset_configs(azure_config,
                               all_azure_dataset_ids=["1", "2"],
                               all_dataset_mountpoints=[],
                               all_local_datasets=[Path("local")])
        assert "Invalid dataset setup" in str(ex)


def test_dataset_consumption2() -> None:
    """
    Creating datasets, case 2: Azure datasets, local folders and mount points given
    """
    azure_config = get_default_azure_config()
    datasets = create_dataset_configs(azure_config,
                                      all_azure_dataset_ids=["1", "2"],
                                      all_dataset_mountpoints=["mp1", "mp2"],
                                      all_local_datasets=[Path("l1"), Path("l2")])
    assert len(datasets) == 2
    assert datasets[0].name == "1"
    assert datasets[1].name == "2"
    assert datasets[0].local_folder == Path("l1")
    assert datasets[1].local_folder == Path("l2")
    assert datasets[0].target_folder == PosixPath("mp1")
    assert datasets[1].target_folder == PosixPath("mp2")


def test_dataset_consumption3() -> None:
    """
    Creating datasets, case 3: local datasets only. This should generate no results
    """
    azure_config = get_default_azure_config()
    datasets = create_dataset_configs(azure_config,
                                      all_azure_dataset_ids=[],
                                      all_dataset_mountpoints=[],
                                      all_local_datasets=[Path("l1"), Path("l2")])
    assert len(datasets) == 0

def test_dataset_consumption4() -> None:
    """
    Creating datasets, case 4: no datasets at all
    """
    azure_config = get_default_azure_config()
    datasets = create_dataset_configs(azure_config,
                                      all_azure_dataset_ids=[],
                                      all_dataset_mountpoints=[],
                                      all_local_datasets=[])
    assert len(datasets) == 0


@pytest.mark.parametrize(("first", "extra", "expected"),
                         [(None, [Path("foo")], [Path("foo")]),
                          (Path("bar"), [Path("foo")], [Path("bar"), Path("foo")])])
def test_dataset_params_local(first: Path, extra: List[Path], expected: List[Path]) -> None:
    """
    Local datasets that are None should be excluded from all_local_dataset_paths
    """
    p = DatasetParams()
    p.local_dataset = first
    p.extra_local_dataset_paths = extra
    assert p.all_local_dataset_paths() == expected


@pytest.mark.parametrize(("first", "extra", "expected"),
                         [("", ["foo"], ["foo"]),
                          ("bar", ["foo"], ["bar", "foo"])])
def test_dataset_params_azure(first: Path, extra: List[Path], expected: List[Path]) -> None:
    """
    Azure datasets that are None or an empty string should be excluded from all_azure_dataset_ids
    """
    p = DatasetParams()
    p.azure_dataset_id = first
    p.extra_azure_dataset_ids = extra
    assert p.all_azure_dataset_ids() == expected
