#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import List
from unittest import mock

import pytest

from InnerEye.Azure.tensorboard_monitor import AMLTensorBoardMonitorConfig
from Tests.ML.util import get_default_azure_config


def patch_and_parse(args: List[str]) -> AMLTensorBoardMonitorConfig:
    """
    Returns a MonitorArguments object created using the mock arguments.
    """
    with mock.patch("sys.argv", [""] + args):
        return AMLTensorBoardMonitorConfig.parse_args()


def test_monitor_args_run_ids() -> None:
    """
    Checks run ids are assigned from commandline args correctly.
    """
    parsed = patch_and_parse(["--run_ids=foo"])
    assert parsed.run_ids == ["foo"]
    parsed = patch_and_parse(["--run_ids", "foo"])
    assert parsed.run_ids == ["foo"]
    assert parsed.experiment_name is None
    assert parsed.run_status == "Running,Completed"


def test_create_azure_config() -> None:
    """
    Tests AzureConfig object can be created correctly.
    """
    get_default_azure_config().get_workspace()


def test_monitor_args_experiment() -> None:
    """
    Check commandline arguments are set correctly in the MonitorArguments object.
    """
    parsed = patch_and_parse(["--experiment_name=foo", "--run_status=Baz", "--port=123"])
    assert parsed.run_ids is None
    assert parsed.experiment_name == "foo"
    assert parsed.run_status == "Baz"
    assert parsed.port == 123


def test_monitor_args_fails() -> None:
    """
    Check MonitorArguments object is not created if both run id and experiment name are not provided.
    """
    with pytest.raises(ValueError) as ex:
        patch_and_parse([])
    assert "list of run ids or an experiment name" in ex.value.args[0]
