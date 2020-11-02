#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Optional

import pytest
from azureml.core import Experiment

from InnerEye.Azure.azure_util import to_azure_friendly_string
from InnerEye.Azure.run_pytest import download_pytest_result
from InnerEye.Common.output_directories import OutputFolderForTests
from Tests.ML.util import get_default_azure_config


def test_download_pytest_file(test_output_dirs: OutputFolderForTests) -> None:
    output_dir = test_output_dirs.root_dir
    azure_config = get_default_azure_config()
    workspace = azure_config.get_workspace()

    def get_run_and_download_pytest(branch: str, number: int) -> Optional[Path]:
        experiment = Experiment(workspace, name=to_azure_friendly_string(branch))
        runs = [run for run in experiment.get_runs() if run.number == number]
        if len(runs) != 1:
            raise ValueError(f"Expected to get exactly 1 run in experiment {experiment.name}")
        return download_pytest_result(runs[0], output_dir)

    # PR 49 is a recent successful build that generated a pytest file.
    # Run 6 in that experiment was canceled did not yet write the pytest file:
    with pytest.raises(ValueError) as ex:
        get_run_and_download_pytest("refs/pull/219/merge", 6)
    assert "No pytest result file" in str(ex)
    downloaded = get_run_and_download_pytest("refs/pull/219/merge", 7)
    assert downloaded is not None
    assert downloaded.exists()
    # Delete the file - it should be cleaned up with the test output directories though.
    # If the file remained, it would be uploaded as a test result file to Azure DevOps
    downloaded.unlink()
