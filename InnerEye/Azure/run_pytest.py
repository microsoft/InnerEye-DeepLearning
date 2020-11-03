#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Tuple

import pytest
from azureml.core import Run

# Test result file from running pytest inside the AzureML job. This file must have a prefix that
# matches the string in the build definition build-pr.yml, in the TrainInAzureML job.
from InnerEye.Azure.azure_util import download_outputs_from_run

PYTEST_RESULTS_FILE = Path("test-results-on-azure-ml.xml")


def run_pytest(pytest_mark: str, outputs_folder: Path) -> Tuple[bool, Path]:
    """
    Runs pytest on the whole test suite, restricting to the tests that have the given PyTest mark.
    :param pytest_mark: The PyTest mark to use for filtering out the tests to run.
    :param outputs_folder: The folder into which the test result XML file should be written.
    :return: True if PyTest found tests to execute and completed successfully, False otherwise.
    Also returns the path to the generated PyTest results file.
    """
    from _pytest.main import ExitCode
    _outputs_file = outputs_folder / PYTEST_RESULTS_FILE
    # Only run on tests in Tests/, to avoid the Tests/ directory if this repo is consumed as a submodule
    pytest_args = ["Tests/", f"--junitxml={str(_outputs_file)}"]

    if pytest_mark:
        pytest_args += ["-m", pytest_mark]
    logging.info(f"Starting pytest, with args: {pytest_args}")
    status_code = pytest.main(pytest_args)
    if status_code == ExitCode.NO_TESTS_COLLECTED:
        logging.error(f"PyTest did not find any tests to run, when restricting to tests with this mark: {pytest_mark}")
        return False, _outputs_file
    return status_code == ExitCode.OK, _outputs_file


def download_pytest_result(run: Run, destination_folder: Path = Path.cwd()) -> Path:
    """
    Downloads the pytest result file that is stored in the output folder of the given AzureML run.
    If there is no pytest result file, throw an Exception.
    :param run: The run from which the files should be read.
    :param destination_folder: The folder into which the PyTest result file is downloaded.
    :return: The path (folder and filename) of the downloaded file.
    """
    logging.info(f"Downloading pytest result file: {PYTEST_RESULTS_FILE}")
    try:
        return download_outputs_from_run(
            PYTEST_RESULTS_FILE,
            destination=destination_folder,
            run=run,
            is_file=True)
    except:
        raise ValueError(f"No pytest result file {PYTEST_RESULTS_FILE} was found for run {run.id}")
