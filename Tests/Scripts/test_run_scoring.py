#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from unittest import mock

import pytest

from InnerEye.Scripts import download_model_and_run_scoring


def get_test_script() -> Path:
    """
    Returns the full path to a testing script that lives inside of the test suite.
    :return:
    """
    current = Path(__file__).parent
    script = current / "script_for_tests.py"
    assert script.is_file(), f"File {script} not found"
    return script


@pytest.mark.parametrize(["script_arg", "expect_failure"],
                         [("correct", False),
                          ("failing", True)])
def test_run_scoring(script_arg: str, expect_failure: bool) -> None:
    """
    Test if we can invoke a script via the scoring pipeline. Passing invalid arguments should make cause failure.
    """
    scoring_script = Path(__file__).parent / "script_for_tests.py"
    assert scoring_script.is_file(), f"The script to invoke does not exist: {scoring_script}"
    scoring_folder = str(scoring_script.parent)
    # Invoke the script, and pass in a single string as the argument. Based on that, the script will either fail
    # or succeed.
    args = ["--model-folder", str(scoring_folder), scoring_script.name, script_arg]
    with mock.patch("sys.argv", [""] + args):
        with pytest.raises(SystemExit) as ex:
            download_model_and_run_scoring.run()
        expected_exit_code = 1 if expect_failure else 0
        assert ex.value.args[0] == expected_exit_code
