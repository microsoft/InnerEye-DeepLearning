#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path

import pytest

from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.baselines_util import REGRESSION_TEST_AZUREML_FOLDER, REGRESSION_TEST_AZUREML_PARENT_FOLDER, \
    compare_folder_contents
from InnerEye.ML.run_ml import MLRunner
from Tests.ML.configs.lightning_test_containers import DummyContainerWithModel


def test_regression_test(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that the file comparison for regression tests is actually called in the workflow.
    """
    container = DummyContainerWithModel()
    container.local_dataset = test_output_dirs.root_dir
    container.regression_test_folder = Path("foo")
    runner = MLRunner(container=container)
    runner.setup(use_mount_or_download_dataset=False)
    with pytest.raises(ValueError) as ex:
        runner.run()
    assert "Folder does not exist" in str(ex)


def test_compare_folder_exists(test_output_dirs: OutputFolderForTests) -> None:
    does_not_exist = test_output_dirs.root_dir / "foo"
    with pytest.raises(ValueError) as ex:
        compare_folder_contents(expected=does_not_exist, actual=test_output_dirs.root_dir)
    assert "Folder does not exist" in str(ex)


def test_compare_folder(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test the comparison of folders that we use for regression tests.
    """
    # Create a test of expected and actual files on the fly.
    expected = test_output_dirs.root_dir / "expected"
    actual = test_output_dirs.root_dir / "actual"
    matching = "matching.txt"
    missing = "missing.txt"
    ignored = "ignored.txt"
    # Comparison should cover at least .csv and .txt files
    mismatch = "mismatch.csv"
    extra = "extra.txt"
    subfolder = Path("folder")
    for folder in [expected, actual]:
        folder.mkdir()
        # This file exists in both expected and actual, should not raise any alerts because it contents matches.
        matching_file = folder / subfolder / matching
        matching_file.parent.mkdir()
        # Use Windows linebreaks here. Tests should run on both Windows and Linux, and should be invariant to linesep
        matching_file.write_text(f"Line1\r\nLine2")
    # This file only exists in the expected results, and should create an error
    (expected / subfolder / missing).write_text("missing")
    (actual / extra).write_text("extra")
    # This file exists in both actual and expected, but has different contents, hence should create an error
    (expected / subfolder / mismatch).write_text("contents1")
    (actual / subfolder / mismatch).write_text("contents2")
    # Create folders that hold the expected results in the AzureML run context. These should be ignored when doing
    # the file-by-file comparison on the local output files.
    for ignored_folder in [REGRESSION_TEST_AZUREML_FOLDER, REGRESSION_TEST_AZUREML_PARENT_FOLDER]:
        azureml1 = expected / ignored_folder / ignored
        azureml1.parent.mkdir()
        azureml1.touch()

    with pytest.raises(ValueError) as ex:
        compare_folder_contents(expected=expected, actual=actual)
    message = ex.value.args[0].splitlines()
    # No message expected
    assert matching not in str(ex)
    assert extra not in str(ex)
    assert ignored not in str(ex)
    # Folders should be skipped in the comparison
    assert f"Missing file: {subfolder}" not in message
    assert f"Missing file: {subfolder / missing}" in message
    assert f"Contents mismatch: {subfolder / mismatch}" in message
