#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path

import pytest

from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.baselines_util import compare_folder_contents
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
    expected = test_output_dirs.root_dir / "expected"
    actual = test_output_dirs.root_dir / "actual"
    matching = "matching.txt"
    missing = "missing.txt"
    # Comparison should cover at least .csv and .txt files
    mismatch = "mismatch.csv"
    extra = "extra.txt"
    for folder in [expected, actual]:
        folder.mkdir()
        # This file exists in both expected and actual, should not raise any alerts
        matching_file = folder / "folder" / matching
        matching_file.parent.mkdir()
        matching_file.write_text(f"Line1{os.linesep}Line2")
    # This file only exists in the expected results
    (expected / "folder" / missing).write_text("missing")
    (actual / extra).write_text("extra")
    # This file exists in both actual and expected, but has different contents
    (expected / "folder" / mismatch).write_text("contents1")
    (actual / "folder" / mismatch).write_text("contents2")

    with pytest.raises(ValueError) as ex:
        compare_folder_contents(expected=expected, actual=actual)
    assert matching not in str(ex)
    assert missing in str(ex)
    assert mismatch in str(ex)
    assert extra not in str(ex)
