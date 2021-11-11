#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import uuid
from pathlib import Path
from unittest import mock

import pytest

from InnerEye.Common.common_util import CROSSVAL_RESULTS_FOLDER, logging_to_stdout
from InnerEye.Common.fixed_paths import MODEL_INFERENCE_JSON_FILE_NAME
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML import baselines_util
from InnerEye.ML.baselines_util import REGRESSION_TEST_AZUREML_FOLDER, REGRESSION_TEST_AZUREML_PARENT_FOLDER, \
    REGRESSION_TEST_OUTPUT_FOLDER, compare_files, compare_folder_contents, compare_folders_and_run_outputs
from InnerEye.ML.deep_learning_config import FINAL_MODEL_FOLDER
from InnerEye.ML.run_ml import MLRunner
from Tests.AfterTraining.test_after_training import FALLBACK_ENSEMBLE_RUN, FALLBACK_SINGLE_RUN, get_most_recent_run
from Tests.ML.configs.lightning_test_containers import DummyContainerWithModel


def create_folder_and_write_text(file: Path, text: str) -> None:
    """
    Writes the given text to a file. The folders in which the file lives are created too, unless they exist already.
    Writing the text keeps the line separators as-is (no translation)
    """
    file.parent.mkdir(exist_ok=True, parents=True)
    with file.open(mode="wt", newline="") as f:
        f.write(text)


def test_regression_test(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that the file comparison for regression tests is actually called in the workflow.
    """
    container = DummyContainerWithModel()
    container.local_dataset = test_output_dirs.root_dir
    container.regression_test_folder = Path(str(uuid.uuid4().hex))
    runner = MLRunner(container=container)
    runner.setup()
    with pytest.raises(ValueError) as ex:
        runner.run()
    assert "Folder with expected files does not exist" in str(ex)


@pytest.mark.parametrize("file_extension", baselines_util.TEXT_FILE_SUFFIXES)
def test_compare_files_text(test_output_dirs: OutputFolderForTests, file_extension: str) -> None:
    """
    Checks the basic code to compare the contents of two text files.
    :param test_output_dirs:
    :param file_extension: The extension of the file to create.
    """
    logging_to_stdout(log_level=logging.DEBUG)
    expected = test_output_dirs.root_dir / f"expected{file_extension}"
    actual = test_output_dirs.root_dir / "actual.does_not_matter"
    # Make sure that we test different line endings - the files should still match
    create_folder_and_write_text(expected, "Line1\r\nLine2")
    create_folder_and_write_text(actual, "Line1\nLine2")
    assert compare_files(expected=expected, actual=actual) == ""
    actual.write_text("does_not_match")
    assert compare_files(expected=expected, actual=actual) == baselines_util.CONTENTS_MISMATCH


def test_compare_files_csv(test_output_dirs: OutputFolderForTests) -> None:
    expected = test_output_dirs.root_dir / "expected.csv"
    actual = test_output_dirs.root_dir / "actual.does_not_matter"
    expected.write_text("""foo,bar
1.0,10.0""")
    actual.write_text("""foo,bar
1.0001,10.001""")
    assert compare_files(expected=expected, actual=actual, csv_relative_tolerance=1e-2) == ""
    assert compare_files(expected=expected, actual=actual, csv_relative_tolerance=1e-3) == ""
    assert compare_files(expected=expected, actual=actual, csv_relative_tolerance=2e-4) == ""
    assert compare_files(expected=expected, actual=actual,
                         csv_relative_tolerance=9e-5) == baselines_util.CONTENTS_MISMATCH


@pytest.mark.parametrize("file_extension", [".png", ".whatever"])
def test_compare_files_binary(test_output_dirs: OutputFolderForTests, file_extension: str) -> None:
    """
    Checks the comparison of files that are not recognized as text files, for example images.
    :param test_output_dirs:
    :param file_extension: The extension of the file to create.
    """
    logging_to_stdout(log_level=logging.DEBUG)
    expected = test_output_dirs.root_dir / f"expected{file_extension}"
    actual = test_output_dirs.root_dir / "actual.does_not_matter"
    data1 = bytes([1, 2, 3])
    data2 = bytes([4, 5, 6])
    expected.write_bytes(data1)
    actual.write_bytes(data1)
    assert compare_files(expected=expected, actual=actual) == ""
    actual.write_bytes(data2)
    assert compare_files(expected=expected, actual=actual) == baselines_util.CONTENTS_MISMATCH


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
    # This file exists in both expected and actual, should not raise any alerts because it contents matches
    # apart from linebreaks
    create_folder_and_write_text(expected / subfolder / matching, "Line1\r\nLine2")
    create_folder_and_write_text(actual / subfolder / matching, "Line1\nLine2")
    # This file only exists in the expected results, and should create an error
    (expected / subfolder / missing).write_text("missing")
    (actual / extra).write_text("extra")
    # This file exists in both actual and expected, but has different contents, hence should create an error
    (expected / subfolder / mismatch).write_text("contents1")
    (actual / subfolder / mismatch).write_text("contents2")

    messages = compare_folder_contents(expected_folder=expected, actual_folder=actual,
                                       csv_relative_tolerance=0.0)
    all_messages = " ".join(messages)
    # No issues expected
    assert matching not in all_messages
    assert extra not in all_messages
    assert ignored not in all_messages
    # Folders should be skipped in the comparison
    assert f"{baselines_util.MISSING_FILE}: {subfolder}" not in messages
    assert f"{baselines_util.MISSING_FILE}: {subfolder}/{missing}" in messages
    assert f"{baselines_util.CONTENTS_MISMATCH}: {subfolder}/{mismatch}" in messages


def test_compare_plain_outputs(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if we can compare that a set of files from the job outputs.
    """
    logging_to_stdout(log_level=logging.DEBUG)
    expected = test_output_dirs.root_dir / REGRESSION_TEST_OUTPUT_FOLDER
    actual = test_output_dirs.root_dir / "my_output"
    for folder in [expected, actual]:
        file1 = folder / "output.txt"
        create_folder_and_write_text(file1, "Something")
    # First comparison should pass
    compare_folders_and_run_outputs(expected=expected, actual=actual, csv_relative_tolerance=0.0)
    # Now add a file to the set of expected files that does not exist in the run: comparison should now fail
    no_such_file = "no_such_file.txt"
    file2 = expected / no_such_file
    create_folder_and_write_text(file2, "foo")
    with pytest.raises(ValueError) as ex:
        compare_folders_and_run_outputs(expected=test_output_dirs.root_dir, actual=Path.cwd())
    message = ex.value.args[0].splitlines()
    assert f"{baselines_util.MISSING_FILE}: {no_such_file}" in message


@pytest.mark.after_training_single_run
def test_compare_folder_against_run(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if we can compare that a set of files exists in an AML run.
    """
    logging_to_stdout(log_level=logging.DEBUG)
    run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_SINGLE_RUN)
    file1 = test_output_dirs.root_dir / REGRESSION_TEST_AZUREML_FOLDER / \
            FINAL_MODEL_FOLDER / MODEL_INFERENCE_JSON_FILE_NAME
    create_folder_and_write_text(file1,
                                 '{"model_name": "BasicModel2Epochs", "checkpoint_paths": ['
                                 '"checkpoints/best_checkpoint.ckpt"], '
                                 '"model_configs_namespace": "InnerEye.ML.configs.segmentation.BasicModel2Epochs"}')
    with mock.patch("InnerEye.ML.baselines_util.RUN_CONTEXT", run):
        # First comparison only on the .json file should pass
        compare_folders_and_run_outputs(expected=test_output_dirs.root_dir, actual=Path.cwd(),
                                        csv_relative_tolerance=0.0)
        # Now add a file to the set of expected files that does not exist in the run: comparison should now fail
        no_such_file = "no_such_file.txt"
        file2 = test_output_dirs.root_dir / REGRESSION_TEST_AZUREML_FOLDER / no_such_file
        create_folder_and_write_text(file2, "foo")
        with pytest.raises(ValueError) as ex:
            compare_folders_and_run_outputs(expected=test_output_dirs.root_dir, actual=Path.cwd())
        message = ex.value.args[0].splitlines()
        assert f"{baselines_util.MISSING_FILE}: {no_such_file}" in message
    # Now run the same comparison that failed previously, without mocking the RUN_CONTEXT. This should now
    # realize that the present run is an offline run, and skip the comparison
    compare_folders_and_run_outputs(expected=test_output_dirs.root_dir, actual=Path.cwd(),
                                    csv_relative_tolerance=0.0)


@pytest.mark.after_training_ensemble_run
def test_compare_folder_against_parent_run(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if we can compare that a set of files exists in an AML run.
    """
    logging_to_stdout(log_level=logging.DEBUG)
    parent_run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_ENSEMBLE_RUN)
    file1 = test_output_dirs.root_dir / REGRESSION_TEST_AZUREML_PARENT_FOLDER / \
            CROSSVAL_RESULTS_FOLDER / "Test_outliers.txt"
    create_folder_and_write_text(file1, """

=== METRIC: Dice ===

No outliers found

=== METRIC: HausdorffDistance_mm ===

No outliers found""")
    with mock.patch("InnerEye.ML.baselines_util.PARENT_RUN_CONTEXT", parent_run):
        # No plain files to compare. The file Test_outliers.txt should be compared and found to match.
        compare_folders_and_run_outputs(expected=test_output_dirs.root_dir, actual=Path.cwd())
        create_folder_and_write_text(file1, "foo")
        with pytest.raises(ValueError) as ex:
            compare_folders_and_run_outputs(expected=test_output_dirs.root_dir, actual=Path.cwd())
        message = ex.value.args[0].splitlines()
        assert f"{baselines_util.CONTENTS_MISMATCH}: {CROSSVAL_RESULTS_FOLDER}/{file1.name}" in message
        # Now add a file to the set of expected files that does not exist in the run: comparison should now fail
        no_such_file = "no_such_file.txt"
        file2 = test_output_dirs.root_dir / REGRESSION_TEST_AZUREML_PARENT_FOLDER / no_such_file
        create_folder_and_write_text(file2, "foo")
        with pytest.raises(ValueError) as ex:
            compare_folders_and_run_outputs(expected=test_output_dirs.root_dir, actual=Path.cwd())
        message = ex.value.args[0].splitlines()
        assert f"{baselines_util.MISSING_FILE}: {no_such_file}" in message
    # Now run the same comparison without mocking the PARENT_RUN_CONTEXT. This should now
    # realize that the present run is a crossval child run
    with pytest.raises(ValueError) as ex:
        compare_folders_and_run_outputs(expected=test_output_dirs.root_dir, actual=Path.cwd())
    assert "no (parent) run to compare against" in str(ex)
