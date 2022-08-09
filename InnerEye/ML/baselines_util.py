#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from azureml.core import Run

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_util import AZUREML_RUN_FOLDER_PREFIX, PARENT_RUN_CONTEXT, RUN_CONTEXT, \
    get_comparison_baseline_paths, is_offline_run_context, strip_prefix
from InnerEye.Common import common_util
from InnerEye.Common.Statistics import wilcoxon_signed_rank_test
from InnerEye.Common.Statistics.wilcoxon_signed_rank_test import WilcoxonTestConfig
from InnerEye.Common.common_util import BASELINE_WILCOXON_RESULTS_FILE, FULL_METRICS_DATAFRAME_FILE, ModelProcessing, \
    SUBJECT_METRICS_FILE_NAME, get_best_epoch_results_path, remove_file_or_directory
from InnerEye.Common.fixed_paths import DEFAULT_AML_UPLOAD_DIR
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.visualizers.metrics_scatterplot import write_to_scatterplot_directory
from InnerEye.ML.visualizers.plot_cross_validation import convert_rows_for_comparisons, may_write_lines_to_file

REGRESSION_TEST_OUTPUT_FOLDER = "OUTPUT"
REGRESSION_TEST_AZUREML_FOLDER = "AZUREML_OUTPUT"
REGRESSION_TEST_AZUREML_PARENT_FOLDER = "AZUREML_PARENT_OUTPUT"
CONTENTS_MISMATCH = "Contents mismatch"
FILE_FORMAT_ERROR = "File format error"
MISSING_FILE = "Missing"
CSV_SUFFIX = ".csv"
TEXT_FILE_SUFFIXES = [".txt", ".json", ".html", ".md"]

INFERENCE_DISABLED_WARNING = "Not performing comparison of model against baseline(s), because inference is currently " \
                             "disabled. If comparison is required, use either the inference_on_test_set or " \
                             "ensemble_inference_on_test_set option, as appropriate."


@dataclass
class DiceScoreComparisonResult:
    """
    Values returned from perform_score_comparisons.
        dataframe: the values (from one or more metrics.csv files) on which comparisons were done
        did_comparisons: whether any comparisons were done - there may have been only one dataset
        wilcoxon_lines: lines containing Wilcoxon test results
        plots: scatterplots from comparisons.
    """
    dataframe: pd.DataFrame
    did_comparisons: bool
    wilcoxon_lines: List[str]
    plots: Dict[str, plt.Figure]

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)


def compare_scores_against_baselines(model_config: SegmentationModelBase, azure_config: AzureConfig,
                                     model_proc: ModelProcessing) -> None:
    """
    If the model config has any baselines to compare against, loads the metrics.csv file that should just have
    been written for the last epoch of the current run, and its dataset.csv. Do the same for all the baselines,
    whose corresponding files should be in the repository already. For each baseline, call the Wilcoxon signed-rank test
    on pairs consisting of Dice scores from the current model and the baseline, and print out comparisons to
    the Wilcoxon results file.
    """
    # The attribute will only be present for a segmentation model; and it might be None or empty even for that.
    comparison_blob_storage_paths = model_config.comparison_blob_storage_paths
    if not comparison_blob_storage_paths:
        return
    outputs_path = model_config.outputs_folder / get_best_epoch_results_path(ModelExecutionMode.TEST, model_proc)
    if not outputs_path.is_dir():
        if not model_config.is_inference_required(model_proc, ModelExecutionMode.TEST):
            logging.info(INFERENCE_DISABLED_WARNING)
            return
        raise FileNotFoundError(
            f"Cannot compare scores against baselines: no best epoch results found at {outputs_path}")
    model_metrics_path = outputs_path / SUBJECT_METRICS_FILE_NAME
    model_dataset_path = outputs_path / DATASET_CSV_FILE_NAME
    if not model_dataset_path.exists():
        raise FileNotFoundError(f"Not comparing with baselines because no {model_dataset_path} file found for this run")
    if not model_metrics_path.exists():
        raise FileNotFoundError(f"Not comparing with baselines because no {model_metrics_path} file found for this run")
    model_metrics_df = pd.read_csv(model_metrics_path)
    model_dataset_df = pd.read_csv(model_dataset_path)
    comparison_result = download_and_compare_scores(outputs_path,
                                                    azure_config, comparison_blob_storage_paths, model_dataset_df,
                                                    model_metrics_df)
    full_metrics_path = str(outputs_path / FULL_METRICS_DATAFRAME_FILE)
    comparison_result.dataframe.to_csv(full_metrics_path)
    if comparison_result.did_comparisons:
        wilcoxon_path = outputs_path / BASELINE_WILCOXON_RESULTS_FILE
        logging.info(
            f"Wilcoxon tests of current {model_proc.value} model against baseline(s), "
            f"written to {wilcoxon_path}:")
        for line in comparison_result.wilcoxon_lines:
            logging.info(line)
        logging.info("End of Wilcoxon test results")
        may_write_lines_to_file(comparison_result.wilcoxon_lines, wilcoxon_path)
    write_to_scatterplot_directory(outputs_path, comparison_result.plots)


def download_and_compare_scores(outputs_folder: Path, azure_config: AzureConfig,
                                comparison_blob_storage_paths: List[Tuple[str, str]], model_dataset_df: pd.DataFrame,
                                model_metrics_df: pd.DataFrame) -> DiceScoreComparisonResult:
    """
    :param azure_config: Azure configuration to use for downloading data
    :param comparison_blob_storage_paths: list of paths to directories containing metrics.csv and dataset.csv files,
        each of the form run_recovery_id/rest_of_path
    :param model_dataset_df: dataframe containing contents of dataset.csv for the current model
    :param model_metrics_df: dataframe containing contents of metrics.csv for the current model
    :return: a dataframe for all the data (current model and all baselines); whether any comparisons were
        done, i.e. whether a valid baseline was found; and the text lines to be written to the Wilcoxon results
        file.
    """
    comparison_baselines = get_comparison_baselines(outputs_folder, azure_config, comparison_blob_storage_paths)
    result = perform_score_comparisons(model_dataset_df, model_metrics_df, comparison_baselines)
    for baseline in comparison_baselines:
        run_rec_path = outputs_folder / baseline.run_recovery_id
        if run_rec_path.exists():
            logging.info(f"Removing directory {run_rec_path}")
            remove_file_or_directory(run_rec_path)
    return result


@dataclass
class ComparisonBaseline:
    """
    Structure to represent baseline data to compare the current run against.
    name: short name as given in the first item of each member of comparison_blob_storage_paths
    dataset_df: in-core copy of dataset.csv of the baseline
    metrics_df: in-core copy of metrics.csv of the baseline
    run_recovery_id: run-rec ID of the baseline run
    """
    name: str
    dataset_df: pd.DataFrame
    metrics_df: pd.DataFrame
    run_recovery_id: str

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)


def perform_score_comparisons(model_dataset_df: pd.DataFrame, model_metrics_df: pd.DataFrame,
                              comparison_baselines: List[ComparisonBaseline]) -> \
        DiceScoreComparisonResult:
    all_runs_df = convert_rows_for_comparisons('CURRENT', model_dataset_df, model_metrics_df)
    if not comparison_baselines:
        return DiceScoreComparisonResult(all_runs_df, False, [], {})
    for baseline in comparison_baselines:
        to_compare = convert_rows_for_comparisons(baseline.name, baseline.dataset_df, baseline.metrics_df)
        all_runs_df = all_runs_df.append(to_compare)
    config = WilcoxonTestConfig(data=all_runs_df, with_scatterplots=True, against=['CURRENT'])
    wilcoxon_lines, plots = wilcoxon_signed_rank_test.wilcoxon_signed_rank_test(config)
    return DiceScoreComparisonResult(all_runs_df, True, wilcoxon_lines, plots)


def get_comparison_baselines(outputs_folder: Path, azure_config: AzureConfig,
                             comparison_blob_storage_paths: List[Tuple[str, str]]) -> \
        List[ComparisonBaseline]:
    comparison_baselines = []
    for (comparison_name, comparison_path) in comparison_blob_storage_paths:
        # Discard the experiment part of the run rec ID, if any.
        comparison_path = comparison_path.split(":")[-1]
        run_rec_id, blob_path_str = comparison_path.split("/", 1)
        run_rec_id = strip_prefix(run_rec_id, AZUREML_RUN_FOLDER_PREFIX)
        blob_path = Path(strip_prefix(blob_path_str, DEFAULT_AML_UPLOAD_DIR + "/"))
        run = azure_config.fetch_run(run_rec_id)
        (comparison_dataset_path, comparison_metrics_path) = get_comparison_baseline_paths(outputs_folder, blob_path,
                                                                                           run, DATASET_CSV_FILE_NAME)
        # If both dataset.csv and metrics.csv were downloaded successfully, read their contents and
        # add a tuple to the comparison data.
        if comparison_dataset_path is not None and comparison_metrics_path is not None and \
                comparison_dataset_path.exists() and comparison_metrics_path.exists():
            comparison_baselines.append(ComparisonBaseline(
                comparison_name,
                pd.read_csv(comparison_dataset_path),
                pd.read_csv(comparison_metrics_path),
                run_rec_id))
        else:
            raise ValueError(f"could not find comparison data for run {run_rec_id}")
    return comparison_baselines


def compare_files(expected: Path, actual: Path, csv_relative_tolerance: float = 0.0) -> str:
    """
    Compares two individual files for regression testing. It returns an empty string if the two files appear identical.
    If the files are not identical, an error message with details is return. This handles known text file formats,
    where it ignores differences in line breaks. All other files are treated as binary, and compared on a byte-by-byte
    basis.

    :param expected: A file that contains the expected contents. The type of comparison (text or binary) is chosen
        based on the extension of this file.

    :param actual: A file that contains the actual contents.
    :param csv_relative_tolerance: When comparing CSV files, use this as the maximum allowed relative discrepancy.
        If 0.0, do not allow any discrepancy.
    :return: An empty string if the files appear identical, or otherwise an error message with details.
    """

    def print_lines(prefix: str, lines: List[str]) -> None:
        num_lines = len(lines)
        count = min(5, num_lines)
        logging.info(f"{prefix} {num_lines} lines, first {count} of those:")
        logging.info(os.linesep.join(lines[:count]))

    def try_read_csv(prefix: str, file: Path) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(file)
        except Exception as ex:
            logging.info(f"{prefix} file can't be read as CSV: {str(ex)}")
            return None

    if expected.suffix == CSV_SUFFIX:
        expected_df = try_read_csv("Expected", expected)
        actual_df = try_read_csv("Actual", actual)
        if expected_df is None or actual_df is None:
            return FILE_FORMAT_ERROR
        try:
            pd.testing.assert_frame_equal(actual_df, expected_df, rtol=csv_relative_tolerance)
        except Exception as ex:
            logging.info(str(ex))
            return CONTENTS_MISMATCH
    elif expected.suffix in TEXT_FILE_SUFFIXES:
        # Compare line-by-line to avoid issues with line separators
        expected_lines = expected.read_text().splitlines()
        actual_lines = actual.read_text().splitlines()
        if expected_lines != actual_lines:
            print_lines("Expected", expected_lines)
            print_lines("Actual", actual_lines)
            return CONTENTS_MISMATCH
    else:
        expected_binary = expected.read_bytes()
        actual_binary = actual.read_bytes()
        if expected_binary != actual_binary:
            logging.info(f"Expected {len(expected_binary)} bytes, actual {len(actual_binary)} bytes")
            return CONTENTS_MISMATCH
    return ""


def compare_folder_contents(expected_folder: Path,
                            csv_relative_tolerance: float,
                            actual_folder: Optional[Path] = None,
                            run: Optional[Run] = None) -> List[str]:
    """
    Compares a set of files in a folder, against files in either the other folder or files stored in the given
    AzureML run. Each file that is present in the "expected" folder must be also present in the "actual" folder
    (or the AzureML run), with exactly the same contents, in the same folder structure.
    For example, if there is a file "<expected>/foo/bar/contents.txt", then there must also be a file
    "<actual>/foo/bar/contents.txt"

    :param expected_folder: A folder with files that are expected to be present.
    :param actual_folder: The output folder with the actually produced files.
    :param run: An AzureML run
    :param csv_relative_tolerance: When comparing CSV files, use this as the maximum allowed relative discrepancy.
        If 0.0, do not allow any discrepancy.
    :return: A list of human readable error messages, with message and file path. If no errors are found, the list is
        empty.
    """
    messages = []
    if run and is_offline_run_context(run):
        logging.warning("Skipping file comparison because the given run context is an AzureML offline run.")
        return []
    files_in_run: List[str] = run.get_file_names() if run else []
    temp_folder = Path(tempfile.mkdtemp()) if run else None
    for file in expected_folder.rglob("*"):
        # rglob also returns folders, skip those
        if file.is_dir():
            continue
        # All files stored in AzureML runs use Linux-style path
        file_relative = file.relative_to(expected_folder).as_posix()
        if actual_folder:
            actual_file = actual_folder / file_relative
        elif temp_folder is not None and run is not None:
            actual_file = temp_folder / file_relative
            if file_relative in files_in_run:
                run.download_file(name=str(file_relative), output_file_path=str(actual_file))
        else:
            raise ValueError("One of the two arguments run, actual_folder must be provided.")
        message = compare_files(expected=file, actual=actual_file,
                                csv_relative_tolerance=csv_relative_tolerance) if actual_file.exists() else MISSING_FILE
        if message:
            messages.append(f"{message}: {file_relative}")
        logging.info(f"File {file_relative}: {message or 'OK'}")
    if temp_folder:
        shutil.rmtree(temp_folder)
    return messages


def compare_folders_and_run_outputs(expected: Path, actual: Path, csv_relative_tolerance: float) -> None:
    """
    Compares the actual set of run outputs in the `actual` folder against an expected set of files in the `expected`
    folder. The `expected` folder can have two special subfolders AZUREML_OUTPUT and AZUREML_PARENT_OUTPUT, that
    contain files that are expected to be present in the AzureML run context of the present run (AZUREML_OUTPUT)
    or the run context of the parent run (AZUREML_PARENT_OUTPUT).
    If a file is missing, or does not have the expected contents, an exception is raised.

    :param expected: A folder with files that are expected to be present.
    :param actual: The output folder with the actually produced files.
    :param csv_relative_tolerance: When comparing CSV files, use this as the maximum allowed relative discrepancy.
        If 0.0, do not allow any discrepancy.
    """
    if not expected.is_dir():
        raise ValueError(f"Folder with expected files does not exist: {expected}")
    logging.debug(f"Current working directory: {Path.cwd()}")
    messages = []
    for (subfolder, message_prefix, actual_folder, run_to_compare) in \
            [(REGRESSION_TEST_OUTPUT_FOLDER, "run output files", actual, None),
             (REGRESSION_TEST_AZUREML_FOLDER, "AzureML outputs in present run", None, RUN_CONTEXT),
             (REGRESSION_TEST_AZUREML_PARENT_FOLDER, "AzureML outputs in parent run", None, PARENT_RUN_CONTEXT)]:
        folder = expected / subfolder
        if folder.is_dir():
            logging.info(f"Comparing results in {folder} against {message_prefix}:")
            if actual_folder is None and run_to_compare is None:
                raise ValueError(f"The set of expected test results in {expected} contains a folder "
                                 f"{subfolder}, but there is no (parent) run to compare against.")
            new_messages = compare_folder_contents(folder,
                                                   actual_folder=actual_folder,
                                                   run=run_to_compare,
                                                   csv_relative_tolerance=csv_relative_tolerance)
            if new_messages:
                messages.append(f"Issues in {message_prefix}:")
                messages.extend(new_messages)
        else:
            logging.info(f"Folder {subfolder} not found, skipping comparison against {message_prefix}.")
    if messages:
        raise ValueError(f"Some expected files were missing or did not have the expected contents:{os.linesep}"
                         f"{os.linesep.join(messages)}")
