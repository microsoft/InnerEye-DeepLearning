#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_util import AZUREML_RUN_FOLDER_PREFIX, download_outputs_from_run, fetch_run, strip_prefix
from InnerEye.Common import common_util
from InnerEye.Common.Statistics import wilcoxon_signed_rank_test
from InnerEye.Common.Statistics.wilcoxon_signed_rank_test import WilcoxonTestConfig
from InnerEye.Common.common_util import BASELINE_WILCOXON_RESULTS_FILE, ENSEMBLE_SPLIT_NAME, \
    EPOCH_FOLDER_NAME_PATTERN, \
    FULL_METRICS_DATAFRAME_FILE, \
    METRICS_FILE_NAME, \
    ModelProcessing, OTHER_RUNS_SUBDIR_NAME, remove_file_or_directory
from InnerEye.Common.fixed_paths import DEFAULT_AML_UPLOAD_DIR
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.visualizers.metrics_scatterplot import write_to_scatterplot_directory
from InnerEye.ML.visualizers.plot_cross_validation import convert_rows_for_comparisons, may_write_lines_to_file


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
    comparison_blob_storage_paths = getattr(model_config, 'comparison_blob_storage_paths')
    if not comparison_blob_storage_paths:
        return
    outputs_path = model_config.outputs_folder
    if model_proc == ModelProcessing.ENSEMBLE_CREATION:
        outputs_path = outputs_path / OTHER_RUNS_SUBDIR_NAME / ENSEMBLE_SPLIT_NAME
    model_epoch_paths = sorted(outputs_path.glob(EPOCH_FOLDER_NAME_PATTERN))
    if not model_epoch_paths:
        logging.warning("Cannot compare scores against baselines: no matches found for "
                        f"{outputs_path}/{EPOCH_FOLDER_NAME_PATTERN}")
        return
    # Use the last (highest-numbered) epoch path for the current run.
    model_epoch_path = model_epoch_paths[-1]
    model_metrics_path = model_epoch_path / ModelExecutionMode.TEST.value / METRICS_FILE_NAME
    model_dataset_path = model_epoch_path / ModelExecutionMode.TEST.value / DATASET_CSV_FILE_NAME
    if not model_dataset_path.exists():
        logging.warning(f"Not comparing with baselines because no {model_dataset_path} file found for this run")
        return
    if not model_metrics_path.exists():
        logging.warning(f"Not comparing with baselines because no {model_metrics_path} file found for this run")
        return
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
    workspace = azure_config.get_workspace()
    comparison_baselines = []
    for (comparison_name, comparison_path) in comparison_blob_storage_paths:
        # Discard the experiment part of the run rec ID, if any.
        comparison_path = comparison_path.split(":")[-1]
        run_rec_id, blob_path_str = comparison_path.split("/", 1)
        run_rec_id = strip_prefix(run_rec_id, AZUREML_RUN_FOLDER_PREFIX)
        blob_path = Path(strip_prefix(blob_path_str, DEFAULT_AML_UPLOAD_DIR + "/"))
        run = fetch_run(workspace, run_rec_id)
        # We usually find dataset.csv in the same directory as metrics.csv, but we sometimes
        # have to look higher up.
        comparison_dataset_path: Optional[Path] = None
        comparison_metrics_path: Optional[Path] = None
        destination_folder = outputs_folder / run_rec_id / blob_path
        # Look for dataset.csv inside epoch_NNN/Test, epoch_NNN/ and at top level
        for blob_path_parent in step_up_directories(blob_path):
            try:
                comparison_dataset_path = download_outputs_from_run(
                    blob_path_parent / DATASET_CSV_FILE_NAME, destination_folder, run, True)
                break
            except ValueError:
                logging.warning(f"cannot find {DATASET_CSV_FILE_NAME} at {blob_path_parent} in {run_rec_id}")
                pass
            except NotADirectoryError:
                logging.warning(f"{blob_path_parent} is not a directory")
                break
            if comparison_dataset_path is None:
                logging.warning(f"cannot find {DATASET_CSV_FILE_NAME} at or above {blob_path} in {run_rec_id}")
        # Look for epoch_NNN/Test/metrics.csv
        try:
            comparison_metrics_path = download_outputs_from_run(
                blob_path / METRICS_FILE_NAME, destination_folder, run, True)
        except ValueError:
            logging.warning(f"cannot find {METRICS_FILE_NAME} at {blob_path} in {run_rec_id}")
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
            logging.warning(f"could not find comparison data for run {run_rec_id}")
            for key, path in ("dataset", comparison_dataset_path), ("metrics", comparison_metrics_path):
                logging.warning(f"path to {key} data is {path}")
                # noinspection PyUnresolvedReferences
                if path is not None and not path.exists():
                    logging.warning("    ... but it does not exist")
    return comparison_baselines


def step_up_directories(path: Path) -> Generator[Path, None, None]:
    """
    Generates the provided directory and all its parents. Needed because dataset.csv
    files are sometimes not where we expect them to be, but higher up.
    """
    while True:
        yield path
        parent = path.parent
        if parent == path:
            break
        path = parent
