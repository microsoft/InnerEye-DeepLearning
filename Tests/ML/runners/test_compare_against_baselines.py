#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pandas as pd
import pytest
from typing import List, Optional, Tuple
from unittest import mock

from InnerEye.Common import common_util
from InnerEye.Common.common_util import BASELINE_WILCOXON_RESULTS_FILE, BEST_EPOCH_FOLDER_NAME, ENSEMBLE_SPLIT_NAME, \
    FULL_METRICS_DATAFRAME_FILE, OTHER_RUNS_SUBDIR_NAME, SUBJECT_METRICS_FILE_NAME, ModelProcessing
from InnerEye.Common.fixed_paths import DEFAULT_AML_UPLOAD_DIR
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.baselines_util import ComparisonBaseline, compare_scores_against_baselines, get_comparison_baselines, \
    perform_score_comparisons
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from Tests.ML.util import get_default_azure_config
from Tests.AfterTraining.test_after_training import get_most_recent_run_id


def create_dataset_df() -> pd.DataFrame:
    """
    Create a test dataframe for DATASET_CSV_FILE_NAME.

    :return: Test dataframe.
    """
    dataset_df = pd.DataFrame()
    dataset_df['subject'] = list(range(10))
    dataset_df['seriesId'] = [f"s{i}" for i in range(10)]
    dataset_df['institutionId'] = ["xyz"] * 10
    return dataset_df


def create_metrics_df() -> pd.DataFrame:
    """
    Create a test dataframe for SUBJECT_METRICS_FILE_NAME.

    :return: Test dataframe.
    """
    metrics_df = pd.DataFrame()
    metrics_df['Patient'] = list(range(10))
    metrics_df['Structure'] = ['appendix'] * 10
    metrics_df['Dice'] = [0.5 + i * 0.02 for i in range(10)]
    return metrics_df


def create_comparison_metrics_df() -> pd.DataFrame:
    """
    Create a test dataframe for comparison metrics.

    :return: Test dataframe.
    """
    comparison_metrics_df = pd.DataFrame()
    comparison_metrics_df['Patient'] = list(range(10))
    comparison_metrics_df['Structure'] = ['appendix'] * 10
    comparison_metrics_df['Dice'] = [0.51 + i * 0.02 for i in range(10)]
    return comparison_metrics_df


def create_comparison_baseline(dataset_df: pd.DataFrame) -> ComparisonBaseline:
    """
    Create a test ComparisonBaseline.

    :param dataset_df: Dataset dataframe.
    :return: New test ComparisonBaseline
    """
    comparison_metrics_df = create_comparison_metrics_df()
    comparison_name = "DefaultName"
    comparison_run_rec_id = "DefaultRunRecId"
    return ComparisonBaseline(comparison_name, dataset_df, comparison_metrics_df, comparison_run_rec_id)


def check_wilcoxon_lines(wilcoxon_lines: List[str], baseline: ComparisonBaseline) -> None:
    """
    Assert that the wilcoxon lines are as expected.

    :param wilcoxon_lines: Lines to test.
    :param baseline: Expected comparison baseline.
    """
    assert wilcoxon_lines[0] == f"Run 1: {baseline.name}"
    assert wilcoxon_lines[1] == "Run 2: CURRENT"
    assert wilcoxon_lines[3].find("WORSE") > 0


@pytest.mark.skipif(common_util.is_windows(), reason="Loading tk sometimes fails on Windows")
def test_perform_score_comparisons() -> None:
    dataset_df = create_dataset_df()
    metrics_df = create_metrics_df()
    baseline = create_comparison_baseline(dataset_df)
    result = perform_score_comparisons(dataset_df, metrics_df, [baseline])
    assert result.did_comparisons
    assert len(result.wilcoxon_lines) == 5
    check_wilcoxon_lines(result.wilcoxon_lines, baseline)
    assert list(result.plots.keys()) == [f"{baseline.name}_vs_CURRENT"]


@pytest.mark.after_training_single_run
def test_get_comparison_data(test_output_dirs: OutputFolderForTests) -> None:
    azure_config = get_default_azure_config()
    comparison_name = "DefaultName"
    comparison_path = get_most_recent_run_id() + \
                      f"/{DEFAULT_AML_UPLOAD_DIR}/{BEST_EPOCH_FOLDER_NAME}/{ModelExecutionMode.TEST.value}"
    baselines = get_comparison_baselines(test_output_dirs.root_dir,
                                         azure_config, [(comparison_name, comparison_path)])
    assert len(baselines) == 1
    assert baselines[0].name == comparison_name


@pytest.mark.parametrize('model_proc_split_infer', [(ModelProcessing.DEFAULT, 0, None),
                                                    (ModelProcessing.DEFAULT, 3, True),
                                                    (ModelProcessing.ENSEMBLE_CREATION, 3, None)])
def test_compare_scores_against_baselines_throws(model_proc_split_infer: Tuple[ModelProcessing, int, Optional[bool]],
                                                 test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that exceptions are raised if the files necessary for baseline comparison are missing,
    then test that when all required files are present that baseline comparison files are written.

    :param model_proc: Model processing to test.
    :param test_output_dirs: Test output directories.
    :return: None.
    """
    (model_proc, number_of_cross_validation_splits, inference_on_test_set) = model_proc_split_infer
    config = SegmentationModelBase(should_validate=False,
                                   comparison_blob_storage_paths=[
                                       ('Single', 'dummy_blob_single/outputs/epoch_120/Test'),
                                       ('5fold', 'dummy_blob_ensemble/outputs/epoch_120/Test')],
                                   number_of_cross_validation_splits=number_of_cross_validation_splits,
                                   inference_on_test_set=inference_on_test_set)
    config.set_output_to(test_output_dirs.root_dir)

    azure_config = get_default_azure_config()

    # If the BEST_EPOCH_FOLDER_NAME folder is missing, expect an exception to be raised.
    with pytest.raises(FileNotFoundError) as ex:
        compare_scores_against_baselines(
            model_config=config,
            azure_config=azure_config, model_proc=model_proc)
    assert "Cannot compare scores against baselines: no best epoch results found at" in str(ex)

    best_epoch_folder_path = config.outputs_folder
    if model_proc == ModelProcessing.ENSEMBLE_CREATION:
        best_epoch_folder_path = best_epoch_folder_path / OTHER_RUNS_SUBDIR_NAME / ENSEMBLE_SPLIT_NAME
    best_epoch_folder_path = best_epoch_folder_path / BEST_EPOCH_FOLDER_NAME / ModelExecutionMode.TEST.value

    best_epoch_folder_path.mkdir(parents=True)

    # If the BEST_EPOCH_FOLDER_NAME folder exists but DATASET_CSV_FILE_NAME is missing,
    # expect an exception to be raised.
    with pytest.raises(FileNotFoundError) as ex:
        compare_scores_against_baselines(
            model_config=config,
            azure_config=azure_config, model_proc=model_proc)
    assert "Not comparing with baselines because no " in str(ex)
    assert DATASET_CSV_FILE_NAME in str(ex)

    model_dataset_path = best_epoch_folder_path / DATASET_CSV_FILE_NAME
    dataset_df = create_dataset_df()
    dataset_df.to_csv(model_dataset_path)

    # If the BEST_EPOCH_FOLDER_NAME folder exists but SUBJECT_METRICS_FILE_NAME is missing,
    # expect an exception to be raised.
    with pytest.raises(FileNotFoundError) as ex:
        compare_scores_against_baselines(
            model_config=config,
            azure_config=azure_config, model_proc=model_proc)
    assert "Not comparing with baselines because no " in str(ex)
    assert SUBJECT_METRICS_FILE_NAME in str(ex)

    model_metrics_path = best_epoch_folder_path / SUBJECT_METRICS_FILE_NAME
    metrics_df = create_metrics_df()
    metrics_df.to_csv(model_metrics_path)

    baseline = create_comparison_baseline(dataset_df)

    # Patch get_comparison_baselines to return the baseline above.
    with mock.patch('InnerEye.ML.baselines_util.get_comparison_baselines', return_value=[baseline]):
        compare_scores_against_baselines(
            model_config=config,
            azure_config=azure_config, model_proc=model_proc)

    # Check the wilcoxoon results file is present and has expected contents.
    wilcoxon_path = best_epoch_folder_path / BASELINE_WILCOXON_RESULTS_FILE
    assert wilcoxon_path.is_file()

    wilcoxon_lines = [line.strip() for line in wilcoxon_path.read_text().splitlines()]
    check_wilcoxon_lines(wilcoxon_lines, baseline)

    # Check the full metrics results file is present.
    full_metrics_path = best_epoch_folder_path / FULL_METRICS_DATAFRAME_FILE
    assert full_metrics_path.is_file()
