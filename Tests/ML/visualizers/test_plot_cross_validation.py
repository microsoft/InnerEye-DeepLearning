#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import pytest
from azureml.core import Run
from pandas.core.dtypes.common import is_string_dtype

from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, fetch_run
from InnerEye.Common.common_util import CROSSVAL_RESULTS_FOLDER, FULL_METRICS_DATAFRAME_FILE, METRICS_AGGREGATES_FILE, \
    METRICS_FILE_NAME, logging_to_stdout
from InnerEye.Common.fixed_paths import DEFAULT_AML_UPLOAD_DIR
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.ML.deep_learning_config import ModelCategory
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.utils.csv_util import CSV_INSTITUTION_HEADER, CSV_SERIES_HEADER
from InnerEye.ML.utils.metrics_constants import LoggingColumns
from InnerEye.ML.visualizers.plot_cross_validation import COL_MODE, \
    METRICS_BY_MODE_AND_STRUCTURE_FILE, METRICS_BY_MODE_FILE, \
    OfflineCrossvalConfigAndFiles, PORTAL_QUERY_TEMPLATE, PlotCrossValidationConfig, RUN_RECOVERY_ID_KEY, \
    RunResultFiles, add_comparison_data, check_result_file_counts, create_portal_query_for_outliers, \
    create_results_breakdown, download_crossval_result_files, get_split_id, load_dataframes, \
    plot_cross_validation_from_files, save_outliers
from Tests.Common.test_util import DEFAULT_ENSEMBLE_RUN_RECOVERY_ID, DEFAULT_RUN_RECOVERY_ID
from Tests.ML.models.architectures.sequential.test_rnn_classifier import ToyMultiLabelSequenceModel, \
    _get_multi_label_sequence_dataframe
from Tests.ML.util import assert_text_files_match, get_default_azure_config
from Tests.fixed_paths_for_tests import full_ml_test_data_path


@pytest.fixture
def test_config_ensemble() -> PlotCrossValidationConfig:
    return PlotCrossValidationConfig(
        run_recovery_id=DEFAULT_ENSEMBLE_RUN_RECOVERY_ID,
        epoch=1,
        model_category=ModelCategory.Segmentation
    )


@pytest.fixture
def test_config() -> PlotCrossValidationConfig:
    return PlotCrossValidationConfig(
        run_recovery_id=DEFAULT_RUN_RECOVERY_ID,
        epoch=1,
        model_category=ModelCategory.Segmentation
    )


@pytest.fixture
def test_config_comparison() -> PlotCrossValidationConfig:
    return PlotCrossValidationConfig(
        run_recovery_id=DEFAULT_ENSEMBLE_RUN_RECOVERY_ID + "_0",
        epoch=1,
        comparison_run_recovery_ids=[DEFAULT_ENSEMBLE_RUN_RECOVERY_ID + "_1"],
        comparison_epochs=[1],
        model_category=ModelCategory.Segmentation
    )


def _get_metrics_df(mode: ModelExecutionMode) -> pd.DataFrame:
    metrics_df = pd.read_csv(full_ml_test_data_path("{}_agg_splits.csv".format(mode.value)))
    # noinspection PyUnresolvedReferences
    metrics_df.split = [DEFAULT_ENSEMBLE_RUN_RECOVERY_ID + "_" + index for index in metrics_df.split.astype(str)]
    return metrics_df.sort_values(list(metrics_df.columns), ascending=True).reset_index(drop=True)


def download_metrics(config: PlotCrossValidationConfig) -> \
        Tuple[Dict[ModelExecutionMode, Optional[pd.DataFrame]], Path]:
    result_files, root_folder = download_crossval_result_files(config)
    dataframes = load_dataframes(result_files, config)
    return dataframes, root_folder


def create_run_result_file_list(config: PlotCrossValidationConfig, folder: str,
                                perform_sub_fold_cross_validation: bool = False) -> List[RunResultFiles]:
    """
    Creates a list of input files for cross validation analysis, from files stored inside of the test data folder.
    :param config: The overall cross validation config
    :param folder: The folder to read from, inside of test_data/plot_cross_validation.
    :param perform_sub_fold_cross_validation: If True then create input files for sub fold cross validation analysis.
    :return:
    """
    full_folder = full_ml_test_data_path("plot_cross_validation") / folder
    files: List[RunResultFiles] = []
    previous_dataset_file = None
    for split in ["0", "1", "1", "1"] if perform_sub_fold_cross_validation else ["0", "1"]:
        for mode in config.execution_modes_to_download():
            metrics_file = full_folder / split / mode.value / METRICS_FILE_NAME
            dataset_file: Optional[Path] = full_folder / split / DATASET_CSV_FILE_NAME
            if dataset_file.exists():  # type: ignore
                # Reduce amount of checked-in large files. dataset files can be large, and usually duplicate across
                # runs. Store only a copy in split 0, re-use in split 1.
                previous_dataset_file = dataset_file
            else:
                dataset_file = previous_dataset_file
            if metrics_file.exists():
                file = RunResultFiles(execution_mode=mode,
                                      metrics_file=metrics_file,
                                      dataset_csv_file=dataset_file,
                                      run_recovery_id=config.run_recovery_id + "_" + split,  # type: ignore
                                      split_index=split)
                files.append(file)
    return files


def create_file_list_for_segmentation_recovery_run(test_config_ensemble: PlotCrossValidationConfig) -> \
        List[RunResultFiles]:
    return create_run_result_file_list(config=test_config_ensemble,
                                       folder="master_1570466706163110")


def test_metrics_preparation_for_segmentation(test_config_ensemble: PlotCrossValidationConfig) -> None:
    """
    Test if metrics dataframes can be loaded and prepared. The files in question are checked in, but
    were downloaded from a run, ID given in DEFAULT_ENSEMBLE_RUN_RECOVERY_ID.
    """
    files = create_file_list_for_segmentation_recovery_run(test_config_ensemble)
    downloaded_metrics = load_dataframes(files, test_config_ensemble)
    for mode in test_config_ensemble.execution_modes_to_download():
        expected_df = _get_metrics_df(mode)
        # Drop the "mode" column, because that was added after creating the test data
        metrics = downloaded_metrics[mode]
        assert metrics is not None
        actual_df = metrics.drop(COL_MODE, axis=1)
        actual_df = actual_df.sort_values(list(actual_df.columns), ascending=True).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected_df, actual_df, check_like=True, check_dtype=False)


def load_result_files_for_classification(perform_sub_fold_cross_validation: bool = False) -> \
        Tuple[List[RunResultFiles], PlotCrossValidationConfig]:
    plotting_config = PlotCrossValidationConfig(
        run_recovery_id="local_branch:HD_cfff5ceb-a227-41d6-a23c-0ebbc33b6301",
        epoch=3,
        model_category=ModelCategory.Classification
    )
    files = create_run_result_file_list(config=plotting_config,
                                        folder="HD_cfff5ceb-a227-41d6-a23c-0ebbc33b6301",
                                        perform_sub_fold_cross_validation=perform_sub_fold_cross_validation)
    return files, plotting_config


@pytest.mark.parametrize("perform_sub_fold_cross_validation", [True, False])
def test_metrics_preparation_for_classification(perform_sub_fold_cross_validation: bool) -> None:
    """
    Test if metrics from classification models can be loaded and prepared. The files in question are checked in,
    and were downloaded from a run on AzureML.
    """
    files, plotting_config = load_result_files_for_classification(perform_sub_fold_cross_validation)
    downloaded_metrics = load_dataframes(files, plotting_config)
    assert ModelExecutionMode.TEST not in downloaded_metrics
    metrics = downloaded_metrics[ModelExecutionMode.VAL]
    assert metrics is not None
    expected_metrics_file = "metrics_preparation_for_sub_fold_classification_VAL.csv" \
        if perform_sub_fold_cross_validation else "metrics_preparation_for_classification_VAL.csv"
    expected_df_csv = full_ml_test_data_path("plot_cross_validation") / expected_metrics_file
    metrics = metrics.sort_values(list(metrics.columns), ascending=True).reset_index(drop=True)
    # To write new test results:
    # metrics.to_csv(expected_df_csv, index=False)
    expected_df = pd.read_csv(expected_df_csv).sort_values(list(metrics.columns), ascending=True).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_df, metrics, check_like=True, check_dtype=False)


def _test_result_aggregation_for_classification(files: List[RunResultFiles],
                                                plotting_config: PlotCrossValidationConfig,
                                                expected_aggregate_metrics: List[str],
                                                expected_epochs: Set[int]) -> None:
    """
    Test how metrics are aggregated for cross validation runs on classification models.
    """
    print(f"Writing aggregated metrics to {plotting_config.outputs_directory}")
    root_folder = plotting_config.outputs_directory
    plot_cross_validation_from_files(OfflineCrossvalConfigAndFiles(config=plotting_config, files=files),
                                     root_folder=root_folder)
    aggregates_file = root_folder / METRICS_AGGREGATES_FILE
    actual_aggregates = aggregates_file.read_text().splitlines()
    header_line = "prediction_target,area_under_roc_curve,area_under_pr_curve,accuracy_at_optimal_threshold," \
                  "false_positive_rate_at_optimal_threshold,false_negative_rate_at_optimal_threshold," \
                  "optimal_threshold,cross_entropy,accuracy_at_threshold_05,subject_count,data_split,epoch"
    expected_aggregate_metrics = [header_line] + expected_aggregate_metrics
    assert len(actual_aggregates) == len(expected_aggregate_metrics), "Number of lines in aggregated metrics file"
    for i, (actual, expected) in enumerate(zip(actual_aggregates, expected_aggregate_metrics)):
        assert actual == expected, f"Mismatch in aggregate metrics at index {i}"
    per_subject_metrics = pd.read_csv(root_folder / FULL_METRICS_DATAFRAME_FILE)
    assert LoggingColumns.Label.value in per_subject_metrics
    assert set(per_subject_metrics[LoggingColumns.Label.value].unique()) == {0.0, 1.0}
    assert LoggingColumns.ModelOutput.value in per_subject_metrics
    assert LoggingColumns.Patient.value in per_subject_metrics
    assert len(per_subject_metrics[LoggingColumns.Patient.value].unique()) == 356
    assert LoggingColumns.Epoch.value in per_subject_metrics
    assert set(per_subject_metrics[LoggingColumns.Epoch.value].unique()) == expected_epochs
    assert LoggingColumns.CrossValidationSplitIndex.value in per_subject_metrics
    assert set(per_subject_metrics[LoggingColumns.CrossValidationSplitIndex.value].unique()) == {0, 1}
    assert LoggingColumns.DataSplit.value in per_subject_metrics
    assert per_subject_metrics[LoggingColumns.DataSplit.value].unique() == ["Val"]


def test_result_aggregation_for_classification(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test how metrics are aggregated for cross validation runs on classification models.
    """
    files, plotting_config = load_result_files_for_classification()
    plotting_config.outputs_directory = test_output_dirs.root_dir
    plotting_config.epoch = 3
    expected_aggregates = ["Default,0.75740,0.91814,0.66854,0.23684,0.35357,0.44438,0.73170,0.33427,356.00000,Val,3"]
    _test_result_aggregation_for_classification(files, plotting_config,
                                                expected_aggregate_metrics=expected_aggregates,
                                                expected_epochs={plotting_config.epoch})
    dataset_csv = plotting_config.outputs_directory / DATASET_CSV_FILE_NAME
    assert dataset_csv.exists()


def test_invalid_number_of_cv_files() -> None:
    """
    Test that an error is raised if the expected number of cross validation fold
    is not equal to the number of results files provided.
    """
    files, plotting_config = load_result_files_for_classification()
    plotting_config.number_of_cross_validation_splits = 4
    print(f"Writing aggregated metrics to {plotting_config.outputs_directory}")
    with pytest.raises(ValueError):
        plot_cross_validation_from_files(OfflineCrossvalConfigAndFiles(config=plotting_config, files=files),
                                         root_folder=plotting_config.outputs_directory)


def test_check_result_file_counts() -> None:
    """
    More tests on the function that checks the number of files of each ModeExecutionMode.
    """
    val_files, plotting_config = load_result_files_for_classification()
    # This test assumes that the loaded val_files all have mode Val
    assert all(file.execution_mode == ModelExecutionMode.VAL for file in val_files)
    plotting_config.number_of_cross_validation_splits = len(val_files)
    # Check that when just the Val files are present, the check does not throw
    config_and_files1 = OfflineCrossvalConfigAndFiles(config=plotting_config, files=val_files)
    check_result_file_counts(config_and_files1)
    # Check that when we add the same number of Test files, the check does not throw
    test_files = [RunResultFiles(execution_mode=ModelExecutionMode.TEST,
                                 metrics_file=file.metrics_file,
                                 dataset_csv_file=file.dataset_csv_file,
                                 run_recovery_id=file.run_recovery_id,
                                 split_index=file.split_index) for file in val_files]
    config_and_files2 = OfflineCrossvalConfigAndFiles(config=plotting_config, files=val_files + test_files)
    check_result_file_counts(config_and_files2)
    # Check that when we have the same number of files as the number of splits, but they are from a mixture
    # of modes, the check does throw
    config_and_files3 = OfflineCrossvalConfigAndFiles(config=plotting_config, files=val_files[:1] + test_files[1:])
    with pytest.raises(ValueError):
        check_result_file_counts(config_and_files3)


def test_result_aggregation_for_classification_all_epochs(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test how metrics are aggregated for classification models, when no epoch is specified.
    """
    files, plotting_config = load_result_files_for_classification()
    plotting_config.outputs_directory = test_output_dirs.root_dir
    plotting_config.epoch = None
    expected_aggregates = \
        ["Default,0.72361,0.90943,0.55618,0.13158,0.52500,0.33307,0.95800,0.21348,356.00000,Val,1",
         "Default,0.75919,0.91962,0.65169,0.19737,0.38571,0.38873,0.82669,0.21348,356.00000,Val,2",
         "Default,0.75740,0.91814,0.66854,0.23684,0.35357,0.44438,0.73170,0.33427,356.00000,Val,3"]
    _test_result_aggregation_for_classification(files, plotting_config,
                                                expected_aggregate_metrics=expected_aggregates,
                                                expected_epochs={1, 2, 3})


def test_add_comparison_data(test_config_comparison: PlotCrossValidationConfig) -> None:
    test_config_comparison.epoch = 2
    test_config_comparison.comparison_epochs = [2]
    metrics_df, root_folder = download_metrics(test_config_comparison)
    initial_metrics = pd.concat(list(metrics_df.values()))
    all_metrics, focus_splits = add_comparison_data(test_config_comparison, initial_metrics)
    focus_split = test_config_comparison.run_recovery_id
    comparison_split = test_config_comparison.comparison_run_recovery_ids[0]
    assert focus_splits == [focus_split]
    assert set(all_metrics.split) == {focus_split, comparison_split}


def test_save_outliers(test_config_ensemble: PlotCrossValidationConfig,
                       test_output_dirs: OutputFolderForTests) -> None:
    """Test to make sure the outlier file for a split is as expected"""
    test_config_ensemble.outputs_directory = test_output_dirs.root_dir
    test_config_ensemble.outlier_range = 0
    dataset_split_metrics = {x: _get_metrics_df(x) for x in [ModelExecutionMode.VAL]}
    save_outliers(test_config_ensemble, dataset_split_metrics, test_config_ensemble.outputs_directory)
    f = f"{ModelExecutionMode.VAL.value}_outliers.txt"
    assert_text_files_match(full_file=test_config_ensemble.outputs_directory / f,
                            expected_file=full_ml_test_data_path(f))


def test_create_portal_query_for_outliers() -> None:
    test_df = pd.DataFrame({
        CSV_INSTITUTION_HEADER: range(2),
        CSV_SERIES_HEADER: range(3, 5),
        "other": range(2)
    })
    expected = PORTAL_QUERY_TEMPLATE.format('r.InstitutionId = "0" OR r.InstitutionId = "1"',
                                            'STARTSWITH(r.VersionedDicomImageSeries.Latest.Series.InstanceUID,"3") OR '
                                            'STARTSWITH(r.VersionedDicomImageSeries.Latest.Series.InstanceUID,"4")')

    assert expected == create_portal_query_for_outliers(test_df)


def test_create_summary(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that summaries of CV performance per mode, and per mode per structure, look like they should.
    """
    root = test_output_dirs.root_dir
    test_file = full_ml_test_data_path("MetricsAcrossAllRuns.csv")
    df = pd.read_csv(test_file)
    file1, file2 = create_results_breakdown(df, root)
    expected1 = full_ml_test_data_path(METRICS_BY_MODE_AND_STRUCTURE_FILE)
    expected2 = full_ml_test_data_path(METRICS_BY_MODE_FILE)
    assert file1.read_text() == expected1.read_text()
    assert file2.read_text() == expected2.read_text()


def test_plot_config() -> None:
    """
    Test that plotting configurations have the correct error handling.
    """
    with pytest.raises(ValueError):
        PlotCrossValidationConfig()
    with pytest.raises(ValueError):
        PlotCrossValidationConfig(run_recovery_id="foo")
    with pytest.raises(ValueError):
        PlotCrossValidationConfig(epoch=1)
    PlotCrossValidationConfig(run_recovery_id="foo", epoch=1)


def test_get_split_index() -> None:
    """
    Test that get_split_index returns the full run ID only when the
    split index itself is negative.
    """
    tags = {CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY: "-1",
            RUN_RECOVERY_ID_KEY: "foo_bar_23"}
    assert get_split_id(tags) == "foo_bar_23"
    tags = {CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY: "42",
            RUN_RECOVERY_ID_KEY: "foo_bar_23"}
    assert get_split_id(tags) == "42"


@pytest.mark.parametrize("is_current_run", [True, False])
def test_download_or_get_local_blobs(is_current_run: bool,
                                     test_config: PlotCrossValidationConfig,
                                     test_output_dirs: OutputFolderForTests) -> None:
    azure_config = get_default_azure_config()
    azure_config.get_workspace()
    assert test_config.run_recovery_id is not None
    run = Run.get_context() if is_current_run else fetch_run(azure_config.get_workspace(),
                                                             test_config.run_recovery_id)
    run_outputs_dir = full_ml_test_data_path() if is_current_run else Path(DEFAULT_AML_UPLOAD_DIR)
    test_config.outputs_directory = run_outputs_dir
    dst = test_config.download_or_get_local_file(
        blob_to_download="dataset.csv",
        destination=test_output_dirs.root_dir,
        run=run
    )
    assert dst is not None
    assert dst.exists()


def test_download_or_get_local_file_2(test_output_dirs: OutputFolderForTests) -> None:
    config = PlotCrossValidationConfig(run_recovery_id=None,
                                       model_category=ModelCategory.Classification,
                                       epoch=None,
                                       should_validate=False)
    download_to_folder = test_output_dirs.root_dir / CROSSVAL_RESULTS_FOLDER
    config.outputs_directory = download_to_folder
    local_results = full_ml_test_data_path("plot_cross_validation") / "HD_cfff5ceb-a227-41d6-a23c-0ebbc33b6301"
    config.local_run_results = str(local_results)
    # A file that sits in the root folder of the local_results should be downloaded into the
    # root of the download_to folder
    file1 = "dummy.txt"
    file_in_folder = config.download_or_get_local_file(None,
                                                       file1,
                                                       download_to_folder)
    assert file_in_folder is not None
    assert file_in_folder == download_to_folder / file1

    # Copying a file in a sub-folder of the local_results: The full path to the file should be
    # preserved and created in the download_to folder.
    file2 = Path("0") / "Val" / "metrics.csv"
    file_in_folder = config.download_or_get_local_file(None,
                                                       file2,
                                                       download_to_folder)
    assert file_in_folder is not None
    assert file_in_folder == download_to_folder / file2


@pytest.mark.skip(reason="This test is only used to create input for test_load_files_with_prediction_target")
def test_run_ml_with_multi_label_sequence_in_crossval(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test training and testing of sequence models that predicts at multiple time points,
    including aggregation of cross validation results.
    """
    logging_to_stdout()
    config = ToyMultiLabelSequenceModel(should_validate=False)
    assert config.get_target_indices() == [1, 2, 3]
    expected_prediction_targets = ["Seq_pos 01", "Seq_pos 02", "Seq_pos 03"]
    target_indices = config.get_target_indices()
    assert target_indices
    assert len(target_indices) == len(expected_prediction_targets)
    config.set_output_to(test_output_dirs.root_dir)
    config.dataset_data_frame = _get_multi_label_sequence_dataframe()
    config.pre_process_dataset_dataframe()
    config.num_epochs = 1
    config.number_of_cross_validation_splits = 2
    azure_config = get_default_azure_config()
    azure_config.train = True
    MLRunner(config, azure_config).run()


def test_load_files_with_prediction_target() -> None:
    """
    For multi-week RNNs that predict at multiple sequence points: Test that the dataframes
    including the prediction_target column can be loaded.
    """
    folder = "multi_label_sequence_in_crossval"
    plotting_config = PlotCrossValidationConfig(
        run_recovery_id="foo",
        epoch=1,
        model_category=ModelCategory.Classification
    )
    files = create_run_result_file_list(plotting_config, folder)

    downloaded_metrics = load_dataframes(files, plotting_config)
    assert ModelExecutionMode.TEST not in downloaded_metrics
    metrics = downloaded_metrics[ModelExecutionMode.VAL]
    assert metrics is not None
    assert LoggingColumns.Hue.value in metrics
    # The prediction target column should always be read as a string, because we will later use it to create
    # hue values for a MetricsDict.
    assert is_string_dtype(metrics[LoggingColumns.Hue.value].dtype)
    assert LoggingColumns.Epoch.value in metrics
    assert LoggingColumns.Patient.value in metrics
    assert len(metrics[LoggingColumns.Hue.value].unique()) == 3
    # Each of the two CV folds has 2 distinct subjects
    assert len(metrics[LoggingColumns.Patient.value].unique()) == 4


def test_aggregate_files_with_prediction_target(test_output_dirs: OutputFolderForTests) -> None:
    """
    For multi-week RNNs that predict at multiple sequence points: Test that the dataframes
    including the prediction_target column can be aggregated.
    """
    plotting_config = PlotCrossValidationConfig(
        run_recovery_id="foo",
        epoch=1,
        model_category=ModelCategory.Classification
    )
    files = create_run_result_file_list(plotting_config, "multi_label_sequence_in_crossval")

    root_folder = test_output_dirs.root_dir
    print(f"Writing result files to {root_folder}")
    plot_cross_validation_from_files(OfflineCrossvalConfigAndFiles(config=plotting_config, files=files),
                                     root_folder=root_folder)
