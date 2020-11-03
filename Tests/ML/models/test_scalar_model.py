#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import io
import logging
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional
from unittest import mock

import pandas as pd
import pytest
import torch
from more_itertools import flatten

from InnerEye.Common import common_util, fixed_paths
from InnerEye.Common.common_util import CROSSVAL_RESULTS_FOLDER, EPOCH_METRICS_FILE_NAME, METRICS_AGGREGATES_FILE, \
    METRICS_FILE_NAME, logging_to_stdout, epoch_folder_name
from InnerEye.Common.metrics_dict import MetricType, MetricsDict, ScalarMetricsDict
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML import model_testing, model_training, runner
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.dataset.scalar_dataset import ScalarDataset
from InnerEye.ML.metrics import InferenceMetricsForClassification, binary_classification_accuracy, \
    compute_scalar_metrics
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.scalar_config import ScalarLoss, ScalarModelBase
from InnerEye.ML.utils.config_util import ModelConfigLoader
from InnerEye.ML.utils.metrics_constants import LoggingColumns
from InnerEye.ML.visualizers.plot_cross_validation import EpochMetricValues, get_config_and_results_for_offline_runs, \
    unroll_aggregate_metrics

from Tests.ML.configs.ClassificationModelForTesting import ClassificationModelForTesting
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import get_default_azure_config, machine_has_gpu, get_default_checkpoint_handler
from Tests.fixed_paths_for_tests import full_ml_test_data_path


@pytest.mark.cpu_and_gpu
@pytest.mark.parametrize("use_mixed_precision", [False, True])
def test_train_classification_model(test_output_dirs: OutputFolderForTests,
                                    use_mixed_precision: bool) -> None:
    """
    Test training and testing of classification models, asserting on the individual results from training and testing.
    Expected test results are stored for GPU with and without mixed precision.
    """
    logging_to_stdout(logging.DEBUG)
    config = ClassificationModelForTesting()
    config.set_output_to(test_output_dirs.root_dir)
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=Path(test_output_dirs.root_dir))
    # Train for 4 epochs, checkpoints at epochs 2 and 4
    config.num_epochs = 4
    config.use_mixed_precision = use_mixed_precision
    config.save_start_epoch = 2
    config.save_step_epochs = 2
    config.test_start_epoch = 2
    config.test_step_epochs = 2
    config.test_diff_epochs = 2
    expected_epochs = [2, 4]
    assert config.get_test_epochs() == expected_epochs
    model_training_result = model_training.model_train(config, checkpoint_handler=checkpoint_handler)
    assert model_training_result is not None
    expected_learning_rates = [0.0001, 9.99971e-05, 9.99930e-05, 9.99861e-05]
    use_mixed_precision_and_gpu = use_mixed_precision and machine_has_gpu
    if use_mixed_precision_and_gpu:
        expected_train_loss = [0.686614, 0.686465, 0.686316, 0.686167]
        expected_val_loss = [0.737039, 0.736721, 0.736339, 0.735957]
    else:
        expected_train_loss = [0.686614, 0.686465, 0.686316, 0.686167]
        expected_val_loss = [0.737061, 0.736690, 0.736321, 0.735952]

    def extract_loss(results: List[MetricsDict]) -> List[float]:
        return [d.values()[MetricType.LOSS.value][0] for d in results]

    actual_train_loss = extract_loss(model_training_result.train_results_per_epoch)
    actual_val_loss = extract_loss(model_training_result.val_results_per_epoch)
    actual_learning_rates = list(flatten(model_training_result.learning_rates_per_epoch))
    assert actual_train_loss == pytest.approx(expected_train_loss, abs=1e-6)
    assert actual_val_loss == pytest.approx(expected_val_loss, abs=1e-6)
    assert actual_learning_rates == pytest.approx(expected_learning_rates, rel=1e-5)
    test_results = model_testing.model_test(config, ModelExecutionMode.TRAIN,
                                            checkpoint_handler=checkpoint_handler)
    assert isinstance(test_results, InferenceMetricsForClassification)
    assert list(test_results.epochs.keys()) == expected_epochs
    if use_mixed_precision_and_gpu:
        expected_metrics = {
            2: [0.635942, 0.736691],
            4: [0.636085, 0.735952],
        }
    else:
        expected_metrics = {
            2: [0.635941, 0.736690],
            4: [0.636084, 0.735952],
        }
    for epoch in expected_epochs:
        assert test_results.epochs[epoch].values()[MetricType.CROSS_ENTROPY.value] == \
               pytest.approx(expected_metrics[epoch], abs=1e-6)
    # Run detailed logs file check only on CPU, it will contain slightly different metrics on GPU, but here
    # we want to mostly assert that the files look reasonable
    if not machine_has_gpu:
        # Check log EPOCH_METRICS_FILE_NAME
        epoch_metrics_path = config.outputs_folder / ModelExecutionMode.TRAIN.value / EPOCH_METRICS_FILE_NAME
        # Auto-format will break the long header line, hence the strange way of writing it!
        expected_epoch_metrics = \
            "loss,cross_entropy,accuracy_at_threshold_05,seconds_per_batch,seconds_per_epoch,learning_rate," + \
            "area_under_roc_curve,area_under_pr_curve,accuracy_at_optimal_threshold," \
            "false_positive_rate_at_optimal_threshold,false_negative_rate_at_optimal_threshold," \
            "optimal_threshold,subject_count,epoch,cross_validation_split_index\n" + \
            """0.6866141557693481,0.6866141557693481,0.5,0,0,0.0001,1.0,1.0,0.5,0.0,0.0,0.529514,2.0,1,-1
            0.6864652633666992,0.6864652633666992,0.5,0,0,9.999712322065557e-05,1.0,1.0,0.5,0.0,0.0,0.529475,2.0,2,-1
            0.6863163113594055,0.6863162517547607,0.5,0,0,9.999306876841536e-05,1.0,1.0,0.5,0.0,0.0,0.529437,2.0,3,-1
            0.6861673593521118,0.6861673593521118,0.5,0,0,9.998613801725043e-05,1.0,1.0,0.5,0.0,0.0,0.529399,2.0,4,-1
            """
        check_log_file(epoch_metrics_path, expected_epoch_metrics,
                       ignore_columns=[LoggingColumns.SecondsPerBatch.value, LoggingColumns.SecondsPerEpoch.value])

        # Check log METRICS_FILE_NAME
        metrics_path = config.outputs_folder / ModelExecutionMode.TRAIN.value / METRICS_FILE_NAME
        metrics_expected = \
            """prediction_target,epoch,subject,model_output,label,cross_validation_split_index,data_split
Default,1,S4,0.5216594338417053,0.0,-1,Train
Default,1,S2,0.5295137763023376,1.0,-1,Train
Default,2,S4,0.5214819312095642,0.0,-1,Train
Default,2,S2,0.5294750332832336,1.0,-1,Train
Default,3,S4,0.5213046073913574,0.0,-1,Train
Default,3,S2,0.5294366478919983,1.0,-1,Train
Default,4,S4,0.5211275815963745,0.0,-1,Train
Default,4,S2,0.5293986201286316,1.0,-1,Train
"""
        check_log_file(metrics_path, metrics_expected, ignore_columns=[])

        # Check log METRICS_FILE_NAME inside of the folder epoch_004/Train, which is written when we run model_test.
        # Normally, we would run it on the Test and Val splits, but for convenience we test on the train split here.
        inference_metrics_path = config.outputs_folder / Path(epoch_folder_name(config.num_epochs)) / \
                           ModelExecutionMode.TRAIN.value / METRICS_FILE_NAME
        inference_metrics_expected = \
            """prediction_target,epoch,subject,model_output,label,cross_validation_split_index,data_split
Default,4,S2,0.5293986201286316,1.0,-1,Train
Default,4,S4,0.5211275815963745,0.0,-1,Train
"""
        check_log_file(inference_metrics_path, inference_metrics_expected, ignore_columns=[])


def check_log_file(path: Path, expected_csv: str, ignore_columns: List[str]) -> None:
    df_expected = pd.read_csv(StringIO(expected_csv))
    df_epoch_metrics_actual = pd.read_csv(path)
    for ignore_column in ignore_columns:
        assert ignore_column in df_epoch_metrics_actual
        # We cannot compare time because in different machines this takes different times
        del df_epoch_metrics_actual[ignore_column]
        if ignore_column in df_expected:
            del df_expected[ignore_column]
    pd.testing.assert_frame_equal(df_expected, df_epoch_metrics_actual, check_less_precise=True)


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
@pytest.mark.parametrize("model_name", ["DummyClassification", "DummyRegression"])
@pytest.mark.parametrize("number_of_offline_cross_validation_splits", [2])
@pytest.mark.parametrize("number_of_cross_validation_splits_per_fold", [2])
def test_run_ml_with_classification_model(test_output_dirs: OutputFolderForTests,
                                          number_of_offline_cross_validation_splits: int,
                                          number_of_cross_validation_splits_per_fold: int,
                                          model_name: str) -> None:
    """
    Test training and testing of classification models, when it is started together via run_ml.
    """
    logging_to_stdout()
    azure_config = get_default_azure_config()
    azure_config.train = True
    train_config: ScalarModelBase = ModelConfigLoader[ScalarModelBase]() \
        .create_model_config_from_name(model_name)
    train_config.number_of_cross_validation_splits = number_of_offline_cross_validation_splits
    train_config.number_of_cross_validation_splits_per_fold = number_of_cross_validation_splits_per_fold
    train_config.set_output_to(test_output_dirs.root_dir)
    if train_config.perform_sub_fold_cross_validation:
        train_config.local_dataset = full_ml_test_data_path("classification_data_sub_fold_cv")
    MLRunner(train_config, azure_config).run()
    _check_offline_cross_validation_output_files(train_config)

    if train_config.is_regression_model:
        assert (train_config.outputs_folder / "0" / "error_plot_4.png").is_file()

    if train_config.perform_cross_validation:
        # Test that the result files can be correctly picked up by the cross validation routine.
        # For that, we point the downloader to the local results folder. The core download method
        # recognizes run_recovery_id == None as the signal to read from the local_run_results folder.
        config_and_files = get_config_and_results_for_offline_runs(train_config)
        result_files = config_and_files.files
        # One file for VAL and one for TRAIN for each child run
        assert len(result_files) == train_config.get_total_number_of_cross_validation_runs() * 2
        for file in result_files:
            assert file.execution_mode != ModelExecutionMode.TEST
            assert file.dataset_csv_file is not None
            assert file.dataset_csv_file.exists()
            assert file.metrics_file is not None
            assert file.metrics_file.exists()


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
def test_run_ml_with_segmentation_model(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test training and testing of segmentation models, when it is started together via run_ml.
    """
    train_config = DummyModel()
    train_config.num_dataload_workers = 0
    train_config.restrict_subjects = "1"
    # Increasing the test crop size should not have any effect on the results.
    # This is for a bug in an earlier version of the code where the wrong execution mode was used to
    # compute the expected mask size at training time.
    train_config.test_crop_size = (75, 75, 75)
    train_config.perform_training_set_inference = False
    train_config.perform_validation_and_test_set_inference = True
    train_config.set_output_to(test_output_dirs.root_dir)
    azure_config = get_default_azure_config()
    azure_config.train = True
    MLRunner(train_config, azure_config).run()


def test_runner1(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test starting a classification model via the commandline runner. Test if we can provide overrides
    for parameters that live inside the DeepLearningConfig, and ones that are specific to classification models.
    :return:
    """
    set_from_commandline = 12345
    scalar1 = '["label"]'
    model_name = "DummyClassification"
    initial_config = ModelConfigLoader[ScalarModelBase]().create_model_config_from_name(model_name)
    assert initial_config.non_image_feature_channels == []
    output_root = str(test_output_dirs.root_dir)
    args = ["",
            "--model", model_name,
            "--train", "True",
            "--random_seed", str(set_from_commandline),
            "--non_image_feature_channels", scalar1,
            "--output_to", output_root,
            ]
    with mock.patch("sys.argv", args):
        config, _ = runner.run(project_root=fixed_paths.repository_root_directory(),
                               yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)
    assert isinstance(config, ScalarModelBase)
    assert config.model_name == "DummyClassification"
    assert config.get_effective_random_seed() == set_from_commandline
    assert config.non_image_feature_channels == ["label"]
    assert str(config.outputs_folder).startswith(output_root)
    assert (config.logs_folder / runner.LOG_FILE_NAME).exists()


def test_runner2(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test starting a classification model via the commandline runner, and provide the same arguments
    that would be passed in via the YAML files.
    :return:
    """
    output_root = str(test_output_dirs.root_dir)
    args = ["",
            "--model", "DummyClassification",
            "--train", "True",
            "--output_to", output_root,
            ]
    with mock.patch("sys.argv", args):
        config, _ = runner.run(project_root=fixed_paths.repository_root_directory(),
                               yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)
    assert isinstance(config, ScalarModelBase)
    assert config.name.startswith("DummyClassification")


@pytest.mark.skipif(common_util.is_windows(), reason="Has issues on windows build")
@pytest.mark.gpu
@pytest.mark.parametrize(["output_values_list", "expected_accuracy"],
                         [([0.4, 0.9], 1.0),
                          ([0.9, 0.4], 0.0),
                          ([0.4, 0.4], 0.5)]
                         )
def test_binary_classification_accuracy(output_values_list: List, expected_accuracy: float) -> None:
    labels = torch.tensor([0.3, 1.0])
    model_output = torch.tensor(output_values_list)
    if machine_has_gpu:
        labels = labels.cuda()
        model_output = model_output.cuda()
    actual_accuracy = binary_classification_accuracy(model_output, labels)
    assert actual_accuracy == pytest.approx(expected_accuracy, abs=1e-8)


@pytest.mark.gpu
@pytest.mark.parametrize("has_hues", [True, False])
@pytest.mark.parametrize("is_classification", [True, False])
def test_scalar_metrics(has_hues: bool, is_classification: bool) -> None:
    hues = ["Foo", "Bar", "Car"]
    if is_classification:
        values = [[0.4, 0.9, 0.3], [0.9, 0.4, 0.2]]
        labels = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        expected_accuracy = [1.0, 0.0, 0.5]
        accuracy_metric_key = MetricType.ACCURACY_AT_THRESHOLD_05.value
        expected_info_format_strs = [
            "CrossEntropy: 0.3081, AccuracyAtThreshold05: 1.0000",
            "CrossEntropy: 1.6094, AccuracyAtThreshold05: 0.0000",
            "CrossEntropy: 0.9831, AccuracyAtThreshold05: 0.5000",
        ]
    else:
        values = [[1.5, -1.0, 2.0], [1.5, 0.0, 1.0]]
        labels = [[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]
        expected_accuracy = [0.25, 5, 0]
        accuracy_metric_key = MetricType.MEAN_SQUARED_ERROR.value
        expected_info_format_strs = [
            "MeanSquaredError: 0.2500, MeanAbsoluteError: 0.5000, r2Score: 0.0000",
            "MeanSquaredError: 5.0000, MeanAbsoluteError: 2.0000, r2Score: -19.0000",
            "MeanSquaredError: 0.0000, MeanAbsoluteError: 0.0000, r2Score: 1.0000"
        ]

    def _get_expected_info_str(index: Optional[int] = None) -> str:
        if index is not None:
            df = pd.DataFrame.from_dict({
                MetricsDict.DATAFRAME_COLUMNS[0]: MetricsDict.DEFAULT_HUE_KEY,
                MetricsDict.DATAFRAME_COLUMNS[1]: [expected_info_format_strs[index]]
            })
        else:
            df = pd.DataFrame.from_dict({
                MetricsDict.DATAFRAME_COLUMNS[0]: hues,
                MetricsDict.DATAFRAME_COLUMNS[1]: expected_info_format_strs
            })
        return df.to_string(index=False)

    if has_hues:
        metrics_dict = _compute_scalar_metrics(
            output_values_list=values, labels=labels, hues=hues, is_classification=is_classification
        )
        for i, hue in enumerate(hues):
            assert metrics_dict.values(hue=hue)[accuracy_metric_key] == [expected_accuracy[i]]
        assert metrics_dict.to_string(tabulate=False) == _get_expected_info_str()
    else:
        for i, hue in enumerate(hues):
            _values = [[x[i]] for x in values]
            _labels = [[x[i]] for x in labels]
            metrics_dict = _compute_scalar_metrics(output_values_list=_values, labels=_labels,
                                                   is_classification=is_classification)
            assert metrics_dict.values()[accuracy_metric_key] == [expected_accuracy[i]]
            assert metrics_dict.to_string(tabulate=False) == _get_expected_info_str(index=i)


def _compute_scalar_metrics(output_values_list: List[List[float]],
                            labels: List[List[float]],
                            is_classification: bool,
                            hues: Optional[List[str]] = None) -> ScalarMetricsDict:
    model_output = torch.tensor(output_values_list)
    _labels = torch.tensor(labels)
    if machine_has_gpu:
        _labels = _labels.cuda()
        model_output = model_output.cuda()
    metrics_dict = ScalarMetricsDict(hues=hues, is_classification_metrics=is_classification)
    subject_ids = list(map(str, range(model_output.shape[0])))
    loss_type = ScalarLoss.BinaryCrossEntropyWithLogits if is_classification else ScalarLoss.MeanSquaredError
    compute_scalar_metrics(metrics_dict, subject_ids, model_output, _labels, loss_type=loss_type)
    return metrics_dict


@pytest.mark.parametrize("offline_parent_cv_run", [True, False])
def test_is_offline_cross_val_parent_run(offline_parent_cv_run: bool) -> None:
    train_config = DummyModel()
    train_config.number_of_cross_validation_splits = 2 if offline_parent_cv_run else 0
    assert MLRunner(train_config).is_offline_cross_val_parent_run() == offline_parent_cv_run


def _check_offline_cross_validation_output_files(train_config: ScalarModelBase) -> None:
    metrics: Dict[ModelExecutionMode, List[pd.DataFrame]] = dict()
    root = Path(train_config.file_system_config.outputs_folder)
    for x in range(train_config.get_total_number_of_cross_validation_runs()):
        expected_outputs_folder = root / str(x)
        assert expected_outputs_folder.exists()
        for m in [ModelExecutionMode.TRAIN, ModelExecutionMode.VAL]:
            metrics_path = expected_outputs_folder / m.value / METRICS_FILE_NAME
            assert metrics_path.exists()
            split_metrics = pd.read_csv(metrics_path)
            if m in metrics:
                # check that metrics for any two folds is not the same
                assert not any([split_metrics.equals(x) for x in metrics[m]])
            metrics[m] = [split_metrics]
    if train_config.perform_cross_validation:
        # test aggregates are as expected
        aggregate_metrics_path = root / CROSSVAL_RESULTS_FOLDER / METRICS_AGGREGATES_FILE
        assert aggregate_metrics_path.is_file()
        # since we aggregate the outputs of each of the child folds
        # we need to compare the outputs w.r.t to the parent folds
        child_folds = train_config.number_of_cross_validation_splits_per_fold
        if train_config.perform_sub_fold_cross_validation:
            train_config.number_of_cross_validation_splits_per_fold = 0
        _dataset_splits = train_config.get_dataset_splits()
        train_config.number_of_cross_validation_splits_per_fold = child_folds

        _val_dataset_split_count = len(_dataset_splits.val[train_config.subject_column].unique()) + len(
            _dataset_splits.train[train_config.subject_column].unique())
        _aggregates_csv = pd.read_csv(aggregate_metrics_path)
        _counts_for_splits = list(_aggregates_csv[LoggingColumns.SubjectCount.value])
        assert all([x == _val_dataset_split_count for x in _counts_for_splits])
        _epochs = list(_aggregates_csv[LoggingColumns.Epoch.value])
        # Each epoch is recorded twice once for the training split and once for the validation
        # split
        assert len(_epochs) == train_config.num_epochs * 2
        assert all([x + 1 in _epochs for x in list(range(train_config.num_epochs)) * 2])
        # Only the validation mode is kept for unrolled aggregates
        unrolled = unroll_aggregate_metrics(_aggregates_csv)
        if train_config.is_classification_model:
            expected_metrics = {LoggingColumns.CrossEntropy.value,
                                LoggingColumns.AreaUnderPRCurve.value,
                                LoggingColumns.AreaUnderRocCurve.value,
                                LoggingColumns.FalseNegativeRateAtOptimalThreshold.value,
                                LoggingColumns.FalsePositiveRateAtOptimalThreshold.value,
                                LoggingColumns.AccuracyAtOptimalThreshold.value,
                                LoggingColumns.OptimalThreshold.value,
                                LoggingColumns.AccuracyAtThreshold05.value}
        else:
            expected_metrics = {LoggingColumns.MeanAbsoluteError.value,
                                LoggingColumns.MeanSquaredError.value,
                                LoggingColumns.R2Score.value}
        expected_metrics = expected_metrics.union({LoggingColumns.SubjectCount.value})
        assert len(unrolled) == train_config.num_epochs * len(expected_metrics)
        actual_metrics = set(m.metric_name for m in unrolled)
        assert actual_metrics == expected_metrics
        actual_epochs = set(m.epoch for m in unrolled)
        assert actual_epochs == set(_epochs)


def test_unroll_aggregates() -> None:
    # This is an output file of a CV run on a classification model, shuffled such that epochs are not in the right
    # order.
    file = io.StringIO("""area_under_roc_curve,area_under_pr_curve,cross_entropy,subject_count,data_split,epoch
1.00000,1.00000,0.70290,3,Val,4
1.00000,1.00000,0.70339,3,Val,1
1.00000,1.00000,0.70323,3,Val,2
1.00000,1.00000,0.70306,3,Val,3
""")
    df = pd.read_csv(file)
    unrolled = unroll_aggregate_metrics(df)
    expected_metrics = {LoggingColumns.CrossEntropy.value,
                        LoggingColumns.AreaUnderPRCurve.value,
                        LoggingColumns.AreaUnderRocCurve.value,
                        LoggingColumns.SubjectCount.value}
    expected_epochs = set(range(1, 5))
    assert len(unrolled) == len(expected_epochs) * len(expected_metrics)
    actual_metrics = set(m.metric_name for m in unrolled)
    assert actual_metrics == expected_metrics
    actual_epochs = set(m.epoch for m in unrolled)
    assert actual_epochs == expected_epochs
    assert unrolled[0] == EpochMetricValues(1, LoggingColumns.AreaUnderPRCurve.value, 1.0)
    assert unrolled[-2] == EpochMetricValues(4, LoggingColumns.CrossEntropy.value, 0.7029)
    assert unrolled[-1] == EpochMetricValues(4, LoggingColumns.SubjectCount.value, 3)


def test_dataset_stats_hook(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if the flexible hook for computing dataset statistics is called correctly in create_and_set_torch_datasets
    """
    model = ClassificationModelForTesting()
    root_dir = test_output_dirs.root_dir
    out_file = root_dir / "stats.txt"

    def hook(datasets: Dict[ModelExecutionMode, ScalarDataset]) -> None:
        # Assert on types to ensure that the hook is called with the right arguments
        assert isinstance(datasets, Dict)
        lines = []
        for mode in ModelExecutionMode:
            assert mode in datasets
            assert isinstance(datasets[mode], ScalarDataset)
            lines.append(f"{mode.value}: {len(datasets[mode].items)}")
        out_file.write_text("\n".join(lines))

    model.dataset_stats_hook = hook

    model.create_and_set_torch_datasets()
    assert out_file.is_file()
    assert out_file.read_text() == "\n".join(["Train: 2", "Test: 1", "Val: 1"])


def test_dataset_stats_hook_failing(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if the hook for computing dataset statistics can safely fail.
    """
    model = ClassificationModelForTesting()

    def hook(_: Dict[ModelExecutionMode, ScalarDataset]) -> None:
        raise ValueError()

    model.dataset_stats_hook = hook
    model.create_and_set_torch_datasets()


def test_get_dataset_splits() -> None:
    """
    Test if dataset splits are created as expected for scalar models.
    """
    model = ClassificationModelForTesting()
    model.local_dataset = full_ml_test_data_path("classification_data_sub_fold_cv")
    model.number_of_cross_validation_splits = 2
    dataset_splits = model.get_dataset_splits()
    assert list(dataset_splits[ModelExecutionMode.TRAIN].subjectID.unique()) == ['S4', 'S5', 'S2', 'S10']
    assert list(dataset_splits[ModelExecutionMode.VAL].subjectID.unique()) == ['S1', 'S6', 'S7', 'S8']
    assert list(dataset_splits[ModelExecutionMode.TEST].subjectID.unique()) == ['S3', 'S9']
    # check if sub-folds are created as expected
    model.number_of_cross_validation_splits_per_fold = 2
    sub_fold_dataset_splits = model.get_dataset_splits()
    # the validation and the test set must be the same for parent and sub fold
    pd.testing.assert_frame_equal(dataset_splits.val, sub_fold_dataset_splits.val,
                                         check_like=True, check_dtype=False)
    pd.testing.assert_frame_equal(dataset_splits.test,
                                         sub_fold_dataset_splits.test, check_like=True,
                                         check_dtype=False)
    # make sure the training set is the expected subset of the parent
    assert list(sub_fold_dataset_splits[ModelExecutionMode.TRAIN].subjectID.unique()) == ['S2', 'S10']
