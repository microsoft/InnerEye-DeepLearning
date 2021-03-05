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

from InnerEye.Common import common_util, fixed_paths
from InnerEye.Common.common_util import CROSSVAL_RESULTS_FOLDER, EPOCH_METRICS_FILE_NAME, METRICS_AGGREGATES_FILE, \
    SUBJECT_METRICS_FILE_NAME, get_epoch_results_path, logging_to_stdout
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.metrics_constants import LoggingColumns, MetricType
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML import model_testing, model_training, runner
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.configs.classification.DummyMulticlassClassification import DummyMulticlassClassification
from InnerEye.ML.dataset.scalar_dataset import ScalarDataset
from InnerEye.ML.metrics import InferenceMetricsForClassification, binary_classification_accuracy, \
    compute_scalar_metrics
from InnerEye.ML.metrics_dict import MetricsDict, ScalarMetricsDict
from InnerEye.ML.reports.notebook_report import get_ipynb_report_name, get_html_report_name, \
    generate_classification_notebook, generate_classification_multilabel_notebook
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.scalar_config import ScalarLoss, ScalarModelBase
from InnerEye.ML.utils.config_util import ModelConfigLoader
from InnerEye.ML.visualizers.plot_cross_validation import EpochMetricValues, get_config_and_results_for_offline_runs, \
    unroll_aggregate_metrics
from Tests.ML.configs.ClassificationModelForTesting import ClassificationModelForTesting
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import get_default_azure_config, get_default_checkpoint_handler, machine_has_gpu


@pytest.mark.cpu_and_gpu
def test_train_classification_model(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test training and testing of classification models, asserting on the individual results from training and
    testing.
    Expected test results are stored for GPU with and without mixed precision.
    """
    logging_to_stdout(logging.DEBUG)
    config = ClassificationModelForTesting()
    config.set_output_to(test_output_dirs.root_dir)
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=Path(test_output_dirs.root_dir))
    # Train for 4 epochs, checkpoints at epochs 2 and 4
    config.num_epochs = 4
    model_training_result = model_training.model_train(config, checkpoint_handler=checkpoint_handler)
    assert model_training_result is not None
    expected_learning_rates = [0.0001, 9.99971e-05, 9.99930e-05, 9.99861e-05]
    expected_train_loss = [0.686614, 0.686465, 0.686316, 0.686167]
    expected_val_loss = [0.737061, 0.736691, 0.736321, 0.735952]
    # Ensure that all metrics are computed on both training and validation set
    assert len(model_training_result.train_results_per_epoch) == config.num_epochs
    assert len(model_training_result.val_results_per_epoch) == config.num_epochs
    assert len(model_training_result.train_results_per_epoch[0]) >= 11
    assert len(model_training_result.val_results_per_epoch[0]) >= 11
    for metric in [MetricType.ACCURACY_AT_THRESHOLD_05,
                   MetricType.ACCURACY_AT_OPTIMAL_THRESHOLD,
                   MetricType.AREA_UNDER_PR_CURVE,
                   MetricType.AREA_UNDER_ROC_CURVE,
                   MetricType.CROSS_ENTROPY,
                   MetricType.LOSS,
                   # For unknown reasons, we don't get seconds_per_batch for the training data.
                   # MetricType.SECONDS_PER_BATCH,
                   MetricType.SECONDS_PER_EPOCH,
                   MetricType.SUBJECT_COUNT,
                   ]:
        assert metric.value in model_training_result.train_results_per_epoch[0], f"{metric.value} not in training"
        assert metric.value in model_training_result.val_results_per_epoch[0], f"{metric.value} not in validation"
    actual_train_loss = model_training_result.get_metric(is_training=True, metric_type=MetricType.LOSS.value)
    actual_val_loss = model_training_result.get_metric(is_training=False, metric_type=MetricType.LOSS.value)
    actual_lr = model_training_result.get_metric(is_training=True, metric_type=MetricType.LEARNING_RATE.value)
    assert actual_train_loss == pytest.approx(expected_train_loss, abs=1e-6), "Training loss"
    assert actual_val_loss == pytest.approx(expected_val_loss, abs=1e-6), "Validation loss"
    assert actual_lr == pytest.approx(expected_learning_rates, rel=1e-5), "Learning rates"
    test_results = model_testing.model_test(config, ModelExecutionMode.TRAIN,
                                            checkpoint_handler=checkpoint_handler)
    assert isinstance(test_results, InferenceMetricsForClassification)
    expected_metrics = [0.636085, 0.735952]
    assert test_results.metrics.values()[MetricType.CROSS_ENTROPY.value] == \
           pytest.approx(expected_metrics, abs=1e-5)
    # Run detailed logs file check only on CPU, it will contain slightly different metrics on GPU, but here
    # we want to mostly assert that the files look reasonable
    if machine_has_gpu:
        return
    # Check epoch_metrics.csv
    epoch_metrics_path = config.outputs_folder / ModelExecutionMode.TRAIN.value / EPOCH_METRICS_FILE_NAME
    # Auto-format will break the long header line, hence the strange way of writing it!
    expected_epoch_metrics = \
        "loss,cross_entropy,accuracy_at_threshold_05,seconds_per_epoch,learning_rate," + \
        "area_under_roc_curve,area_under_pr_curve,accuracy_at_optimal_threshold," \
        "false_positive_rate_at_optimal_threshold,false_negative_rate_at_optimal_threshold," \
        "optimal_threshold,subject_count,epoch,cross_validation_split_index\n" + \
        """0.6866141557693481,0.6866141557693481,0.5,0,0.0001,1.0,1.0,0.5,0.0,0.0,0.529514,2.0,0,-1	
        0.6864652633666992,0.6864652633666992,0.5,0,9.999712322065557e-05,1.0,1.0,0.5,0.0,0.0,0.529475,2.0,1,-1	
        0.6863163113594055,0.6863162517547607,0.5,0,9.999306876841536e-05,1.0,1.0,0.5,0.0,0.0,0.529437,2.0,2,-1	
        0.6861673593521118,0.6861673593521118,0.5,0,9.998613801725043e-05,1.0,1.0,0.5,0.0,0.0,0.529399,2.0,3,-1	
        """
    # We cannot compare columns like "seconds_per_epoch" because timing will obviously vary between machines.
    # Column must still be present, though.
    check_log_file(epoch_metrics_path, expected_epoch_metrics,
                   ignore_columns=[LoggingColumns.SecondsPerEpoch.value])
    # Check metrics.csv: This contains the per-subject per-epoch model outputs
    # Randomization comes out slightly different on Windows, hence only execute the test on Linux
    if common_util.is_windows():
        return
    metrics_path = config.outputs_folder / ModelExecutionMode.TRAIN.value / SUBJECT_METRICS_FILE_NAME
    metrics_expected = \
        """prediction_target,epoch,subject,model_output,label,cross_validation_split_index,data_split
Default,0,S2,0.5295137763023376,1.0,-1,Train
Default,0,S4,0.5216594338417053,0.0,-1,Train
Default,1,S4,0.5214819312095642,0.0,-1,Train
Default,1,S2,0.5294750332832336,1.0,-1,Train
Default,2,S2,0.5294366478919983,1.0,-1,Train
Default,2,S4,0.5213046073913574,0.0,-1,Train
Default,3,S2,0.5293986201286316,1.0,-1,Train
Default,3,S4,0.5211275815963745,0.0,-1,Train
"""
    check_log_file(metrics_path, metrics_expected, ignore_columns=[])
    # Check log METRICS_FILE_NAME inside of the folder epoch_004/Train, which is written when we run model_test.
    # Normally, we would run it on the Test and Val splits, but for convenience we test on the train split here.
    inference_metrics_path = config.outputs_folder / get_epoch_results_path(ModelExecutionMode.TRAIN) / \
                             SUBJECT_METRICS_FILE_NAME
    inference_metrics_expected = \
        """prediction_target,subject,model_output,label,cross_validation_split_index,data_split
Default,S2,0.5293986201286316,1.0,-1,Train
Default,S4,0.5211275815963745,0.0,-1,Train
"""
    check_log_file(inference_metrics_path, inference_metrics_expected, ignore_columns=[])


@pytest.mark.cpu_and_gpu
def test_train_classification_multilabel_model(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test training and testing of classification models, asserting on the individual results from training and
    testing.
    Expected test results are stored for GPU with and without mixed precision.
    """
    logging_to_stdout(logging.DEBUG)
    config = DummyMulticlassClassification()
    config.set_output_to(test_output_dirs.root_dir)
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=Path(test_output_dirs.root_dir))
    # Train for 4 epochs, checkpoints at epochs 2 and 4
    config.num_epochs = 4
    model_training_result = model_training.model_train(config, checkpoint_handler=checkpoint_handler)
    assert model_training_result is not None
    expected_learning_rates = [0.0001, 9.99971e-05, 9.99930e-05, 9.99861e-05]
    expected_train_loss = [0.699870228767395, 0.6239662170410156, 0.551329493522644, 0.4825132489204407]
    expected_val_loss = [0.6299371719360352, 0.5546272993087769, 0.4843321740627289, 0.41909298300743103]
    # Ensure that all metrics are computed on both training and validation set
    assert len(model_training_result.train_results_per_epoch) == config.num_epochs
    assert len(model_training_result.val_results_per_epoch) == config.num_epochs
    assert len(model_training_result.train_results_per_epoch[0]) >= 11
    assert len(model_training_result.val_results_per_epoch[0]) >= 11
    for class_name in config.class_names:
        for metric in [MetricType.ACCURACY_AT_THRESHOLD_05,
                       MetricType.ACCURACY_AT_OPTIMAL_THRESHOLD,
                       MetricType.AREA_UNDER_PR_CURVE,
                       MetricType.AREA_UNDER_ROC_CURVE,
                       MetricType.CROSS_ENTROPY]:
            assert f'{metric.value}/{class_name}' in model_training_result.train_results_per_epoch[
                0], f"{metric.value} not in training"
            assert f'{metric.value}/{class_name}' in model_training_result.val_results_per_epoch[
                0], f"{metric.value} not in validation"
    for metric in [MetricType.LOSS,
                   MetricType.SECONDS_PER_EPOCH,
                   MetricType.SUBJECT_COUNT]:
        assert metric.value in model_training_result.train_results_per_epoch[0], f"{metric.value} not in training"
        assert metric.value in model_training_result.val_results_per_epoch[0], f"{metric.value} not in validation"

    actual_train_loss = model_training_result.get_metric(is_training=True, metric_type=MetricType.LOSS.value)
    actual_val_loss = model_training_result.get_metric(is_training=False, metric_type=MetricType.LOSS.value)
    actual_lr = model_training_result.get_metric(is_training=True, metric_type=MetricType.LEARNING_RATE.value)
    assert actual_train_loss == pytest.approx(expected_train_loss, abs=1e-6), "Training loss"
    assert actual_val_loss == pytest.approx(expected_val_loss, abs=1e-6), "Validation loss"
    assert actual_lr == pytest.approx(expected_learning_rates, rel=1e-5), "Learning rates"
    test_results = model_testing.model_test(config, ModelExecutionMode.TRAIN,
                                            checkpoint_handler=checkpoint_handler)
    assert isinstance(test_results, InferenceMetricsForClassification)

    expected_metrics = {MetricType.CROSS_ENTROPY: [1.3996, 5.2966, 1.4020, 0.3553, 0.6908],
                        MetricType.ACCURACY_AT_THRESHOLD_05: [0.0000, 0.0000, 0.0000, 1.0000, 1.0000]
                        }

    for i, class_name in enumerate(config.class_names):
        for metric in expected_metrics.keys():
            assert expected_metrics[metric][i] == pytest.approx(
                                                        test_results.metrics.get_single_metric(
                                                            metric_name=metric,
                                                            hue=class_name), 1e-4)

    def get_epoch_path(mode: ModelExecutionMode) -> Path:
        p = get_epoch_results_path(mode=mode)
        return config.outputs_folder / p / SUBJECT_METRICS_FILE_NAME

    path_to_best_epoch_train = get_epoch_path(ModelExecutionMode.TRAIN)
    path_to_best_epoch_val = get_epoch_path(ModelExecutionMode.VAL)
    path_to_best_epoch_test = get_epoch_path(ModelExecutionMode.TEST)
    generate_classification_notebook(result_notebook=config.outputs_folder / get_ipynb_report_name(config.model_category.value),
                                     config=config,
                                     train_metrics=path_to_best_epoch_train,
                                     val_metrics=path_to_best_epoch_val,
                                     test_metrics=path_to_best_epoch_test)
    assert (config.outputs_folder / get_html_report_name(config.model_category.value)).exists()

    report_name_multilabel = f"{config.model_category.value}_multilabel"
    generate_classification_multilabel_notebook(result_notebook=config.outputs_folder / get_ipynb_report_name(report_name_multilabel),
                                                config=config,
                                                train_metrics=path_to_best_epoch_train,
                                                val_metrics=path_to_best_epoch_val,
                                                test_metrics=path_to_best_epoch_test)
    assert (config.outputs_folder / get_html_report_name(report_name_multilabel)).exists()


def check_log_file(path: Path, expected_csv: str, ignore_columns: List[str]) -> None:
    df_expected = pd.read_csv(StringIO(expected_csv))
    df_epoch_metrics_actual = pd.read_csv(path)
    for ignore_column in ignore_columns:
        assert ignore_column in df_epoch_metrics_actual, f"Column {ignore_column} will be ignored, but must still be" \
                                                         f"present in the dataframe"
        del df_epoch_metrics_actual[ignore_column]
        if ignore_column in df_expected:
            del df_expected[ignore_column]
    pd.testing.assert_frame_equal(df_expected, df_epoch_metrics_actual, check_less_precise=True, check_like=True)


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
@pytest.mark.parametrize("model_name", ["DummyClassification", "DummyRegression"])
@pytest.mark.parametrize("number_of_offline_cross_validation_splits", [2])
def test_run_ml_with_classification_model(test_output_dirs: OutputFolderForTests,
                                          number_of_offline_cross_validation_splits: int,
                                          model_name: str) -> None:
    """
    Test training and testing of classification models, when it is started together via run_ml.
    """
    logging_to_stdout()
    azure_config = get_default_azure_config()
    azure_config.train = True
    config: ScalarModelBase = ModelConfigLoader[ScalarModelBase]() \
        .create_model_config_from_name(model_name)
    config.number_of_cross_validation_splits = number_of_offline_cross_validation_splits
    config.set_output_to(test_output_dirs.root_dir)
    # Trying to run DDP from the test suite hangs, hence restrict to single GPU.
    config.max_num_gpus = 1
    MLRunner(config, azure_config).run()
    _check_offline_cross_validation_output_files(config)

    if config.perform_cross_validation:
        # Test that the result files can be correctly picked up by the cross validation routine.
        # For that, we point the downloader to the local results folder. The core download method
        # recognizes run_recovery_id == None as the signal to read from the local_run_results folder.
        config_and_files = get_config_and_results_for_offline_runs(config)
        result_files = config_and_files.files
        # One file for VAL and one for TRAIN for each child run
        assert len(result_files) == config.get_total_number_of_cross_validation_runs() * 2
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
    config = DummyModel()
    config.num_dataload_workers = 0
    config.restrict_subjects = "1"
    # Increasing the test crop size should not have any effect on the results.
    # This is for a bug in an earlier version of the code where the wrong execution mode was used to
    # compute the expected mask size at training time.
    config.test_crop_size = (75, 75, 75)
    config.perform_training_set_inference = False
    config.perform_validation_and_test_set_inference = True
    config.set_output_to(test_output_dirs.root_dir)
    azure_config = get_default_azure_config()
    azure_config.train = True
    MLRunner(config, azure_config).run()


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
            "--max_num_gpus", "1"
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
            "--max_num_gpus", "1"
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
        # Issue #373: We have odd values here for ExplainedVariance, and had already for r2score
        expected_info_format_strs = [
            "MeanSquaredError: 0.2500, MeanAbsoluteError: 0.5000, ExplainedVariance: 0.0000",
            "MeanSquaredError: 5.0000, MeanAbsoluteError: 2.0000, ExplainedVariance: -19.0000",
            "MeanSquaredError: 0.0000, MeanAbsoluteError: 0.0000, ExplainedVariance: 1.0000"
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
            metrics_path = expected_outputs_folder / m.value / SUBJECT_METRICS_FILE_NAME
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
        _dataset_splits = train_config.get_dataset_splits()

        _val_dataset_split_count = len(_dataset_splits.val[train_config.subject_column].unique()) + len(
            _dataset_splits.train[train_config.subject_column].unique())
        _aggregates_csv = pd.read_csv(aggregate_metrics_path)
        _counts_for_splits = list(_aggregates_csv[LoggingColumns.SubjectCount.value])
        assert all([x == _val_dataset_split_count for x in _counts_for_splits])
        _epochs = list(_aggregates_csv[LoggingColumns.Epoch.value])
        # Each epoch is recorded twice once for the training split and once for the validation
        # split
        assert len(_epochs) == train_config.num_epochs * 2
        assert _epochs == list(range(train_config.num_epochs)) * 2
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
                                LoggingColumns.ExplainedVariance.value}
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
    model.local_dataset = full_ml_test_data_path("classification_data_generated_random")
    model.number_of_cross_validation_splits = 2
    dataset_splits = model.get_dataset_splits()
    assert list(dataset_splits[ModelExecutionMode.TRAIN].subjectID.unique()) == ['4', '5', '2', '10']
    assert list(dataset_splits[ModelExecutionMode.VAL].subjectID.unique()) == ['1', '6', '7', '8']
    assert list(dataset_splits[ModelExecutionMode.TEST].subjectID.unique()) == ['3', '9']
