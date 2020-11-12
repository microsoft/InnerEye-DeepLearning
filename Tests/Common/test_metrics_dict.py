#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from io import StringIO
from statistics import mean
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, roc_curve

from InnerEye.Common.common_util import DataframeLogger
from InnerEye.Common.metrics_dict import Hue, MetricType, MetricsDict, PredictionEntry, ScalarMetricsDict, \
    SequenceMetricsDict, average_metric_values
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML import metrics
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import BACKGROUND_CLASS_NAME
from InnerEye.ML.utils.io_util import tabulate_dataframe
from InnerEye.ML.utils.metrics_constants import LoggingColumns


def test_average_metric_values() -> None:
    """
    Test averages are computed correctly.
    """
    assert np.isnan(average_metric_values([], skip_nan_when_averaging=True))
    assert np.isnan(average_metric_values([], skip_nan_when_averaging=False))
    valid = [1, 2, 3.14]
    m1 = average_metric_values(valid, skip_nan_when_averaging=True)
    assert m1 == mean(valid)
    assert average_metric_values(valid, skip_nan_when_averaging=True) == m1
    with_nan = [1, np.nan]
    assert average_metric_values(with_nan, skip_nan_when_averaging=True) == 1
    assert np.isnan(average_metric_values(with_nan, skip_nan_when_averaging=False))


def test_metrics_dict1() -> None:
    """
    Test insertion of scalar values into a MetricsDict.
    """
    m = MetricsDict()
    assert m.get_hue_names() == [MetricsDict.DEFAULT_HUE_KEY]
    name = "foo"
    v1 = 2.7
    v2 = 3.14
    m.add_metric(name, v1)
    m.add_metric(name, v2)
    assert m.values()[name] == [v1, v2]
    with pytest.raises(ValueError) as ex:
        # noinspection PyTypeChecker
        m.add_metric(name, [1.0])  # type: ignore
    assert "Expected the metric to be a scalar" in str(ex)
    assert m.skip_nan_when_averaging[name] is False
    v3 = 3.0
    name2 = "bar"
    m.add_metric(name2, v3, skip_nan_when_averaging=True)
    assert m.skip_nan_when_averaging[name2] is True
    # Expected average: Metric "foo" averages over two values v1 and v2. For "bar", we only inserted one value anyhow
    average = m.average()
    mean_v1_v2 = mean([v1, v2])
    assert average.values() == {name: [mean_v1_v2], name2: [v3]}
    num_entries = m.num_entries()
    assert num_entries == {name: 2, name2: 1}


@pytest.mark.parametrize("hues", [None, ["A", "B"]])
def test_metrics_dict_flatten(hues: Optional[List[str]]) -> None:
    m = MetricsDict(hues=hues)
    _hues = hues or [MetricsDict.DEFAULT_HUE_KEY] * 2
    m.add_metric("foo", 1.0, hue=_hues[0])
    m.add_metric("foo", 2.0, hue=_hues[1])
    m.add_metric("bar", 3.0, hue=_hues[0])
    m.add_metric("bar", 4.0, hue=_hues[1])

    if hues is None:
        average = m.average(across_hues=True)
        # We should be able to flatten out all the singleton values that the `average` operation returns
        all_values = list(average.enumerate_single_values())
        assert all_values == [(MetricsDict.DEFAULT_HUE_KEY, "foo", 1.5), (MetricsDict.DEFAULT_HUE_KEY, "bar", 3.5)]
        # When trying to flatten off a dictionary that has two values, this should fail:
        with pytest.raises(ValueError) as ex:
            list(m.enumerate_single_values())
        assert "only hold 1 item" in str(ex)
    else:
        average = m.average(across_hues=False)
        all_values = list(average.enumerate_single_values())
        assert all_values == [('A', 'foo', 1.0), ('A', 'bar', 3.0), ('B', 'foo', 2.0), ('B', 'bar', 4.0)]


def test_metrics_dict_average_metrics_averaging() -> None:
    """
    Test if averaging metrics avoid NaN as expected.
    """
    m = MetricsDict()
    metric1 = "foo"
    v1 = 1.0
    m.add_metric(metric1, v1)
    m.add_metric(metric1, np.nan, skip_nan_when_averaging=True)
    metric2 = "bar"
    v2 = 2.0
    m.add_metric(metric2, v2)
    m.add_metric(metric2, np.nan, skip_nan_when_averaging=False)
    average = m.average()
    assert average.values()[metric1] == [v1]
    assert np.isnan(average.values()[metric2])


def test_metrics_dict_roc() -> None:
    """
    Test if adding ROC entries to a MetricsDict instance works, and returns the correct AUC.
    """
    # Prepare a vector of predictions and labels. We can compute AUC off those to compare.
    # MetricsDict will get that supplied in 3 chunks, and should return the same AUC value.
    predictions = np.array([0.5, 0.6, 0.1, 0.8, 0.2, 0.9])
    labels = np.array([0, 1.0, 0, 0, 1, 1], dtype=np.float)
    split_length = [3, 2, 1]
    assert sum(split_length) == len(predictions)
    summed = np.cumsum(split_length)
    m = MetricsDict()
    for i, end in enumerate(summed):
        start = 0 if i == 0 else summed[i - 1]
        pred = predictions[start:end]
        label = labels[start:end]
        subject_ids = list(map(str, range(len(pred))))
        m.add_predictions(subject_ids, pred, label)
    assert m.has_prediction_entries
    actual_auc = m.get_roc_auc()
    expected_auc = roc_auc_score(labels, predictions)
    assert actual_auc == pytest.approx(expected_auc, 1e-6)
    actual_pr_auc = m.get_pr_auc()
    expected_pr_auc = 0.7111111
    assert actual_pr_auc == pytest.approx(expected_pr_auc, 1e-6)


def test_metrics_dict_roc_degenerate() -> None:
    """
    Test if adding ROC entries to a MetricsDict instance works, if there is only 1 class present.
    """
    # Prepare a vector of predictions and labels. We can compute AUC off those to compare.
    # MetricsDict will get that supplied in 3 chunks, and should return the same AUC value.
    predictions = np.array([0.5, 0.6, 0.1, 0.8, 0.2, 0.9])
    m = MetricsDict()
    subject_ids = list(map(str, range(len(predictions))))
    m.add_predictions(subject_ids, predictions, np.ones_like(predictions))
    assert m.has_prediction_entries
    assert m.get_roc_auc() == 1.0
    assert m.get_pr_auc() == 1.0


def test_metrics_dict_add_integer() -> None:
    """
    Adding a scalar metric where the value is an integer by accident should still store the metric.
    """
    m = MetricsDict()
    m.add_metric("foo", 1)
    assert "foo" in m.values()
    assert m.values()["foo"] == [1.0]


def test_delete_metric() -> None:
    """
    Deleting a set of metrics from the dictionary.
    """
    m = MetricsDict()
    m.add_metric(MetricType.LOSS, 1)
    assert m.values()[MetricType.LOSS.value] == [1.0]
    m.delete_metric(MetricType.LOSS)
    assert MetricType.LOSS.value not in m.values()


def test_load_metrics_from_df() -> None:
    expected_epochs = [1] * 2 + [2] * 2
    expected_modes = [ModelExecutionMode.VAL, ModelExecutionMode.TEST] * 2
    expected_labels = [1] * 4
    expected_subjects = ["A"] * 4

    test_df = pd.DataFrame.from_dict({
        LoggingColumns.Epoch.value: expected_epochs,
        LoggingColumns.DataSplit.value: [x.value for x in expected_modes],
        LoggingColumns.ModelOutput.value: [0.1, 0.2, 0.3, 0.4],
        LoggingColumns.Label.value: expected_labels,
        LoggingColumns.Patient.value: expected_subjects
    })
    metrics = ScalarMetricsDict.load_execution_mode_metrics_from_df(test_df, is_classification_metrics=True)
    for x in set(expected_modes):
        for e in set(expected_epochs):
            expected_df = test_df[
                (test_df[LoggingColumns.DataSplit.value] == x.value) & (test_df[LoggingColumns.Epoch.value] == e)]
            metrics_dict = metrics[x][e]
            assert np.alltrue(expected_df[LoggingColumns.ModelOutput.value].values == metrics_dict.get_predictions())
            assert np.alltrue(expected_df[LoggingColumns.Label.value].values == metrics_dict.get_labels())
            assert np.alltrue(expected_df[LoggingColumns.Patient.value].values == metrics_dict.subject_ids())


def test_load_metrics_from_df_with_hue() -> None:
    """
    Test loading of per-epoch predictions from a dataframe when the dataframe contains a prediction_target column.
    """
    hue_name = "foo"
    hues = [MetricsDict.DEFAULT_HUE_KEY] * 2 + [hue_name] * 2
    expected_epoch = 1
    expected_mode = ModelExecutionMode.VAL
    expected_labels = [1]
    expected_subjects = ["A"]
    model_outputs_1 = [0.1, 0.2]
    model_outputs_2 = [0.3, 0.4]
    test_df = pd.DataFrame.from_dict({
        LoggingColumns.Hue.value: hues,
        LoggingColumns.Epoch.value: [expected_epoch] * 4,
        LoggingColumns.DataSplit.value: [expected_mode.value] * 4,
        LoggingColumns.ModelOutput.value: model_outputs_1 + model_outputs_2,
        LoggingColumns.Label.value: expected_labels * 4,
        LoggingColumns.Patient.value: expected_subjects * 4
    })
    metrics = ScalarMetricsDict.load_execution_mode_metrics_from_df(test_df, is_classification_metrics=True)
    assert expected_mode in metrics
    assert expected_epoch in metrics[expected_mode]
    metrics_dict = metrics[expected_mode][expected_epoch]
    assert metrics_dict.get_hue_names(include_default=False) == [hue_name]
    assert metrics_dict.get_predictions().flatten().tolist() == model_outputs_1
    assert metrics_dict.get_predictions(hue=hue_name).flatten().tolist() == model_outputs_2


def test_metrics_dict_average_additional_metrics() -> None:
    """
    Test if computing the ROC entries and metrics at optimal threshold with MetricsDict.average() works
    as expected and returns the correct values.
    """
    # Prepare a vector of predictions and labels.
    predictions = np.array([0.5, 0.6, 0.1, 0.8, 0.2, 0.9])
    labels = np.array([0, 1.0, 0, 0, 1, 1], dtype=np.float)
    split_length = [3, 2, 1]

    # Get MetricsDict
    assert sum(split_length) == len(predictions)
    summed = np.cumsum(split_length)
    # MetricsDict will get that supplied in 3 chunks.
    m = MetricsDict()
    for i, end in enumerate(summed):
        start = 0 if i == 0 else summed[i - 1]
        pred = predictions[start:end]
        label = labels[start:end]
        subject_ids = list(map(str, range(len(pred))))
        m.add_predictions(subject_ids, pred, label)
    assert m.has_prediction_entries

    # Compute average MetricsDict
    averaged = m.average()

    # Compute additional expected metrics for the averaged MetricsDict
    expected_auc = roc_auc_score(labels, predictions)
    expected_fpr, expected_tpr, thresholds = roc_curve(labels, predictions)
    expected_optimal_idx = np.argmax(expected_tpr - expected_fpr)
    expected_optimal_threshold = float(thresholds[expected_optimal_idx])
    expected_accuracy = np.mean((predictions > expected_optimal_threshold) == labels)

    # Check computed values against expected
    assert averaged.values()[MetricType.OPTIMAL_THRESHOLD.value][0] == pytest.approx(expected_optimal_threshold)
    assert averaged.values()[MetricType.ACCURACY_AT_OPTIMAL_THRESHOLD.value][0] == pytest.approx(expected_accuracy)
    assert averaged.values()[MetricType.FALSE_POSITIVE_RATE_AT_OPTIMAL_THRESHOLD.value][0] == \
           pytest.approx(expected_fpr[expected_optimal_idx])
    assert averaged.values()[MetricType.FALSE_NEGATIVE_RATE_AT_OPTIMAL_THRESHOLD.value][0] == \
           pytest.approx(1 - expected_tpr[expected_optimal_idx])
    assert averaged.values()[MetricType.AREA_UNDER_ROC_CURVE.value][0] == pytest.approx(expected_auc, 1e-6)


def test_metrics_dict_get_hues() -> None:
    """
    Test to make sure metrics dict is configured properly with/without hues
    """
    m = MetricsDict()
    assert m.get_hue_names() == [MetricsDict.DEFAULT_HUE_KEY]
    assert m.get_hue_names(include_default=False) == []
    _hues = ["A", "B", "C"]
    m = MetricsDict(hues=_hues)
    assert m.get_hue_names() == _hues + [MetricsDict.DEFAULT_HUE_KEY]
    assert m.get_hue_names(include_default=False) == _hues


def test_metrics_store_mixed_hues() -> None:
    """
    Test to make sure metrics dict is able to handle default and non-default hues
    """
    m = MetricsDict(hues=["A", "B"])
    m.add_metric("foo", 1)
    m.add_metric("foo", 1, hue="B")
    m.add_metric("bar", 2, hue="A")
    assert list(m.enumerate_single_values()) == \
           [('A', 'bar', 2), ('B', 'foo', 1), (MetricsDict.DEFAULT_HUE_KEY, 'foo', 1)]


def test_metrics_dict_to_string() -> None:
    """
    Test to make sure metrics dict is able to be stringified correctly
    """
    m = MetricsDict()
    m.add_metric("foo", 1.0)
    m.add_metric("bar", math.pi)
    info_df = pd.DataFrame(columns=MetricsDict.DATAFRAME_COLUMNS)
    info_df = info_df.append({MetricsDict.DATAFRAME_COLUMNS[0]: MetricsDict.DEFAULT_HUE_KEY,
                              MetricsDict.DATAFRAME_COLUMNS[1]: "foo: 1.0000, bar: 3.1416"}, ignore_index=True)
    assert m.to_string() == tabulate_dataframe(info_df)
    assert m.to_string(tabulate=False) == info_df.to_string(index=False)


def test_metrics_dict_to_string_with_hues() -> None:
    """
    Test to make sure metrics dict is able to be stringified correctly with hues
    """
    m = MetricsDict(hues=["G1"])
    m.add_metric("foo", 1.0)
    m.add_metric("bar", math.pi, hue="G1")
    m.add_metric("baz", 2.0, hue="G1")
    info_df = pd.DataFrame(columns=MetricsDict.DATAFRAME_COLUMNS)
    info_df = info_df.append({MetricsDict.DATAFRAME_COLUMNS[0]: "G1",
                              MetricsDict.DATAFRAME_COLUMNS[1]: "bar: 3.1416, baz: 2.0000"}, ignore_index=True)
    info_df = info_df.append({MetricsDict.DATAFRAME_COLUMNS[0]: MetricsDict.DEFAULT_HUE_KEY,
                              MetricsDict.DATAFRAME_COLUMNS[1]: "foo: 1.0000"}, ignore_index=True)
    assert m.to_string() == tabulate_dataframe(info_df)
    assert m.to_string(tabulate=False) == info_df.to_string(index=False)


def test_classification_metrics_avg() -> None:
    hue1 = "H1"
    hue2 = "H2"
    m = MetricsDict(hues=[hue1, hue2], is_classification_metrics=True)
    m.add_metric("foo", 1.0)
    m.add_metric("foo", 2.0)
    # Perfect predictions for hue1, should give AUC == 1.0
    m.add_predictions(["S1", "S2"], np.array([0.0, 1.0]), np.array([0.0, 1.0]), hue=hue1)
    expected_hue1_auc = 1.0
    # Worst possible predictions for hue2, should give AUC == 0.0
    m.add_predictions(["S1", "S2"], np.array([1.0, 0.0]), np.array([0.0, 1.0]), hue=hue2)
    expected_hue2_auc = 0.0
    averaged = m.average(across_hues=False)
    g1_averaged = averaged.values(hue=hue1)
    assert MetricType.AREA_UNDER_ROC_CURVE.value in g1_averaged
    assert g1_averaged[MetricType.AREA_UNDER_ROC_CURVE.value] == [expected_hue1_auc]
    assert MetricType.AREA_UNDER_PR_CURVE.value in g1_averaged
    assert MetricType.SUBJECT_COUNT.value in g1_averaged
    assert g1_averaged[MetricType.SUBJECT_COUNT.value] == [2.0]
    default_averaged = averaged.values()
    assert default_averaged == {"foo": [1.5]}
    can_enumerate = list(averaged.enumerate_single_values())
    assert len(can_enumerate) >= 8
    assert can_enumerate[0] == (hue1, MetricType.AREA_UNDER_ROC_CURVE.value, 1.0)
    assert can_enumerate[-1] == (MetricsDict.DEFAULT_HUE_KEY, "foo", 1.5)

    g2_averaged = averaged.values(hue=hue2)
    assert MetricType.AREA_UNDER_ROC_CURVE.value in g2_averaged
    assert g2_averaged[MetricType.AREA_UNDER_ROC_CURVE.value] == [expected_hue2_auc]

    averaged_across_hues = m.average(across_hues=True)
    assert averaged_across_hues.get_hue_names() == [MetricsDict.DEFAULT_HUE_KEY]
    assert MetricType.AREA_UNDER_ROC_CURVE.value in averaged_across_hues.values()
    expected_averaged_auc = 0.5 * (expected_hue1_auc + expected_hue2_auc)
    assert averaged_across_hues.values()[MetricType.AREA_UNDER_ROC_CURVE.value] == [expected_averaged_auc]


def test_metrics_dict_per_subject() -> None:
    """
    Ensure that adding per-subject predictions can correctly handle subject IDs
    """
    hue1 = "H1"
    hue2 = "H2"
    m = ScalarMetricsDict(hues=[hue1, hue2], is_classification_metrics=True)
    m.add_predictions(["S1", "S2"], np.array([0.0, 1.0]), np.array([0.0, 1.0]), hue=hue1)
    m.add_predictions(["S1", "S2"], np.array([1.0, 0.0]), np.array([0.0, 1.0]), hue=hue2)
    predictions = m.get_predictions_and_labels_per_subject(hue=hue1)
    assert len(predictions) == 2


def test_metrics_dic_subject_ids() -> None:
    hue1 = "H1"
    m = ScalarMetricsDict(hues=[hue1], is_classification_metrics=True)
    m.add_predictions(subject_ids=['0'], predictions=np.zeros(1), labels=np.zeros(1), hue=hue1)
    assert m.subject_ids() == []
    assert m.subject_ids(hue=hue1) == ['0']


def test_hue_entries() -> None:
    hue = Hue(name="foo")
    assert not hue.has_prediction_entries
    assert len(hue.get_predictions()) == 0
    assert len(hue.get_labels()) == 0
    assert len(hue.get_predictions_and_labels_per_subject()) == 0
    hue.add_predictions(["S1", "S2"], np.ones(2), np.ones(2))
    hue.add_predictions(["S3"], np.zeros(1), np.zeros(1))
    assert hue.has_prediction_entries
    assert len(hue.subject_ids) == 3
    assert len(hue.labels) == 2
    assert len(hue.labels[0]) == 2
    assert len(hue.predictions) == 2
    assert len(hue.predictions[0]) == 2

    assert len(hue.get_predictions()) == 3
    assert len(hue.get_labels()) == 3
    entries = hue.get_predictions_and_labels_per_subject()
    assert entries == [
        PredictionEntry("S1", 1.0, 1.0),
        PredictionEntry("S2", 1.0, 1.0),
        PredictionEntry("S3", 0.0, 0.0),
    ]


def test_load_metrics_from_df_with_hues(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if we can re-create a MetricsDict object with model predictions and labels, when the data file contains
    a prediction target value.
    """
    df_str = """prediction_target,epoch,subject,model_output,label,cross_validation_split_index,data_split
01,1,2137.00005,0.54349,1.0,0,Val
01,1,2137.00125,0.54324,0.0,1,Val
01,1,3250.00005,0.50822,0.0,0,Val
01,1,3250.12345,0.47584,0.0,1,Val
02,1,2137.00005,0.55538,1.0,0,Val
02,1,2137.00125,0.55759,0.0,1,Val
02,1,3250.00005,0.47255,0.0,0,Val
02,1,3250.12345,0.46996,0.0,1,Val
03,1,2137.00005,0.56670,1.0,0,Val
03,1,2137.00125,0.57003,0.0,1,Val
03,1,3250.00005,0.46321,0.0,0,Val
03,1,3250.12345,0.47309,0.0,1,Val
"""
    df = pd.read_csv(StringIO(df_str), converters={LoggingColumns.Hue.value: lambda x: x})
    metrics = ScalarMetricsDict.load_execution_mode_metrics_from_df(df, is_classification_metrics=True)
    mode = ModelExecutionMode.VAL
    epoch = 1
    assert mode in metrics
    assert epoch in metrics[mode]
    metrics_dict = metrics[mode][epoch]
    expected_hues = ["01", "02", "03"]
    assert metrics_dict.get_hue_names(include_default=False) == expected_hues
    for hue in expected_hues:
        assert len(metrics_dict._get_hue(hue).get_predictions()) == 4
    logger_output_file = test_output_dirs.create_file_or_folder_path("output.csv")
    logger = DataframeLogger(csv_path=logger_output_file)
    ScalarMetricsDict.aggregate_and_save_execution_mode_metrics(metrics, logger)
    output = pd.read_csv(logger_output_file, dtype=str)
    assert LoggingColumns.Hue.value in output
    assert list(output[LoggingColumns.Hue.value]) == expected_hues
    assert LoggingColumns.DataSplit.value in output
    assert list(output[LoggingColumns.DataSplit.value].unique()) == [ModelExecutionMode.VAL.value]
    assert LoggingColumns.Epoch.value in output
    assert list(output[LoggingColumns.Epoch.value].unique()) == ["1"]
    assert LoggingColumns.AreaUnderPRCurve.value in output
    assert list(output[LoggingColumns.AreaUnderPRCurve.value]) == ['1.00000', '0.25000', '0.25000']


def test_get_hue_name_from_target_index() -> None:
    """
    Tests if we can create metrics hue names from sequence indices, and get them back from the string.
    """
    index = 7
    hue_name = SequenceMetricsDict.get_hue_name_from_target_index(index)
    assert hue_name == "Seq_pos 07"
    assert SequenceMetricsDict.get_target_index_from_hue_name(hue_name) == index
    with pytest.raises(ValueError):
        SequenceMetricsDict.get_target_index_from_hue_name("foo 07")
    with pytest.raises(ValueError):
        SequenceMetricsDict.get_target_index_from_hue_name("Seq_pos ab")


def test_metrics_dict_with_default_hue() -> None:
    hue_name = "foo"
    metrics_dict = MetricsDict(hues=[hue_name, MetricsDict.DEFAULT_HUE_KEY])
    assert metrics_dict.get_hue_names(include_default=True) == [hue_name, MetricsDict.DEFAULT_HUE_KEY]
    assert metrics_dict.get_hue_names(include_default=False) == [hue_name]


def test_diagnostics() -> None:
    """
    Test if we can store diagnostic values (no restrictions on data types) in the metrics dict.
    """
    name = "foo"
    value1 = "something"
    value2 = (1, 2, 3)
    m = MetricsDict()
    m.add_diagnostics(name, value1)
    m.add_diagnostics(name, value2)
    assert m.diagnostics == {name: [value1, value2]}


def test_delete_hue() -> None:
    h1 = "a"
    h2 = "b"
    a = MetricsDict(hues=[h1, h2])
    a.add_metric("foo", 1.0, hue=h1)
    a.add_metric("bar", 2.0, hue=h2)
    a.delete_hue(h1)
    assert a.get_hue_names(include_default=False) == [h2]
    assert list(a.enumerate_single_values()) == [(h2, "bar", 2.0)]


def test_get_single_metric() -> None:
    h1 = "a"
    m = MetricsDict(hues=[h1])
    m1, v1 = ("foo", 1.0)
    m2, v2 = (MetricType.LOSS, 2.0)
    m.add_metric(m1, v1, hue=h1)
    m.add_metric(m2, v2)
    assert m.get_single_metric(m1, h1) == v1
    assert m.get_single_metric(m2) == v2
    with pytest.raises(KeyError) as ex1:
        m.get_single_metric(m1, "no such hue")
    assert "no such hue" in str(ex1)
    with pytest.raises(KeyError) as ex2:
        m.get_single_metric("no such metric", h1)
    assert "no such metric" in str(ex2)
    m.add_metric(m2, v2)
    with pytest.raises(ValueError) as ex3:
        m.get_single_metric(m2)
    assert "Expected a single entry" in str(ex3)


def test_aggregate_segmentation_metrics() -> None:
    """
    Test how per-epoch segmentation metrics are aggregated to computed foreground dice and voxel count proportions.
    """
    g1 = "Liver"
    g2 = "Lung"
    ground_truth_ids = [BACKGROUND_CLASS_NAME, g1, g2]
    dice = [0.85, 0.75, 0.55]
    voxels_proportion = [0.85, 0.10, 0.05]
    loss = 3.14
    other_metric = 2.71
    m = MetricsDict(hues=ground_truth_ids)
    voxel_count = 200
    # Add 3 values per metric, but such that the averages are back at the value given in dice[i]
    for i in range(3):
        delta = (i - 1) * 0.05
        for j, ground_truth_id in enumerate(ground_truth_ids):
            m.add_metric(MetricType.DICE, dice[j] + delta, hue=ground_truth_id)
            m.add_metric(MetricType.VOXEL_COUNT, int(voxels_proportion[j] * voxel_count), hue=ground_truth_id)
        m.add_metric(MetricType.LOSS, loss + delta)
        m.add_metric("foo", other_metric)
    m.add_diagnostics("foo", "bar")
    aggregate = metrics.aggregate_segmentation_metrics(m)
    assert aggregate.diagnostics == m.diagnostics
    enumerated = list((g, s, v) for g, s, v in aggregate.enumerate_single_values())
    expected = [
        # Dice and voxel count per foreground structure should be retained during averaging
        (g1, MetricType.DICE.value, dice[1]),
        (g1, MetricType.VOXEL_COUNT.value, voxels_proportion[1] * voxel_count),
        # Proportion of foreground voxels is computed during averaging
        (g1, MetricType.PROPORTION_FOREGROUND_VOXELS.value, voxels_proportion[1]),
        (g2, MetricType.DICE.value, dice[2]),
        (g2, MetricType.VOXEL_COUNT.value, voxels_proportion[2] * voxel_count),
        (g2, MetricType.PROPORTION_FOREGROUND_VOXELS.value, voxels_proportion[2]),
        # Loss is present in the default metrics group, and should be retained.
        (MetricsDict.DEFAULT_HUE_KEY, MetricType.LOSS.value, loss),
        (MetricsDict.DEFAULT_HUE_KEY, "foo", other_metric),
        # Dice averaged across the foreground structures is added during the function call, as is proportion of voxels
        (MetricsDict.DEFAULT_HUE_KEY, MetricType.DICE.value, 0.5 * (dice[1] + dice[2])),
        (MetricsDict.DEFAULT_HUE_KEY, MetricType.PROPORTION_FOREGROUND_VOXELS.value,
         voxels_proportion[1] + voxels_proportion[2]),
    ]
    assert len(enumerated) == len(expected)
    # Numbers won't match up precisely because of rounding during averaging
    for (actual, e) in zip(enumerated, expected):
        assert actual[0:2] == e[0:2]
        assert actual[2] == pytest.approx(e[2])


def test_add_foreground_dice() -> None:
    g1 = "Liver"
    g2 = "Lung"
    ground_truth_ids = [BACKGROUND_CLASS_NAME, g1, g2]
    dice = [0.85, 0.75, 0.55]
    m = MetricsDict(hues=ground_truth_ids)
    for j, ground_truth_id in enumerate(ground_truth_ids):
        m.add_metric(MetricType.DICE, dice[j], hue=ground_truth_id)
    metrics.add_average_foreground_dice(m)
    assert m.get_single_metric(MetricType.DICE) == 0.5 * (dice[1] + dice[2])
