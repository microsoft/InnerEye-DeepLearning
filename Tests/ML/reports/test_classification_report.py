#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import shutil
import pytest
import math

from pathlib import Path

from InnerEye.Common.output_directories import TestOutputDirectories
from InnerEye.ML.reports.notebook_report import generate_classification_notebook
from InnerEye.ML.reports.classification_report import ReportedMetrics, get_results, get_metric, \
    get_k_best_and_worst_performing, get_correct_and_misclassified_examples
from InnerEye.ML.utils.metrics_constants import LoggingColumns


def test_generate_classification_report(test_output_dirs: TestOutputDirectories) -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"
    val_metrics_file = reports_folder / "val_metrics_classification.csv"
    current_dir = Path(test_output_dirs.make_sub_dir("test_classification_report"))
    result_file = current_dir / "report.ipynb"
    result_html = generate_classification_notebook(result_notebook=result_file,
                                                   val_metrics=val_metrics_file,
                                                   test_metrics=test_metrics_file)
    assert result_file.is_file()
    assert result_html.is_file()
    assert result_html.suffix == ".html"


def test_get_results() -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"

    results = get_results(test_metrics_file)
    assert all([results.subject_ids[i] == i for i in range(12)])
    assert all([results.labels[i] == label for i, label in enumerate([1]*6 + [0]*6)])
    assert all([results.model_outputs[i] == op for i, op in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0] * 2)])


def test_functions_with_invalid_csv() -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"
    val_metrics_file = reports_folder / "val_metrics_classification.csv"
    invalid_metrics_file = reports_folder / "invalid_metrics_classification.csv"
    shutil.copyfile(test_metrics_file, invalid_metrics_file)
    # Duplicate a subject
    with open(invalid_metrics_file, "a") as file:
        file.write("Default,1,5,1.0,1,-1,Test")

    with pytest.raises(ValueError):
        get_results(invalid_metrics_file)

    with pytest.raises(ValueError):
        get_correct_and_misclassified_examples(invalid_metrics_file, test_metrics_file)

    with pytest.raises(ValueError):
        get_correct_and_misclassified_examples(val_metrics_file, invalid_metrics_file)


def test_get_metric() -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"
    val_metrics_file = reports_folder / "val_metrics_classification.csv"

    optimal_threshold = get_metric(test_metrics_csv=test_metrics_file,
                                   val_metrics_csv=val_metrics_file,
                                   metric=ReportedMetrics.OptimalThreshold)

    assert optimal_threshold == 0.6

    auc_roc = get_metric(test_metrics_csv=test_metrics_file,
                         val_metrics_csv=val_metrics_file,
                         metric=ReportedMetrics.AUC_ROC)
    assert auc_roc == 0.5

    auc_pr = get_metric(test_metrics_csv=test_metrics_file,
                        val_metrics_csv=val_metrics_file,
                        metric=ReportedMetrics.AUC_PR)

    assert math.isclose(auc_pr, 13/24, abs_tol=1e-15)

    accuracy = get_metric(test_metrics_csv=test_metrics_file,
                          val_metrics_csv=val_metrics_file,
                          metric=ReportedMetrics.Accuracy)

    assert accuracy == 0.5

    fpr = get_metric(test_metrics_csv=test_metrics_file,
                     val_metrics_csv=val_metrics_file,
                     metric=ReportedMetrics.FalsePositiveRate)

    assert fpr == 0.5

    fnr = get_metric(test_metrics_csv=test_metrics_file,
                     val_metrics_csv=val_metrics_file,
                     metric=ReportedMetrics.FalseNegativeRate)

    assert fnr == 0.5


def test_get_correct_and_misclassified_examples() -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"
    val_metrics_file = reports_folder / "val_metrics_classification.csv"

    results = get_correct_and_misclassified_examples(val_metrics_csv=val_metrics_file,
                                                     test_metrics_csv=test_metrics_file)

    true_positives = [item[LoggingColumns.Patient.value] for _, item in results.true_positives.iterrows()]
    assert all([i in true_positives for i in [3, 4, 5]])

    true_negatives = [item[LoggingColumns.Patient.value] for _, item in results.true_negatives.iterrows()]
    assert all([i in true_negatives for i in [6, 7, 8]])

    false_positives = [item[LoggingColumns.Patient.value] for _, item in results.false_positives.iterrows()]
    assert all([i in false_positives for i in [9, 10, 11]])

    false_negatives = [item[LoggingColumns.Patient.value] for _, item in results.false_negatives.iterrows()]
    assert all([i in false_negatives for i in [0, 1, 2]])


def test_get_k_best_and_worst_performing() -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"
    val_metrics_file = reports_folder / "val_metrics_classification.csv"

    results = get_k_best_and_worst_performing(val_metrics_csv=val_metrics_file,
                                              test_metrics_csv=test_metrics_file,
                                              k=2)

    best_true_positives = [item[LoggingColumns.Patient.value] for _, item in results.true_positives.iterrows()]
    assert best_true_positives == [5, 4]

    best_true_negatives = [item[LoggingColumns.Patient.value] for _, item in results.true_negatives.iterrows()]
    assert best_true_negatives == [6, 7]

    worst_false_positives = [item[LoggingColumns.Patient.value] for _, item in results.false_positives.iterrows()]
    assert worst_false_positives == [11, 10]

    worst_false_negatives = [item[LoggingColumns.Patient.value] for _, item in results.false_negatives.iterrows()]
    assert worst_false_negatives == [0, 1]
