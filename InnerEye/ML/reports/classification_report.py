#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from dataclasses import dataclass
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score
from enum import Enum
from pathlib import Path

from InnerEye.ML.reports.notebook_report import print_header
from InnerEye.Common.metrics_dict import MetricsDict
from InnerEye.ML.utils.metrics_constants import LoggingColumns
from InnerEye.ML.pipelines.scalar_inference import ScalarInferencePipelineBase


@dataclass
class Results:
    true_positives: pd.DataFrame
    false_positives: pd.DataFrame
    true_negatives: pd.DataFrame
    false_negatives: pd.DataFrame


class ReportedMetrics(Enum):
    OptimalThreshold = "optimal_threshold"
    AUC_PR = "auc_pr"
    AUC_ROC = "auc_roc"


def get_results(csv: Path) -> ScalarInferencePipelineBase.Result:
    df = pd.read_csv(csv)
    labels = df[LoggingColumns.Label.value]
    model_outputs = df[LoggingColumns.ModelOutput.value]
    subjects = df[LoggingColumns.Patient.value]
    return ScalarInferencePipelineBase.Result(subject_ids=subjects, labels=labels, model_outputs=model_outputs)


def plot_auc(x_values: np.ndarray, y_values: np.ndarray, title: str, ax: Axes):
    ax.plot(x_values, y_values)
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)
    ax.set_title(title)

    # write values of points
    for x, y in zip (x_values, y_values):
        ax.annotate(f"{x:0.3f}, {y:0.3f}", xy = (x, y), xytext=(15, 0), textcoords='offset points')


def plot_pr_and_roc_curves_from_csv(metrics_csv: Path):
    print_header("ROC and PR curves", level=3)
    results = get_results(metrics_csv)

    _, ax = plt.subplots(1, 2)

    fpr, tpr, thresholds = roc_curve(results.labels, results.model_outputs)
    plot_auc(fpr, tpr, "ROC Curve", ax[0])
    precision, recall, thresholds = precision_recall_curve(results.labels, results.model_outputs)
    plot_auc(precision, recall, "PR Curve", ax[1])

    plt.show()


def get_metric(val_metrics_csv: Path, test_metrics_csv: Path, metric: ReportedMetrics):
    results_val = get_results(val_metrics_csv)
    fpr, tpr, thresholds = roc_curve(results_val.labels, results_val.model_outputs)
    optimal_idx = MetricsDict.get_optimal_idx(fpr=fpr, tpr=tpr)
    optimal_threshold = thresholds[optimal_idx]

    if metric is ReportedMetrics.OptimalThreshold:
        return optimal_threshold

    results_test = get_results(test_metrics_csv)

    if metric is ReportedMetrics.AUC_ROC:
        return roc_auc_score(results_test.labels, results_test.model_outputs)
    elif metric is ReportedMetrics.AUC_PR:
        precision, recall, _ = precision_recall_curve(results_test.labels, results_test.model_outputs)
        return auc(recall, precision)
    else:
        raise ValueError("Unknown metric")


def print_metrics(val_metrics_csv: Path, test_metrics_csv: Path):
    print_header("Test metrics", level=3)
    optimal_threshold = get_metric(val_metrics_csv=val_metrics_csv,
                                   test_metrics_csv=test_metrics_csv,
                                   metric=ReportedMetrics.OptimalThreshold)

    print(f"Optimal threshold: {optimal_threshold}")

    roc_auc = get_metric(val_metrics_csv=val_metrics_csv,
                                   test_metrics_csv=test_metrics_csv,
                                   metric=ReportedMetrics.AUC_ROC)
    print(f"AUC PR: {roc_auc}")

    pr_auc = get_metric(val_metrics_csv=val_metrics_csv,
                                   test_metrics_csv=test_metrics_csv,
                                   metric=ReportedMetrics.AUC_PR)
    print(f"AUC PR: {pr_auc}")


def get_correct_and_misclassified_examples(val_metrics_csv: Path, test_metrics_csv: Path):
    df_val = pd.read_csv(val_metrics_csv)
    fpr, tpr, thresholds = roc_curve(df_val[LoggingColumns.Label.value], df_val[LoggingColumns.ModelOutput.value])
    optimal_idx = MetricsDict.get_optimal_idx(fpr=fpr, tpr=tpr)
    optimal_threshold = thresholds[optimal_idx]

    df_test = pd.read_csv(test_metrics_csv)
    df_test["predicted"] = df_test.apply(lambda x: int(x[LoggingColumns.ModelOutput.value] >= optimal_threshold), axis=1)

    true_positives = df_test[(df_test["predicted"] == 1) & (df_test[LoggingColumns.Label.value] == 1)]
    false_positives = df_test[(df_test["predicted"] == 1) & (df_test[LoggingColumns.Label.value] == 0)]
    false_negatives = df_test[(df_test["predicted"] == 0) & (df_test[LoggingColumns.Label.value] == 1)]
    true_negatives = df_test[(df_test["predicted"] == 0) & (df_test[LoggingColumns.Label.value] == 0)]

    return Results(true_positives=true_positives,
                   true_negatives=true_negatives,
                   false_positives=false_positives,
                   false_negatives=false_negatives)


def get_k_best_and_worst_performing(val_metrics_csv: Path, test_metrics_csv: Path, k: int):
    results = get_correct_and_misclassified_examples(val_metrics_csv=val_metrics_csv,
                                                     test_metrics_csv=test_metrics_csv)

    # sort by model_output
    sorted = Results(true_positives=
                     results.true_positives.sort_values(by=LoggingColumns.ModelOutput.value, ascending=False).head(k),
                     true_negatives=
                     results.true_negatives.sort_values(by=LoggingColumns.ModelOutput.value, ascending=True).head(k),
                     false_positives=
                     results.false_positives.sort_values(by=LoggingColumns.ModelOutput.value, ascending=False).head(k),
                     false_negatives=
                     results.false_negatives.sort_values(by=LoggingColumns.ModelOutput.value, ascending=True).head(k))
    return sorted


def print_k_best_and_worst_performing(val_metrics_csv: Path, test_metrics_csv: Path, k: int):

    results = get_k_best_and_worst_performing(val_metrics_csv=val_metrics_csv,
                                              test_metrics_csv=test_metrics_csv,
                                              k=k)

    print_header(f"Top {k} false positives")
    print(results.false_positives[LoggingColumns.Patient.value])

    print_header(f"Top {k} false negatives")
    print(results.false_negatives[LoggingColumns.Patient.value])

    print_header(f"Top {k} true positives")
    print(results.true_positives[LoggingColumns.Patient.value])

    print_header(f"Top {k} true negatives")
    print(results.true_negatives[LoggingColumns.Patient.value])
