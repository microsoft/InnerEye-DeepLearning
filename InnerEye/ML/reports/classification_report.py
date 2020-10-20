#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score

from InnerEye.ML.reports.notebook_report import print_header
from InnerEye.Common.metrics_dict import MetricsDict
from InnerEye.ML.utils.metrics_constants import LoggingColumns


@dataclass
class Results:
    true_positives: pd.DataFrame
    false_positives: pd.DataFrame
    true_negatives: pd.DataFrame
    false_negatives: pd.DataFrame


def plot_auc(x_values, y_values, title, ax):
    ax.plot(x_values, y_values)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_title(title)

    # write values of points
    for x, y in zip (x_values, y_values):
        ax.annotate(f"{x:0.3f}, {y:0.3f}", xy = (x, y), xytext=(15, 0), textcoords='offset points')


def plot_pr_and_roc_curves_from_csv(metrics_csv):
    print_header("ROC and PR curves", level=3)
    df = pd.read_csv(metrics_csv)

    _, ax = plt.subplots(1, 2)

    fpr, tpr, thresholds = roc_curve(df[LoggingColumns.Label.value], df[LoggingColumns.ModelOutput.value])
    plot_auc(fpr, tpr, "ROC Curve", ax[0])
    precision, recall, thresholds = precision_recall_curve(df[LoggingColumns.Label.value],
                                                           df[LoggingColumns.ModelOutput.value])
    plot_auc(precision, recall, "PR Curve", ax[1])

    plt.show()


def print_metrics(val_metrics_csv, test_metrics_csv):
    print_header("Test metrics", level=3)
    df_val = pd.read_csv(val_metrics_csv)
    fpr, tpr, thresholds = roc_curve(df_val[LoggingColumns.Label.value], df_val[LoggingColumns.ModelOutput.value])
    optimal_idx = MetricsDict.get_optimal_idx(fpr=fpr, tpr=tpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"Optimal threshold: {optimal_threshold}")

    df_test = pd.read_csv(test_metrics_csv)
    roc_auc = roc_auc_score(df_test[LoggingColumns.Label.value], df_test[LoggingColumns.ModelOutput.value])
    print(f"AUC PR: {roc_auc}")
    precision, recall, _ = precision_recall_curve(df_test[LoggingColumns.Label.value],
                                                  df_test[LoggingColumns.ModelOutput.value])
    pr_auc = auc(recall, precision)
    print(f"AUC PR: {roc_auc}")


def get_correct_and_misclassified_examples(val_metrics_csv, test_metrics_csv):
    df_val = pd.read_csv(val_metrics_csv)
    fpr, tpr, thresholds = roc_curve(df_val[LoggingColumns.Label.value], df_val[LoggingColumns.ModelOutput.value])
    optimal_idx = MetricsDict.get_optimal_idx(fpr=fpr, tpr=tpr)
    optimal_threshold = thresholds[optimal_idx]

    df_test = pd.read_csv(test_metrics_csv)
    df_test["predicted"] = df_test.apply(lambda x: int(x[LoggingColumns.ModelOutput.value] > optimal_threshold), axis=1)

    true_positives = df_test[(df_test["predicted"] == 1) & (df_test[LoggingColumns.Label.value] == 1)]
    false_positives = df_test[(df_test["predicted"] == 1) & (df_test[LoggingColumns.Label.value] == 0)]
    false_negatives = df_test[(df_test["predicted"] == 0) & (df_test[LoggingColumns.Label.value] == 1)]
    true_negatives = df_test[(df_test["predicted"] == 0) & (df_test[LoggingColumns.Label.value] == 0)]

    return Results(true_positives=true_positives,
                   true_negatives=true_negatives,
                   false_positives=false_positives,
                   false_negatives=false_negatives)


def print_top_best_and_worst(val_metrics_csv, test_metrics_csv, k: int):
    results = get_correct_and_misclassified_examples(val_metrics_csv=val_metrics_csv,
                                                     test_metrics_csv=test_metrics_csv)

    # sort by model_output
    results.true_positives = results.true_positives.sort_values(by=LoggingColumns.Label.value, ascending=False)
    results.true_negatives = results.true_negatives.sort_values(by=LoggingColumns.Label.value, ascending=True)
    results.false_positives = results.true_positives.sort_values(by=LoggingColumns.Label.value, ascending=False)
    results.false_negatives = results.true_negatives.sort_values(by=LoggingColumns.Label.value, ascending=True)

    print_header(f"Top {k} false positives")
    print(results.false_positives[LoggingColumns.Patient.value].head(k))

    print_header(f"Top {k} false negatives")
    print(results.false_negatives[LoggingColumns.Patient.value].head(k))

    print_header(f"Top {k} true positives")
    print(results.true_positives[LoggingColumns.Patient.value].head(k))

    print_header(f"Top {k} true negatives")
    print(results.true_negatives[LoggingColumns.Patient.value].head(k))
