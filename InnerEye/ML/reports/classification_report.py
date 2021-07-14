#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from PIL import Image
from matplotlib.axes import Axes
from sklearn.metrics import auc, precision_recall_curve, recall_score, roc_auc_score, roc_curve

from InnerEye.Common.common_util import BEST_EPOCH_FOLDER_NAME
from InnerEye.Common.metrics_constants import LoggingColumns
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.dataset.scalar_dataset import ScalarDataset
from InnerEye.ML.metrics_dict import MetricsDict, binary_classification_accuracy
from InnerEye.ML.reports.notebook_report import print_header, print_table
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.utils.io_util import load_image_in_known_formats


@dataclass
class LabelsAndPredictions:
    subject_ids: np.ndarray
    labels: np.ndarray
    model_outputs: np.ndarray


@dataclass
class Results:
    true_positives: pd.DataFrame
    false_positives: pd.DataFrame
    true_negatives: pd.DataFrame
    false_negatives: pd.DataFrame


class ReportedScalarMetrics(Enum):
    """
    Different metrics displayed in the report.
    """
    AUC_PR = "Area under PR Curve", False
    AUC_ROC = "Area under ROC Curve", False
    OptimalThreshold = "Optimal threshold", False
    AccuracyAtOptimalThreshold = "Accuracy at optimal threshold", True
    AccuracyAtThreshold05 = "Accuracy at threshold 0.5", True
    Sensitivity = "Sensitivity at optimal threshold", True
    Specificity = "Specificity at optimal threshold", True

    def __init__(self, description: str, requires_threshold: bool) -> None:
        self.description = description
        self.requires_threshold = requires_threshold


def read_csv_and_filter_prediction_target(csv: Path, prediction_target: str,
                                          crossval_split_index: Optional[int] = None,
                                          data_split: Optional[ModelExecutionMode] = None,
                                          epoch: Optional[int] = None) -> pd.DataFrame:
    """
    Given one of the CSV files written during inference time, read it and select only those rows which belong to the
    given prediction_target. Also check that the final subject IDs are unique.

    :param csv: Path to the metrics CSV file. Must contain at least the following columns (defined in the LoggingColumns
        enum): LoggingColumns.Patient, LoggingColumns.Hue.
    :param prediction_target: Target ("hue") by which to filter.
    :param crossval_split_index: If specified, filter rows only for the respective run (requires
        LoggingColumns.CrossValidationSplitIndex).
    :param data_split: If specified, filter rows by Train/Val/Test (requires LoggingColumns.DataSplit).
    :param epoch: If specified, filter rows for given epoch (default: last epoch only; requires LoggingColumns.Epoch).
    :return: Filtered dataframe.
    """

    def check_column_present(dataframe: pd.DataFrame, column: LoggingColumns) -> None:
        if column.value not in dataframe:
            raise ValueError(f"Missing {column.value} column.")

    df = pd.read_csv(csv)
    df = df[df[LoggingColumns.Hue.value] == prediction_target]  # Filter by prediction target
    df = df[~df[LoggingColumns.Label.value].isna()]  # Filter missing labels

    # Filter by crossval split index
    if crossval_split_index is not None:
        check_column_present(df, LoggingColumns.CrossValidationSplitIndex)
        df = df[df[LoggingColumns.CrossValidationSplitIndex.value] == crossval_split_index]

    # Filter by Train/Val/Test
    if data_split is not None:
        check_column_present(df, LoggingColumns.DataSplit)
        df = df[df[LoggingColumns.DataSplit.value] == data_split.value]

    # Filter by epoch
    if LoggingColumns.Epoch.value in df:
        # In a FULL_METRICS_DATAFRAME_FILE, the epoch column will be BEST_EPOCH_FOLDER_NAME (string) for the Test split.
        # Here we cast the whole column to integer, mapping BEST_EPOCH_FOLDER_NAME to -1.
        epochs = df[LoggingColumns.Epoch.value].apply(lambda e: -1 if e == BEST_EPOCH_FOLDER_NAME else int(e))
        if epoch is None:
            epoch = epochs.max()  # Take last epoch if unspecified
        df = df[epochs == epoch]
    elif epoch is not None:
        raise ValueError(f"Specified epoch {epoch} but missing {LoggingColumns.Epoch.value} column.")

    if not df[LoggingColumns.Patient.value].is_unique:
        raise ValueError(f"Subject IDs should be unique, but found duplicate entries "
                         f"in column {LoggingColumns.Patient.value} in the csv file.")
    return df


def get_labels_and_predictions(csv: Path, prediction_target: str,
                               crossval_split_index: Optional[int] = None,
                               data_split: Optional[ModelExecutionMode] = None,
                               epoch: Optional[int] = None) -> LabelsAndPredictions:
    """
    Given a CSV file, reads the subject IDs, ground truth labels and model outputs for each subject
    for the given prediction target.

    :param csv: Path to the metrics CSV file. Must contain at least the following columns (defined in the LoggingColumns
        enum): LoggingColumns.Patient, LoggingColumns.Hue.
    :param prediction_target: Target ("hue") by which to filter.
    :param crossval_split_index: If specified, filter rows only for the respective run (requires
        LoggingColumns.CrossValidationSplitIndex).
    :param data_split: If specified, filter rows by Train/Val/Test (requires LoggingColumns.DataSplit).
    :param epoch: If specified, filter rows for given epoch (default: last epoch only; requires LoggingColumns.Epoch).
    :return: Filtered labels and model outputs.
    """
    df = read_csv_and_filter_prediction_target(csv, prediction_target, crossval_split_index, data_split, epoch)
    return get_labels_and_predictions_from_dataframe(df)


def get_labels_and_predictions_from_dataframe(df: pd.DataFrame) -> LabelsAndPredictions:
    """
    Given a dataframe, reads the subject IDs, ground truth labels and model outputs for each subject.
    NOTE: This dataframe should have results from a single epoch, as in the metrics files written during inference, not
    like the ones written while training. It must have at least the following columns (defined in the LoggingColumns
    enum):
    LoggingColumns.Patient, LoggingColumns.Label, LoggingColumns.ModelOutput.
    """
    labels = df[LoggingColumns.Label.value].to_numpy()
    model_outputs = df[LoggingColumns.ModelOutput.value].to_numpy()
    subjects = df[LoggingColumns.Patient.value].to_numpy()
    return LabelsAndPredictions(subject_ids=subjects, labels=labels, model_outputs=model_outputs)


def format_pr_or_roc_axes(plot_type: str, ax: Axes) -> None:
    """
    Format PR or ROC plot with appropriate title, axis labels, limits, and grid.
    :param plot_type: Either 'pr' or 'roc'.
    :param ax: Axes object to format.
    """
    if plot_type == 'pr':
        title, xlabel, ylabel = "PR Curve", "Recall", "Precision"
    elif plot_type == 'roc':
        title, xlabel, ylabel = "ROC Curve", "False positive rate", "True positive rate"
    else:
        raise ValueError(f"Plot type must be either 'pr' or 'roc' (received '{plot_type}')")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(lw=1, color='lightgray')


def plot_pr_and_roc_curves(labels_and_model_outputs: LabelsAndPredictions, axs: Optional[Sequence[Axes]] = None,
                           plot_kwargs: Optional[Dict[str, Any]] = None) -> None:
    """
    Given labels and model outputs, plot the ROC and PR curves.
    :param labels_and_model_outputs:
    :param axs: Pair of axes objects onto which to plot the ROC and PR curves, respectively. New axes are created by
    default.
    :param plot_kwargs: Plotting options to be passed to both `ax.plot(...)` calls.
    """
    if axs is None:
        _, axs = plt.subplots(1, 2)
    if plot_kwargs is None:
        plot_kwargs = {}

    fpr, tpr, thresholds = roc_curve(labels_and_model_outputs.labels, labels_and_model_outputs.model_outputs)
    axs[0].plot(fpr, tpr, **plot_kwargs)
    format_pr_or_roc_axes('roc', axs[0])

    precision, recall, thresholds = precision_recall_curve(labels_and_model_outputs.labels,
                                                           labels_and_model_outputs.model_outputs)
    axs[1].plot(recall, precision, **plot_kwargs)
    format_pr_or_roc_axes('pr', axs[1])

    plt.show()


def plot_scores_and_summary(all_labels_and_model_outputs: Sequence[LabelsAndPredictions],
                            scoring_fn: Callable[[LabelsAndPredictions], Tuple[np.ndarray, np.ndarray]],
                            interval_width: float = .8,
                            ax: Optional[Axes] = None) -> Tuple[List, Any]:
    """
    Plot a collection of score curves along with the (vertical) median and confidence interval (CI).

    Each plotted curve is interpolated onto a common horizontal grid, and the median and CI are computed vertically
    at each horizontal location.
    :param all_labels_and_model_outputs: Collection of ground-truth labels and model predictions (e.g. for various
    cross-validation runs).
    :param scoring_fn: A scoring function mapping a `LabelsAndPredictions` object to X and Y coordinates for plotting.
    :param interval_width: A value in [0, 1] representing what fraction of the data should be contained in
    the shaded area. The edges of the interval are `median +/- interval_width/2`.
    :param ax: Axes object onto which to plot (default: use current axes).
    :return: A tuple of `(line_handles, summary_handle)` to use in setting a legend for the plot: `line_handles` is a
    list corresponding to the curves for each `LabelsAndPredictions`, and `summary_handle` references the median line
    and shaded CI area.
    """
    if ax is None:
        ax = plt.gca()
    x_grid = np.linspace(0, 1, 101)
    interp_ys = []
    line_handles = []
    for index, labels_and_model_outputs in enumerate(all_labels_and_model_outputs):
        x_values, y_values = scoring_fn(labels_and_model_outputs)
        interp_ys.append(np.interp(x_grid, x_values, y_values))
        handle, = ax.plot(x_values, y_values, lw=1)
        line_handles.append(handle)

    interval_quantiles = [.5 - interval_width / 2, .5, .5 + interval_width / 2]
    y_lo, y_mid, y_hi = np.quantile(interp_ys, interval_quantiles, axis=0)
    h1 = ax.fill_between(x_grid, y_lo, y_hi, color='k', alpha=.2, lw=0)
    h2, = ax.plot(x_grid, y_mid, 'k', lw=2)
    summary_handle = (h1, h2)
    return line_handles, summary_handle


def plot_pr_and_roc_curves_crossval(all_labels_and_model_outputs: Sequence[LabelsAndPredictions],
                                    axs: Optional[Sequence[Axes]] = None) -> None:
    """
    Given a list of LabelsAndPredictions objects, plot the corresponding ROC and PR curves, along with median line and
    shaded 80% confidence interval (computed over TPRs and precisions for each fixed FPR and recall value).
    :param all_labels_and_model_outputs: Collection of ground-truth labels and model predictions (e.g. for various
    cross-validation runs).
    :param axs: Pair of axes objects onto which to plot the ROC and PR curves, respectively. New axes are created by
    default.
    """
    if axs is None:
        _, axs = plt.subplots(1, 2)

    def get_roc_xy(labels_and_model_outputs: LabelsAndPredictions) -> Tuple[np.ndarray, np.ndarray]:
        fpr, tpr, thresholds = roc_curve(labels_and_model_outputs.labels, labels_and_model_outputs.model_outputs)
        return fpr, tpr

    def get_pr_xy(labels_and_model_outputs: LabelsAndPredictions) -> Tuple[np.ndarray, np.ndarray]:
        precision, recall, thresholds = precision_recall_curve(labels_and_model_outputs.labels,
                                                               labels_and_model_outputs.model_outputs)
        return recall[::-1], precision[::-1]  # inverted to be in ascending order

    interval_width = .8
    line_handles, summary_handle = plot_scores_and_summary(all_labels_and_model_outputs,
                                                           scoring_fn=get_roc_xy, ax=axs[0],
                                                           interval_width=interval_width)
    plot_scores_and_summary(all_labels_and_model_outputs, scoring_fn=get_pr_xy, ax=axs[1],
                            interval_width=interval_width)

    line_labels = [f"Split {split_index}" for split_index in range(len(all_labels_and_model_outputs))]
    axs[0].legend(line_handles + [summary_handle],
                  line_labels + [f"Median \u00b1 {50 * interval_width:g}%"])

    format_pr_or_roc_axes('roc', axs[0])
    format_pr_or_roc_axes('pr', axs[1])

    plt.show()


def plot_pr_and_roc_curves_from_csv(metrics_csv: Path, config: ScalarModelBase,
                                    data_split: Optional[ModelExecutionMode] = None,
                                    is_crossval_report: bool = False) -> None:
    """
    Given the CSV written during inference time and the model config, plot the ROC and PR curves for all prediction
    targets.
    :param metrics_csv: Path to the metrics CSV file.
    :param config: Model config.
    :param data_split: Whether to filter the CSV file for Train, Val, or Test results (default: no filtering).
    :param is_crossval_report: If True, assumes CSV contains results for multiple cross-validation runs and plots the
    curves with median and confidence intervals. Otherwise, plots curves for a single run.
    """
    for prediction_target in config.target_names:
        print_header(f"Class: {prediction_target}", level=3)
        if is_crossval_report:
            all_metrics = [get_labels_and_predictions(metrics_csv, prediction_target,
                                                      crossval_split_index=crossval_split, data_split=data_split)
                           for crossval_split in range(config.number_of_cross_validation_splits)]
            plot_pr_and_roc_curves_crossval(all_metrics)
        else:
            metrics = get_labels_and_predictions(metrics_csv, prediction_target, data_split=data_split)
            plot_pr_and_roc_curves(metrics)


def get_metric(predictions_to_set_optimal_threshold: LabelsAndPredictions,
               predictions_to_compute_metrics: LabelsAndPredictions,
               metric: ReportedScalarMetrics,
               optimal_threshold: Optional[float] = None) -> float:
    """
    Given LabelsAndPredictions objects for the validation and test sets, return the specified metric.
    :param predictions_to_set_optimal_threshold: This set of ground truth labels and model predictions is used to
    determine the optimal threshold for classification.
    :param predictions_to_compute_metrics: The set of labels and model outputs to calculate metrics for.
    :param metric: The name of the metric to calculate.
    :param optimal_threshold: If provided, use this threshold instead of calculating an optimal threshold.
    """
    if not optimal_threshold:
        fpr, tpr, thresholds = roc_curve(predictions_to_set_optimal_threshold.labels,
                                         predictions_to_set_optimal_threshold.model_outputs)
        optimal_idx = MetricsDict.get_optimal_idx(fpr=fpr, tpr=tpr)
        optimal_threshold = thresholds[optimal_idx]

    assert optimal_threshold  # for mypy, we have already calculated optimal threshold if it was set to None

    if metric is ReportedScalarMetrics.OptimalThreshold:
        return optimal_threshold

    only_one_class_present = len(set(predictions_to_compute_metrics.labels)) < 2

    if metric is ReportedScalarMetrics.AUC_ROC:
        return math.nan if only_one_class_present else roc_auc_score(predictions_to_compute_metrics.labels,
                                                                     predictions_to_compute_metrics.model_outputs)
    elif metric is ReportedScalarMetrics.AUC_PR:
        if only_one_class_present:
            return math.nan
        precision, recall, _ = precision_recall_curve(predictions_to_compute_metrics.labels,
                                                      predictions_to_compute_metrics.model_outputs)
        return auc(recall, precision)
    elif metric is ReportedScalarMetrics.AccuracyAtOptimalThreshold:
        return binary_classification_accuracy(model_output=predictions_to_compute_metrics.model_outputs,
                                              label=predictions_to_compute_metrics.labels,
                                              threshold=optimal_threshold)
    elif metric is ReportedScalarMetrics.AccuracyAtThreshold05:
        return binary_classification_accuracy(model_output=predictions_to_compute_metrics.model_outputs,
                                              label=predictions_to_compute_metrics.labels,
                                              threshold=0.5)
    elif metric is ReportedScalarMetrics.Specificity:
        return recall_score(predictions_to_compute_metrics.labels,
                            predictions_to_compute_metrics.model_outputs >= optimal_threshold, pos_label=0)
    elif metric is ReportedScalarMetrics.Sensitivity:
        return recall_score(predictions_to_compute_metrics.labels,
                            predictions_to_compute_metrics.model_outputs >= optimal_threshold)
    else:
        raise ValueError("Unknown metric")


def get_all_metrics(predictions_to_set_optimal_threshold: LabelsAndPredictions,
                    predictions_to_compute_metrics: LabelsAndPredictions,
                    is_thresholded: bool = False) -> Dict[str, float]:
    """
    Given LabelsAndPredictions objects for the validation and test sets, compute some metrics.
    :param predictions_to_set_optimal_threshold: This is used to determine the optimal threshold for classification.
    :param predictions_to_compute_metrics: Metrics are calculated for this set.
    :param is_thresholded: Whether the model outputs are binary (they have been thresholded at some point)
                           or are floating point numbers.
    :return: Dictionary mapping metric descriptions to computed values.
    """
    optimal_threshold = 0.5 if is_thresholded else \
        get_metric(predictions_to_set_optimal_threshold=predictions_to_set_optimal_threshold,
                   predictions_to_compute_metrics=predictions_to_compute_metrics,
                   metric=ReportedScalarMetrics.OptimalThreshold)

    metrics = {}
    for metric in ReportedScalarMetrics:  # type: ReportedScalarMetrics
        if is_thresholded and not metric.requires_threshold:
            continue
        metrics[metric.description] = get_metric(
            predictions_to_set_optimal_threshold=predictions_to_set_optimal_threshold,
            predictions_to_compute_metrics=predictions_to_compute_metrics,
            metric=metric, optimal_threshold=optimal_threshold)

    return metrics


def print_metrics(predictions_to_set_optimal_threshold: LabelsAndPredictions,
                  predictions_to_compute_metrics: LabelsAndPredictions,
                  is_thresholded: bool = False) -> None:
    """
    Given LabelsAndPredictions objects for the validation and test sets, print out some metrics.
    :param predictions_to_set_optimal_threshold: This is used to determine the optimal threshold for classification.
    :param predictions_to_compute_metrics: Metrics are calculated for this set.
    :param is_thresholded: Whether the model outputs are binary (they have been thresholded at some point)
                           or are floating point numbers.
    """
    metrics = get_all_metrics(predictions_to_set_optimal_threshold, predictions_to_compute_metrics, is_thresholded)
    rows = [[description, f"{value:.4f}"] for description, value in metrics.items()]
    print_table(rows)


def get_metrics_table_for_prediction_target(csv_to_set_optimal_threshold: Path,
                                            csv_to_compute_metrics: Path,
                                            config: ScalarModelBase,
                                            prediction_target: str,
                                            data_split_to_set_optimal_threshold: Optional[ModelExecutionMode] = None,
                                            data_split_to_compute_metrics: Optional[ModelExecutionMode] = None,
                                            is_thresholded: bool = False,
                                            is_crossval_report: bool = False) -> Tuple[List[List[str]], List[str]]:
    """
    Given CSVs written during inference for the validation and test sets, compute and format metrics as a table.

    :param csv_to_set_optimal_threshold: CSV written during inference time for the val set. This is used to determine
        the optimal threshold for classification.
    :param csv_to_compute_metrics: CSV written during inference time for the test set. Metrics are calculated for
        this CSV.
    :param config: Model config
    :param prediction_target: The prediction target for which to compute metrics.
    :param data_split_to_set_optimal_threshold: Whether to filter the validation CSV file for Train, Val, or Test
        results (default: no filtering).
    :param data_split_to_compute_metrics: Whether to filter the test CSV file for Train, Val, or Test results
        (default: no filtering).
    :param is_thresholded: Whether the model outputs are binary (they have been thresholded at some point)
        or are floating point numbers.
    :param is_crossval_report: If True, assumes CSVs contain results for multiple cross-validation runs and formats the
        metrics along with means and standard deviations. Otherwise, collect metrics for a single run.
    :return: Tuple of rows and header, where each row and the header are lists of strings of same length (2 if
        `is_crossval_report` is False, `config.number_of_cross_validation_splits`+2 otherwise).
    """

    def get_metrics_for_crossval_split(prediction_target: str,
                                       crossval_split: Optional[int] = None) -> Dict[str, float]:
        predictions_to_set_optimal_threshold = get_labels_and_predictions(csv_to_set_optimal_threshold,
                                                                          prediction_target,
                                                                          crossval_split_index=crossval_split,
                                                                          data_split=data_split_to_set_optimal_threshold)
        predictions_to_compute_metrics = get_labels_and_predictions(csv_to_compute_metrics, prediction_target,
                                                                    crossval_split_index=crossval_split,
                                                                    data_split=data_split_to_compute_metrics)
        return get_all_metrics(predictions_to_set_optimal_threshold, predictions_to_compute_metrics, is_thresholded)

    # Compute metrics for all crossval splits or single run, and initialise table header
    all_metrics: List[Dict[str, float]] = []
    header = ["Metric"]
    if is_crossval_report:
        for crossval_split in range(config.number_of_cross_validation_splits):
            all_metrics.append(get_metrics_for_crossval_split(prediction_target, crossval_split))
            header.append(f"Split {crossval_split}")
    else:
        all_metrics.append(get_metrics_for_crossval_split(prediction_target))
        header.append("Value")
    computed_metrics = all_metrics[0].keys()

    # Format table rows
    rows = [[metric] + [f"{fold_metrics[metric]:.4f}" for fold_metrics in all_metrics]
            for metric in computed_metrics]

    # Add aggregation column and header
    if is_crossval_report:
        for row, metric in zip(rows, computed_metrics):
            values = [fold_metrics[metric] for fold_metrics in all_metrics]
            row.append(f"{np.mean(values):.4f} ({np.std(values):.4f})")
        header.append("Mean (std)")

    return rows, header


def print_metrics_for_all_prediction_targets(csv_to_set_optimal_threshold: Path,
                                             csv_to_compute_metrics: Path,
                                             config: ScalarModelBase,
                                             data_split_to_set_optimal_threshold: Optional[ModelExecutionMode] = None,
                                             data_split_to_compute_metrics: Optional[ModelExecutionMode] = None,
                                             is_thresholded: bool = False,
                                             is_crossval_report: bool = False) -> None:
    """
    Given CSVs written during inference for the validation and test sets, print out metrics for every prediction target
    in the config.

    :param csv_to_set_optimal_threshold: CSV written during inference time for the val set. This is used to determine
        the optimal threshold for classification.
    :param csv_to_compute_metrics: CSV written during inference time for the test set. Metrics are calculated for
        this CSV.
    :param config: Model config
    :param data_split_to_set_optimal_threshold: Whether to filter the validation CSV file for Train, Val, or Test
        results (default: no filtering).
    :param data_split_to_compute_metrics: Whether to filter the test CSV file for Train, Val, or Test results
        (default: no filtering).
    :param is_thresholded: Whether the model outputs are binary (they have been thresholded at some point)
        or are floating point numbers.
    :param is_crossval_report: If True, assumes CSVs contain results for multiple cross-validation runs and prints the
        metrics along with means and standard deviations. Otherwise, prints metrics for a single run.
    """
    for prediction_target in config.target_names:
        print_header(f"Class: {prediction_target}", level=3)
        rows, header = get_metrics_table_for_prediction_target(
            csv_to_set_optimal_threshold=csv_to_set_optimal_threshold,
            data_split_to_set_optimal_threshold=data_split_to_set_optimal_threshold,
            csv_to_compute_metrics=csv_to_compute_metrics,
            data_split_to_compute_metrics=data_split_to_compute_metrics,
            config=config,
            prediction_target=prediction_target,
            is_thresholded=is_thresholded,
            is_crossval_report=is_crossval_report)
        print_table(rows, header)


def get_correct_and_misclassified_examples(val_metrics_csv: Path, test_metrics_csv: Path,
                                           prediction_target: str = MetricsDict.DEFAULT_HUE_KEY) -> Optional[Results]:
    """
    Given the paths to the metrics files for the validation and test sets, get a list of true positives,
    false positives, false negatives and true negatives.
    The threshold for classification is obtained by looking at the validation file, and applied to the test set to get
    label predictions.
    The validation and test csvs must have at least the following columns (defined in the LoggingColumns enum):
    LoggingColumns.Hue, LoggingColumns.Patient, LoggingColumns.Label, LoggingColumns.ModelOutput.
    """
    df_val = read_csv_and_filter_prediction_target(val_metrics_csv, prediction_target)

    if len(df_val) == 0:
        return None

    fpr, tpr, thresholds = roc_curve(df_val[LoggingColumns.Label.value], df_val[LoggingColumns.ModelOutput.value])
    optimal_idx = MetricsDict.get_optimal_idx(fpr=fpr, tpr=tpr)
    optimal_threshold = thresholds[optimal_idx]

    df_test = read_csv_and_filter_prediction_target(test_metrics_csv, prediction_target)

    if len(df_test) == 0:
        return None

    df_test["predicted"] = df_test.apply(lambda x: int(x[LoggingColumns.ModelOutput.value] >= optimal_threshold),
                                         axis=1)

    true_positives = df_test[(df_test["predicted"] == 1) & (df_test[LoggingColumns.Label.value] == 1)]
    false_positives = df_test[(df_test["predicted"] == 1) & (df_test[LoggingColumns.Label.value] == 0)]
    false_negatives = df_test[(df_test["predicted"] == 0) & (df_test[LoggingColumns.Label.value] == 1)]
    true_negatives = df_test[(df_test["predicted"] == 0) & (df_test[LoggingColumns.Label.value] == 0)]

    return Results(true_positives=true_positives,
                   true_negatives=true_negatives,
                   false_positives=false_positives,
                   false_negatives=false_negatives)


def get_k_best_and_worst_performing(val_metrics_csv: Path, test_metrics_csv: Path, k: int,
                                    prediction_target: str = MetricsDict.DEFAULT_HUE_KEY) -> Optional[Results]:
    """
    Get the top "k" best predictions (i.e. correct classifications where the model was the most certain) and the
    top "k" worst predictions (i.e. misclassifications where the model was the most confident).
    """
    results = get_correct_and_misclassified_examples(val_metrics_csv=val_metrics_csv,
                                                     test_metrics_csv=test_metrics_csv,
                                                     prediction_target=prediction_target)
    if results is None:
        return None

    # sort by model_output
    sorted = Results(true_positives=results.true_positives.sort_values(by=LoggingColumns.ModelOutput.value,
                                                                       ascending=False).head(k),
                     true_negatives=results.true_negatives.sort_values(by=LoggingColumns.ModelOutput.value,
                                                                       ascending=True).head(k),
                     false_positives=results.false_positives.sort_values(by=LoggingColumns.ModelOutput.value,
                                                                         ascending=False).head(k),
                     false_negatives=results.false_negatives.sort_values(by=LoggingColumns.ModelOutput.value,
                                                                         ascending=True).head(k))
    return sorted


def print_k_best_and_worst_performing(val_metrics_csv: Path,
                                      test_metrics_csv: Path,
                                      k: int,
                                      prediction_target: str) -> None:
    """
    Print the top "k" best predictions (i.e. correct classifications where the model was the most certain) and the
    top "k" worst predictions (i.e. misclassifications where the model was the most confident).
    :param val_metrics_csv: Path to one of the metrics csvs written during inference. This set of metrics will be
                            used to determine the thresholds for predicting labels on the test set. The best and worst
                            performing subjects will not be printed out for this csv.
    :param test_metrics_csv: Path to one of the metrics csvs written during inference. This is the csv for which
                            best and worst performing subjects will be printed out.
    :param k: Number of subjects of each category to print out.
    :param prediction_target: The class label to filter on
    """
    results = get_k_best_and_worst_performing(val_metrics_csv=val_metrics_csv,
                                              test_metrics_csv=test_metrics_csv,
                                              k=k,
                                              prediction_target=prediction_target)
    if results is None:
        print_header("Empty validation or test set", level=2)
        return

    print_header(f"Top {k} false positives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.false_positives[LoggingColumns.Patient.value],
                                                        results.false_positives[LoggingColumns.ModelOutput.value])):
        print_header(f"{index + 1}. ID {subject} Score: {model_output:.5f}", level=4)

    print_header(f"Top {k} false negatives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.false_negatives[LoggingColumns.Patient.value],
                                                        results.false_negatives[LoggingColumns.ModelOutput.value])):
        print_header(f"{index + 1}. ID {subject} Score: {model_output:.5f}", level=4)

    print_header(f"Top {k} true positives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.true_positives[LoggingColumns.Patient.value],
                                                        results.true_positives[LoggingColumns.ModelOutput.value])):
        print_header(f"{index + 1}. ID {subject} Score: {model_output:.5f}", level=4)

    print_header(f"Top {k} true negatives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.true_negatives[LoggingColumns.Patient.value],
                                                        results.true_negatives[LoggingColumns.ModelOutput.value])):
        print_header(f"{index + 1}. ID {subject} Score: {model_output:.5f}", level=4)


def get_image_filepath_from_subject_id(subject_id: str,
                                       dataset: ScalarDataset,
                                       config: ScalarModelBase) -> List[Path]:
    """
    Return the filepaths for images associated with a subject. If the subject is not found, raises a ValueError.
    :param subject_id: Subject to retrive image for
    :param dataset: scalar dataset object
    :param config: model config
    :return: List of paths to the image files for the patient.
    """
    for item in dataset.items:
        if item.metadata.id == subject_id:
            return item.get_all_image_filepaths(root_path=config.local_dataset,
                                                file_mapping=dataset.file_to_full_path)

    raise ValueError(f"Could not find subject {subject_id} in the dataset.")


def get_image_labels_from_subject_id(subject_id: str,
                                     dataset: ScalarDataset,
                                     config: ScalarModelBase) -> List[str]:
    """
    Return the ground truth labels associated with a subject. If the subject is not found, raises a ValueError.
    :param subject_id: Subject to retrive image for
    :param dataset: scalar dataset object
    :param config: model config
    :return: List of labels for the patient.
    """
    labels = None

    for item in dataset.items:
        if item.metadata.id == subject_id:
            labels = torch.flatten(torch.nonzero(item.label)).tolist()
            break

    if labels is None:
        raise ValueError(f"Could not find subject {subject_id} in the dataset.")

    return [config.class_names[int(label)] for label in labels
            if not math.isnan(label)]


def get_image_outputs_from_subject_id(subject_id: str,
                                      metrics_df: pd.DataFrame) -> List[Tuple[str, int]]:
    """
    Return a list of tuples (Label class name, model output for the class) for a single subject.
    """

    filtered = metrics_df[metrics_df[LoggingColumns.Patient.value] == subject_id]
    outputs = list(zip(filtered[LoggingColumns.Hue.value].values.tolist(),
                       filtered[LoggingColumns.ModelOutput.value].values.astype(float).tolist()))
    return outputs


def plot_image_from_filepath(filepath: Path, im_width: int) -> bool:
    """
    Plots a 2D image given the filepath. Returns false if the image could not be plotted (for example, if it was 3D).
    :param filepath: Path to image
    :param im_width: Display width for image
    :return: True if image was plotted, False otherwise
    """

    image = load_image_in_known_formats(filepath, load_segmentation=False).images
    if not image.ndim == 2 and not (image.ndim == 3 and image.shape[0] == 1):
        print_header(f"Image has unsupported shape {image.shape}", level=0)
        return False

    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)

    # normalize to make sure pixels are plottable
    min = image.min()
    max = image.max()
    image = (image - min) / (max - min)

    image = image * 255.
    image = np.repeat(image[:, :, None], 3, 2)
    image = image.astype(np.uint8)
    h, w = image.shape[:2]
    im_height = int(im_width * h / w)
    display(Image.fromarray(image).resize((im_width, im_height)))
    return True


def plot_image_for_subject(subject_id: str,
                           dataset: ScalarDataset,
                           im_width: int,
                           model_output: float,
                           header: Optional[str],
                           config: ScalarModelBase,
                           metrics_df: Optional[pd.DataFrame] = None) -> None:
    """
    Given a subject ID, plots the corresponding image.
    :param subject_id: Subject to plot image for
    :param dataset: scalar dataset object
    :param im_width: Display width for image
    :param model_output: The predicted value for this image
    :param header: Optional header printed along with the subject ID and score for the image.
    :param config: model config
    :param metrics_df: dataframe with the metrics written out during inference time
    """
    print_header("", level=4)
    if header:
        print_header(header, level=4)

    labels = get_image_labels_from_subject_id(subject_id=subject_id,
                                              dataset=dataset,
                                              config=config)

    print_header(f"True labels: {', '.join(labels) if labels else 'Negative'}", level=4)

    if metrics_df is not None:
        all_model_outputs = get_image_outputs_from_subject_id(subject_id=subject_id,
                                                              metrics_df=metrics_df)
        print_header(f"ID: {subject_id}", level=4)
        print_header(f"Model output: {', '.join([':'.join([str(x) for x in output]) for output in all_model_outputs])}",
                     level=4)
    else:
        print_header(f"ID: {subject_id} Score: {model_output}", level=4)

    filepaths = get_image_filepath_from_subject_id(subject_id=str(subject_id),
                                                   dataset=dataset,
                                                   config=config)

    if not filepaths:
        print_header(f"Subject ID {subject_id} not found."
                     f"Note: Reports with datasets that use channel columns in the dataset.csv "
                     f"are not yet supported.", level=0)
        return

    for filepath in filepaths:
        success = plot_image_from_filepath(filepath, im_width=im_width)
        if not success:
            print_header("Unable to plot image: image must be 2D with shape [w, h] or [1, w, h].", level=0)


def plot_k_best_and_worst_performing(val_metrics_csv: Path, test_metrics_csv: Path, k: int,
                                     prediction_target: str, config: ScalarModelBase) -> None:
    """
    Plot images for the top "k" best predictions (i.e. correct classifications where the model was the most certain)
    and the top "k" worst predictions (i.e. misclassifications where the model was the most confident).
    :param val_metrics_csv: Path to one of the metrics csvs written during inference. This set of metrics will be
                            used to determine the thresholds for predicting labels on the test set. The best and worst
                            performing subjects will not be printed out for this csv.
    :param test_metrics_csv: Path to one of the metrics csvs written during inference. This is the csv for which
                            best and worst performing subjects will be printed out.
    :param k: Number of subjects of each category to print out.
    :param prediction_target: The class label to filter on
    :param config: scalar model config object
    """
    results = get_k_best_and_worst_performing(val_metrics_csv=val_metrics_csv,
                                              test_metrics_csv=test_metrics_csv,
                                              k=k,
                                              prediction_target=prediction_target)
    if results is None:
        print_header("Empty validation or test set", level=4)
        return

    test_metrics = pd.read_csv(test_metrics_csv, dtype=str)

    df = config.read_dataset_if_needed()
    dataset = ScalarDataset(args=config, data_frame=df)

    im_width = 800

    print_header("", level=2)
    print_header(f"Top {k} false positives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.false_positives[LoggingColumns.Patient.value],
                                                        results.false_positives[LoggingColumns.ModelOutput.value])):
        plot_image_for_subject(subject_id=str(subject),
                               dataset=dataset,
                               im_width=im_width,
                               model_output=model_output,
                               header="False Positive",
                               config=config,
                               metrics_df=test_metrics)

    print_header(f"Top {k} false negatives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.false_negatives[LoggingColumns.Patient.value],
                                                        results.false_negatives[LoggingColumns.ModelOutput.value])):
        plot_image_for_subject(subject_id=str(subject),
                               dataset=dataset,
                               im_width=im_width,
                               model_output=model_output,
                               header="False Negative",
                               config=config,
                               metrics_df=test_metrics)

    print_header(f"Top {k} true positives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.true_positives[LoggingColumns.Patient.value],
                                                        results.true_positives[LoggingColumns.ModelOutput.value])):
        plot_image_for_subject(subject_id=str(subject),
                               dataset=dataset,
                               im_width=im_width,
                               model_output=model_output,
                               header="True Positive",
                               config=config,
                               metrics_df=test_metrics)

    print_header(f"Top {k} true negatives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.true_negatives[LoggingColumns.Patient.value],
                                                        results.true_negatives[LoggingColumns.ModelOutput.value])):
        plot_image_for_subject(subject_id=str(subject),
                               dataset=dataset,
                               im_width=im_width,
                               model_output=model_output,
                               header="True Negative",
                               config=config,
                               metrics_df=test_metrics)
