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
import scipy.interpolate
import torch
from IPython.display import display
from PIL import Image
from matplotlib.axes import Axes
from sklearn.metrics import auc, precision_recall_curve, recall_score, roc_auc_score, roc_curve

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


class ReportedMetrics(Enum):
    """
    Different metrics displayed in the report.
    """
    OptimalThreshold = "optimal_threshold"
    AUC_PR = "auc_pr"
    AUC_ROC = "auc_roc"
    Accuracy = "accuracy"
    FalsePositiveRate = "false_positive_rate"
    FalseNegativeRate = "false_negative_rate"


def quantile(x, q, axis=-1):
    x = np.sort(x, axis=axis)
    rank = np.linspace(0, 1, x.shape[axis])
    return scipy.interpolate.interp1d(rank, x, axis=axis)(q)


def read_csv_and_filter_prediction_target(csv: Path, prediction_target: str, crossval_split_index: Optional[int] = None,
                                          data_split: Optional[ModelExecutionMode] = None) -> pd.DataFrame:
    """
    Given one of the csv files written during inference time, read it and select only those rows which belong to the
    given prediction_target. If crossval_split_index is provided, will additionally filter rows only for the respective
    run, and data_split can further filter by Train/Val/Test. Also check that the final subject IDs are unique.
    The csv must have at least the following columns (defined in the LoggingColumns enum):
    LoggingColumns.Hue, LoggingColumns.Patient, LoggingColumns.CrossValidationSplitIndex (if crossval_split_index is
    given), LoggingColumns.DataSplit (if data_split is given).
    """
    df = pd.read_csv(csv)
    df = df[df[LoggingColumns.Hue.value] == prediction_target]  # Filter by prediction target
    if crossval_split_index is not None:
        if LoggingColumns.CrossValidationSplitIndex.value not in df:
            raise ValueError(f"Missing {LoggingColumns.CrossValidationSplitIndex.value} column.")
        # Filter by crossval split index
        df = df[df[LoggingColumns.CrossValidationSplitIndex.value] == crossval_split_index]
    if data_split is not None:
        if LoggingColumns.DataSplit.value not in df:
            raise ValueError(f"Missing {LoggingColumns.DataSplit.value} column.")
        df = df[df[LoggingColumns.DataSplit.value] == data_split.value]  # Filter by Train/Val/Test
    if not df[LoggingColumns.Patient.value].is_unique:
        raise ValueError(f"Subject IDs should be unique, but found duplicate entries "
                         f"in column {LoggingColumns.Patient.value} in the csv file.")
    return df


def get_labels_and_predictions(csv: Path, prediction_target: str, crossval_split_index: Optional[int] = None,
                               data_split: Optional[ModelExecutionMode] = None) -> LabelsAndPredictions:
    """
    Given a CSV file, reads the subject IDs, ground truth labels and model outputs for each subject
    for the given prediction target.
    NOTE: This CSV file should have results from a single epoch, as in the metrics files written during inference, not
    like the ones written while training. It must have at least the following columns (defined in the LoggingColumns
    enum):
    LoggingColumns.Hue, LoggingColumns.Patient, LoggingColumns.Label, LoggingColumns.ModelOutput.
    """
    df = read_csv_and_filter_prediction_target(csv, prediction_target, crossval_split_index, data_split)
    return get_labels_and_predictions_from_dataframe(df)


def get_labels_and_predictions_from_dataframe(df: pd.DataFrame) -> LabelsAndPredictions:
    """
    Given a dataframe, reads the subject IDs, ground truth labels and model outputs for each subject.
    NOTE: This dataframe should have results from a single epoch, as in the metrics files written during inference, not
    like the ones written while training. It must have at least the following columns (defined in the LoggingColumns
    enum):
    LoggingColumns.Patient, LoggingColumns.Label, LoggingColumns.ModelOutput.
    If present, the LoggingColumns.CrossValidationSplitIndex column will be read into the resulting crossval_folds
    field, otherwise it will be filled with -1 by default.
    """
    labels = df[LoggingColumns.Label.value].to_numpy()
    model_outputs = df[LoggingColumns.ModelOutput.value].to_numpy()
    subjects = df[LoggingColumns.Patient.value].to_numpy()
    return LabelsAndPredictions(subject_ids=subjects, labels=labels, model_outputs=model_outputs)


def plot_auc(x_values: np.ndarray, y_values: np.ndarray,  ax: Axes, print_coords: bool = False,
             **plot_kwargs: Any) -> None:
    """
    Plot a curve given the x and y values of each point.
    :param x_values: x coordinate of each data point to be plotted
    :param y_values: y coordinate of each data point to be plotted
    :param ax: matplotlib.axes.Axes object for plotting
    :param print_coords: If true, prints out the coordinates of each point on the graph.
    """
    ax.plot(x_values, y_values, **plot_kwargs)

    if print_coords:
        # write values of points
        for x, y in zip(x_values, y_values):
            ax.annotate(f"{x:0.3f}, {y:0.3f}", xy=(x, y), xytext=(15, 0), textcoords='offset points')


def format_pr_or_roc_axes(plot_type: str, ax: Axes) -> None:
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
    Given a LabelsAndPredictions object, plot the ROC and PR curves.
    """
    if axs is None:
        _, axs = plt.subplots(1, 2)
    if plot_kwargs is None:
        plot_kwargs = {}

    fpr, tpr, thresholds = roc_curve(labels_and_model_outputs.labels, labels_and_model_outputs.model_outputs)
    plot_auc(fpr, tpr, axs[0], **plot_kwargs)
    format_pr_or_roc_axes('roc', axs[0])

    precision, recall, thresholds = precision_recall_curve(labels_and_model_outputs.labels,
                                                           labels_and_model_outputs.model_outputs)
    plot_auc(recall, precision, axs[1], **plot_kwargs)
    format_pr_or_roc_axes('pr', axs[1])

    plt.show()


def plot_scores_and_summary(all_labels_and_model_outputs: Sequence[LabelsAndPredictions],
                            scoring_fn: Callable[[LabelsAndPredictions], Tuple[np.ndarray, np.ndarray]],
                            confidence_interval_width: float = .8,
                            ax: Optional[Axes] = None) -> Tuple[List, Any]:
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

    confidence_interval_quantiles = [.5 - confidence_interval_width / 2, .5, .5 + confidence_interval_width / 2]
    y_lo, y_mid, y_hi = quantile(interp_ys, confidence_interval_quantiles, axis=0)
    h1 = ax.fill_between(x_grid, y_lo, y_hi, color='k', alpha=.2, lw=0)
    h2, = ax.plot(x_grid, y_mid, 'k', lw=2)
    summary_handle = (h1, h2)
    return line_handles, summary_handle


def plot_pr_and_roc_curves_crossval(all_labels_and_model_outputs: Sequence[LabelsAndPredictions],
                                    axs: Optional[Sequence[Axes]] = None) -> None:
    """
    Given a list of LabelsAndPredictions objects, plot the ROC and PR curves.
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

    confidence_interval_width = .8
    line_handles, summary_handle = plot_scores_and_summary(all_labels_and_model_outputs,
                                                           scoring_fn=get_roc_xy, ax=axs[0],
                                                           confidence_interval_width=confidence_interval_width)
    plot_scores_and_summary(all_labels_and_model_outputs, scoring_fn=get_pr_xy, ax=axs[1],
                            confidence_interval_width=confidence_interval_width)

    line_labels = [f"Split {split_index}" for split_index in range(len(all_labels_and_model_outputs))]
    axs[0].legend(line_handles + [summary_handle],
                  line_labels + [f"Median, {100 * confidence_interval_width:g}% CI"])

    format_pr_or_roc_axes('roc', axs[0])
    format_pr_or_roc_axes('pr', axs[1])

    plt.show()


def plot_pr_and_roc_curves_from_csv(metrics_csv: Path, config: ScalarModelBase,
                                    data_split: Optional[ModelExecutionMode] = None,
                                    is_crossval_report: bool = False) -> None:
    """
    Given the csv written during inference time and the model config,
    plot the ROC and PR curves for all prediction targets.
    """
    for prediction_target in config.class_names:
        print_header(f"Class: {prediction_target}", level=3)
        if is_crossval_report:
            all_metrics = [get_labels_and_predictions(metrics_csv, prediction_target,
                                                      crossval_split_index=crossval_split, data_split=data_split)
                           for crossval_split in range(config.number_of_cross_validation_splits)]
            plot_pr_and_roc_curves_crossval(all_metrics)
        else:
            metrics = get_labels_and_predictions(metrics_csv, prediction_target, data_split=data_split)
            plot_pr_and_roc_curves(metrics)


def get_metric(val_labels_and_predictions: LabelsAndPredictions,
               test_labels_and_predictions: LabelsAndPredictions,
               metric: ReportedMetrics,
               optimal_threshold: Optional[float] = None) -> float:
    """
    Given LabelsAndPredictions objects for the validation and test sets, return the specified metric.
    :param val_labels_and_predictions: This set of ground truth labels and model predictions is used to determine the
    optimal threshold for classification.
    :param test_labels_and_predictions: The set of labels and model outputs to calculate metrics for.
    :param metric: The name of the metric to calculate.
    :param optimal_threshold: If provided, use this threshold instead of calculating an optimal threshold.
    """
    if not optimal_threshold:
        fpr, tpr, thresholds = roc_curve(val_labels_and_predictions.labels, val_labels_and_predictions.model_outputs)
        optimal_idx = MetricsDict.get_optimal_idx(fpr=fpr, tpr=tpr)
        optimal_threshold = thresholds[optimal_idx]

    assert optimal_threshold  # for mypy, we have already calculated optimal threshold if it was set to None

    if metric is ReportedMetrics.OptimalThreshold:
        return optimal_threshold

    only_one_class_present = len(set(test_labels_and_predictions.labels)) < 2

    if metric is ReportedMetrics.AUC_ROC:
        return math.nan if only_one_class_present else roc_auc_score(test_labels_and_predictions.labels, test_labels_and_predictions.model_outputs)
    elif metric is ReportedMetrics.AUC_PR:
        if only_one_class_present:
            return math.nan
        precision, recall, _ = precision_recall_curve(test_labels_and_predictions.labels, test_labels_and_predictions.model_outputs)
        return auc(recall, precision)
    elif metric is ReportedMetrics.Accuracy:
        return binary_classification_accuracy(model_output=test_labels_and_predictions.model_outputs,
                                              label=test_labels_and_predictions.labels,
                                              threshold=optimal_threshold)
    elif metric is ReportedMetrics.FalsePositiveRate:
        tnr = recall_score(test_labels_and_predictions.labels, test_labels_and_predictions.model_outputs >= optimal_threshold, pos_label=0)
        return 1 - tnr
    elif metric is ReportedMetrics.FalseNegativeRate:
        return 1 - recall_score(test_labels_and_predictions.labels, test_labels_and_predictions.model_outputs >= optimal_threshold)
    else:
        raise ValueError("Unknown metric")


def get_all_metrics(val_labels_and_predictions: LabelsAndPredictions,
                    test_labels_and_predictions: LabelsAndPredictions,
                    is_thresholded: bool = False) -> Dict[str, float]:
    """
    Given LabelsAndPredictions objects for the validation and test sets, compute some metrics.
    :param val_labels_and_predictions: LabelsAndPredictions object for the val set. This is used to determine the
    optimal threshold for classification.
    :param test_labels_and_predictions: LabelsAndPredictions object for the test set. Metrics are calculated for this
    set.
    :param is_thresholded: Whether the model outputs are binary (they have been thresholded at some point)
                           or are floating point numbers.
    :return:
    """
    optimal_threshold = 0.5 if is_thresholded else None

    metrics = {}
    if not is_thresholded:
        roc_auc = get_metric(val_labels_and_predictions=val_labels_and_predictions,
                             test_labels_and_predictions=test_labels_and_predictions,
                             metric=ReportedMetrics.AUC_ROC)
        metrics["Area under ROC Curve"] = roc_auc

        pr_auc = get_metric(val_labels_and_predictions=val_labels_and_predictions,
                            test_labels_and_predictions=test_labels_and_predictions,
                            metric=ReportedMetrics.AUC_PR)
        metrics["Area under PR Curve"] = pr_auc

        optimal_threshold = get_metric(val_labels_and_predictions=val_labels_and_predictions,
                                       test_labels_and_predictions=test_labels_and_predictions,
                                       metric=ReportedMetrics.OptimalThreshold)
        metrics["Optimal threshold"] = optimal_threshold

    accuracy = get_metric(val_labels_and_predictions=val_labels_and_predictions,
                          test_labels_and_predictions=test_labels_and_predictions,
                          metric=ReportedMetrics.Accuracy,
                          optimal_threshold=optimal_threshold)
    metrics["Accuracy at optimal threshold"] = accuracy

    fpr = get_metric(val_labels_and_predictions=val_labels_and_predictions,
                     test_labels_and_predictions=test_labels_and_predictions,
                     metric=ReportedMetrics.FalsePositiveRate,
                     optimal_threshold=optimal_threshold)
    metrics["Specificity at optimal threshold"] = 1 - fpr

    fnr = get_metric(val_labels_and_predictions=val_labels_and_predictions,
                     test_labels_and_predictions=test_labels_and_predictions,
                     metric=ReportedMetrics.FalseNegativeRate,
                     optimal_threshold=optimal_threshold)
    metrics["Sensitivity at optimal threshold"] = 1 - fnr

    return metrics


def print_metrics(val_labels_and_predictions: LabelsAndPredictions,
                  test_labels_and_predictions: LabelsAndPredictions,
                  is_thresholded: bool = False) -> None:
    """
    Given LabelsAndPredictions objects for the validation and test sets, print out some metrics.
    :param val_labels_and_predictions: LabelsAndPredictions object for the val set. This is used to determine the
    optimal threshold for classification.
    :param test_labels_and_predictions: LabelsAndPredictions object for the test set. Metrics are calculated for this
    set.
    :param is_thresholded: Whether the model outputs are binary (they have been thresholded at some point)
                           or are floating point numbers.
    :return:
    """
    metrics = get_all_metrics(val_labels_and_predictions, test_labels_and_predictions, is_thresholded)
    rows = [[description, f"{value:.4f}"] for description, value in metrics.items()]
    print_table(rows)


def print_metrics_for_all_prediction_targets(val_metrics_csv: Path,
                                             test_metrics_csv: Path,
                                             config: ScalarModelBase,
                                             val_data_split: Optional[ModelExecutionMode] = None,
                                             test_data_split: Optional[ModelExecutionMode] = None,
                                             is_thresholded: bool = False,
                                             is_crossval_report: bool = False) -> None:
    """
    Given csvs written during inference for the validation and test sets, print out metrics for every prediction target
    in the config.

    :param val_metrics_csv: Csv written during inference time for the val set. This is used to determine the
    optimal threshold for classification.
    :param test_metrics_csv: Csv written during inference time for the test set. Metrics are calculated for this csv.
    :param config: Model config
    :param is_thresholded: Whether the model outputs are binary (they have been thresholded at some point)
                           or are floating point numbers.
    """
    def get_metrics_for_fold(prediction_target: str, crossval_split: Optional[int] = None) -> Dict[str, float]:
        val_labels_and_predictions = get_labels_and_predictions(val_metrics_csv, prediction_target,
                                                                crossval_split_index=crossval_split,
                                                                data_split=val_data_split)
        test_labels_and_predictions = get_labels_and_predictions(test_metrics_csv, prediction_target,
                                                                 crossval_split_index=crossval_split,
                                                                 data_split=test_data_split)
        return get_all_metrics(val_labels_and_predictions, test_labels_and_predictions, is_thresholded)

    for prediction_target in config.class_names:
        print_header(f"Class: {prediction_target}", level=3)
        all_metrics = []
        if is_crossval_report:
            header = ["Metric"]
            for crossval_split in range(config.number_of_cross_validation_splits):
                all_metrics.append(get_metrics_for_fold(prediction_target, crossval_split))
                header.append(f"Split {crossval_split}")
        else:
            all_metrics.append(get_metrics_for_fold(prediction_target))
            header = None
        rows = [[metric] + [f"{fold_metrics[metric]:.4f}" for fold_metrics in all_metrics]
                for metric in all_metrics[0]]
        if is_crossval_report:
            for row, metric in zip(rows, all_metrics[0]):
                values = [fold_metrics[metric] for fold_metrics in all_metrics]
                row.append(f"{np.mean(values):.4f} ({np.std(values):.4f})")
            header.append(f"Mean (std)")
        print_table(rows, header)


def get_correct_and_misclassified_examples(val_metrics_csv: Path, test_metrics_csv: Path,
                                           prediction_target: str = "Default") -> Results:
    """
    Given the paths to the metrics files for the validation and test sets, get a list of true positives,
    false positives, false negatives and true negatives.
    The threshold for classification is obtained by looking at the validation file, and applied to the test set to get
    label predictions.
    The validation and test csvs must have at least the following columns (defined in the LoggingColumns enum):
    LoggingColumns.Hue, LoggingColumns.Patient, LoggingColumns.Label, LoggingColumns.ModelOutput.

    """
    df_val = read_csv_and_filter_prediction_target(val_metrics_csv, prediction_target)

    fpr, tpr, thresholds = roc_curve(df_val[LoggingColumns.Label.value], df_val[LoggingColumns.ModelOutput.value])
    optimal_idx = MetricsDict.get_optimal_idx(fpr=fpr, tpr=tpr)
    optimal_threshold = thresholds[optimal_idx]

    df_test = read_csv_and_filter_prediction_target(test_metrics_csv, prediction_target)

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
                                    prediction_target: str = MetricsDict.DEFAULT_HUE_KEY) -> Results:
    """
    Get the top "k" best predictions (i.e. correct classifications where the model was the most certain) and the
    top "k" worst predictions (i.e. misclassifications where the model was the most confident).
    """
    results = get_correct_and_misclassified_examples(val_metrics_csv=val_metrics_csv,
                                                     test_metrics_csv=test_metrics_csv,
                                                     prediction_target=prediction_target)

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


def print_k_best_and_worst_performing(val_metrics_csv: Path, test_metrics_csv: Path, k: int, prediction_target: str) -> None:
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
