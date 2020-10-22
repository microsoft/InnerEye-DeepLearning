#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from dataclasses import dataclass
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, recall_score
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
from IPython.display import display
from PIL import Image

from InnerEye.ML.reports.notebook_report import print_header
from InnerEye.Common.metrics_dict import MetricsDict, binary_classification_accuracy
from InnerEye.ML.utils.metrics_constants import LoggingColumns
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


def get_results(csv: Path) -> LabelsAndPredictions:
    """
    Given a CSV file, reads the subject IDs, ground truth labels and model outputs for each subject.
    NOTE: This CSV file should have results from a single epoch, as in the metrics files written during inference, not
    like the ones written while training.
    """
    df = pd.read_csv(csv)
    labels = df[LoggingColumns.Label.value]
    model_outputs = df[LoggingColumns.ModelOutput.value]
    subjects = df[LoggingColumns.Patient.value]
    if not subjects.is_unique:
        raise ValueError(f"Subject IDs should be unique, but found duplicate entries "
                         f"in column {LoggingColumns.Patient.value} in the csv file.")
    return LabelsAndPredictions(subject_ids=subjects, labels=labels, model_outputs=model_outputs)


def plot_auc(x_values: np.ndarray, y_values: np.ndarray, title: str, ax: Axes, print_coords: bool = False) -> None:
    """
    Plot a curve given the x and y values of each point.
    :param x_values: x coordinate of each data point to be plotted
    :param y_values: y coordinate of each data point to be plotted
    :param title: Title of the plot
    :param ax: matplotlib.axes.Axes object for plotting
    :param print_coords: If true, prints out the coordinates of each point on the graph.
    """
    ax.plot(x_values, y_values)
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)
    ax.set_title(title)

    if print_coords:
        # write values of points
        for x, y in zip(x_values, y_values):
            ax.annotate(f"{x:0.3f}, {y:0.3f}", xy=(x, y), xytext=(15, 0), textcoords='offset points')


def plot_pr_and_roc_curves_from_csv(metrics_csv: Path) -> None:
    """
    Given a csv file, read the predicted values and ground truth labels and plot the ROC and PR curves.
    """
    print_header("ROC and PR curves", level=3)
    results = get_results(metrics_csv)

    _, ax = plt.subplots(1, 2)

    fpr, tpr, thresholds = roc_curve(results.labels, results.model_outputs)
    plot_auc(fpr, tpr, "ROC Curve", ax[0])
    precision, recall, thresholds = precision_recall_curve(results.labels, results.model_outputs)
    plot_auc(recall, precision, "PR Curve", ax[1])

    plt.show()


def get_metric(val_metrics_csv: Path, test_metrics_csv: Path, metric: ReportedMetrics) -> float:
    """
    Given a csv file, read the predicted values and ground truth labels and return the specified metric.
    """
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
    elif metric is ReportedMetrics.Accuracy:
        return binary_classification_accuracy(model_output=results_test.model_outputs,
                                              label=results_test.labels,
                                              threshold=optimal_threshold)
    elif metric is ReportedMetrics.FalsePositiveRate:
        tnr = recall_score(results_test.labels, results_test.model_outputs >= optimal_threshold, pos_label=0)
        return 1 - tnr
    elif metric is ReportedMetrics.FalseNegativeRate:
        return 1 - recall_score(results_test.labels, results_test.model_outputs >= optimal_threshold)
    else:
        raise ValueError("Unknown metric")


def print_metrics(val_metrics_csv: Path, test_metrics_csv: Path) -> None:
    """
    Given a csv file, read the predicted values and ground truth labels and print out some metrics.
    """
    roc_auc = get_metric(val_metrics_csv=val_metrics_csv,
                         test_metrics_csv=test_metrics_csv,
                         metric=ReportedMetrics.AUC_ROC)
    print_header(f"Area under ROC Curve: {roc_auc:.4f}", level=4)

    pr_auc = get_metric(val_metrics_csv=val_metrics_csv,
                        test_metrics_csv=test_metrics_csv,
                        metric=ReportedMetrics.AUC_PR)
    print_header(f"Area under PR Curve: {pr_auc:.4f}", level=4)

    optimal_threshold = get_metric(val_metrics_csv=val_metrics_csv,
                                   test_metrics_csv=test_metrics_csv,
                                   metric=ReportedMetrics.OptimalThreshold)

    print_header(f"Optimal threshold: {optimal_threshold: .4f}", level=4)

    accuracy = get_metric(val_metrics_csv=val_metrics_csv,
                          test_metrics_csv=test_metrics_csv,
                          metric=ReportedMetrics.Accuracy)
    print_header(f"Accuracy at optimal threshold: {accuracy:.4f}", level=4)

    fpr = get_metric(val_metrics_csv=val_metrics_csv,
                     test_metrics_csv=test_metrics_csv,
                     metric=ReportedMetrics.FalsePositiveRate)
    print_header(f"Specificity at optimal threshold: {1-fpr:.4f}", level=4)

    fnr = get_metric(val_metrics_csv=val_metrics_csv,
                     test_metrics_csv=test_metrics_csv,
                     metric=ReportedMetrics.FalseNegativeRate)
    print_header(f"Sensitivity at optimal threshold: {1-fnr:.4f}", level=4)


def get_correct_and_misclassified_examples(val_metrics_csv: Path, test_metrics_csv: Path) -> Results:
    """
    Given the paths to the metrics files for the validation and test sets, get a list of true positives,
    false positives, false negatives and true negatives.
    The threshold for classification is obtained by looking at the validation file, and applied to the test set to get
    label predictions.
    """
    df_val = pd.read_csv(val_metrics_csv)

    if not df_val[LoggingColumns.Patient.value].is_unique:
        raise ValueError(f"Subject IDs should be unique, but found duplicate entries "
                         f"in column {LoggingColumns.Patient.value} in the csv file.")

    fpr, tpr, thresholds = roc_curve(df_val[LoggingColumns.Label.value], df_val[LoggingColumns.ModelOutput.value])
    optimal_idx = MetricsDict.get_optimal_idx(fpr=fpr, tpr=tpr)
    optimal_threshold = thresholds[optimal_idx]

    df_test = pd.read_csv(test_metrics_csv)

    if not df_test[LoggingColumns.Patient.value].is_unique:
        raise ValueError(f"Subject IDs should be unique, but found duplicate entries "
                         f"in column {LoggingColumns.Patient.value} in the csv file.")

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


def get_k_best_and_worst_performing(val_metrics_csv: Path, test_metrics_csv: Path, k: int) -> Results:
    """
    Get the top "k" best predictions (i.e. correct classifications where the model was the most certain) and the
    top "k" worst predictions (i.e. misclassifications where the model was the most confident).
    """
    results = get_correct_and_misclassified_examples(val_metrics_csv=val_metrics_csv,
                                                     test_metrics_csv=test_metrics_csv)

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


def print_k_best_and_worst_performing(val_metrics_csv: Path, test_metrics_csv: Path, k: int) -> None:
    """
    Print the top "k" best predictions (i.e. correct classifications where the model was the most certain) and the
    top "k" worst predictions (i.e. misclassifications where the model was the most confident).
    """
    results = get_k_best_and_worst_performing(val_metrics_csv=val_metrics_csv,
                                              test_metrics_csv=test_metrics_csv,
                                              k=k)

    print_header(f"Top {k} false positives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.false_positives[LoggingColumns.Patient.value],
                                                        results.false_positives[LoggingColumns.ModelOutput.value])):
        print_header(f"{index+1}. ID {subject} Score: {model_output:.5f}", level=4)

    print_header(f"Top {k} false negatives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.false_negatives[LoggingColumns.Patient.value],
                                                        results.false_negatives[LoggingColumns.ModelOutput.value])):
        print_header(f"{index+1}. ID {subject} Score: {model_output:.5f}", level=4)

    print_header(f"Top {k} true positives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.true_positives[LoggingColumns.Patient.value],
                                                        results.true_positives[LoggingColumns.ModelOutput.value])):
        print_header(f"{index+1}. ID {subject} Score: {model_output:.5f}", level=4)

    print_header(f"Top {k} true negatives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.true_negatives[LoggingColumns.Patient.value],
                                                        results.true_negatives[LoggingColumns.ModelOutput.value])):
        print_header(f"{index+1}. ID {subject} Score: {model_output:.5f}", level=4)


def get_image_filepath_from_subject_id(subject_id: str,
                                       dataset_df: pd.DataFrame,
                                       dataset_subject_column: str,
                                       dataset_file_column: str,
                                       dataset_dir: Path) -> Optional[Path]:
    """
    Returns the filepath for the image associated with a subject. If the subject is not found, return None.
    If the csv contains multiple entries per subject (which may happen if the csv uses the channels column) then
    return None as we do not support these csv types yet.
    :param subject_id: Subject to retrive image for
    :param dataset_df: Dataset dataframe (from the datset.csv file)
    :param dataset_subject_column: Name of the column with the subject IDs
    :param dataset_file_column: Name of the column with the image filepaths
    :param dataset_dir: Path to the dataset
    :return: path to the image file for the patient or None if it is not found.
    """

    if not dataset_df[dataset_subject_column].is_unique:
        return None

    dataset_df[dataset_subject_column] = dataset_df.apply(lambda x: str(x[dataset_subject_column]), axis=1)

    if subject_id not in dataset_df[dataset_subject_column].unique():
        return None

    filtered = dataset_df[dataset_df[dataset_subject_column] == subject_id]
    filepath = filtered.iloc[0][dataset_file_column]
    return dataset_dir / Path(filepath)


def plot_image_from_filepath(filepath: Path, im_size: Tuple) -> bool:
    """
    Plots a 2D image given the filepath. Returns false if the image could not be plotted (for example, if it was 3D).
    :param filepath: Path to image
    :param im_size: Display size for image
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
    display(Image.fromarray(image).resize(im_size))
    return True


def plot_image_for_subject(subject_id: str,
                           dataset_df: pd.DataFrame,
                           dataset_subject_column: str,
                           dataset_file_column: str,
                           dataset_dir: Path,
                           im_size: Tuple,
                           model_output: float) -> None:
    """
    Given a subject ID, plots the corresponding image.
    :param subject_id: Subject to plot image for
    :param dataset_df: Dataset dataframe (from the datset.csv file)
    :param dataset_subject_column: Name of the column with the subject IDs
    :param dataset_file_column: Name of the column with the image filepaths
    :param dataset_dir: Path to the dataset
    :param im_size: Display size for image
    :param model_output: The predicted value for this image
    """
    print_header("", level=4)
    print_header(f"ID: {subject_id} Score: {model_output}", level=4)

    filepath = get_image_filepath_from_subject_id(subject_id=str(subject_id),
                                                  dataset_df=dataset_df,
                                                  dataset_subject_column=dataset_subject_column,
                                                  dataset_file_column=dataset_file_column,
                                                  dataset_dir=dataset_dir)
    if not filepath:
        print_header(f"Subject ID {subject_id} not found, or found duplicate entries for this subject "
                     f"in column {dataset_subject_column} in the csv file. "
                     f"Note: Reports with datasets that use channel columns in the dataset.csv "
                     f"are not yet supported.")
        return

    success = plot_image_from_filepath(filepath, im_size=im_size)
    if not success:
        print_header("Unable to plot image: image must be 2D with shape [w, h] or [1, w, h].", level=0)


def plot_k_best_and_worst_performing(val_metrics_csv: Path, test_metrics_csv: Path, k: int, dataset_csv_path: Path,
                                     dataset_subject_column: str, dataset_file_column: str) -> None:
    """
    Plot images for the top "k" best predictions (i.e. correct classifications where the model was the most certain)
    and the top "k" worst predictions (i.e. misclassifications where the model was the most confident).
    """
    results = get_k_best_and_worst_performing(val_metrics_csv=val_metrics_csv,
                                              test_metrics_csv=test_metrics_csv,
                                              k=k)

    dataset_df = pd.read_csv(dataset_csv_path)
    dataset_dir = dataset_csv_path.parent

    im_size = (800, 800)

    print_header("", level=2)
    print_header(f"Top {k} false positives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.false_positives[LoggingColumns.Patient.value],
                                                        results.false_positives[LoggingColumns.ModelOutput.value])):
        plot_image_for_subject(subject_id=str(subject),
                               dataset_df=dataset_df,
                               dataset_subject_column=dataset_subject_column,
                               dataset_file_column=dataset_file_column,
                               dataset_dir=dataset_dir,
                               im_size=im_size,
                               model_output=model_output)

    print_header(f"Top {k} false negatives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.false_negatives[LoggingColumns.Patient.value],
                                                        results.false_negatives[LoggingColumns.ModelOutput.value])):
        plot_image_for_subject(subject_id=str(subject),
                               dataset_df=dataset_df,
                               dataset_subject_column=dataset_subject_column,
                               dataset_file_column=dataset_file_column,
                               dataset_dir=dataset_dir,
                               im_size=im_size,
                               model_output=model_output)

    print_header(f"Top {k} true positives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.true_positives[LoggingColumns.Patient.value],
                                                        results.true_positives[LoggingColumns.ModelOutput.value])):
        plot_image_for_subject(subject_id=str(subject),
                               dataset_df=dataset_df,
                               dataset_subject_column=dataset_subject_column,
                               dataset_file_column=dataset_file_column,
                               dataset_dir=dataset_dir,
                               im_size=im_size,
                               model_output=model_output)

    print_header(f"Top {k} true negatives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.true_negatives[LoggingColumns.Patient.value],
                                                        results.true_negatives[LoggingColumns.ModelOutput.value])):
        plot_image_for_subject(subject_id=str(subject),
                               dataset_df=dataset_df,
                               dataset_subject_column=dataset_subject_column,
                               dataset_file_column=dataset_file_column,
                               dataset_dir=dataset_dir,
                               im_size=im_size,
                               model_output=model_output)
