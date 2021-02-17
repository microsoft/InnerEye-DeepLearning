#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image
from matplotlib.axes import Axes
from sklearn.metrics import auc, precision_recall_curve, recall_score, roc_auc_score, roc_curve

from InnerEye.Common.metrics_constants import LoggingColumns
from InnerEye.ML.metrics_dict import MetricsDict, binary_classification_accuracy
from InnerEye.ML.reports.notebook_report import print_header
from InnerEye.ML.utils.io_util import load_image_in_known_formats
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.dataset.scalar_dataset import extract_label_classification


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


def read_csv_and_filter_hue(csv: Path, hue: str):
    df = pd.read_csv(csv)
    df = df[df[LoggingColumns.Hue.value] == hue]  # Filter Hue
    if not df[LoggingColumns.Patient.value].is_unique:
        raise ValueError(f"Subject IDs should be unique, but found duplicate entries "
                         f"in column {LoggingColumns.Patient.value} in the csv file.")
    return df


def generate_psuedo_labels(csv: Path,
                           hues: List[str],
                           all_hues: List[str],
                           per_class_thresholds: List[float]) -> pd.DataFrame:

    def get_pseudo_label(df: pd.DataFrame):
        df_to_return = df.iloc[0]

        pred_positives = df[df[LoggingColumns.Hue.value].isin(hues)][LoggingColumns.ModelOutput.value].values
        pred_negatives = df[~df[LoggingColumns.Hue.value].isin(hues)][LoggingColumns.ModelOutput.value].values

        if all(pred_positives) and not any(pred_negatives):
            df_to_return[LoggingColumns.ModelOutput.value] = 1
        else:
            df_to_return[LoggingColumns.ModelOutput.value] = 0

        true_positives = df[df[LoggingColumns.Hue.value].isin(hues)][LoggingColumns.Label.value].values
        true_negatives = df[~df[LoggingColumns.Hue.value].isin(hues)][LoggingColumns.Label.value].values

        if all(true_positives) and not any(true_negatives):
            df_to_return[LoggingColumns.Label.value] = 1
        else:
            df_to_return[LoggingColumns.Label.value] = 0

        return df_to_return

    if not hues:
        print_header("Empty hue list", level=0)

    df = pd.read_csv(csv)
    for i in range(len(per_class_thresholds)):
        hue_rows = df[LoggingColumns.Hue.value] == all_hues[i]
        df.loc[hue_rows, LoggingColumns.ModelOutput.value] = \
            df.loc[hue_rows, LoggingColumns.ModelOutput.value] > per_class_thresholds[i]

    df = df.groupby(LoggingColumns.Patient.value).apply(get_pseudo_label)
    df[LoggingColumns.Hue.value] = "|".join(hues)
    return df


def get_unique_label_combinations(dataset_csv: Path, config: ScalarModelBase) -> List[List[str]]:
    df = pd.read_csv(dataset_csv)
    labels = list(set(df[config.label_value_column].values))

    labels = [[config.class_names[int(i)]
                for i in np.array(extract_label_classification(label_string=label,
                                                               sample_id="",
                                                               num_classes=len(config.class_names),
                                                               labels_exclusive=config.labels_exclusive,
                                                               is_classification_dataset=config.is_classification_model))
                                                               .nonzero()[0]]
              for label in labels if not isinstance(label, float) or not math.isnan(label)]
    return labels


def get_results(csv: Path, hues: List[str],
                all_hues: Optional[List[str]] = None,
                thresholds: Optional[List[float]] = None,
                exclusive: bool = False) -> LabelsAndPredictions:
    """
    Given a CSV file, reads the subject IDs, ground truth labels and model outputs for each subject.
    NOTE: This CSV file should have results from a single epoch, as in the metrics files written during inference, not
    like the ones written while training.
    """
    if len(hues) == 1 and not exclusive:
        df = read_csv_and_filter_hue(csv, hues[0])
    else:
        df = generate_psuedo_labels(csv=csv, hues=hues, all_hues=all_hues, per_class_thresholds=thresholds)

    labels = df[LoggingColumns.Label.value].to_numpy()
    model_outputs = df[LoggingColumns.ModelOutput.value].to_numpy()
    subjects = df[LoggingColumns.Patient.value].to_numpy()
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


def plot_pr_and_roc_curves_from_csv(metrics_csv: Path, hue: str) -> None:
    """
    Given a csv file, read the predicted values and ground truth labels and plot the ROC and PR curves.
    """
    print_header("ROC and PR curves", level=3)
    results = get_results(metrics_csv, [hue])

    _, ax = plt.subplots(1, 2)

    fpr, tpr, thresholds = roc_curve(results.labels, results.model_outputs)

    plot_auc(fpr, tpr, "ROC Curve", ax[0])
    precision, recall, thresholds = precision_recall_curve(results.labels, results.model_outputs)
    plot_auc(recall, precision, "PR Curve", ax[1])

    plt.show()


def get_metric(val_metrics_csv: Path,
               test_metrics_csv: Path,
               metric: ReportedMetrics,
               hues: List[str],
               all_hues: Optional[List[str]] = None,
               thresholds: Optional[List[float]] = None,
               exclusive: bool = False) -> float:
    """
    Given a csv file, read the predicted values and ground truth labels and return the specified metric.
    """
    if len(hues) == 1 and not exclusive:
        results_val = get_results(val_metrics_csv, hues, all_hues, thresholds, exclusive)
        fpr, tpr, thresholds = roc_curve(results_val.labels, results_val.model_outputs)
        optimal_idx = MetricsDict.get_optimal_idx(fpr=fpr, tpr=tpr)
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 0.5

    if metric is ReportedMetrics.OptimalThreshold:
        return optimal_threshold

    results_test = get_results(test_metrics_csv, hues, all_hues, thresholds, exclusive)
    only_one_class_present = len(set(results_test.labels)) < 2

    if metric is ReportedMetrics.AUC_ROC:
        return math.nan if only_one_class_present else roc_auc_score(results_test.labels, results_test.model_outputs)
    elif metric is ReportedMetrics.AUC_PR:
        if only_one_class_present:
            return math.nan
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


def print_metrics(val_metrics_csv: Path, test_metrics_csv: Path,
                  hues: List[str],
                  all_hues,
                  thresholds,
                  exclusive: bool = False) -> None:
    """
    Given a csv file, read the predicted values and ground truth labels and print out some metrics.
    """
    if not exclusive and len(hues) == 1:
        roc_auc = get_metric(val_metrics_csv=val_metrics_csv,
                             test_metrics_csv=test_metrics_csv,
                             metric=ReportedMetrics.AUC_ROC,
                             hues=hues,
                             all_hues=all_hues,
                             thresholds=thresholds,
                             exclusive=exclusive)
        print_header(f"Area under ROC Curve: {roc_auc:.4f}", level=4)

        pr_auc = get_metric(val_metrics_csv=val_metrics_csv,
                            test_metrics_csv=test_metrics_csv,
                            metric=ReportedMetrics.AUC_PR,
                            hues=hues,
                            all_hues=all_hues,
                            thresholds=thresholds,
                            exclusive=exclusive)
        print_header(f"Area under PR Curve: {pr_auc:.4f}", level=4)

        optimal_threshold = get_metric(val_metrics_csv=val_metrics_csv,
                                       test_metrics_csv=test_metrics_csv,
                                       metric=ReportedMetrics.OptimalThreshold,
                                       hues=hues,
                                       all_hues=all_hues,
                                       thresholds=thresholds,
                                       exclusive=exclusive)

        print_header(f"Optimal threshold: {optimal_threshold: .4f}", level=4)

    accuracy = get_metric(val_metrics_csv=val_metrics_csv,
                          test_metrics_csv=test_metrics_csv,
                          metric=ReportedMetrics.Accuracy,
                          hues=hues,
                          all_hues=all_hues,
                          thresholds=thresholds,
                          exclusive=exclusive)
    print_header(f"Accuracy at optimal threshold: {accuracy:.4f}", level=4)

    fpr = get_metric(val_metrics_csv=val_metrics_csv,
                     test_metrics_csv=test_metrics_csv,
                     metric=ReportedMetrics.FalsePositiveRate,
                     hues=hues,
                     all_hues=all_hues,
                     thresholds=thresholds,
                     exclusive=exclusive)
    print_header(f"Specificity at optimal threshold: {1 - fpr:.4f}", level=4)

    fnr = get_metric(val_metrics_csv=val_metrics_csv,
                     test_metrics_csv=test_metrics_csv,
                     metric=ReportedMetrics.FalseNegativeRate,
                     hues=hues,
                     all_hues=all_hues,
                     thresholds=thresholds,
                     exclusive=exclusive)
    print_header(f"Sensitivity at optimal threshold: {1 - fnr:.4f}", level=4)
    print_header("", level=4)


def get_correct_and_misclassified_examples(val_metrics_csv: Path, test_metrics_csv: Path,
                                           hue: str = "Default") -> Results:
    """
    Given the paths to the metrics files for the validation and test sets, get a list of true positives,
    false positives, false negatives and true negatives.
    The threshold for classification is obtained by looking at the validation file, and applied to the test set to get
    label predictions.
    """
    df_val = read_csv_and_filter_hue(val_metrics_csv, hue)

    fpr, tpr, thresholds = roc_curve(df_val[LoggingColumns.Label.value], df_val[LoggingColumns.ModelOutput.value])
    optimal_idx = MetricsDict.get_optimal_idx(fpr=fpr, tpr=tpr)
    optimal_threshold = thresholds[optimal_idx]

    df_test = read_csv_and_filter_hue(test_metrics_csv, hue)

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
                                    hue: str = "Default") -> Results:
    """
    Get the top "k" best predictions (i.e. correct classifications where the model was the most certain) and the
    top "k" worst predictions (i.e. misclassifications where the model was the most confident).
    """
    results = get_correct_and_misclassified_examples(val_metrics_csv=val_metrics_csv,
                                                     test_metrics_csv=test_metrics_csv,
                                                     hue=hue)

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


def print_k_best_and_worst_performing(val_metrics_csv: Path, test_metrics_csv: Path, k: int, hue: str) -> None:
    """
    Print the top "k" best predictions (i.e. correct classifications where the model was the most certain) and the
    top "k" worst predictions (i.e. misclassifications where the model was the most confident).
    """
    results = get_k_best_and_worst_performing(val_metrics_csv=val_metrics_csv,
                                              test_metrics_csv=test_metrics_csv,
                                              k=k,
                                              hue=hue)

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
                                       dataset_df: pd.DataFrame,
                                       config: ScalarModelBase,
                                       dataset_dir: Path) -> List[Path]:
    """
    Returns the filepath for the image associated with a subject. If the subject is not found, return None.
    If the csv contains multiple entries per subject (which may happen if the csv uses the channels column) then
    return None as we do not support these csv types yet.
    :param subject_id: Subject to retrive image for
    :param dataset_df: Dataset dataframe (from the datset.csv file)
    :param dataset_dir: Path to the dataset
    :return: path to the image file for the patient or None if it is not found.
    """

    dataset_df[config.subject_column] = dataset_df.apply(lambda x: str(x[config.subject_column]), axis=1)
    filtered = dataset_df[dataset_df[config.subject_column] == subject_id]
    filepaths = filtered[config.image_file_column].values

    return [dataset_dir / Path(filepath) for filepath in filepaths]


def get_image_labels_from_subject_id(subject_id: str,
                                    dataset_df: pd.DataFrame,
                                    config: ScalarModelBase) -> List[str]:

    dataset_df[config.subject_column] = dataset_df.apply(lambda x: str(x[config.subject_column]), axis=1)
    filtered = dataset_df[dataset_df[config.subject_column] == subject_id]
    labels = list(set(filtered[config.label_value_column].values))

    if len(labels) > 1:
        raise ValueError(f"Labels inconsistent, expect the same set of labels for all hues per subject. "
                         f"Got labels {labels}")

    labels = np.array(extract_label_classification(label_string=labels[0],
                                                   sample_id=subject_id,
                                                   num_classes=len(config.class_names),
                                                   labels_exclusive=config.labels_exclusive,
                                                   is_classification_dataset=config.is_classification_model)
                      ).nonzero()[0]

    return [config.class_names[int(label)] for label in labels
            if not isinstance(labels, float) or not math.isnan(label)]


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
                           dataset_df: pd.DataFrame,
                           dataset_dir: Path,
                           im_width: int,
                           model_output: float,
                           header: Optional[str],
                           config: ScalarModelBase) -> None:
    """
    Given a subject ID, plots the corresponding image.
    :param subject_id: Subject to plot image for
    :param dataset_df: Dataset dataframe (from the datset.csv file)
    :param dataset_dir: Path to the dataset
    :param im_width: Display width for image
    :param model_output: The predicted value for this image
    :param header: Optional header printed along with the subject ID and score for the image.
    """
    print_header("", level=4)
    if header:
        print_header(header, level=4)

    labels = get_image_labels_from_subject_id(subject_id=subject_id,
                                              dataset_df=dataset_df,
                                              config=config)

    print_header(f"ID: {subject_id} Score: {model_output}", level=4)
    print_header(f"True labels: {', '.join(labels)}", level=4)

    filepaths = get_image_filepath_from_subject_id(subject_id=str(subject_id),
                                                   dataset_df=dataset_df,
                                                   config=config,
                                                   dataset_dir=dataset_dir)

    if not filepaths:
        print_header(f"Subject ID {subject_id} not found."
                     f"Note: Reports with datasets that use channel columns in the dataset.csv "
                     f"are not yet supported.", level=0)
        return

    for filepath in filepaths:
        success = plot_image_from_filepath(filepath, im_width=im_width)
        if not success:
            print_header("Unable to plot image: image must be 2D with shape [w, h] or [1, w, h].", level=0)


def plot_k_best_and_worst_performing(val_metrics_csv: Path, test_metrics_csv: Path, k: int, dataset_csv_path: Path,
                                    hue: str, config: ScalarModelBase) -> None:
    """
    Plot images for the top "k" best predictions (i.e. correct classifications where the model was the most certain)
    and the top "k" worst predictions (i.e. misclassifications where the model was the most confident).
    """
    results = get_k_best_and_worst_performing(val_metrics_csv=val_metrics_csv,
                                              test_metrics_csv=test_metrics_csv,
                                              k=k,
                                              hue=hue)

    dataset_df = pd.read_csv(dataset_csv_path)
    dataset_dir = dataset_csv_path.parent

    im_width = 800

    print_header("", level=2)
    print_header(f"Top {k} false positives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.false_positives[LoggingColumns.Patient.value],
                                                        results.false_positives[LoggingColumns.ModelOutput.value])):
        plot_image_for_subject(subject_id=str(subject),
                               dataset_df=dataset_df,
                               dataset_dir=dataset_dir,
                               im_width=im_width,
                               model_output=model_output,
                               header="False Positive",
                               config=config)

    print_header(f"Top {k} false negatives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.false_negatives[LoggingColumns.Patient.value],
                                                        results.false_negatives[LoggingColumns.ModelOutput.value])):
        plot_image_for_subject(subject_id=str(subject),
                               dataset_df=dataset_df,
                               dataset_dir=dataset_dir,
                               im_width=im_width,
                               model_output=model_output,
                               header="False Negative",
                               config=config)

    print_header(f"Top {k} true positives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.true_positives[LoggingColumns.Patient.value],
                                                        results.true_positives[LoggingColumns.ModelOutput.value])):
        plot_image_for_subject(subject_id=str(subject),
                               dataset_df=dataset_df,
                               dataset_dir=dataset_dir,
                               im_width=im_width,
                               model_output=model_output,
                               header="True Positive",
                               config=config)

    print_header(f"Top {k} true negatives", level=2)
    for index, (subject, model_output) in enumerate(zip(results.true_negatives[LoggingColumns.Patient.value],
                                                        results.true_negatives[LoggingColumns.ModelOutput.value])):
        plot_image_for_subject(subject_id=str(subject),
                               dataset_df=dataset_df,
                               dataset_dir=dataset_dir,
                               im_width=im_width,
                               model_output=model_output,
                               header="True Negative",
                               config=config)
