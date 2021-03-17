#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import List, Set, FrozenSet

import pandas as pd
import torch
import math

from InnerEye.Common.metrics_constants import LoggingColumns
from InnerEye.ML.dataset.scalar_dataset import ScalarDataset
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.reports.classification_report import LabelsAndPredictions, print_metrics, get_labels_and_predictions, \
    get_metric, ReportedMetrics
from InnerEye.ML.reports.notebook_report import print_header


def get_unique_prediction_target_combinations(config: ScalarModelBase) -> Set[FrozenSet[str]]:
    """
    Get a list of all the combinations of labels that exist in the dataset.

    For multilabel classification tasks, this function will return all unique combinations of labels that
    occur in the dataset csv.
    For example, if there are 6 samples in the dataset with the following ground truth labels
    Sample1: class1, class2
    Sample2: class0
    Sample3: class1
    Sample4: class2, class3
    Sample5: (all label classes are negative in Sample 5)
    Sample6: class1, class2
    This function will return {{"class1", "class2"}, {"class0"}, {"class1"},  {"class2", "class3"}, {}}

    For binary classification tasks (assume class_names has not been changed from ["Default"]):
    This function will return a set with two members - {{"Default"}, {}} if there is at least one positive example
    in the dataset. If there are no positive examples, it returns {{}}.
    """
    df = config.read_dataset_if_needed()
    dataset = ScalarDataset(args=config, data_frame=df)

    all_labels = [torch.flatten(torch.nonzero(item.label)).tolist() for item in dataset.items]
    label_set = set(frozenset([config.class_names[i] for i in labels if not math.isnan(i)])
                    for labels in all_labels)

    return label_set


def get_dataframe_with_exact_label_matches(metrics_df: pd.DataFrame,
                                           prediction_target_set_to_match: List[str],
                                           all_prediction_targets: List[str],
                                           thresholds_per_prediction_target: List[float]) -> pd.DataFrame:
    """
    Given a set of prediction targets, for each sample find
        (i) if the set of ground truth labels matches this set exactly,
        (ii) if the predicted model outputs (after thresholding) match this set exactly

    Generates an output dataframe with the rows:
    LoggingColumns.Patient, LoggingColumns.Label, LoggingColumns.ModelOutput, LoggingColumns.Hue

    The output dataframe is generated according to the following rules:
      - LoggingColumns.Patient: For each sample, the sample id is copied over into this field
      - LoggingColumns.Label: For each sample, this field is set to 1 if the set of ground truth labels for the sample
        correspond exactly with the given set of prediction targets, otherwise it is set to 0.
      - LoggingColumns.ModelOutput: For each sample, this field is set to 1 if the model predicts a value exceeding the
        prediction target threshold for every prediction target in the given set and lower for all other prediction
        targets. It is set to 0 otherwise.
      - LoggingColumns.Hue: For every sample, this is set to "|".join(prediction_target_set_to_match)

    :param metrics_df: Dataframe with the model predictions (read from the csv written by the inference pipeline)
                       The dataframe must have at least the following columns (defined in the LoggingColumns enum):
                       LoggingColumns.Hue, LoggingColumns.Patient, LoggingColumns.Label, LoggingColumns.ModelOutput.
                       Any other columns will be ignored.
    :param prediction_target_set_to_match: The set of prediction targets to which each sample is compared
    :param all_prediction_targets: The entire set of prediction targets on which the model is trained
    :param thresholds_per_prediction_target: Thresholds per prediction target to decide if model has predicted True or
                                             False for the specific prediction target
    :return: Dataframe with generated label and model outputs per sample
    """

    def get_exact_label_match(df: pd.DataFrame) -> pd.DataFrame:
        values_to_return = {LoggingColumns.Patient.value: [df.iloc[0][LoggingColumns.Patient.value]]}

        pred_positives = df[df[LoggingColumns.Hue.value].isin(prediction_target_set_to_match)][LoggingColumns.ModelOutput.value].values
        pred_negatives = df[~df[LoggingColumns.Hue.value].isin(prediction_target_set_to_match)][LoggingColumns.ModelOutput.value].values

        if all(pred_positives) and not any(pred_negatives):
            values_to_return[LoggingColumns.ModelOutput.value] = [1]
        else:
            values_to_return[LoggingColumns.ModelOutput.value] = [0]

        true_positives = df[df[LoggingColumns.Hue.value].isin(prediction_target_set_to_match)][LoggingColumns.Label.value].values
        true_negatives = df[~df[LoggingColumns.Hue.value].isin(prediction_target_set_to_match)][LoggingColumns.Label.value].values

        if all(true_positives) and not any(true_negatives):
            values_to_return[LoggingColumns.Label.value] = [1]
        else:
            values_to_return[LoggingColumns.Label.value] = [0]

        return pd.DataFrame.from_dict(values_to_return)

    df = metrics_df.copy()
    for i in range(len(thresholds_per_prediction_target)):
        df_for_prediction_target = df[LoggingColumns.Hue.value] == all_prediction_targets[i]
        df.loc[df_for_prediction_target, LoggingColumns.ModelOutput.value] = \
            df.loc[df_for_prediction_target, LoggingColumns.ModelOutput.value] > thresholds_per_prediction_target[i]

    df = df.groupby(LoggingColumns.Patient.value, as_index=False).apply(get_exact_label_match).reset_index(drop=True)
    df[LoggingColumns.Hue.value] = ["|".join(prediction_target_set_to_match)] * len(df)
    return df


def get_labels_and_predictions_for_prediction_target_set(csv: Path,
                                                         prediction_target_set_to_match: List[str],
                                                         all_prediction_targets: List[str],
                                                         thresholds_per_prediction_target: List[float]) -> LabelsAndPredictions:
    """
    Given a CSV file, generate a set of labels and model predictions for the combination of prediction targets.
    NOTE: This CSV file should have results from a single epoch, as in the metrics files written during inference, not
    like the ones written while training.
    """
    metrics_df = pd.read_csv(csv)
    df = get_dataframe_with_exact_label_matches(metrics_df=metrics_df,
                                                prediction_target_set_to_match=prediction_target_set_to_match,
                                                all_prediction_targets=all_prediction_targets,
                                                thresholds_per_prediction_target=thresholds_per_prediction_target)

    labels = df[LoggingColumns.Label.value].to_numpy()
    model_outputs = df[LoggingColumns.ModelOutput.value].to_numpy()
    subjects = df[LoggingColumns.Patient.value].to_numpy()
    return LabelsAndPredictions(subject_ids=subjects, labels=labels, model_outputs=model_outputs)


def print_metrics_for_thresholded_output_for_all_prediction_targets(val_metrics_csv: Path,
                                                                    test_metrics_csv: Path,
                                                                    config: ScalarModelBase) -> None:

    """
    Given csvs written during inference for the validation and test sets, print out metrics for every combination of
    prediction targets that exist in the dataset.

    :param val_metrics_csv: Csv written during inference time for the val set. This is used to determine the
    optimal threshold for classification.
    :param test_metrics_csv: Csv written during inference time for the test set. Metrics are calculated for this csv.
    :param config: Model config
    """

    unique_prediction_target_combinations = get_unique_prediction_target_combinations(config)
    all_prediction_target_combinations = list(set([(prediction_target,) for prediction_target in config.class_names])
                                              | set(map(tuple, unique_prediction_target_combinations)))

    thresholds_per_prediction_target = []
    for label in config.class_names:
        val_metrics = get_labels_and_predictions(val_metrics_csv, label)
        test_metrics = get_labels_and_predictions(test_metrics_csv, label)
        thresholds_per_prediction_target.append(get_metric(val_labels_and_predictions=val_metrics,
                                                test_labels_and_predictions=test_metrics,
                                                metric=ReportedMetrics.OptimalThreshold))

    for labels in all_prediction_target_combinations:
        print_header(f"Class {'|'.join(labels) or 'Negative'}", level=3)
        val_metrics = get_labels_and_predictions_for_prediction_target_set(val_metrics_csv, labels, config.class_names, thresholds_per_prediction_target)
        test_metrics = get_labels_and_predictions_for_prediction_target_set(test_metrics_csv, labels, config.class_names, thresholds_per_prediction_target)
        print_metrics(val_labels_and_predictions=val_metrics, test_labels_and_predictions=test_metrics, is_thresholded=True)
