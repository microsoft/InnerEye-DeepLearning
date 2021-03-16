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
from InnerEye.ML.reports.classification_report import LabelsAndPredictions


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


def get_dataframe_with_exact_label_matches(csv: Path,
                                           prediction_target_set_to_match: List[str],
                                           all_prediction_targets: List[str],
                                           thresholds_per_prediction_target: List[float]) -> pd.DataFrame:
    """
    Generate a pseudo dataset, which has the ground truth and model predictions for a particular combination of labels.
    The ground truth for a sample is set to True if the set of ground truth labels for the sample corresponds exactly
    with the given list of labels, otherwise it is set to False. Similarly, the model output is set to True if the model
    predicts a value exceeding the label threshold for every label in the given set and for no other labels, and False
    otherwise.
    :param csv: csv with the model predictions (written by the inference pipeline)
    :param prediction_target_set_to_match: A combination of labels to calculate the ground truth and model prediction
    for
    :param all_prediction_targets: The entire set of labels on which the model is trained
    :param thresholds_per_prediction_target: Thresholds per label class to decide if model has predicted True or
    False for the specific label class
    :return: Dataframe with generated ground truth and model outputs per sample
    """

    def get_exact_label_match(df: pd.DataFrame) -> pd.DataFrame:
        df_to_return = df.iloc[0]

        pred_positives = df[df[LoggingColumns.Hue.value].isin(prediction_target_set_to_match)][LoggingColumns.ModelOutput.value].values
        pred_negatives = df[~df[LoggingColumns.Hue.value].isin(prediction_target_set_to_match)][LoggingColumns.ModelOutput.value].values

        if all(pred_positives) and not any(pred_negatives):
            df_to_return[LoggingColumns.ModelOutput.value] = 1
        else:
            df_to_return[LoggingColumns.ModelOutput.value] = 0

        true_positives = df[df[LoggingColumns.Hue.value].isin(prediction_target_set_to_match)][LoggingColumns.Label.value].values
        true_negatives = df[~df[LoggingColumns.Hue.value].isin(prediction_target_set_to_match)][LoggingColumns.Label.value].values

        if all(true_positives) and not any(true_negatives):
            df_to_return[LoggingColumns.Label.value] = 1
        else:
            df_to_return[LoggingColumns.Label.value] = 0

        return df_to_return

    df = pd.read_csv(csv)
    for i in range(len(thresholds_per_prediction_target)):
        df_for_prediction_target = df[LoggingColumns.Hue.value] == all_prediction_targets[i]
        df.loc[df_for_prediction_target, LoggingColumns.ModelOutput.value] = \
            df.loc[df_for_prediction_target, LoggingColumns.ModelOutput.value] > thresholds_per_prediction_target[i]

    df = df.groupby(LoggingColumns.Patient.value).apply(get_exact_label_match)
    df[LoggingColumns.Hue.value] = "|".join(prediction_target_set_to_match)
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
    df = get_dataframe_with_exact_label_matches(csv=csv,
                                                prediction_target_set_to_match=prediction_target_set_to_match,
                                                all_prediction_targets=all_prediction_targets,
                                                thresholds_per_prediction_target=thresholds_per_prediction_target)

    labels = df[LoggingColumns.Label.value].to_numpy()
    model_outputs = df[LoggingColumns.ModelOutput.value].to_numpy()
    subjects = df[LoggingColumns.Patient.value].to_numpy()
    return LabelsAndPredictions(subject_ids=subjects, labels=labels, model_outputs=model_outputs)
