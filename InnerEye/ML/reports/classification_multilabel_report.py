#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from InnerEye.Common.metrics_constants import LoggingColumns
from InnerEye.ML.dataset.scalar_dataset import extract_label_classification
from InnerEye.ML.reports.notebook_report import print_header
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.reports.classification_report import LabelsAndPredictions


def get_unique_label_combinations(dataset_csv: Path, config: ScalarModelBase) -> List[List[str]]:
    """
    Get a list of all the combinations of labels that exist in the dataset.
    """
    df = pd.read_csv(dataset_csv, dtype=str)
    labels = list(set(df[config.label_value_column].values))

    labels = [[config.class_names[int(i)]
                for i in np.array(extract_label_classification(label_string=label,
                                                               sample_id="",
                                                               num_classes=len(config.class_names),
                                                               is_classification_dataset=config.is_classification_model))
                                                               .nonzero()[0]]
              for label in labels if not isinstance(label, float) or not math.isnan(label)]
    return labels


def generate_psuedo_labels(csv: Path,
                           hues: List[str],
                           all_hues: List[str],
                           per_class_thresholds: List[float]) -> pd.DataFrame:
    """
    Generate a pseudo dataset, which has the ground truth and model predictions for a particular combination of labels.
    The ground truth for a sample is set to True if the set of ground truth labels for the sample corresponds exactly
    with the given list of labels, otherwise it is set to False. Similarly, the model output is set to True if the model
    predicts a value exceeding the label threshold for every label in the given set and for no other labels, and False
    otherwise.
    :param csv: csv with the model predictions (written by the inference pipeline)
    :param hues: A combination of labels to calculate the ground truth and model prediction for
    :param all_hues: The entire set of labels on which the model is trained
    :param per_class_thresholds: Thresholds per label class to decide if model has predicted True of False for the
    specific label class
    :return: Dataframe with generated ground truth and model outputs per sample
    """

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


def get_psuedo_labels_and_predictions(csv: Path,
                                      hues: List[str],
                                      all_hues: List[str],
                                      thresholds: List[float]) -> LabelsAndPredictions:
    """
    Given a CSV file, generate a set of labels and model predictions for the combination of label classes.
    NOTE: This CSV file should have results from a single epoch, as in the metrics files written during inference, not
    like the ones written while training.
    """
    df = generate_psuedo_labels(csv=csv, hues=hues, all_hues=all_hues, per_class_thresholds=thresholds)

    labels = df[LoggingColumns.Label.value].to_numpy()
    model_outputs = df[LoggingColumns.ModelOutput.value].to_numpy()
    subjects = df[LoggingColumns.Patient.value].to_numpy()
    return LabelsAndPredictions(subject_ids=subjects, labels=labels, model_outputs=model_outputs)
