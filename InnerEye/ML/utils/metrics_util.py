#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.metrics import r2_score as sklearn_r2_score

from InnerEye.Common.metrics_constants import MetricsFileColumns
from InnerEye.Common.type_annotations import TupleFloat3


class MetricsPerPatientWriter:
    """
    Stores information about metrics eg: Dice, Mean and Hausdorff Distances, broken down by patient and structure.
    """

    def __init__(self) -> None:
        self.columns: Dict[str, Any] = {MetricsFileColumns.Patient.value: [],
                                        MetricsFileColumns.Structure.value: [],
                                        MetricsFileColumns.Dice.value: [],
                                        MetricsFileColumns.HausdorffDistanceMM.value: [],
                                        MetricsFileColumns.MeanDistanceMM.value: []}
        self.float_format = "%.3f"

    def add(self,
            patient: str,
            structure: str,
            dice: float,
            hausdorff_distance_mm: float,
            mean_distance_mm: float) -> None:
        """
        Adds a Dice score, Mean and Hausdorff Distances for a patient + structure combination to the present object.

        :param patient: The name of the patient.
        :param structure: The structure that is predicted for.
        :param dice: The value of the Dice score that was achieved.
        :param hausdorff_distance_mm: The hausdorff distance in mm
        :param mean_distance_mm: The mean surface distance in mm
        """
        self.columns[MetricsFileColumns.Patient.value].append(patient)
        self.columns[MetricsFileColumns.Structure.value].append(structure)
        self.columns[MetricsFileColumns.Dice.value].append(format_metric(dice))
        self.columns[MetricsFileColumns.HausdorffDistanceMM.value].append(format_metric(hausdorff_distance_mm))
        self.columns[MetricsFileColumns.MeanDistanceMM.value].append(format_metric(mean_distance_mm))

    def to_csv(self, file_name: Path) -> None:
        """
        Writes the per-patient per-structure metrics to a CSV file.
        Sorting is done first by structure name, then by Dice score ascending.
        :param file_name: The name of the file to write to.
        """
        df = self.to_data_frame()
        dice_numeric = MetricsFileColumns.DiceNumeric.value
        sort_keys = [MetricsFileColumns.Structure.value, dice_numeric]
        sorted_by_dice = df.sort_values(sort_keys, ascending=True)
        del sorted_by_dice[dice_numeric]
        sorted_by_dice.to_csv(file_name, index=False, float_format=self.float_format)

    def save_aggregates_to_csv(self, file_path: Path, allow_incomplete_labels: bool = False) -> None:
        """
        Writes the per-structure aggregate Dice scores (mean, median, and others) to a CSV file.
        The aggregates are those that are output by the Dataframe 'describe' method.

        :param file_path: The name of the file to write to.
        :param allow_incomplete_labels: boolean flag. If false, all ground truth files must be provided.
        If true, ground truth files are optional and we add a total_patients count column for easy
        comparison. (Defaults to False.)
        """

        stats_columns = ['mean', 'std', 'min', 'max']
        # get aggregates for all metrics
        df = self.to_data_frame()
        aggregates = df.groupby(MetricsFileColumns.Structure.value).describe()

        total_num_patients_column_name = f"total_{MetricsFileColumns.Patient.value}".lower()
        if not total_num_patients_column_name.endswith("s"):
            total_num_patients_column_name += "s"

        def filter_rename_metric_columns(_metric_column: str, is_count_column: bool = False) -> pd.DataFrame:
            _columns = ["count"] + stats_columns if is_count_column else stats_columns
            _df = aggregates[_metric_column][_columns]
            if is_count_column and allow_incomplete_labels:
                # For this condition we add a total_patient count column so that readers can make
                # more sense of aggregated metrics where some patients were missing the label (i.e.
                # partial ground truth).
                num_subjects = len(pd.unique(df[MetricsFileColumns.Patient.value]))
                _df[total_num_patients_column_name] = num_subjects
                _df = _df[["count", total_num_patients_column_name] + stats_columns]
            _columns_to_rename = [x for x in _df.columns if x != "count" and x != total_num_patients_column_name]
            return _df.rename(columns={k: f"{_metric_column}_{k}" for k in _columns_to_rename})

        def _merge_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
            return pd.merge(df1, df2,
                            left_on=MetricsFileColumns.Structure.value,
                            right_on=MetricsFileColumns.Structure.value).drop_duplicates()

        df_dice = filter_rename_metric_columns(MetricsFileColumns.DiceNumeric.value, True)
        df_hd = filter_rename_metric_columns(MetricsFileColumns.HausdorffDistanceMM.value)
        df_md = filter_rename_metric_columns(MetricsFileColumns.MeanDistanceMM.value)
        _merge_df(_merge_df(df_dice, df_hd), df_md).to_csv(file_path, float_format=self.float_format)

    def to_data_frame(self) -> DataFrame:
        """
        Creates a DataFrame that holds all the per-patient per-structure results.
        A numeric column is added, that contains the Dice score as a numeric value.
        """
        # The general opinion (on StackOverflow) appears to be that creating a DataFrame row-by-row is
        # slow, and should be avoided. Hence, work with dictionary as long as possible, and only finally
        # convert to a DataFrame.

        # noinspection PyTypeChecker
        dtypes: Dict[str, Union[Type[float], Type[str]]] = {column: str for column in self.columns}
        dtypes[MetricsFileColumns.Dice.value] = float
        dtypes[MetricsFileColumns.HausdorffDistanceMM.value] = float
        dtypes[MetricsFileColumns.MeanDistanceMM.value] = float
        df = DataFrame(self.columns, dtype=str)
        df = df.astype(dtypes)
        df[MetricsFileColumns.DiceNumeric.value] = df[MetricsFileColumns.Dice.value]
        df = df.sort_values(by=[MetricsFileColumns.Patient.value, MetricsFileColumns.Structure.value])
        return df


def get_number_of_voxels_per_class(labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the number of voxels for each class in a one-hot label map.
    :param labels: one-hot label map in shape Batches x Classes x Z x Y x X or Classes x Z x Y x X
    :return: A tensor of shape [Batches x Classes] containing the number of non-zero voxels along Z, Y, X
    """
    if not len(labels.shape) in [5, 4]:
        raise Exception("labels must have either 4 (Classes x Z x Y x X) "
                        "or 5 dimensions (Batches x Classes x Z x Y x X), found:{}"
                        .format(len(labels.shape)))

    if len(labels.shape) == 4:
        labels = labels[None, ...]

    return torch.count_nonzero(labels, dim=(2, 3, 4))


def get_label_overlap_stats(labels: np.ndarray, label_names: List[str]) -> Dict[str, int]:
    """
    Computes overlap between labelled structures and returns the stats in a dictionary format.

    :param labels: nd-NumPy array containing binary masks for all labels
    :param label_names: A list of strings containing target label names, e.g. [spleen, liver]
    """
    if len(label_names) != labels.shape[0]:
        raise ValueError("Length of input label names and stacked array mismatch.")

    # Check if multiple labels are assigned to same pixel and count the occurrence
    structure_array = labels.reshape(labels.shape[0], -1)
    overlapping_pixels = np.sum(structure_array, axis=0) >= 2
    overlapping_classes, _ = np.where(structure_array[:, overlapping_pixels])
    overlapping_classes, n_counts = np.unique(overlapping_classes, return_counts=True)

    overlap_stats = {label_names[class_id]: n_count
                     for class_id, n_count in zip(overlapping_classes, n_counts)}

    # For a subset of label names, if there is no overlap, set their overlap values to zero
    for label_name in label_names:
        if label_name not in overlap_stats:
            overlap_stats[label_name] = 0

    return overlap_stats


def get_label_volume(labels: np.ndarray, label_names: List[str], label_spacing: TupleFloat3) -> Dict[str, float]:
    """
    Computes volume of ground-truth labels in mL and returns it in a dictionary

    :param labels: nd-NumPy array containing binary masks for all labels
    :param label_names: A list of strings containing target label names, e.g. [spleen, liver]
    :param label_spacing: label spacing
    """
    if len(label_names) != labels.shape[0]:
        raise ValueError("Length of input label names and stacked array mismatch.")

    labels = labels.reshape(labels.shape[0], -1)
    fg_labels = np.sum(labels, axis=1)
    pixel_volume = reduce(lambda x, y: x * y, label_spacing) / 1000.0

    return {label_name: fg_labels[loop_id] * round(pixel_volume, 3)
            for loop_id, label_name in enumerate(label_names)}


def format_metric(metric: float) -> str:
    """
    Returns a readable string from the given Dice or loss function value, rounded to 3 digits.
    """
    return "{:0.3f}".format(metric)


def binary_classification_accuracy(model_output: Union[torch.Tensor, np.ndarray],
                                   label: Union[torch.Tensor, np.ndarray],
                                   threshold: float = 0.5) -> float:
    """
    Computes the accuracy for binary classification from a real-valued model output and a label vector.
    The model output is assumed to be in the range between 0 and 1, a value larger than 0.5 indicates a prediction
    of class 1. The label vector is expected to contain class indices 0 and 1 only, but is also thresholded at 0.5.

    :param model_output: A tensor containing model outputs.
    :param label: A tensor containing class labels.
    :param threshold: the cut-off probability threshold for predictions. If model_ouput is > threshold, the predicted
    class is 1 else 0.
    :return: 1.0 if all predicted classes match the expected classes given in 'labels'. 0.0 if no predicted classes
    match their labels.
    """
    model_output, label = convert_input_and_label(model_output, label)
    predicted_class = model_output > threshold
    label = label > 0.5
    return (predicted_class == label).float().mean().item()


def r2_score(model_output: Union[torch.Tensor, np.ndarray], label: Union[torch.Tensor, np.ndarray]) \
        -> float:
    """
    Computes the coefficient of determination R2. Represents the proportion of variance explained
    by the (independent) variables in the model. R2 = 1 - Mean(SquaredErrors) / Variance(Labels)
    """
    if torch.is_tensor(label):
        label = label.detach().cpu().numpy()
    if torch.is_tensor(model_output):
        model_output = model_output.detach().cpu().numpy()
    return sklearn_r2_score(label, model_output)


def mean_absolute_error(model_output: Union[torch.Tensor, np.ndarray],
                        label: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Computes the mean absolute error i.e. mean(abs(model_output - label))
    """
    model_output, label = convert_input_and_label(model_output, label)
    absolute_errors = torch.abs(model_output - label)
    return absolute_errors.mean().item()


def mean_squared_error(model_output: Union[torch.Tensor, np.ndarray],
                       label: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Computes the mean squared error i.e. mean((model_output - label)^2)
    """
    model_output, label = convert_input_and_label(model_output, label)
    return torch.nn.functional.mse_loss(model_output, label).item()


def convert_input_and_label(model_output: Union[torch.Tensor, np.ndarray],
                            label: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ensures that both model_output and label are tensors of dtype float32.
    :return a Tuple with model_output, label as float tensors.
    """
    if not torch.is_tensor(model_output):
        model_output = torch.tensor(model_output)
    if not torch.is_tensor(label):
        label = torch.tensor(label)
    return model_output.float(), label.float()


def is_missing_ground_truth(ground_truth: np.array) -> bool:
    """
    calculate_metrics_per_class in metrics.py and plot_contours_for_all_classes in plotting.py both
    check whether there is ground truth missing using this simple check for NaN value at 0, 0, 0.
    To avoid duplicate code we bring it here as a utility function.
    :param ground_truth: ground truth binary array with dimensions: [Z x Y x X].
    :param label_id: Integer index of the label to check.
    :returns: True if the label is missing (signified by NaN), False otherwise.
    """
    return np.isnan(ground_truth[0, 0, 0])
