#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import tensorboardX
import torch
from PIL.Image import Image
from pandas import DataFrame
from sklearn.metrics import r2_score as sklearn_r2_score

from InnerEye.Azure.azure_util import DEFAULT_CROSS_VALIDATION_SPLIT_INDEX, PARENT_RUN_CONTEXT, RUN_CONTEXT, \
    is_offline_run_context
from InnerEye.Common.common_util import EPOCH_METRICS_FILE_NAME, METRICS_FILE_NAME
from InnerEye.Common.type_annotations import TupleFloat3
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.utils.metrics_constants import MetricsFileColumns


class DataframeLogger:
    """
    Single DataFrame logger for logging to CSV file
    """

    def __init__(self, csv_path: Path):
        self.records: List[Dict[str, Any]] = []
        self.csv_path = csv_path

    def add_record(self, record: Dict[str, Any]) -> None:
        self.records.append(record)

    def flush(self, log_info: bool = False) -> None:
        """
        Save the internal records to a csv file.
        :param log_info: Log INFO if log_info is True.
        """
        import pandas as pd
        if not self.csv_path.parent.is_dir():
            self.csv_path.parent.mkdir(parents=True)
        # Specifying columns such that the order in which columns appear matches the order in which
        # columns were added in the code.
        columns = self.records[0].keys() if len(self.records) > 0 else None
        df = pd.DataFrame.from_records(self.records, columns=columns)
        df.to_csv(self.csv_path, sep=',', mode='w', index=False)
        if log_info:
            logging.info(f"\n {df.to_string(index=False)}")


class MetricsDataframeLoggers:
    """
    Contains DataframeLogger instances for logging metrics to CSV during training and validation stages respectively
    """

    def __init__(self, outputs_folder: Path):
        self.outputs_folder = outputs_folder
        _train_root = self.outputs_folder / ModelExecutionMode.TRAIN.value
        _val_root = self.outputs_folder / ModelExecutionMode.VAL.value
        # training loggers
        self.train_subject_metrics = DataframeLogger(_train_root / METRICS_FILE_NAME)
        self.train_epoch_metrics = DataframeLogger(_train_root / EPOCH_METRICS_FILE_NAME)
        # validation loggers
        self.val_subject_metrics = DataframeLogger(_val_root / METRICS_FILE_NAME)
        self.val_epoch_metrics = DataframeLogger(_val_root / EPOCH_METRICS_FILE_NAME)
        self._all_metrics = [
            self.train_subject_metrics,
            self.train_epoch_metrics,
            self.val_subject_metrics,
            self.val_epoch_metrics
        ]

    def close_all(self) -> None:
        """
        Save all records for each logger to disk.
        """
        for x in self._all_metrics:
            x.flush()


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
        Adds a Dice score, Mean nad Hausdorff Distances for a patient + structure combination to the present object.

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

    def save_aggregates_to_csv(self, file_path: Path) -> None:
        """
        Writes the per-structure aggregate Dice scores (mean, median, and others) to a CSV file.
        The aggregates are those that are output by the Dataframe 'describe' method.

        :param file_path: The name of the file to write to.
        """

        stats_columns = ['mean', 'std', 'min', 'max']
        # get aggregates for all metrics
        aggregates = self.to_data_frame().groupby(MetricsFileColumns.Structure.value).describe()

        def filter_rename_metric_columns(_metric_column: str, is_count_column: bool = False) -> pd.DataFrame:
            _columns = ["count"] + stats_columns if is_count_column else stats_columns
            _df = aggregates[_metric_column][_columns]
            _columns_to_rename = [x for x in _df.columns if x != "count"]
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

        # dtype is specified as (an instance of) str, not the str class itself, but this seems correct.
        # noinspection PyTypeChecker
        df = DataFrame(self.columns, dtype=str)
        df[MetricsFileColumns.DiceNumeric.value] = pd.Series(data=df[MetricsFileColumns.Dice.value].apply(float))
        df[MetricsFileColumns.HausdorffDistanceMM.value] = pd.Series(
            data=df[MetricsFileColumns.HausdorffDistanceMM.value].apply(float))
        df[MetricsFileColumns.MeanDistanceMM.value] = pd.Series(
            data=df[MetricsFileColumns.MeanDistanceMM.value].apply(float))
        return df


class AzureMLLogger:
    """
    Stores the information that is required to log metrics to AzureML.
    """

    def __init__(self,
                 logging_prefix: str,
                 cross_validation_split_index: int,
                 log_to_parent_run: bool):
        """
        :param logging_prefix: A prefix string that will be added to all metrics names before logging.
        :param cross_validation_split_index: The cross validation split index, or its default value if not running
        inside cross validation.
        :param log_to_parent_run: If true, all metrics will also be written to the Hyperdrive parent run when that
        parent run is present.
        """
        self.logging_prefix = logging_prefix
        self.cross_validation_split_index = cross_validation_split_index
        self.log_to_parent_run = log_to_parent_run

    def log_to_azure(self,
                     label: str,
                     metric: float) -> None:
        """
        Logs a metric as a key/value pair to AzureML.
        :param label: The name of the metric that should be logged
        :param metric: The value of the metric.
        """
        if not is_offline_run_context(RUN_CONTEXT):
            metric_name = self.logging_prefix + label
            RUN_CONTEXT.log(metric_name, metric)
            # When running in a cross validation setting, log all metrics to the hyperdrive parent run too,
            # so that we can easily overlay graphs across runs.
            if self.log_to_parent_run and PARENT_RUN_CONTEXT:
                if self.cross_validation_split_index > DEFAULT_CROSS_VALIDATION_SPLIT_INDEX:
                    PARENT_RUN_CONTEXT.log(f"{metric_name}_Split{self.cross_validation_split_index}",
                                           metric)


class AzureAndTensorboardLogger:
    """
    Contains functionality to log metrics to both Azure run and TensorBoard event file
    for both classification and segmentation models.
    """

    def __init__(self,
                 azureml_logger: AzureMLLogger,
                 tensorboard_logger: tensorboardX.SummaryWriter):
        self.azureml_logger = azureml_logger
        self.tensorboard_logger = tensorboard_logger
        self.epoch = 0

    def close(self) -> None:
        """
        Closes all loggers that require explicit closing.
        """
        self.tensorboard_logger.close()

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the given value as the epoch for all following logging calls.
        :param epoch: The current epoch
        """
        self.epoch = epoch

    def log_to_azure_and_tensorboard(self, label: str, metric: float) -> None:
        """
        Writes a metric to the Azure run and to the TensorBoard event file
        :param label: The string name of the metric.
        :param metric: The value of the metric.
        """
        self.azureml_logger.log_to_azure(label, metric)
        self.log_to_tensorboard(label, metric)

    def log_to_tensorboard(self, label: str, metric: float) -> None:
        """
        Writes a metric to a Tensorboard event file.
        :param label: The string name of the metric.
        :param metric: The value of the metric.
        """
        # TensorBoard does not like tags that contain spaces, and prints out a warning for each logging attempt.
        # Replace space with underscore to reduce logging noise.
        writer = self.tensorboard_logger
        label_without_spaces = label.replace(" ", "_")
        writer.add_scalar(label_without_spaces, metric, self.epoch)

    def log_image(self, name: str, path: str) -> None:
        """
        Logs a PNG image stored in `path` to Azure and Tensorboard.
        """
        if not is_offline_run_context(RUN_CONTEXT):
            RUN_CONTEXT.log_image(name=name, path=path)
        writer = self.tensorboard_logger
        img = Image.open(path).convert("RGB")
        img = np.transpose(np.asarray(img), (2, 0, 1))
        writer.add_image(name, img, self.epoch)


def get_number_of_voxels_per_class(labels: Union[np.ndarray, torch.Tensor]) -> List[int]:
    """
    Computes the number of voxels for each class in a one-hot label map.

    :param labels: one-hot label map in shape Batches x Classes x Z x Y x X or Classes x Z x Y x X
    """
    if labels is None:
        raise Exception("labels cannot be None")
    if not len(labels.shape) in [5, 4]:
        raise Exception("labels must have either 4 (Classes x Z x Y x X) "
                        "or 5 dimensions (Batches x Classes x Z x Y x X), found:{}"
                        .format(len(labels.shape)))

    if len(labels.shape) == 4:
        labels = labels[None, ...]

    # TODO antonsc: Switch to Pytorch 1.7 and use torch.count_nonzero
    return [np.sum(c).item() for c in np.count_nonzero(labels.cpu().numpy(), axis=0)]


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
