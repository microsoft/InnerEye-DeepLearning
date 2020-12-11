#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import SimpleITK as sitk
import math
import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
from PIL import Image
from azureml.core import Run

from InnerEye.Azure.azure_util import DEFAULT_CROSS_VALIDATION_SPLIT_INDEX, PARENT_RUN_CONTEXT, RUN_CONTEXT, \
    get_run_context_or_default, is_offline_run_context
from InnerEye.Common.common_util import DataframeLogger
from InnerEye.Common.metrics_dict import MetricType, MetricsDict, ScalarMetricsDict, get_column_name_for_logging, \
    get_metric_name_with_hue_prefix
from InnerEye.Common.type_annotations import TupleFloat3
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import BACKGROUND_CLASS_NAME
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.scalar_config import ScalarLoss
from InnerEye.ML.utils import metrics_util
from InnerEye.ML.utils.image_util import binaries_from_multi_label_array, check_array_range, is_binary_array
from InnerEye.ML.utils.io_util import reverse_tuple_float3
from InnerEye.ML.utils.metrics_constants import LoggingColumns
from InnerEye.ML.utils.metrics_util import binary_classification_accuracy, mean_absolute_error, r2_score
from InnerEye.ML.utils.ml_util import check_size_matches
from InnerEye.ML.utils.sequence_utils import get_masked_model_outputs_and_labels
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule

TRAIN_STATS_FILE = "train_stats.csv"


@dataclass(frozen=True)
class InferenceMetrics:
    """
    Defined purely to serve as a superclass.
    """
    pass


@dataclass(frozen=True)
class InferenceMetricsForClassification(InferenceMetrics):
    """
    Stores a dictionary mapping from epoch number to the metrics that were achieved in that epoch.
    """
    epochs: Dict[int, MetricsDict]


@dataclass(frozen=True)
class InferenceMetricsForSegmentation(InferenceMetrics):
    """
    Stores metrics for segmentation models, per execution mode and epoch.
    """
    data_split: ModelExecutionMode
    epochs: Dict[int, float]

    def get_best_epoch_dice(self) -> float:
        """
        Gets the Dice score that the model achieves in the best epoch.
        """
        return self.epochs[self.get_best_epoch()]

    def get_best_epoch(self) -> int:
        """
        Gets the epoch that achieves the best (highest) Dice score.
        """
        epoch = max(self.epochs, key=self.epochs.get)
        return epoch

    def get_metrics_log_key(self) -> str:
        """
        Gets a string name for logging the metrics specific to the execution mode (train, val, test)
        :return:
        """
        return f"InferenceMetrics_{self.data_split.value}"

    def log_metrics(self, run_context: Run = None) -> None:
        """
        Log metrics for each epoch to the provided runs logs, or the current run context if None provided
        :param run_context: Run for which to log the metrics to, use the current run context if None provided
        :return:
        """
        run_context = get_run_context_or_default(run_context)
        keys = sorted(self.epochs.keys())

        run_context.log_table(name=self.get_metrics_log_key(), value={
            "Checkpoint": keys,
            "Dice": [self.epochs[x] for x in keys]
        })


@dataclass(frozen=True)
class SegmentationMetricsPerClass:
    """
    Stores different segmentation metrics, as a list where each list entry represents one class.
    """
    dice: List[float] = field(default_factory=list)
    hausdorff_distance_mm: List[float] = field(default_factory=list)
    mean_distance_mm: List[float] = field(default_factory=list)

    def append_nan(self) -> None:
        """
        Adds a NaN for all metrics that are stored, indicating that the target class was not present and no
        output was produced by the model.
        """
        self.dice.append(math.nan)
        self.hausdorff_distance_mm.append(math.nan)
        self.mean_distance_mm.append(math.nan)

    def append(self,
               dice: float,
               hausdorff_distance_mm: float,
               mean_distance_mm: float) -> None:
        """
        Stores the metrics for a class in the present object.
        :param dice: The Dice score between ground truth and model output.
        :param hausdorff_distance_mm: The Hausdorff distance between ground truth and model output, in millimeters.
        :param mean_distance_mm: The mean surface distance between ground truth and model output, in millimeters.
        :return:
        """
        self.dice.append(dice)
        self.hausdorff_distance_mm.append(hausdorff_distance_mm)
        self.mean_distance_mm.append(mean_distance_mm)


class AzureMLLogger:
    """
    Stores the information that is required to log metrics to AzureML.
    """

    def __init__(self,
                 cross_validation_split_index: int,
                 logging_prefix: str,
                 log_to_parent_run: bool):
        """
        :param cross_validation_split_index: The cross validation split index, or its default value if not running
        inside cross validation.
        :param logging_prefix: A prefix string that will be added to all metrics names before logging.
        :param log_to_parent_run: If true, all metrics will also be written to the Hyperdrive parent run when that
        parent run is present.
        """
        self.logging_prefix = logging_prefix
        self.cross_validation_split_index = cross_validation_split_index
        self.log_to_parent_run = log_to_parent_run

    def log_to_azure(self, label: str, metric: float) -> None:
        """
        Logs a metric as a key/value pair to AzureML.
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
                 tensorboard_logger: tensorboardX.SummaryWriter,
                 epoch: int):
        self.azureml_logger = azureml_logger
        self.tensorboard_logger = tensorboard_logger
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

    def log_segmentation_epoch_metrics(self,
                                       metrics: MetricsDict,
                                       learning_rates: List[float]) -> None:
        """
        Logs segmentation metrics (e.g. loss, dice scores, learning rates) to an event file for TensorBoard
        visualization and to the AzureML run context
        :param learning_rates: The logged learning rates.
        :param metrics: The metrics of the specified epoch, averaged along its batches.
        """
        logging_fn = self.log_to_azure_and_tensorboard
        logging_fn(MetricType.LOSS.value, metrics.get_single_metric(MetricType.LOSS))
        logging_fn("Dice/AverageExceptBackground", metrics.get_single_metric(MetricType.DICE))
        logging_fn("Voxels/ProportionForeground", metrics.get_single_metric(MetricType.PROPORTION_FOREGROUND_VOXELS))
        logging_fn("TimePerEpoch_Seconds", metrics.get_single_metric(MetricType.SECONDS_PER_EPOCH))

        if learning_rates is not None:
            for i, lr in enumerate(learning_rates):
                logging_fn("LearningRate/Index_{}".format(i), lr)

        for class_name in metrics.get_hue_names(include_default=False):
            # Tensorboard groups metrics by what is before the slash.
            # With metrics Dice/Foo and Dice/Bar, it will create a section for "Dice",
            # and inside of it, there are graphs for Foo and Bar
            get_label = lambda x, y: "{}/{}".format(x, y)
            logging_fn(get_label("Dice", class_name),
                       metrics.get_single_metric(MetricType.DICE, hue=class_name))
            logging_fn(get_label("Voxels", class_name),
                       metrics.get_single_metric(MetricType.PROPORTION_FOREGROUND_VOXELS, hue=class_name))

    def log_classification_epoch_metrics(self, metrics: MetricsDict) -> None:
        """
        Writes all values from MetricsDict object into a file for Tensorboard visualization,
        and into the AzureML run context.
        :param metrics: dictionary containing the metrics to be logged, averaged over minibatches.
        """
        for hue_name, label, metric in metrics.enumerate_single_values():
            self.log_to_azure_and_tensorboard(get_metric_name_with_hue_prefix(label, hue_name), metric)


def vars_with_scalar_fields_only(o: Any) -> Dict[str, Any]:
    """
    Returns a dictionary similar to vars(o), but with only those fields that either have integer
    or floating point value.
    :param o: The object to process.
    :return: A dictionary mapping from field name to value.
    """

    def is_scalar(f: Any) -> bool:
        return isinstance(f, (int, float))

    return {key: value for key, value in vars(o) if is_scalar(value)}


def surface_distance(seg: sitk.Image, reference_segmentation: sitk.Image) -> float:
    """
    Symmetric surface distances taking into account the image spacing
    https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/34_Segmentation_Evaluation.ipynb
    :param seg: mask 1
    :param reference_segmentation: mask 2
    :return: mean distance
    """
    statistics_image_filter = sitk.StatisticsImageFilter()
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    reference_surface = sitk.LabelContour(reference_segmentation)
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())

    reference_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(reference_segmentation, squaredDistance=False, useImageSpacing=True))
    reference_surface = sitk.LabelContour(reference_segmentation)

    # Symmetric surface distance measures
    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(seg, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(seg)

    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = _add_zero_distances(num_segmented_surface_pixels, seg2ref_distance_map_arr)
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = _add_zero_distances(num_reference_surface_pixels, ref2seg_distance_map_arr)

    all_surface_distances = seg2ref_distances + ref2seg_distances
    return np.mean(all_surface_distances).item()


def _add_zero_distances(num_segmented_surface_pixels: int, seg2ref_distance_map_arr: np.ndarray) -> List[float]:
    """
    # Get all non-zero distances and then add zero distances if required.
    :param num_segmented_surface_pixels:
    :param seg2ref_distance_map_arr:
    :return: list of distances, augmented with zeros.
    """
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    return seg2ref_distances


def calculate_metrics_per_class(segmentation: np.ndarray,
                                ground_truth: np.ndarray,
                                ground_truth_ids: List[str],
                                voxel_spacing: TupleFloat3,
                                patient_id: Optional[int] = None) -> MetricsDict:
    """
    Calculate the dice for all foreground structures (the background class is completely ignored).
    Returns a MetricsDict with metrics for each of the foreground
    structures. Metrics are NaN if both ground truth and prediction are all zero for a class.
    :param ground_truth_ids: The names of all foreground classes.
    :param segmentation: predictions multi-value array with dimensions: [Z x Y x X]
    :param ground_truth: ground truth binary array with dimensions: [C x Z x Y x X]
    :param voxel_spacing: voxel_spacing in 3D Z x Y x X
    :param patient_id: for logging
    """
    number_of_classes = ground_truth.shape[0]
    if len(ground_truth_ids) != (number_of_classes - 1):
        raise ValueError(f"Received {len(ground_truth_ids)} foreground class names, but "
                         f"the label tensor indicates that there are {number_of_classes - 1} classes.")
    binaries = binaries_from_multi_label_array(segmentation, number_of_classes)

    all_classes_are_binary = [is_binary_array(ground_truth[label_id]) for label_id in range(ground_truth.shape[0])]
    if not np.all(all_classes_are_binary):
        raise ValueError("Ground truth values should be 0 or 1")
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    metrics = MetricsDict(hues=ground_truth_ids)
    for i, prediction in enumerate(binaries):
        if i == 0:
            continue
        check_size_matches(prediction, ground_truth[i], arg1_name="prediction", arg2_name="ground_truth")
        if not is_binary_array(prediction):
            raise ValueError("Predictions values should be 0 or 1")
        # simpleitk returns a Dice score of 0 if both ground truth and prediction are all zeros.
        # We want to be able to fish out those cases, and treat them specially later.
        prediction_zero = np.all(prediction == 0)
        gt_zero = np.all(ground_truth[i] == 0)
        dice = mean_surface_distance = hausdorff_distance = math.nan
        if not (prediction_zero and gt_zero):
            prediction_image = sitk.GetImageFromArray(prediction.astype(np.uint8))
            prediction_image.SetSpacing(sitk.VectorDouble(reverse_tuple_float3(voxel_spacing)))
            ground_truth_image = sitk.GetImageFromArray(ground_truth[i].astype(np.uint8))
            ground_truth_image.SetSpacing(sitk.VectorDouble(reverse_tuple_float3(voxel_spacing)))
            overlap_measures_filter.Execute(prediction_image, ground_truth_image)
            dice = overlap_measures_filter.GetDiceCoefficient()
            if prediction_zero or gt_zero:
                hausdorff_distance = mean_surface_distance = math.inf
            else:
                try:
                    hausdorff_distance_filter.Execute(prediction_image, ground_truth_image)
                    hausdorff_distance = hausdorff_distance_filter.GetHausdorffDistance()
                except Exception as e:
                    logging.warning("Cannot calculate Hausdorff distance for "
                                    f"structure {i} of patient {patient_id}: {e}")
                try:
                    mean_surface_distance = surface_distance(prediction_image, ground_truth_image)
                except Exception as e:
                    logging.warning(f"Cannot calculate mean distance for structure {i} of patient {patient_id}: {e}")
            logging.debug(f"Patient {patient_id}, class {i} has Dice score {dice}")

        def add_metric(metric_type: MetricType, value: float) -> None:
            metrics.add_metric(metric_type, value, skip_nan_when_averaging=True, hue=ground_truth_ids[i - 1])

        add_metric(MetricType.DICE, dice)
        add_metric(MetricType.HAUSDORFF_mm, hausdorff_distance)
        add_metric(MetricType.MEAN_SURFACE_DIST_mm, mean_surface_distance)
    return metrics


def compute_dice_across_patches(segmentation: torch.Tensor,
                                ground_truth: torch.Tensor,
                                use_cuda: bool,
                                allow_multiple_classes_for_each_pixel: bool = False) -> torch.Tensor:
    """
    Computes the Dice scores for all classes across all patches in the arguments.
    :param segmentation: Tensor containing class ids predicted by a model.
    :param ground_truth: One-hot encoded torch tensor containing ground-truth label ids.
    :param use_cuda: If set to True, uses CUDA backend for computations
    :param allow_multiple_classes_for_each_pixel: If set to False, ground-truth tensor has
    to contain only one foreground label for each pixel.
    :return A torch tensor of size (Patches, Classes) with the Dice scores. Dice scores are computed for
    all classes including the background class at index 0.
    """
    if use_cuda:
        segmentation = segmentation.cuda()
        ground_truth = ground_truth.cuda()

    check_size_matches(segmentation, ground_truth, 4, 5, [0, -3, -2, -1],
                       arg1_name="segmentation", arg2_name="ground_truth")

    # One-hot encoded ground-truth values should sum up to one for all pixels
    if not allow_multiple_classes_for_each_pixel:
        if not torch.allclose(torch.sum(ground_truth, dim=1).float(),
                              torch.ones(segmentation.shape, device=ground_truth.device).float()):
            raise Exception("Ground-truth one-hot matrix does not sum up to one for all pixels")

    # Convert the ground-truth to one-hot-encoding
    [num_patches, num_classes] = ground_truth.size()[:2]
    one_hot_segmentation = F.one_hot(segmentation, num_classes=num_classes).permute(0, 4, 1, 2, 3)

    # Convert the tensors to bool tensors
    one_hot_segmentation = one_hot_segmentation.bool().view(num_patches, num_classes, -1)
    ground_truth = ground_truth.bool().view(num_patches, num_classes, -1)

    # And operation between segmentation and ground-truth - reduction operation
    # Count the number of samples in segmentation and ground-truth
    intersection = 2.0 * torch.sum(one_hot_segmentation & ground_truth, dim=-1).float()
    union = torch.sum(one_hot_segmentation, dim=-1) + torch.sum(ground_truth, dim=-1).float() + 1.0e-6

    return intersection / union


def format_learning_rates(learning_rates: List[float]) -> str:
    """
    Converts a list of learning rates to a human readable string. Multiple entries are separated by semicolon.
    :param learning_rates: An iterable of learning rate values.
    :return: An empty string if the argument is None or empty, otherwise the string representation of the rates,
    formatted as {:0.2e}
    """
    if learning_rates is None or len(learning_rates) == 0:
        return ""
    return "; ".join("{:0.2e}".format(lr) for lr in learning_rates)


def store_epoch_stats_for_segmentation(outputs_dir: Path,
                                       epoch: int,
                                       learning_rates: List[float],
                                       training_results: MetricsDict,
                                       validation_results: MetricsDict) -> None:
    """
    Writes a dictionary of statistics for a segmentation training run to a file. Successive calls to the function
    append another line of metrics. The first line of the file contains the column headers (names of the metrics).
    :param training_results: A MetricsDict object with all metrics that were achieved on the training set in the
    current epoch.
    :param validation_results: A MetricsDict object with all metrics that were achieved on the validation set in the
    current epoch.
    :param learning_rates: The learning rates that were used in the current epoch.
    :param epoch: The number of the current training epoch.
    :param outputs_dir: The directory in which the statistics file should be created.
    :return:
    """
    epoch_stats = {
        "Epoch": str(epoch),
        "LearningRate": format_learning_rates(learning_rates),
        "TrainLoss": metrics_util.format_metric(training_results.get_single_metric(MetricType.LOSS)),
        "TrainDice": metrics_util.format_metric(training_results.get_single_metric(MetricType.DICE)),
        "ValLoss": metrics_util.format_metric(validation_results.get_single_metric(MetricType.LOSS)),
        "ValDice": metrics_util.format_metric(validation_results.get_single_metric(MetricType.DICE)),
    }
    # When using os.linesep, additional LF characters are inserted. Expected behaviour only when
    # using this on both Windows and Linux.
    line_sep = "\n"
    tab = "\t"
    full_file = outputs_dir / TRAIN_STATS_FILE
    if not full_file.exists():
        header = tab.join(epoch_stats.keys())
        full_file.write_text(header + line_sep)
    line = tab.join(epoch_stats.values())
    with full_file.open("a") as f:
        f.write(line + line_sep)


def validate_and_store_model_parameters(writer: tensorboardX.SummaryWriter, epoch: int,
                                        model: DeviceAwareModule) -> None:
    """
    Validates and writes all model weights to the given TensorBoard writer.
    :param writer: TensorBoard summary writer
    :param epoch: The epoch for which these model parameters correspond to.
    :param model: The model from which to extract the parameters.
    :return:
    """
    for name, param in model.named_parameters():
        param_numpy = param.clone().cpu().data.numpy()
        check_array_range(param_numpy, error_prefix="Parameter {}".format(name))
        writer.add_histogram(name, param_numpy, epoch)


def store_epoch_metrics(azure_and_tensorboard_logger: AzureAndTensorboardLogger,
                        df_logger: DataframeLogger,
                        epoch: int,
                        metrics: MetricsDict,
                        learning_rates: List[float],
                        config: ModelConfigBase) -> None:
    """
    Writes the loss, Dice scores, and learning rates into a file for Tensorboard visualization,
    and into the AzureML run context.
    :param azure_and_tensorboard_logger: An instance of AzureAndTensorboardLogger.
    :param df_logger: An instance of DataframeLogger, for logging results to csv.
    :param epoch: The epoch corresponding to the results.
    :param metrics: The metrics of the specified epoch, averaged along its batches.
    :param learning_rates: The logged learning rates.
    :param config: one of SegmentationModelBase
    """
    if config.is_segmentation_model:
        azure_and_tensorboard_logger.log_segmentation_epoch_metrics(metrics,
                                                                    learning_rates)
        logger_row = {
            LoggingColumns.Dice.value: metrics.get_single_metric(MetricType.DICE),
            LoggingColumns.Loss.value: metrics.get_single_metric(MetricType.LOSS),
            LoggingColumns.SecondsPerEpoch.value: metrics.get_single_metric(MetricType.SECONDS_PER_EPOCH)
        }

    elif config.is_scalar_model:
        assert isinstance(metrics, MetricsDict)
        azure_and_tensorboard_logger.log_classification_epoch_metrics(metrics)
        logger_row: Dict[str, float] = {}  # type: ignore
        for hue_name, metric_name, metric_value in metrics.enumerate_single_values():
            logging_column_name = get_column_name_for_logging(metric_name, hue_name=hue_name)
            logger_row[logging_column_name] = metric_value
    else:
        raise ValueError("Model must be either classification, regression or segmentation model")

    logger_row.update({
        LoggingColumns.Epoch.value: epoch,
        LoggingColumns.CrossValidationSplitIndex.value: config.cross_validation_split_index
    })

    df_logger.add_record(logger_row)


def compute_scalar_metrics(metrics_dict: ScalarMetricsDict,
                           subject_ids: Sequence[str],
                           model_output: torch.Tensor,
                           labels: torch.Tensor,
                           loss_type: ScalarLoss = ScalarLoss.BinaryCrossEntropyWithLogits) -> None:
    """
    Computes various metrics for a binary classification task from real-valued model output and a label vector,
    and stores them in the given `metrics_dict`.
    The model output is assumed to be in the range between 0 and 1, a value larger than 0.5 indicates a prediction
    of class 1. The label vector is expected to contain class indices 0 and 1 only.
    Metrics for each model output channel will be isolated, and a non-default hue for each model output channel is
    expected, and must exist in the provided metrics_dict. The Default hue is used for single model outputs.
    :param metrics_dict: An object that holds all metrics. It will be updated in-place.
    :param subject_ids: Subject ids for the model output and labels.
    :param model_output: A tensor containing model outputs.
    :param labels: A tensor containing class labels.
    :param loss_type: The type of loss that the model uses. This is required to optionally convert 2-dim model output
    to probabilities.
    """
    _model_output_channels = model_output.shape[1]
    model_output_hues = metrics_dict.get_hue_names(include_default=len(metrics_dict.hues_without_default) == 0)

    if len(model_output_hues) < _model_output_channels:
        raise ValueError("Hues must be provided for each model output channel, found "
                         f"{_model_output_channels} channels but only {len(model_output_hues)} hues")

    for i, hue in enumerate(model_output_hues):
        # mask the model outputs and labels if required
        masked_model_outputs_and_labels = get_masked_model_outputs_and_labels(
            model_output[:, i, ...], labels[:, i, ...], subject_ids)

        # compute metrics on valid masked tensors only
        if masked_model_outputs_and_labels is not None:
            _model_output, _labels, _subject_ids = \
                masked_model_outputs_and_labels.model_outputs.data, \
                masked_model_outputs_and_labels.labels.data, \
                masked_model_outputs_and_labels.subject_ids

            if loss_type == ScalarLoss.MeanSquaredError:
                metrics = {
                    MetricType.MEAN_SQUARED_ERROR: F.mse_loss(_model_output, _labels.float(), reduction='mean').item(),
                    MetricType.MEAN_ABSOLUTE_ERROR: mean_absolute_error(_model_output, _labels),
                    MetricType.R2_SCORE: r2_score(_model_output, _labels)
                }
            else:
                metrics = {
                    MetricType.CROSS_ENTROPY: F.binary_cross_entropy(_model_output, _labels.float(),
                                                                     reduction='mean').item(),
                    MetricType.ACCURACY_AT_THRESHOLD_05: binary_classification_accuracy(_model_output, _labels)
                }
            for key, value in metrics.items():
                if key == MetricType.R2_SCORE:
                    # For a batch size 1, R2 score can be nan. We need to ignore nans
                    # when average in case the last batch is of size 1.
                    metrics_dict.add_metric(key, value, skip_nan_when_averaging=True, hue=hue)
                else:
                    metrics_dict.add_metric(key, value, hue=hue)

            assert _subject_ids is not None
            metrics_dict.add_predictions(_subject_ids, _model_output.detach().cpu().numpy(),
                                         _labels.cpu().numpy(), hue=hue)


def aggregate_segmentation_metrics(metrics: MetricsDict) -> MetricsDict:
    """
    Computes aggregate metrics for segmentation models, from a metrics dictionary that contains the results for
    individual minibatches. Specifically, average Dice scores for only the foreground structures and proportions
    of foreground voxels are computed. All metrics for the background class will be removed.
    All other metrics that are already present in the input metrics will be averaged and available in the result.
    Diagnostic values present in the input will be passed through unchanged.
    :param metrics: A metrics dictionary that contains the per-minibatch results.
    """
    class_names_with_background = metrics.get_hue_names(include_default=False)
    has_background_class = class_names_with_background[0] == BACKGROUND_CLASS_NAME
    foreground_classes = class_names_with_background[1:] if has_background_class else class_names_with_background
    result = metrics.average(across_hues=False)
    result.diagnostics = metrics.diagnostics.copy()
    if has_background_class:
        result.delete_hue(BACKGROUND_CLASS_NAME)
    add_average_foreground_dice(result)
    # Total number of voxels per class, including the background class
    total_voxels = []
    voxel_count = MetricType.VOXEL_COUNT.value
    for g in class_names_with_background:
        values = metrics.values(hue=g)
        if voxel_count in values:
            total_voxels.append(sum(values[voxel_count]))
    if len(total_voxels) > 0:
        # Proportion of voxels in foreground classes only
        proportion_foreground = np.array(total_voxels[1:], dtype=float) / sum(total_voxels)
        for i, foreground_class in enumerate(foreground_classes):
            result.add_metric(MetricType.PROPORTION_FOREGROUND_VOXELS, proportion_foreground[i], hue=foreground_class)
        result.add_metric(MetricType.PROPORTION_FOREGROUND_VOXELS, np.sum(proportion_foreground).item())
    return result


def add_average_foreground_dice(metrics: MetricsDict) -> None:
    """
    If the given metrics dictionary contains an entry for Dice score, and only one value for the Dice score per class,
    then add an average Dice score for all foreground classes to the metrics dictionary (modified in place).
    :param metrics: The object that holds metrics. The average Dice score will be written back into this object.
    """
    all_dice = []
    for structure_name in metrics.get_hue_names(include_default=False):
        if structure_name != BACKGROUND_CLASS_NAME:
            all_dice.append(metrics.get_single_metric(MetricType.DICE, hue=structure_name))
    metrics.add_metric(MetricType.DICE, np.nanmean(all_dice).item())
