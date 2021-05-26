#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from azureml.core import Run

from InnerEye.Azure.azure_util import get_run_context_or_default
from InnerEye.Common.metrics_constants import LoggingColumns, MetricType
from InnerEye.Common.type_annotations import DictStrFloat, TupleFloat3
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import BACKGROUND_CLASS_NAME
from InnerEye.ML.metrics_dict import DataframeLogger, INTERNAL_TO_LOGGING_COLUMN_NAMES, MetricsDict, \
    ScalarMetricsDict
from InnerEye.ML.scalar_config import ScalarLoss
from InnerEye.ML.utils.image_util import binaries_from_multi_label_array, is_binary_array
from InnerEye.ML.utils.io_util import reverse_tuple_float3
from InnerEye.ML.utils.metrics_util import binary_classification_accuracy, mean_absolute_error, r2_score
from InnerEye.ML.utils.ml_util import check_size_matches
from InnerEye.ML.utils.sequence_utils import get_masked_model_outputs_and_labels

MAX_ITEM_LOAD_TIME_SEC = 0.5
MAX_LOAD_TIME_WARNINGS = 3
MAX_LOAD_TIME_EPOCHS = 5


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
    metrics: MetricsDict


@dataclass(frozen=True)
class InferenceMetricsForSegmentation(InferenceMetrics):
    """
    Stores metrics for segmentation models, per execution mode and epoch.
    """
    data_split: ModelExecutionMode
    metrics: float

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

        run_context.log_table(name=self.get_metrics_log_key(), value={
            "Dice": self.metrics
        })


@dataclass
class EpochTimers:
    """
    Contains all information necessary to compute the IO metrics: Epoch times, batch times, loading times.
    """
    epoch_start_time: float = time.time()
    epoch_end_time: float = time.time()
    batch_start_time: float = time.time()
    num_load_time_warnings: int = 0
    num_load_time_exceeded: int = 0
    total_extra_load_time: float = 0.0
    total_load_time: float = 0.0
    num_batches: int = 0
    load_time_warning_epochs: Set[int] = field(default_factory=set)

    def reset(self) -> None:
        """
        Resets all timers to the current time, and all counters to 0. The set of epochs for which warnings about
        load time were produced will not be reset.
        """
        current_time = time.time()
        self.epoch_start_time = current_time
        self.epoch_end_time = current_time
        self.batch_start_time = current_time
        self.num_load_time_warnings = 0
        self.num_load_time_exceeded = 0
        self.total_extra_load_time = 0.0
        self.total_load_time = 0.0
        self.num_batches = 0

    def epoch_end(self) -> None:
        """
        Stores the present time in the epoch_end_time field of the object.
        """
        self.epoch_end_time = time.time()

    @property
    def total_epoch_time(self) -> float:
        """
        Gets the time in seconds between epoch start and epoch end.
        """
        return self.epoch_end_time - self.epoch_start_time

    @property
    def should_warn_in_this_epoch(self) -> bool:
        """
        Returns True if warnings about loading time should be printed in the present epoch. Returns False if
        this warning has been printed already in more than MAX_LOAD_TIME_EPOCHS epochs.
        :return:
        """
        return len(self.load_time_warning_epochs) <= MAX_LOAD_TIME_EPOCHS

    def batch_start(self, batch_index: int, epoch: int, message_prefix: str) -> float:
        """
        Called when a minibatch of data has been loaded. This computes the time it took to load the minibatch,
        and adds it to the internal bookkeeping.
        :return: The time it took to load the minibatch, in seconds.
        """
        item_finish_time = time.time()
        item_load_time = item_finish_time - self.batch_start_time
        self.total_load_time += item_load_time
        # Having slow minibatch loading is OK in the very first batch of the every epoch, where processes
        # are spawned. Later, the load time should be zero.
        if batch_index == 0:
            logging.info(f"{message_prefix}: Loaded the first minibatch of data in {item_load_time:0.2f} sec.")
        elif item_load_time > MAX_ITEM_LOAD_TIME_SEC:
            self.load_time_warning_epochs.add(epoch)
            self.num_load_time_exceeded += 1
            self.total_extra_load_time += item_load_time
            if self.num_load_time_warnings < MAX_LOAD_TIME_WARNINGS and self.should_warn_in_this_epoch:
                logging.warning(f"{message_prefix}: Loading minibatch {batch_index} took {item_load_time:0.2f} sec. "
                                "This can mean that there are not enough data loader worker processes, or that there "
                                "is a performance problem in loading. This warning will be printed at most "
                                f"{MAX_LOAD_TIME_WARNINGS} times in at most {MAX_LOAD_TIME_EPOCHS} epochs.")
                self.num_load_time_warnings += 1
        return item_load_time

    def batch_end(self) -> float:
        """
        Called after a minibatch has been processed (training or validation step completed). Returns the time it took
        to process the current batch (including loading).
        :return: The time it took to process the current batch, in seconds.
        """
        current_time = time.time()
        elapsed = current_time - self.batch_start_time
        self.batch_start_time = current_time
        self.num_batches += 1
        return elapsed


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

    binary_classes = [is_binary_array(ground_truth[label_id]) for label_id in range(ground_truth.shape[0])]

    # If ground truth image is nan, then will not be used for metrics computation.
    nan_images = [np.isnan(np.sum(ground_truth[label_id])) for label_id in range(ground_truth.shape[0])]

    # Validates if not binary then nan
    assert np.all(np.array(binary_classes) == ~np.array(nan_images))

    #  Validates that all binary images should be 0 or 1
    if not np.all(np.array(binary_classes)[~np.array(nan_images)]):
        raise ValueError("Ground truth values should be 0 or 1")
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    metrics = MetricsDict(hues=ground_truth_ids)
    for i, prediction in enumerate(binaries):
        # Skips if background image or nan_image
        if i == 0 or nan_images[i]:
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
                                allow_multiple_classes_for_each_pixel: bool = False) -> torch.Tensor:
    """
    Computes the Dice scores for all classes across all patches in the arguments.
    :param segmentation: Tensor containing class ids predicted by a model.
    :param ground_truth: One-hot encoded torch tensor containing ground-truth label ids.
    :param allow_multiple_classes_for_each_pixel: If set to False, ground-truth tensor has
    to contain only one foreground label for each pixel.
    :return A torch tensor of size (Patches, Classes) with the Dice scores. Dice scores are computed for
    all classes including the background class at index 0.
    """
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


def store_epoch_metrics(metrics: DictStrFloat,
                        epoch: int,
                        file_logger: DataframeLogger) -> None:
    """
    Writes all metrics (apart from ones that measure run time) into a CSV file,
    with an additional columns for epoch number.
    :param file_logger: An instance of DataframeLogger, for logging results to csv.
    :param epoch: The epoch corresponding to the results.
    :param metrics: The metrics of the specified epoch, averaged along its batches.
    """
    logger_row = {}
    for key, value in metrics.items():
        tokens = key.split("/")
        if len(tokens) == 1:
            metric_name = tokens[0]
            hue_suffix = ""
        elif len(tokens) == 2:
            metric_name = tokens[0]
            hue_suffix = "/" + tokens[1]
        else:
            raise ValueError(f"Expected key to have format 'metric_name[/optional_suffix_for_hue]', got {key}")

        if metric_name == MetricType.SECONDS_PER_BATCH.value or metric_name == MetricType.SECONDS_PER_EPOCH.value:
            continue
        if metric_name in INTERNAL_TO_LOGGING_COLUMN_NAMES.keys():
            logger_row[INTERNAL_TO_LOGGING_COLUMN_NAMES[metric_name].value + hue_suffix] = value
        else:
            logger_row[metric_name + hue_suffix] = value
    logger_row[LoggingColumns.Epoch.value] = epoch
    file_logger.add_record(logger_row)
    file_logger.flush()


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
            # Convert labels to the same datatype as the model outputs, necessary when running with AMP
            _labels = _labels.to(dtype=_model_output.dtype)
            if loss_type == ScalarLoss.MeanSquaredError:
                metrics = {
                    MetricType.MEAN_SQUARED_ERROR: F.mse_loss(_model_output, _labels, reduction='mean').item(),
                    MetricType.MEAN_ABSOLUTE_ERROR: mean_absolute_error(_model_output, _labels),
                    MetricType.EXPLAINED_VAR: r2_score(_model_output, _labels)
                }
            else:
                metrics = {
                    MetricType.CROSS_ENTROPY: F.binary_cross_entropy(_model_output, _labels, reduction='mean').item(),
                    MetricType.ACCURACY_AT_THRESHOLD_05: binary_classification_accuracy(_model_output, _labels)
                }
            for key, value in metrics.items():
                if key == MetricType.EXPLAINED_VAR:
                    # For a batch size 1, R2 score can be nan. We need to ignore nans
                    # when average in case the last batch is of size 1.
                    metrics_dict.add_metric(key, value, skip_nan_when_averaging=True, hue=hue)
                else:
                    metrics_dict.add_metric(key, value, hue=hue)

            assert _subject_ids is not None
            metrics_dict.add_predictions(_subject_ids, _model_output.detach().cpu().numpy(),
                                         _labels.cpu().numpy(), hue=hue)


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
