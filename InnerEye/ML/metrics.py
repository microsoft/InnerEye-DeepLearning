#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import SimpleITK as sitk
import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
from azureml.core import Run
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional import roc
from pytorch_lightning.metrics.functional.classification import accuracy, auc, auroc, precision_recall_curve

from InnerEye.Azure.azure_util import get_run_context_or_default
from InnerEye.Common.metrics_dict import DataframeLogger, INTERNAL_TO_LOGGING_COLUMN_NAMES, MetricType, MetricsDict, \
    ScalarMetricsDict, \
    get_metric_name_with_hue_prefix
from InnerEye.Common.type_annotations import DictStrFloat, TupleFloat3
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import BACKGROUND_CLASS_NAME
from InnerEye.ML.scalar_config import ScalarLoss
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.ML.utils.image_util import binaries_from_multi_label_array, check_array_range, is_binary_array
from InnerEye.ML.utils.io_util import reverse_tuple_float3
from InnerEye.ML.utils.metrics_constants import LoggingColumns
from InnerEye.ML.utils.metrics_util import binary_classification_accuracy, \
    mean_absolute_error, r2_score
from InnerEye.ML.utils.ml_util import check_size_matches
from InnerEye.ML.utils.sequence_utils import get_masked_model_outputs_and_labels

from pytorch_lightning import metrics

class MeanAbsoluteError(metrics.MeanAbsoluteError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = MetricType.MEAN_ABSOLUTE_ERROR.value

class MeanSquaredError(metrics.MeanSquaredError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = MetricType.MEAN_SQUARED_ERROR.value

class ExplainedVariance(metrics.ExplainedVariance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = MetricType.EXPLAINED_VAR.value


class Accuracy05(metrics.Accuracy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = MetricType.ACCURACY_AT_THRESHOLD_05.value

class ScalarMetricsBase(Metric):
    def __init__(self, dist_sync_on_step: bool = False, name: str = "metric"):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.name = name

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self):
        raise NotImplementedError("Should be implemented in the child classes")

    def _get_metrics_at_optimal_cutoff(self, preds, targets) -> Tuple:
        """
        Computes the ROC to find the optimal cut-off i.e. the probability threshold for which the
        difference between true positive rate and false positive rate is smallest. Then, computes
        the false positive rate, false negative rate and accuracy at this threshold (i.e. when the
        predicted probability is higher than the threshold the predicted label is 1 otherwise 0).
        :param hue: The hue to restrict the values used for computation, otherwise all values will be used.
        :returns: Tuple(optimal_threshold, false positive rate, false negative rate, accuracy)
        """
        if torch.unique(targets).numel() == 1:
            return torch.tensor(np.nan), torch.tensor(np.nan), torch.tensor(np.nan), torch.tensor(np.nan)
        fpr, tpr, thresholds = roc(preds, targets)
        optimal_idx = torch.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        acc = accuracy(preds > optimal_threshold, targets)
        false_negative_optimal = 1 - tpr[optimal_idx]
        false_positive_optimal = fpr[optimal_idx]
        return optimal_threshold, false_positive_optimal, false_negative_optimal, acc


class AccuracyAtOptimalThreshold(ScalarMetricsBase):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step,
                         name=MetricType.ACCURACY_AT_OPTIMAL_THRESHOLD.value)

    def compute(self):
        preds = torch.cat(self.preds)
        target = torch.cat(self.targets)
        return self._get_metrics_at_optimal_cutoff(preds=preds, targets=target)[3]

class OptimalThreshold(ScalarMetricsBase):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, name=MetricType.OPTIMAL_THRESHOLD.value)

    def compute(self):
        preds = torch.cat(self.preds)
        target = torch.cat(self.targets)
        return self._get_metrics_at_optimal_cutoff(preds=preds, targets=target)[0]


class FalsePositiveRateOptimalThreshold(ScalarMetricsBase):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step,
                         name=MetricType.FALSE_POSITIVE_RATE_AT_OPTIMAL_THRESHOLD.value)

    def compute(self):
        preds = torch.cat(self.preds)
        target = torch.cat(self.targets)
        return self._get_metrics_at_optimal_cutoff(preds=preds, targets=target)[1]


class FalseNegativeRateOptimalThreshold(ScalarMetricsBase):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step,
                         name=MetricType.FALSE_NEGATIVE_RATE_AT_OPTIMAL_THRESHOLD.value)

    def compute(self):
        preds = torch.cat(self.preds)
        target = torch.cat(self.targets)
        return self._get_metrics_at_optimal_cutoff(preds=preds, targets=target)[2]


class AreaUnderRocCurve(ScalarMetricsBase):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step,
                         name=MetricType.AREA_UNDER_ROC_CURVE.value)

    def compute(self):
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        if torch.unique(targets).numel() == 1:
            return torch.tensor(np.nan)
        return auroc(preds, targets)



class AreaUnderPRCurve(ScalarMetricsBase):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step,
                         name=MetricType.AREA_UNDER_PR_CURVE.value)

    def compute(self):
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        if torch.unique(targets).numel() == 1:
            return torch.tensor(np.nan)
        prec, recall, _ = precision_recall_curve(preds, targets)
        return auc(recall, prec)

class BinaryCrossEntropy(ScalarMetricsBase):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, name=MetricType.CROSS_ENTROPY.value)

    def compute(self):
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return F.binary_cross_entropy(input=preds, target=targets)

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


def store_model_parameters(writer: tensorboardX.SummaryWriter,
                           epoch: int,
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


def log_classification_epoch_metrics(logging_fn: Callable[[str, float], None],
                                     metrics: MetricsDict) -> None:
    """
    Writes all values from MetricsDict object into a file for Tensorboard visualization,
    and into the AzureML run context.
    :param metrics: dictionary containing the metrics to be logged, averaged over minibatches.
    """
    for hue_name, label, metric in metrics.enumerate_single_values():
        logging_fn(get_metric_name_with_hue_prefix(label, hue_name), metric)


def nanmean(values: Iterable[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if len(valid) == 0:
        # This should only handle the case when all entries in a minibatch have no Dice score.
        return 0.0
    return sum(valid) / len(valid)


def add_average_dice(metrics: DictStrFloat) -> DictStrFloat:
    """
    Takes a dictionary containing metrics, searches for all metrics that have the "Dice/" prefix,
    and adds their average value as a new metric called "dice". If the argument already contains "dice",
    the result is equal to the argument.
    :param metrics: Dictionary with metrics, key is the metric name, value is the value of the metric.
    :return: A new metrics dictionary with a metric "dice" that is the average Dice score.
    """
    result = {}
    if LoggingColumns.Dice.value not in metrics:
        dice_dict = {name: value for name, value in metrics.items() if name.startswith(MetricType.DICE.value)}
        result[LoggingColumns.Dice.value] = nanmean(dice_dict.values())
    result.update(**metrics)
    return result


def store_epoch_metrics(metrics: DictStrFloat,
                        epoch: int,
                        file_logger: DataframeLogger,
                        cross_validation_split_index: int) -> None:
    """
    Writes all metrics into a CSV file, with additional columns for epoch and cross validation split.
    :param file_logger: An instance of DataframeLogger, for logging results to csv.
    :param epoch: The epoch corresponding to the results.
    :param metrics: The metrics of the specified epoch, averaged along its batches.
    :param cross_validation_split_index: The current split if running inside cross validation.
    """
    logger_row = {}
    for key, value in metrics.items():
        if key in INTERNAL_TO_LOGGING_COLUMN_NAMES.keys():
            logger_row[INTERNAL_TO_LOGGING_COLUMN_NAMES[key].value] = value
        else:
            logger_row[key] = value
    logger_row[LoggingColumns.Epoch.value] = epoch
    logger_row[LoggingColumns.CrossValidationSplitIndex.value] = cross_validation_split_index
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
                    MetricType.R2_SCORE: r2_score(_model_output, _labels)
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
