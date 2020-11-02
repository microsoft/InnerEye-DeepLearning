#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from more_itertools import flatten
from sklearn.metrics import auc, log_loss, precision_recall_curve, roc_auc_score, roc_curve

from InnerEye.Azure.azure_util import DEFAULT_CROSS_VALIDATION_SPLIT_INDEX
from InnerEye.Common.common_util import DataframeLogger, check_properties_are_not_none
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.sequence_config import SEQUENCE_POSITION_HUE_NAME_PREFIX, SequenceModelBase
from InnerEye.ML.utils.io_util import tabulate_dataframe
from InnerEye.ML.utils.metrics_constants import LoggingColumns
from InnerEye.ML.utils.metrics_util import binary_classification_accuracy, mean_absolute_error, mean_squared_error, \
    r2_score

FloatOrInt = Union[float, int]
T = TypeVar('T', np.ndarray, float)


def create_metrics_dict_from_config(config: ScalarModelBase) -> Union[ScalarMetricsDict, SequenceMetricsDict]:
    """
    Create an instance of either a ScalarMetricsDict or SequenceMetricsDict, based on the
    type of config provided.
    :param config: Model configuration information.
    """
    if isinstance(config, SequenceModelBase):
        return SequenceMetricsDict.create_from_config(config)
    else:
        return ScalarMetricsDict.create_from_config(config)


def average_metric_values(values: List[float], skip_nan_when_averaging: bool) -> float:
    """
    Returns the average (arithmetic mean) of the values provided. If skip_nan_when_averaging is True, the mean
    will be computed without any possible NaN values in the list.
    :param values: The individual values that should be averaged.
    :param skip_nan_when_averaging: If True, compute mean with any NaN values. If False, any NaN value present
    in the argument will make the function return NaN.
    :return: The average of the provided values. If the argument is an empty list, NaN will be returned.
    """
    if skip_nan_when_averaging:
        return np.nanmean(values).item()
    else:
        return np.mean(values).item()


@dataclass(frozen=True)
class PredictionEntry(Generic[T]):
    subject_id: str
    predictions: T
    labels: T

    def __post_init__(self) -> None:
        check_properties_are_not_none(self)


@unique
class MetricType(Enum):
    """
    Contains the different metrics that are computed.
    """
    # Any result of loss computation, depending on what's configured in the model.
    LOSS = "Loss"

    # Classification metrics
    CROSS_ENTROPY = "CrossEntropy"
    # Classification accuracy assuming that posterior > 0.5 means predicted class 1
    ACCURACY_AT_THRESHOLD_05 = "AccuracyAtThreshold05"
    ACCURACY_AT_OPTIMAL_THRESHOLD = "AccuracyAtOptimalThreshold"
    # Metrics for segmentation
    DICE = "Dice"
    HAUSDORFF_mm = "HausdorffDistance_millimeters"
    MEAN_SURFACE_DIST_mm = "MeanSurfaceDistance_millimeters"
    VOXEL_COUNT = "VoxelCount"
    PROPORTION_FOREGROUND_VOXELS = "ProportionForegroundVoxels"

    PATCH_CENTER = "PatchCenter"

    AREA_UNDER_ROC_CURVE = "AreaUnderRocCurve"
    AREA_UNDER_PR_CURVE = "AreaUnderPRCurve"
    OPTIMAL_THRESHOLD = "OptimalThreshold"
    FALSE_POSITIVE_RATE_AT_OPTIMAL_THRESHOLD = "FalsePositiveRateAtOptimalThreshold"
    FALSE_NEGATIVE_RATE_AT_OPTIMAL_THRESHOLD = "FalseNegativeRateAtOptimalThreshold"

    # Regression metrics
    MEAN_ABSOLUTE_ERROR = "MeanAbsoluteError"
    MEAN_SQUARED_ERROR = "MeanSquaredError"
    R2_SCORE = "r2Score"

    # Common metrics
    SECONDS_PER_BATCH = "SecondsPerBatch"
    SECONDS_PER_EPOCH = "SecondsPerEpoch"
    SUBJECT_COUNT = "SubjectCount"
    LEARNING_RATE = "LearningRate"


MetricTypeOrStr = Union[str, MetricType]

# Mapping from the internal logging column names to the ones used in the outside-facing pieces of code:
# Output data files, logging systems.
INTERNAL_TO_LOGGING_COLUMN_NAMES = {
    MetricType.LOSS.value: LoggingColumns.Loss,
    MetricType.ACCURACY_AT_THRESHOLD_05.value: LoggingColumns.AccuracyAtThreshold05,
    MetricType.CROSS_ENTROPY.value: LoggingColumns.CrossEntropy,
    MetricType.SECONDS_PER_BATCH.value: LoggingColumns.SecondsPerBatch,
    MetricType.SECONDS_PER_EPOCH.value: LoggingColumns.SecondsPerEpoch,
    MetricType.AREA_UNDER_ROC_CURVE.value: LoggingColumns.AreaUnderRocCurve,
    MetricType.AREA_UNDER_PR_CURVE.value: LoggingColumns.AreaUnderPRCurve,
    MetricType.SUBJECT_COUNT.value: LoggingColumns.SubjectCount,
    MetricType.MEAN_SQUARED_ERROR.value: LoggingColumns.MeanSquaredError,
    MetricType.MEAN_ABSOLUTE_ERROR.value: LoggingColumns.MeanAbsoluteError,
    MetricType.R2_SCORE.value: LoggingColumns.R2Score,
    MetricType.LEARNING_RATE.value: LoggingColumns.LearningRate,
    MetricType.ACCURACY_AT_OPTIMAL_THRESHOLD.value: LoggingColumns.AccuracyAtOptimalThreshold,
    MetricType.OPTIMAL_THRESHOLD.value: LoggingColumns.OptimalThreshold,
    MetricType.FALSE_POSITIVE_RATE_AT_OPTIMAL_THRESHOLD.value: LoggingColumns.FalsePositiveRateAtOptimalThreshold,
    MetricType.FALSE_NEGATIVE_RATE_AT_OPTIMAL_THRESHOLD.value: LoggingColumns.FalseNegativeRateAtOptimalThreshold
}


def get_column_name_for_logging(metric_name: Union[str, MetricType],
                                hue_name: Optional[str] = None) -> str:
    """
    Computes the column name that should be used when logging a metric to disk.
    Raises a value error when no column name has yet been defined.
    :param metric_name: The name of the metric.
    :param hue_name: If provided will be used as a prefix hue_name/column_name
    """
    metric_str = metric_name if isinstance(metric_name, str) else metric_name.value
    if metric_str in INTERNAL_TO_LOGGING_COLUMN_NAMES:
        return get_metric_name_with_hue_prefix(INTERNAL_TO_LOGGING_COLUMN_NAMES[metric_str].value, hue_name)
    raise ValueError(f"No column name mapping defined for metric '{metric_str}'")


def get_metric_name_with_hue_prefix(metric_name: str, hue_name: Optional[str] = None) -> str:
    """
    If hue_name is provided and is not equal to the default hue then it will be
    used as a prefix hue_name/column_name, otherwise metric_name will be returned.
    """
    prefix = f"{hue_name}/" if hue_name and hue_name is not MetricsDict.DEFAULT_HUE_KEY else ''
    return f"{prefix}{metric_name}"


@dataclass
class Hue:
    """
    Dataclass to encapsulate hue specific data related for metrics computation.
    """
    name: str
    values: Dict[str, List[FloatOrInt]] = field(default_factory=dict)
    predictions: List[np.ndarray] = field(default_factory=list)
    labels: List[np.ndarray] = field(default_factory=list)
    subject_ids: List[str] = field(default_factory=list)

    @property
    def has_prediction_entries(self) -> bool:
        """
        Returns True if the present object stores any entries for computing the Area Under Roc Curve metric.
        """
        _labels = self.labels
        return len(_labels) > 0 if _labels else False

    def add_predictions(self,
                        subject_ids: Sequence[str],
                        predictions: np.ndarray,
                        labels: np.ndarray) -> None:
        """
        Adds predictions and labels for later computing the area under the ROC curve.
        :param subject_ids: Subject ids associated with the predictions and labels.
        :param predictions: A numpy array with model predictions, of size [N x C] for N samples in C classes, or size
        [N x 1] or size [N] for binary.
        :param labels: A numpy array with labels,  of size [N x C] for N samples in C classes, or size
        [N x 1] or size [N] for binary.
        """
        if predictions.ndim == 1:
            predictions = np.expand_dims(predictions, axis=1)
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=1)
        if not (len(predictions) == len(labels) == len(subject_ids)):
            raise ValueError("predictions, labels and subject_ids must have the same length in dimension 0 "
                             f"found predictions={len(predictions)}, labels={len(labels)}, "
                             f"and subject_ids={len(subject_ids)}")
        self.subject_ids += subject_ids
        self.predictions.append(predictions)
        self.labels.append(labels)

    def get_predictions(self) -> np.ndarray:
        """
        Return a concatenated copy of the roc predictions stored internally.
        """

        return Hue._concat_if_needed(self.predictions)

    def get_labels(self) -> np.ndarray:
        """
        Return a concatenated copy of the roc labels stored internally.
        """
        return Hue._concat_if_needed(self.labels)

    def get_predictions_and_labels_per_subject(self) -> List[PredictionEntry[float]]:
        """
        Gets the per-subject predictions that are stored in the present object.
        """
        predictions = self.get_predictions()
        labels = self.get_labels()
        if not (len(self.subject_ids) == len(labels) == len(predictions)):
            raise ValueError(f"Inconsistent number of predictions stored: "
                             f"{len(self.subject_ids)} subjects, "
                             f"{len(labels)} labels, "
                             f"{len(predictions)} predictions.")
        return [PredictionEntry(subject_id=x,
                                predictions=predictions[i][0],
                                labels=labels[i][0])
                for i, x in enumerate(self.subject_ids)]

    @staticmethod
    def _concat_if_needed(arrays: List[np.ndarray]) -> np.ndarray:
        """
        Joins a list of arrays into a single array, taking empty lists into account correctly.
        :param arrays: Array list to be concatenated.
        """
        if arrays:
            return np.concatenate(arrays, axis=0)
        return np.array([])

    def enumerate_single_values(self) -> Iterable[Tuple[str, float]]:
        """
        Returns an iterator that contains all (metric name, metric value) tuples that are stored in the
        present object. The method assumes that there is exactly 1 metric value stored per name, and throws a
        ValueError if that is not the case.
        :return: An iterator with (metric name, metric value) pairs.
        """
        for metric_name, metric_value in self.values.items():
            if len(metric_value) == 1:
                yield metric_name, metric_value[0]
            else:
                raise ValueError(f"Expected that all metrics lists only hold 1 item, "
                                 f"but got this list for Hue {self.name} : metric "
                                 f"'{metric_name}': {metric_value}")


class MetricsDict:
    """
    This class helps aggregate an arbitrary number of metrics across multiple batches or multiple samples. Metrics are
    identified by a string name. Metrics can have further hues which are isolated metrics records, and can be used
    for cases such as different anatomical structures, where we might want to maintain separate metrics for each
    structure, to perform independent aggregations.
    """

    DEFAULT_HUE_KEY = "Default"
    # the columns used when metrics dict is converted to a data frame/string representation
    DATAFRAME_COLUMNS = [LoggingColumns.Hue.value, "metrics"]

    def __init__(self, hues: Optional[List[str]] = None, is_classification_metrics: bool = True) -> None:
        """
        :param hues: Supported hues for this metrics dict, otherwise all records will belong to the
        default hue.
        :param is_classification_metrics: If this is a classification metrics dict
        """
        if hues and MetricsDict.DEFAULT_HUE_KEY in hues:
            hues.remove(MetricsDict.DEFAULT_HUE_KEY)
        self.hues_without_default = hues or []
        _hue_keys = self.hues_without_default + [MetricsDict.DEFAULT_HUE_KEY]
        self.hues: OrderedDict[str, Hue] = OrderedDict([(x, Hue(name=x)) for x in _hue_keys])
        self.skip_nan_when_averaging: Dict[str, bool] = dict()
        self.row_labels: List[str] = list()
        self.is_classification_metrics = is_classification_metrics
        self.diagnostics: Dict[str, List[Any]] = dict()

    def subject_ids(self, hue: str = DEFAULT_HUE_KEY) -> List[str]:
        """
        Return the subject ids that have metrics associated with them in this dictionary.
        :param hue: If provided then subject ids belonging to this hue only will be returned.
        Otherwise subject ids for the default hue will be returned.
        """
        return self._get_hue(hue=hue).subject_ids

    def get_hue_names(self, include_default: bool = True) -> List[str]:
        """
        Returns all of the hues supported by this metrics dict
        :param include_default: Include the default hue if True, otherwise exclude the default hue.
        """
        _hue_names = list(self.hues.keys())
        if not include_default:
            _hue_names.remove(MetricsDict.DEFAULT_HUE_KEY)
        return _hue_names

    def delete_hue(self, hue: str) -> None:
        """
        Removes all data stored for the given hue from the present object.
        :param hue: The hue to remove.
        """
        del self.hues[hue]

    def get_single_metric(self, metric_name: MetricTypeOrStr, hue: str = DEFAULT_HUE_KEY) -> FloatOrInt:
        """
        Gets the value stored for the given metric. The method assumes that there is a single value stored for the
        metric, and raises a ValueError if that is not the case.
        :param metric_name: The name of the metric to retrieve.
        :param hue: The hue to retrieve the metric from.
        :return:
        """
        name = MetricsDict._metric_name(metric_name)
        values = self.values(hue)[name]
        if len(values) == 1:
            return values[0]
        raise ValueError(f"Expected a single entry for metric '{name}', but got {len(values)}")

    def has_prediction_entries(self, hue: str = DEFAULT_HUE_KEY) -> bool:
        """
        Returns True if the present object stores any entries for computing the Area Under Roc Curve metric.
        :param hue: will be used to check a particular hue otherwise default hue will be used.
        :return: True if entries exist. False otherwise.
        """
        return self._get_hue(hue).has_prediction_entries

    def values(self, hue: str = DEFAULT_HUE_KEY) -> Dict[str, Any]:
        """
        Returns values held currently in the dict
        :param hue: will be used to restrict values for the provided hue otherwise values in the default
        hue will be returned.
        :return: Dictionary of values for this object.
        """
        return self._get_hue(hue).values

    def add_diagnostics(self, name: str, value: Any) -> None:
        """
        Adds a diagnostic value to the present object. Multiple diagnostics can be stored per unique value of name,
        the values get concatenated.
        :param name: The name of the diagnostic value to store.
        :param value: The value to store.
        """
        if name in self.diagnostics:
            # There is already an entry, append to the end of the list
            self.diagnostics[name].append(value)
        else:
            self.diagnostics[name] = [value]

    @staticmethod
    def _metric_name(metric_name: MetricTypeOrStr) -> str:
        """
        Converts a metric name, given either as an enum or a string, to a string.
        """
        if isinstance(metric_name, MetricType):
            return metric_name.value
        return str(metric_name)

    def add_metric(self,
                   metric_name: Union[str, MetricType],
                   metric_value: FloatOrInt,
                   skip_nan_when_averaging: bool = False,
                   hue: str = DEFAULT_HUE_KEY) -> None:
        """
        Adds values for a single metric to the present object, when the metric value is a scalar.
        :param metric_name: The name of the metric to add. This can be a string or a value in the MetricType enum.
        :param metric_value: The values of the metric, as a float or integer.
        :param skip_nan_when_averaging: If True, averaging this metric will skip any NaN (not a number) values.
        If False, NaN will propagate through the mean computation.
        :param hue: The hue for which this record belongs to, default hue will be used if None provided.
        """
        _metric_name = MetricsDict._metric_name(metric_name)
        if isinstance(metric_value, (float, int)):
            _values = self._get_hue(hue).values
            if _metric_name in _values:
                # There is already an entry for this metric, append to the end of the list
                _values[_metric_name].append(metric_value)
            else:
                _values[_metric_name] = [metric_value]
        else:
            raise ValueError(f"Expected the metric to be a scalar (float or int), but got: {type(metric_value)}")
        self.skip_nan_when_averaging[_metric_name] = skip_nan_when_averaging

    def delete_metric(self,
                      metric_name: Union[str, MetricType],
                      hue: str = DEFAULT_HUE_KEY) -> None:
        """
        Deletes all values that are stored for a given metric from the present object.
        :param metric_name: The name of the metric to add. This can be a string or a value in the MetricType enum.
        :param hue: The hue for which this record belongs to, default hue will be used if None provided.
        """
        _metric_name = MetricsDict._metric_name(metric_name)
        del self._get_hue(hue).values[_metric_name]

    def add_predictions(self, subject_ids: Sequence[str],
                        predictions: np.ndarray,
                        labels: np.ndarray,
                        hue: str = DEFAULT_HUE_KEY) -> None:
        """
        Adds predictions and labels for later computing the area under the ROC curve.
        :param subject_ids: Subject ids associated with the predictions and labels.
        :param predictions: A numpy array with model predictions, of size [N x C] for N samples in C classes, or size
        [N x 1] or size [N] for binary.
        :param labels: A numpy array with labels,  of size [N x C] for N samples in C classes, or size
        [N x 1] or size [N] for binary.
        :param hue: The hue this prediction belongs to, default hue will be used if None provided.
        """
        self._get_hue(hue).add_predictions(subject_ids=subject_ids,
                                           labels=labels,
                                           predictions=predictions)

    def num_entries(self, hue: str = DEFAULT_HUE_KEY) -> Dict[str, int]:
        """
        Gets the number of values that are stored for each individual metric.
        :param hue: The hue to count entries for, otherwise all entries will be counted.
        :return: A dictionary mapping from metric name to number of values stored.
        """
        _values = self._get_hue(hue).values
        return {m: len(v) for m, v in _values.items()}

    def average(self,
                add_metrics_from_entries: bool = False,
                across_hues: bool = True) -> MetricsDict:
        """
        Returns a MetricsDict object that only contains the per-metric averages (arithmetic mean) from the present
        object.
        Computing the average will respect the skip_nan_when_averaging value that has been provided when adding
        the metric.
        :param add_metrics_from_entries: average existing metrics in the dict.
        :param across_hues: If True then same metric types will be averaged regardless of hues, otherwise
        separate averages for each metric type for each hue will be computed, Default is True.
        :return: A MetricsDict object with a single-item list for each of the metrics.
        """

        def _get_all_metrics() -> List[Tuple[str, str, Any]]:
            _all_values = {}
            for _hue in self.get_hue_names():
                _values = self.values(_hue)
                if self.has_prediction_entries(_hue):
                    if self.is_classification_metrics:
                        _values[MetricType.AREA_UNDER_ROC_CURVE.value] = [self.get_roc_auc(_hue)]
                        _values[MetricType.AREA_UNDER_PR_CURVE.value] = [self.get_pr_auc(_hue)]
                        # Add metrics at optimal cut-off
                        optimal_threshold, fpr, fnr, accuracy = self.get_metrics_at_optimal_cutoff(_hue)
                        _values[MetricType.ACCURACY_AT_OPTIMAL_THRESHOLD.value] = [accuracy]
                        _values[MetricType.FALSE_POSITIVE_RATE_AT_OPTIMAL_THRESHOLD.value] = [fpr]
                        _values[MetricType.FALSE_NEGATIVE_RATE_AT_OPTIMAL_THRESHOLD.value] = [fnr]
                        _values[MetricType.OPTIMAL_THRESHOLD.value] = [optimal_threshold]

                        if add_metrics_from_entries:
                            if MetricType.CROSS_ENTROPY.value in _values:
                                raise ValueError(
                                    "Unable to add cross entropy because this metric is already present in the dict.")
                            else:
                                _values[MetricType.CROSS_ENTROPY.value] = [self.get_cross_entropy(_hue)]
                                _values[MetricType.ACCURACY_AT_THRESHOLD_05.value] = [self.get_accuracy_at05(_hue)]
                    else:
                        if add_metrics_from_entries:
                            _values[MetricType.MEAN_ABSOLUTE_ERROR.value] = [self.get_mean_absolute_error(_hue)]
                            _values[MetricType.MEAN_SQUARED_ERROR.value] = [self.get_mean_squared_error(_hue)]
                            _values[MetricType.R2_SCORE.value] = [self.get_r2_score(_hue)]

                    _values[MetricType.SUBJECT_COUNT.value] = [len(self.get_predictions(_hue))]
                _all_values[_hue] = _values
            # noinspection PyTypeChecker
            return list(flatten([list(map(lambda x: (k, *x), v.items())) for k, v in _all_values.items()]))  # type: ignore

        def _fill_new_metrics_dict(m: MetricsDict, average: bool = False) -> MetricsDict:
            for _m_hue, _m_metric_name, _m_value in _get_all_metrics():
                skip_nan = self.skip_nan_when_averaging.get(_m_metric_name, False)  # type: ignore
                if average:
                    m.add_metric(_m_metric_name,
                                 average_metric_values(_m_value, skip_nan_when_averaging=skip_nan),
                                 hue=_m_hue)
                else:
                    for _v in _m_value:
                        m.add_metric(_m_metric_name, _v, skip_nan_when_averaging=skip_nan)
            return m

        if across_hues:
            return _fill_new_metrics_dict(MetricsDict()).average(across_hues=False)
        else:
            return _fill_new_metrics_dict(MetricsDict(hues=self.get_hue_names(include_default=False)), average=True)

    def get_accuracy_at05(self, hue: str = DEFAULT_HUE_KEY) -> float:
        """
        Returns the binary classification accuracy at threshold 0.5
        """
        return binary_classification_accuracy(model_output=self.get_predictions(hue=hue),
                                              label=self.get_labels(hue=hue))

    @classmethod
    def get_optimal_idx(cls, fpr: np.ndarray, tpr: np.ndarray) -> np.ndarray:
        """
        Given a list of FPR and TPR values corresponding to different thresholds, compute the index which corresponds
        to the optimal threshold.
        """
        optimal_idx = np.argmax(tpr - fpr)
        return optimal_idx

    def get_metrics_at_optimal_cutoff(self, hue: str = DEFAULT_HUE_KEY) -> Tuple:
        """
        Computes the ROC to find the optimal cut-off i.e. the probability threshold for which the
        difference between true positive rate and false positive rate is smallest. Then, computes
        the false positive rate, false negative rate and accuracy at this threshold (i.e. when the
        predicted probability is higher than the threshold the predicted label is 1 otherwise 0).
        :param hue: The hue to restrict the values used for computation, otherwise all values will be used.
        :returns: Tuple(optimal_threshold, false positive rate, false negative rate, accuracy)
        """
        fpr, tpr, thresholds = roc_curve(self.get_labels(hue=hue), self.get_predictions(hue=hue))
        optimal_idx = MetricsDict.get_optimal_idx(fpr=fpr, tpr=tpr)
        optimal_threshold = float(thresholds[optimal_idx])
        accuracy = binary_classification_accuracy(model_output=self.get_predictions(hue=hue),
                                                  label=self.get_labels(hue=hue),
                                                  threshold=optimal_threshold)
        false_negative_optimal = 1 - tpr[optimal_idx]
        false_positive_optimal = fpr[optimal_idx]
        return optimal_threshold, false_positive_optimal, false_negative_optimal, accuracy

    def get_roc_auc(self, hue: str = DEFAULT_HUE_KEY) -> float:
        """
        Computes the Area Under the ROC curve, from the entries that were supplied in the add_roc_entries method.
        :param hue: The hue to restrict the values used for computation, otherwise all values will be used.
        :return: The AUC score, or np.nan if no entries are available in the present object.
        """
        if not self.has_prediction_entries(hue):
            return np.nan
        predictions = self.get_predictions(hue)
        labels = self.get_labels(hue)
        if predictions.shape[1] == 1 and labels.shape[1] == 1 and len(np.unique(labels)) == 1:
            # We are dealing with a binary classification problem, but there is only a single class present
            # in the data: This happens occasionaly in test data. Return 1.0 because in such cases we could
            # always get a classifier threshold that correctly classifies everything.
            return 1.0
        else:
            return roc_auc_score(labels, predictions)

    def get_pr_auc(self, hue: str = DEFAULT_HUE_KEY) -> float:
        """
        Computes the Area Under the Precision Recall Curve, from the entries that were supplied in the
        add_roc_entries method.
        :param hue: The hue to restrict the values used for computation, otherwise all values will be used.
        :return: The PR AUC score, or np.nan if no entries are available in the present object.
        """
        if not self.has_prediction_entries(hue):
            return np.nan
        predictions = self.get_predictions(hue)
        labels = self.get_labels(hue)
        if predictions.shape[1] == 1 and labels.shape[1] == 1 and len(np.unique(labels)) == 1:
            # We are dealing with a binary classification problem, but there is only a single class present
            # in the data: This happens occasionaly in test data. Return 1.0 because in such cases we could
            # always get a classifier threshold that correctly classifies everything.
            return 1.0
        precision, recall, _ = precision_recall_curve(labels, predictions)
        return auc(recall, precision)

    def get_cross_entropy(self, hue: str = DEFAULT_HUE_KEY) -> float:
        """
        Computes the binary cross entropy from the entries that were supplied in the
        add_roc_entries method.
        :param hue: The hue to restrict the values used for computation, otherwise all values will be used.
        :return: The cross entropy score.
        """
        predictions = self.get_predictions(hue)
        labels = self.get_labels(hue)
        return log_loss(labels, predictions)

    def get_mean_absolute_error(self, hue: str = DEFAULT_HUE_KEY) -> float:
        """
        Get the mean absolute error.
        :param hue: The hue to restrict the values used for computation, otherwise all values will be used.
        :return: Mean absolute error.
        """
        return mean_absolute_error(model_output=self.get_predictions(hue), label=self.get_labels(hue))

    def get_mean_squared_error(self, hue: str = DEFAULT_HUE_KEY) -> float:
        """
        Get the mean squared error.
        :param hue: The hue to restrict the values used for computation, otherwise all values will be used.
        :return: Mean squared error
        """
        return mean_squared_error(model_output=self.get_predictions(hue), label=self.get_labels(hue))

    def get_r2_score(self, hue: str = DEFAULT_HUE_KEY) -> float:
        """
        Get the R2 score.
        :param hue: The hue to restrict the values used for computation, otherwise all values will be used.
        :return: R2 score
        """
        return r2_score(model_output=self.get_predictions(hue), label=self.get_labels(hue))

    def enumerate_single_values(self, hue: Optional[str] = None) -> Iterable[Tuple[str, str, float]]:
        """
        Returns an iterator that contains all (hue name, metric name, metric values) tuples that are stored in the
        present object. This method assumes that for each hue/metric combination there is exactly 1 value, and it
        throws an exception if that is more than 1 value.
        :param hue: The hue to restrict the values, otherwise all values will be used if set to None.
        :return: An iterator with (hue name, metric name, metric values) pairs.
        """
        for _hue, metric_name, values in self._enumerate_values(hue=hue, ensure_singleton_values_only=True):
            yield _hue, metric_name, values[0]

    def _enumerate_values(self, hue: Optional[str] = None,
                          ensure_singleton_values_only: bool = False) \
            -> Iterable[Tuple[str, str, List[float]]]:
        """
        Returns an iterator that contains all (hue name, metric name, metric values) tuples that are stored in the
        present object.
        :param hue: The hue to restrict the values, otherwise all values will be used if set to None.
        :param ensure_singleton_values_only: Ensure that each of the values return is a singleton.
        :return: An iterator with (hue name, metric name, metric values) pairs.
        """
        _hues_to_iterate = [hue] if hue is not None else self.get_hue_names()
        for _hue in _hues_to_iterate:
            _values = self._get_hue(_hue).values
            for metric_name, metric_value in _values.items():
                if ensure_singleton_values_only and len(metric_value) != 1:
                    raise ValueError(f"Expected that all metrics lists only hold 1 item, "
                                     f"but got this list for Hue {_hue} : metric "
                                     f"'{metric_name}': {metric_value}")

                yield _hue, metric_name, metric_value

    def enumerate_single_values_groupwise(self) -> Iterable[Tuple[str, Iterable[Tuple[str, float]]]]:
        """
        Returns an iterator that contains (hue name, metric_name_and_value) tuples that are stored in the
        present object. The second tuple element is again an iterator that returns all metric name and value tuples
        that are stored for that specific hue. This method assumes that for each hue/metric combination there is
        exactly 1 value, and it throws an exception if that is more than 1 value.
        :return: An iterator with (hue name, metric_name_and_value) pairs.
        """
        _hues_to_iterate = [MetricsDict.DEFAULT_HUE_KEY] + self.get_hue_names(include_default=False)
        for _hue in _hues_to_iterate:
            yield _hue, self._get_hue(_hue).enumerate_single_values()

    def get_predictions(self, hue: str = DEFAULT_HUE_KEY) -> np.ndarray:
        """
        Return a concatenated copy of the roc predictions stored internally.
        :param hue: The hue to restrict the values, otherwise all values will be used.
        :return: concatenated roc predictions as np array
        """
        return self._get_hue(hue).get_predictions()

    def get_labels(self, hue: str = DEFAULT_HUE_KEY) -> np.ndarray:
        """
        Return a concatenated copy of the roc labels stored internally.
        :param hue: The hue to restrict the values, otherwise all values will be used.
        :return: roc labels as np array
        """
        return self._get_hue(hue).get_labels()

    def get_predictions_and_labels_per_subject(self, hue: str = DEFAULT_HUE_KEY) \
            -> List[PredictionEntry[float]]:
        """
        Gets the per-subject labels and predictions that are stored in the present object.
        :param hue: The hue to restrict the values, otherwise the default hue will be used.
        :return: List of per-subject labels and predictions
        """
        return self._get_hue(hue).get_predictions_and_labels_per_subject()

    def to_string(self, tabulate: bool = True) -> str:
        """
        Creates a multi-line human readable string from the given metrics.
        :param tabulate: If True then create a pretty printable table string.
        :return: Formatted metrics string
        """
        df = self.to_data_frame()
        return tabulate_dataframe(df) if tabulate else df.to_string(index=False)

    def to_data_frame(self) -> pd.DataFrame:
        """
        Creates a data frame representation of the metrics dict in the format with the
        Hue name as a column and a string representation of all metrics for that hue as a second column.
        """

        def _format_metric_values(x: Union[List[float], float]) -> str:
            x = [x] if isinstance(x, float) else x
            _x = [f"{y:0.4f}" for y in x]
            return str(_x[0] if len(_x) == 1 else _x)

        info_df = pd.DataFrame(columns=MetricsDict.DATAFRAME_COLUMNS)
        for hue in self.get_hue_names():
            info_list = [f"{metric_name}: {_format_metric_values(metric_values)}"
                         for _, metric_name, metric_values in self._enumerate_values(hue=hue)]
            if info_list:
                info_list_str = ", ".join(info_list)
                info_df = info_df.append({MetricsDict.DATAFRAME_COLUMNS[0]: hue,
                                          MetricsDict.DATAFRAME_COLUMNS[1]: info_list_str}, ignore_index=True)
        return info_df

    def _get_hue(self, hue: str = DEFAULT_HUE_KEY) -> Hue:
        """
        Get the hue record for the provided key.
        Raises a KeyError if the provided hue key does not exist.
        :param hue: The hue to retrieve record for
        """
        if hue not in self.hues:
            raise KeyError(f"Unknown hue '{hue}' provided, key value must be one of {self.hues.keys()}")
        else:
            return self.hues[hue]


class ScalarMetricsDict(MetricsDict):
    """
    Specialization of the MetricsDict with Classification related functions.
    """

    def __init__(self, hues: Optional[List[str]] = None, is_classification_metrics: bool = True) -> None:
        super().__init__(hues, is_classification_metrics=is_classification_metrics)

    @staticmethod
    def create_from_config(config: ScalarModelBase) -> ScalarMetricsDict:
        """
        Creates an instance of the ScalarMetricsDict from the provided ScalarModelBase config.
        Label channels for the provided model config will be used to set the hues for this dictionary.
        :param config: ScalarModelBase
        :return: ScalarMetricsDict
        """
        return ScalarMetricsDict(is_classification_metrics=config.is_classification_model)

    def binary_classification_accuracy(self, hue: str = MetricsDict.DEFAULT_HUE_KEY) -> float:
        """
        :param hue: The hue to restrict the values, otherwise all values will be used.
        :return: binary classification accuracy
        """
        return binary_classification_accuracy(model_output=self.get_predictions(hue=hue),
                                              label=self.get_labels(hue=hue))

    def store_metrics_per_subject(self,
                                  epoch: int,
                                  df_logger: DataframeLogger,
                                  mode: ModelExecutionMode,
                                  cross_validation_split_index: int = DEFAULT_CROSS_VALIDATION_SPLIT_INDEX) -> None:
        """
        Store metrics using the provided df_logger at subject level for classification models.
        :param epoch: Epoch these metrics are computed for.
        :param df_logger: A data frame logger to use to write the metrics to disk.
        :param mode: Model execution mode these metrics belong to.
        :param cross_validation_split_index: cross validation split index for the epoch if performing cross val
        :return:
        """
        for hue in self.get_hue_names():
            for prediction_entry in self.get_predictions_and_labels_per_subject(hue=hue):
                df_logger.add_record({
                    LoggingColumns.Hue.value: hue,
                    LoggingColumns.Epoch.value: epoch,
                    LoggingColumns.Patient.value: prediction_entry.subject_id,
                    LoggingColumns.ModelOutput.value: prediction_entry.predictions,
                    LoggingColumns.Label.value: prediction_entry.labels,
                    LoggingColumns.CrossValidationSplitIndex.value: cross_validation_split_index,
                    LoggingColumns.DataSplit.value: mode.value
                })

    @staticmethod
    def load_execution_mode_metrics_from_df(df: pd.DataFrame,
                                            is_classification_metrics: bool) -> Dict[ModelExecutionMode,
                                                                                     Dict[int, ScalarMetricsDict]]:
        """
        Helper function to create BinaryClassificationMetricsDict grouped by ModelExecutionMode and epoch
        from a given dataframe. The following columns must exist in the provided data frame:
        >>> LoggingColumns.DataSplit
        >>> LoggingColumns.Epoch

        :param df: DataFrame to use for creating the metrics dict.
        :param is_classification_metrics: If the current metrics are for classification or not.
        """
        has_hue_column = LoggingColumns.Hue.value in df
        group_columns = [LoggingColumns.DataSplit.value, LoggingColumns.Epoch.value]
        if has_hue_column:
            group_columns.append(LoggingColumns.Hue.value)
        grouped = df.groupby(group_columns)
        result: Dict[ModelExecutionMode, Dict[int, ScalarMetricsDict]] = dict()
        hues = []
        if has_hue_column:
            hues = [h for h in df[LoggingColumns.Hue.value].unique() if h]
        for name, group in grouped:
            if has_hue_column:
                mode_str, epoch, hue = name
            else:
                mode_str, epoch = name
                hue = MetricsDict.DEFAULT_HUE_KEY
            mode = ModelExecutionMode(mode_str)
            if mode not in result:
                result[mode] = dict()
            if epoch not in result[mode]:
                result[mode][epoch] = ScalarMetricsDict(is_classification_metrics=is_classification_metrics,
                                                        hues=hues)
            subjects = list(group[LoggingColumns.Patient.value].values)
            predictions = group[LoggingColumns.ModelOutput.value].to_numpy(dtype=np.float)
            labels = group[LoggingColumns.Label.value].to_numpy(dtype=np.float)
            result[mode][epoch].add_predictions(subjects, predictions, labels, hue=hue)

        return result

    @staticmethod
    def aggregate_and_save_execution_mode_metrics(metrics: Dict[ModelExecutionMode, Dict[int, ScalarMetricsDict]],
                                                  data_frame_logger: DataframeLogger,
                                                  log_info: bool = True) -> None:
        """
        Given metrics dicts for execution modes and epochs, compute the aggregate metrics that are computed
        from the per-subject predictions. The metrics are written to the dataframe logger with the string labels
        (column names) taken from the `MetricType` enum.
        :param metrics: Mapping between epoch and subject level metrics
        :param data_frame_logger: DataFrame logger to write to and flush
        :param log_info: If True then log results as an INFO string to the default logger also.
        :return:
        """
        for mode, epoch_metrics in metrics.items():
            for epoch, metrics_dict in epoch_metrics.items():
                # Compute the aggregate metrics using the .average method of the dictionary,
                # to ensure that we are averaging over the same metrics that would be written in training.
                averaged = metrics_dict.average(add_metrics_from_entries=True, across_hues=False)
                for hue, values_within_hue in averaged.enumerate_single_values_groupwise():
                    record: Dict[str, Any] = {
                        LoggingColumns.Hue.value: hue,
                    }
                    has_any_values = False
                    for key, value in values_within_hue:
                        has_any_values = True
                        value_str = str(value) if isinstance(value, int) else f"{value:0.5f}"
                        metric_name = get_column_name_for_logging(key)
                        record[metric_name] = value_str
                    # Do not create a row at all if there are no metrics in a particular hue. This could happen
                    # for example when using multi-step RNN, where no data is in the default hue.
                    if has_any_values:
                        # Add epoch last to more easily navigate visually
                        record[LoggingColumns.DataSplit.value] = mode.value
                        record[LoggingColumns.Epoch.value] = epoch
                        data_frame_logger.add_record(record)

        # save results to disk
        data_frame_logger.flush(log_info=log_info)


class SequenceMetricsDict(ScalarMetricsDict):
    """
    Specialization of the MetricsDict with Sequence related functions.
    """

    def __init__(self, hues: Optional[List[str]] = None, is_classification_metrics: bool = True) -> None:
        super().__init__(hues, is_classification_metrics=is_classification_metrics)

    @staticmethod
    def create_from_config(config: SequenceModelBase) -> SequenceMetricsDict:
        # Create labels for the different prediction target positions that give numerically increasing positions
        # when using string sorting
        hues = [SequenceMetricsDict.get_hue_name_from_target_index(p)
                for p in config.sequence_target_positions]
        return SequenceMetricsDict(hues=hues, is_classification_metrics=config.is_classification_model)

    @staticmethod
    def get_hue_name_from_target_index(target_index: int) -> str:
        """
        Creates a metrics hue name for sequence models, from a target index. For a sequence model that predicts
        at index 7, the hue name would be "Seq_pos 07"
        """
        return f"{SEQUENCE_POSITION_HUE_NAME_PREFIX} {target_index:02}"

    @staticmethod
    def get_target_index_from_hue_name(hue_name: str) -> int:
        """
        Extracts a sequence target index from a metrics hue name. For example, from metrics hue "Seq_pos 07",
        it would return 7.
        :param hue_name: hue name containing sequence target index
        """
        if hue_name.startswith(SEQUENCE_POSITION_HUE_NAME_PREFIX):
            try:
                return int(hue_name[len(SEQUENCE_POSITION_HUE_NAME_PREFIX):])
            except:
                pass
        raise ValueError(f"Unable to extract target index from this string: {hue_name}")
