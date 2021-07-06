#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from typing import Any, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics as metrics
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional import accuracy, auc, auroc, precision_recall_curve, roc
from torch.nn import ModuleList

from InnerEye.Common.metrics_constants import AVERAGE_DICE_SUFFIX, MetricType, TRAIN_PREFIX, VALIDATION_PREFIX


def nanmean(values: torch.Tensor) -> torch.Tensor:
    """
    Computes the average of all values in the tensor, skipping those entries that are NaN (not a number).
    If all values are NaN, the result is also NaN.
    :param values: The values to average.
    :return: A scalar tensor containing the average.
    """
    valid = values[~torch.isnan(values.view((-1,)))]
    if valid.numel() == 0:
        return torch.tensor([math.nan]).type_as(values)
    return valid.mean()


class MeanAbsoluteError(metrics.MeanAbsoluteError):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = MetricType.MEAN_ABSOLUTE_ERROR.value

    @property
    def has_predictions(self) -> bool:
        """
        Returns True if the present object stores at least 1 prediction (self.update has been called at least once),
        or False if no predictions are stored.
        """
        return self.total > 0  # type: ignore


class MeanSquaredError(metrics.MeanSquaredError):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = MetricType.MEAN_SQUARED_ERROR.value

    @property
    def has_predictions(self) -> bool:
        """
        Returns True if the present object stores at least 1 prediction (self.update has been called at least once),
        or False if no predictions are stored.
        """
        return self.total > 0  # type: ignore


class ExplainedVariance(metrics.ExplainedVariance):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = MetricType.EXPLAINED_VAR.value

    @property
    def has_predictions(self) -> bool:
        """
        Returns True if the present object stores at least 1 prediction (self.update has been called at least once),
        or False if no predictions are stored.
        """
        return self.n_obs > 0  # type: ignore


class Accuracy05(metrics.Accuracy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = MetricType.ACCURACY_AT_THRESHOLD_05.value

    @property
    def has_predictions(self) -> bool:
        """
        Returns True if the present object stores at least 1 prediction (self.update has been called at least once),
        or False if no predictions are stored.
        """
        return (self.total) or (self.tp + self.fp + self.tn + self.fn) > 0  # type: ignore


class AverageWithoutNan(Metric):
    """
    A generic metric computer that keep track of the average of all values excluding those that are NaN.
    """

    def __init__(self, dist_sync_on_step: bool = False, name: str = ""):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.name = name

    def update(self, value: torch.Tensor) -> None:  # type: ignore
        """
        Stores all the given individual elements of the given tensor in the present object.
        """
        for v in value.view((-1,)):
            if not torch.isnan(v):
                self.sum = self.sum + v  # type: ignore
                self.count = self.count + 1  # type: ignore

    def compute(self) -> torch.Tensor:
        if self.count == 0.0:
            raise ValueError("No values stored, or only NaN values have so far been fed into this object.")
        return self.sum / self.count


class ScalarMetricsBase(Metric):
    """
    A base class for all metrics that can only be computed once the complete set of model predictions and labels
    is available. The base class provides an `update` method, and synchronized storage for predictions (field `preds`)
    and labels (field `targets`). Derived classes need to override the `compute` method.
    """

    def __init__(self, name: str = "", compute_from_logits: bool = False):
        super().__init__(dist_sync_on_step=False)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.name = name
        self.compute_from_logits = compute_from_logits

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:  # type: ignore
        self.preds.append(preds)  # type: ignore
        self.targets.append(targets)  # type: ignore

    def compute(self) -> torch.Tensor:
        """
        Computes a metric from the stored predictions and targets.
        """
        raise NotImplementedError("Should be implemented in the child classes")

    def _get_preds_and_targets(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets a tuple of (concatenated predictions, concatenated targets).
        """
        preds, targets = torch.cat(self.preds), torch.cat(self.targets)  # type: ignore

        # Handles the case where we have a binary problem and predictions are specified [1-p, p] as predictions
        # where p is probability of class 1. Instead of just specifying p.
        if preds.dim() == 2 and preds.shape[1] == 2:
            assert preds.shape[0] == targets.shape[0]
            return preds[:, 1], targets

        assert preds.dim() == targets.dim() == 1 and preds.shape[0] == targets.shape[0]
        return preds, targets

    @property
    def has_predictions(self) -> bool:
        """
        Returns True if the present object stores at least 1 prediction (self.update has been called at least once),
        or False if no predictions are stored.
        """
        return len(self.preds) > 0  # type: ignore

    def _get_metrics_at_optimal_cutoff(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the ROC to find the optimal cut-off i.e. the probability threshold for which the
        difference between true positive rate and false positive rate is smallest. Then, computes
        the false positive rate, false negative rate and accuracy at this threshold (i.e. when the
        predicted probability is higher than the threshold the predicted label is 1 otherwise 0).
        :returns: Tuple(optimal_threshold, false positive rate, false negative rate, accuracy)
        """
        preds, targets = self._get_preds_and_targets()
        if torch.unique(targets).numel() == 1:
            return torch.tensor(np.nan), torch.tensor(np.nan), torch.tensor(np.nan), torch.tensor(np.nan)
        fpr, tpr, thresholds = roc(preds, targets)
        assert isinstance(fpr, torch.Tensor)
        assert isinstance(tpr, torch.Tensor)
        assert isinstance(thresholds, torch.Tensor)
        optimal_idx = torch.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        acc = accuracy(preds > optimal_threshold, targets)
        false_negative_optimal = 1 - tpr[optimal_idx]
        false_positive_optimal = fpr[optimal_idx]
        return optimal_threshold, false_positive_optimal, false_negative_optimal, acc


class AccuracyAtOptimalThreshold(ScalarMetricsBase):
    """
    Computes the binary classification accuracy at an optimal cut-off point.
    """

    def __init__(self) -> None:
        super().__init__(name=MetricType.ACCURACY_AT_OPTIMAL_THRESHOLD.value)

    def compute(self) -> torch.Tensor:
        return self._get_metrics_at_optimal_cutoff()[3]


class OptimalThreshold(ScalarMetricsBase):
    """
    Computes the optimal cut-off point for a binary classifier.
    """

    def __init__(self) -> None:
        super().__init__(name=MetricType.OPTIMAL_THRESHOLD.value)

    def compute(self) -> torch.Tensor:
        return self._get_metrics_at_optimal_cutoff()[0]


class FalsePositiveRateOptimalThreshold(ScalarMetricsBase):
    """
    Computes the false positive rate when choosing the optimal cut-off point for a binary classifier.
    """

    def __init__(self) -> None:
        super().__init__(name=MetricType.FALSE_POSITIVE_RATE_AT_OPTIMAL_THRESHOLD.value)

    def compute(self) -> torch.Tensor:
        return self._get_metrics_at_optimal_cutoff()[1]


class FalseNegativeRateOptimalThreshold(ScalarMetricsBase):
    """
    Computes the false negative rate when choosing the optimal cut-off point for a binary classifier.
    """

    def __init__(self) -> None:
        super().__init__(name=MetricType.FALSE_NEGATIVE_RATE_AT_OPTIMAL_THRESHOLD.value)

    def compute(self) -> torch.Tensor:
        return self._get_metrics_at_optimal_cutoff()[2]


class AreaUnderRocCurve(ScalarMetricsBase):
    """
    Computes the area under the receiver operating curve (ROC).
    """

    def __init__(self) -> None:
        super().__init__(name=MetricType.AREA_UNDER_ROC_CURVE.value)

    def compute(self) -> torch.Tensor:
        preds, targets = self._get_preds_and_targets()
        if torch.unique(targets).numel() == 1:
            return torch.tensor(np.nan)
        return auroc(preds, targets)


class AreaUnderPrecisionRecallCurve(ScalarMetricsBase):
    """
    Computes the area under the precision-recall-curve.
    """

    def __init__(self) -> None:
        super().__init__(name=MetricType.AREA_UNDER_PR_CURVE.value)

    def compute(self) -> torch.Tensor:
        preds, targets = self._get_preds_and_targets()
        if torch.unique(targets).numel() == 1:
            return torch.tensor(np.nan)
        prec, recall, _ = precision_recall_curve(preds, targets)
        return auc(recall, prec)  # type: ignore


class BinaryCrossEntropyWithLogits(ScalarMetricsBase):
    """
    Computes the cross entropy for binary classification.
    This metric must be computed off the model output logits.
    """

    def __init__(self) -> None:
        super().__init__(name=MetricType.CROSS_ENTROPY.value, compute_from_logits=True)

    def compute(self) -> torch.Tensor:
        preds, targets = self._get_preds_and_targets()
        # All classification metrics work with integer targets, but this one does not. Convert to float.
        return F.binary_cross_entropy_with_logits(input=preds, target=targets.to(dtype=preds.dtype))


class MetricForMultipleStructures(torch.nn.Module):
    """
    Stores a metric for multiple structures, and an average Dice score across all structures.
    The class consumes pre-computed metric values, and only keeps an aggregate for later computing the
    averages. When averaging, metric values that are NaN are skipped.
    """

    def __init__(self, ground_truth_ids: List[str], is_training: bool,
                 metric_name: str = MetricType.DICE.value,
                 use_average_across_structures: bool = True) -> None:
        """
        Creates a new MetricForMultipleStructures object.
        :param ground_truth_ids: The list of anatomical structures that should be stored.
        :param metric_name: The name of the metric that should be stored. This is used in the names of the individual
        metrics.
        :param is_training: If true, use "train/" as the prefix for all metric names, otherwise "val/"
        :param use_average_across_structures: If True, keep track of the average metric value across structures,
        while skipping NaNs. If false, only store the per-structure metric values.
        """
        super().__init__()
        prefix = (TRAIN_PREFIX if is_training else VALIDATION_PREFIX) + metric_name + "/"
        # All Metric classes must be
        self.average_per_structure = ModuleList([AverageWithoutNan(name=prefix + g) for g in ground_truth_ids])
        self.use_average_across_structures = use_average_across_structures
        if use_average_across_structures:
            self.average_all = AverageWithoutNan(name=prefix + AVERAGE_DICE_SUFFIX)
        self.count = len(ground_truth_ids)

    def update(self, values_per_structure: torch.Tensor) -> None:
        """
        Stores a vector of per-structure Dice scores in the present object. It updates the per-structure values,
        and the aggregate value across all structures.
        :param values_per_structure: A row tensor that has as many entries as there are ground truth IDs.
        """
        if values_per_structure.dim() != 1 or values_per_structure.numel() != self.count:
            raise ValueError(f"Expected a tensor with {self.count} elements, but "
                             f"got shape {values_per_structure.shape}")
        for i, v in enumerate(values_per_structure.view((-1,))):
            self.average_per_structure[i].update(v)
        if self.use_average_across_structures:
            self.average_all.update(nanmean(values_per_structure))

    def __iter__(self) -> Iterator[Metric]:
        """
        Enumerates all the metrics that the present object holds: First the average across all structures,
        then the per-structure Dice scores.
        """
        if self.use_average_across_structures:
            yield self.average_all
        yield from self.average_per_structure

    def compute_all(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Calls the .compute() method on all the metrics that the present object holds, and returns a sequence
        of (metric name, metric value) tuples. This will automatically also call .reset() on the metrics.
        The first returned metric is the average across all structures, then come the per-structure values.
        """
        for d in self:
            yield d.name, d.compute()  # type: ignore

    def reset(self) -> None:
        """
        Calls the .reset() method on all the metrics that the present object holds.
        """
        for d in self:
            d.reset()
