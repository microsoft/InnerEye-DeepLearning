#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from typing import List, Optional

import numpy as np
import pytest
import torch
from sklearn.metrics import auc, log_loss, precision_recall_curve, roc_curve

from InnerEye.Common.metrics_dict import INTERNAL_TO_LOGGING_COLUMN_NAMES, MetricType, MetricsDict, \
    get_column_name_for_logging
from InnerEye.Common.type_annotations import TupleFloat3
from InnerEye.ML import metrics
from InnerEye.ML.configs.classification.DummyClassification import DummyClassification
from InnerEye.ML.configs.regression.DummyRegression import DummyRegression
from InnerEye.ML.lightning_models import ScalarLightning


def test_calculate_dice1() -> None:
    """
    Ground truth is class 1 at every voxel. Predicting class 0 at every voxel should give
    Dice of zero for both classes (background and foreground), because for both classes, either the predicted
    structure or
    the true structure is non-empty. Predicting 1 at every voxel gives NaN for class 0 because
    both predicted and true are empty, and (obviously) 1.0 for class 1.
    """
    g1 = "g1"
    zero = np.zeros((3, 3, 3))
    one = np.ones((3, 3, 3))

    # ground truth is expected in one-hot encoding, but the segmentation is a map with class indices in each voxel
    def assert_metrics(segmentation: np.ndarray, ground_truth: np.ndarray, expected_dice: float) -> None:
        a = metrics.calculate_metrics_per_class(segmentation, ground_truth,
                                                voxel_spacing=(1, 1, 1), ground_truth_ids=[g1])
        assert a.get_hue_names(include_default=False) == [g1]
        assert equal_respecting_nan(a.get_single_metric(MetricType.DICE, hue=g1), expected_dice)

    # Case 1: Ground truth says everything is class 1, and segmentation says the same
    assert_metrics(one, np.stack([zero, one]), expected_dice=1.0)
    # Case 2: Ground truth says everything is class 0, but segmentation says it's class 1
    assert_metrics(one, np.stack([one, zero]), expected_dice=0.0)
    # Case 3: Ground truth says everything is class 0, and segmentation says the same: This means that class 1
    # is correctly predicted, but empty ground truth and empty prediction are indicated by Dice NaN
    assert_metrics(zero, np.stack([one, zero]), expected_dice=math.nan)


def equal_respecting_nan(v1: float, v2: float) -> bool:
    if math.isnan(v1) and math.isnan(v2):
        return True
    return v1 == v2


@pytest.mark.parametrize(["prediction_list", "expected_dice"],
                         [([0, 0, 1], 1.0),  # prediction same as GT
                          ([1, 0, 1], 2 / 3),  # 2*1/(1+2)
                          ([1, 1, 1], 0.5)])  # 2*1/(3+1)
def test_calculate_dice2(prediction_list: list, expected_dice: float) -> None:
    g1 = "g1"

    # Turns a row vector into a single Z-slice 3D array, by copying along dimension 1 and extending.
    # Without doing that, computation of the Hausdorff distance fails.
    def expand(a: List[float]) -> np.ndarray:
        return np.repeat(np.array([[a]]), 3, 1)

    # Ground truth is same as (i.e. one-hot version of) prediction.
    ground_truth_values = expand([0, 0, 1])
    ground_truth = np.stack([1 - ground_truth_values, ground_truth_values])
    prediction = expand(prediction_list)
    m = metrics.calculate_metrics_per_class(prediction, ground_truth, voxel_spacing=(1, 1, 1), ground_truth_ids=[g1])
    assert m.get_single_metric(MetricType.DICE, hue=g1) == expected_dice


def test_calculate_hd() -> None:
    g1 = "g1"
    np.random.seed(0)
    prediction0 = np.zeros((10, 5, 2))
    prediction1 = np.ones_like(prediction0)
    gt_all_zero = np.stack([prediction1, prediction0])
    gt_all_one = np.stack([prediction0, prediction1])

    def assert_metrics(prediction: np.ndarray, ground_truth: np.ndarray, expected: Optional[float],
                       voxel_spacing: TupleFloat3 = (1, 1, 1)) -> float:
        m = metrics.calculate_metrics_per_class(prediction, ground_truth, voxel_spacing=voxel_spacing,
                                                ground_truth_ids=[g1])
        actual = m.get_single_metric(MetricType.HAUSDORFF_mm, hue=g1)
        if expected is not None:
            assert actual == expected
        return actual

    # check an infinity value if either the prediction or gt have no foreground
    assert_metrics(prediction0, gt_all_one, math.inf)
    assert_metrics(prediction1, gt_all_zero, math.inf)

    def generate_random_prediction() -> np.ndarray:
        result = np.round(np.random.uniform(size=prediction0.shape))
        # Ensure not all the same value
        if result.min() == result.max():
            result[0, 0, 0] = 1 - result[0, 0, 0]
        return result

    random_prediction = generate_random_prediction()
    matching_gt = np.stack([1 - random_prediction, random_prediction])
    assert_metrics(random_prediction, matching_gt, 0.0)
    # check voxel spacing is being used as expected
    random_prediction2 = generate_random_prediction()
    non_matching_gt = np.stack([1 - random_prediction2, random_prediction2])
    without_spacing = assert_metrics(random_prediction, non_matching_gt, voxel_spacing=(1, 1, 1), expected=None)
    with_spacing = assert_metrics(random_prediction, non_matching_gt, voxel_spacing=(2.0, 2.0, 2.0), expected=None)
    assert without_spacing != with_spacing


def test_calculate_hd_exact() -> None:
    prediction = np.array([[[0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1]]])
    ground_truth = np.array([[[1, 0, 0],
                              [1, 0, 0],
                              [1, 0, 0]]])

    ground_truth = np.stack(np.stack([1 - ground_truth, ground_truth]))
    g1 = "g1"
    m = metrics.calculate_metrics_per_class(prediction, ground_truth, voxel_spacing=(1, 2, 3), ground_truth_ids=[g1])
    assert m.get_single_metric(MetricType.HAUSDORFF_mm, hue=g1) == 6
    assert m.get_single_metric(MetricType.MEAN_SURFACE_DIST_mm, hue=g1) == 6


def test_compute_dice_across_patches() -> None:
    patches = 2
    # Ground truth has 3 classes, all entries are 1. Cardinality of each GT is 3
    num_classes = 3
    ground_truth = torch.from_numpy(np.ones((patches, num_classes, 4, 1, 1)))
    # Prediction in patch 0 has 3 entries for class 0, 1 for class 1, 0 for class 2.
    # In patch 1, all predictions are left at class 0.
    prediction_argmax = torch.from_numpy(np.zeros((patches, 4, 1, 1))).long()
    prediction_argmax[0, 1, 0, 0] = 1
    dice = metrics.compute_dice_across_patches(prediction_argmax,
                                               ground_truth,
                                               allow_multiple_classes_for_each_pixel=True).cpu().numpy()

    expected_dice_patch0 = np.array([2 * 3 / (4 + 3), 2 * 1 / (4 + 1), 0])
    expected_dice_patch1 = np.array([1, 0, 0])

    assert dice.shape == (patches, num_classes)

    expected_dice = np.vstack([expected_dice_patch0, expected_dice_patch1])
    assert np.allclose(dice, expected_dice, rtol=1.e-5, atol=1.e-8)


def test_get_column_name_for_logging() -> None:
    metric_name = MetricType.LOSS.value
    expected_metric_name = INTERNAL_TO_LOGGING_COLUMN_NAMES[metric_name].value
    assert expected_metric_name \
           == get_column_name_for_logging(metric_name=metric_name) \
           == get_column_name_for_logging(metric_name=metric_name, hue_name=MetricsDict.DEFAULT_HUE_KEY)
    hue_name = "foo"
    assert f"{hue_name}/{expected_metric_name}" == \
           get_column_name_for_logging(metric_name=metric_name, hue_name=hue_name)


def test_classification_metrics() -> None:
    classification_module = ScalarLightning(DummyClassification())
    metrics = classification_module._get_metrics_classes()
    outputs = [torch.tensor([0.9, 0.8, 0.6]), torch.tensor([0.3, 0.9, 0.4])]
    labels = [torch.tensor([1., 1., 0.]), torch.tensor([0., 0., 0.])]
    for output, label in zip(outputs, labels):
        for metric in metrics:
            metric.update(output, label)
    accuracy_05, accuracy_opt, threshold, fpr, fnr, roc_auc, pr_auc, cross_entropy = [metric.compute() for metric in
                                                                                      metrics]
    all_labels = torch.cat(labels).numpy()
    all_outputs = torch.cat(outputs).numpy()
    expected_accuracy_at_05 = np.mean((all_outputs > 0.5) == all_labels)
    expected_binary_cross_entropy = log_loss(y_true=all_labels, y_pred=all_outputs)
    expected_fpr, expected_tpr, expected_thresholds = roc_curve(y_true=all_labels, y_score=all_outputs)
    expected_roc_auc = auc(expected_fpr, expected_tpr)
    expected_optimal_idx = np.argmax(expected_tpr - expected_fpr)
    expected_optimal_threshold = expected_thresholds[expected_optimal_idx]
    expected_accuracy = np.mean((all_outputs > expected_optimal_threshold) == all_labels)
    expected_optimal_fpr = expected_fpr[expected_optimal_idx]
    expected_optimal_fnr = 1 - expected_tpr[expected_optimal_idx]
    prec, recall, _ = precision_recall_curve(y_true=all_labels, probas_pred=all_outputs)
    expected_pr_auc = auc(recall, prec)
    assert accuracy_opt == expected_accuracy
    assert threshold == expected_optimal_threshold
    assert fpr == expected_optimal_fpr
    assert fnr == expected_optimal_fnr
    assert roc_auc == expected_roc_auc
    assert pr_auc == expected_pr_auc
    assert cross_entropy == expected_binary_cross_entropy
    assert accuracy_05 == expected_accuracy_at_05

def test_regression_metrics() -> None:
    regression_module = ScalarLightning(DummyRegression())
    metrics = regression_module._get_metrics_classes()
    outputs = [torch.tensor([1., 2., 1.]), torch.tensor([4., 0., 2.])]
    labels = [torch.tensor([1., 1., 0.]), torch.tensor([2., 0., 2.])]
    for output, label in zip(outputs, labels):
        for metric in metrics:
            metric.update(output, label)
    MAE, MSE, ExpVar = [metric.compute() for metric in metrics]
    all_labels = torch.cat(labels)
    all_outputs = torch.cat(outputs)
    expected_mae = torch.mean(torch.abs(all_labels - all_outputs))
    expected_mse = torch.mean(torch.square(all_labels - all_outputs))
    # ExpVar 1 - Var(y_pred - y_true) / Var(y_true)
    expected_expVar = 1 - torch.var(all_outputs - all_labels) / torch.var(all_labels)
    assert expected_mae == MAE
    assert expected_mse == MSE
    assert torch.isclose(expected_expVar, ExpVar, atol=1e-5)
