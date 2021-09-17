#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np


def compute_label_entropy(label_counts: np.ndarray) -> np.ndarray:
    """
    :param label_counts: Input label histogram (n_samples x n_classes)
    """
    label_distribution = label_counts / np.sum(label_counts, axis=-1, keepdims=True)
    return cross_entropy(label_distribution, label_distribution)


def compute_accuracy(source: np.ndarray, target: np.ndarray) -> float:
    """
    Computes the agreement rate between two tensors
    :param source: source array (n_samples, n_classes)
    :param target: target array (n_samples, n_classes)
    """
    # Compare single label case against all available labels
    source = np.argmax(source, axis=1)
    target = np.argmax(target, axis=1)

    # Accuracy
    acc = 100.0 * np.sum(source == target) / source.size

    return acc

def compute_model_disagreement_score(posteriors: np.ndarray) -> np.ndarray:
    """
    Measure model disagreement score (Ref BALD)
    :param posteriors: numpy array (shape: (model_candidates, batch_size, num_classes))
    :return: Disagreement score (BALD) for each sample (shape: (batch))
    """
    def _entropy(x: np.ndarray, log_base: int, epsilon: float = 1e-12) -> np.ndarray:
        return -np.sum(x * (np.log(x + epsilon) / np.log(log_base)), axis=-1)
    num_classes = int(posteriors.shape[-1])
    avg_posteriors = np.mean(posteriors, axis=0)
    avg_entropy = _entropy(avg_posteriors, num_classes)
    exp_conditional_entropy = np.mean(_entropy(posteriors, num_classes), axis=0)
    bald_score = avg_entropy - exp_conditional_entropy
    return bald_score


def cross_entropy(predicted_distribution: np.ndarray, target_distribution: np.ndarray) -> np.ndarray:
    """
    Compute the normalised cross-entropy between the predicted and target distributions
    :param predicted_distribution: Predicted distribution shape = (num_samples, num_classes)
    :param target_distribution: Target distribution shape = (num_samples, num_classes)
    :return: The cross-entropy for each sample
    """
    num_classes = predicted_distribution.shape[1]
    return -np.sum(target_distribution * np.log(predicted_distribution + 1e-12) / np.log(num_classes), axis=-1)

def max_prediction_error(predicted_distribution: np.ndarray, target_distribution: np.ndarray) -> np.ndarray:
    """
    Compute the max (class-wise) prediction error between the predicted and target distributions
    :param predicted_distribution: Predicted distribution shape = (num_samples, num_classes)
    :param target_distribution: Target distribution shape = (num_samples, num_classes)
    :return: The max (class-wise) prediction error for each sample
    """
    current_target_class = np.argmax(target_distribution, axis=1)
    current_target_pred_prob = predicted_distribution[range(len(current_target_class)), current_target_class]
    prediction_errors = 1.0 - current_target_pred_prob
    return prediction_errors

def total_variation(predicted_distribution: np.ndarray, target_distribution: np.ndarray) -> np.ndarray:
    """
    Compute the total variation error between the predicted and target distributions
    :param predicted_distribution: Predicted distribution shape = (num_samples, num_classes)
    :param target_distribution: Target distribution shape = (num_samples, num_classes)
    :return: The total variation for each sample
    """
    return np.sum(np.abs(predicted_distribution - target_distribution), axis=-1) / 2.
