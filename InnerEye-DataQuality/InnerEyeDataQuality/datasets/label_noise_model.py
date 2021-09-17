#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np

from default_paths import CIFAR10_ROOT_DIR
from InnerEyeDataQuality.datasets.cifar10h import CIFAR10H
from InnerEyeDataQuality.selection.simulation_statistics import get_ambiguous_sample_ids

from sklearn.metrics import confusion_matrix


def get_cifar10h_confusion_matrix(temperature: float = 1.0, only_difficult_cases: bool = False) -> np.ndarray:
    """
    Generates a class confusion matrix based on the label distribution in CIFAR10H.
    """
    cifar10h_labels = CIFAR10H.download_cifar10h_labels(str(CIFAR10_ROOT_DIR))

    if only_difficult_cases:
        ambiguous_sample_ids = get_ambiguous_sample_ids(cifar10h_labels)
        cifar10h_labels = cifar10h_labels[ambiguous_sample_ids, :]

    # Temperature scale the original distribution
    if temperature > 1.0:
        orig_distribution = cifar10h_labels / np.sum(cifar10h_labels, axis=1, keepdims=True)
        _d = np.power(orig_distribution, 1. / temperature)
        scaled_distribution = _d / np.sum(_d, axis=1, keepdims=True)
        sample_counts = (scaled_distribution * np.sum(cifar10h_labels, axis=1, keepdims=True)).astype(np.int64)
    else:
        sample_counts = cifar10h_labels

    y_pred, y_true = list(), list()
    for image_index in range(sample_counts.shape[0]):
        image_label_counts = sample_counts[image_index]
        for _iter, _el in enumerate(image_label_counts.tolist()):
            y_pred.extend([_iter] * _el)
        y_true.extend([np.argmax(image_label_counts)] * np.sum(image_label_counts))
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    return cm


def get_cifar10_asym_noise_model(eta: float = 0.3) -> np.ndarray:
    """
    CLASS-DEPENDENT ASYMMETRIC LABEL NOISE
    https://proceedings.neurips.cc/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf
    TRUCK -> AUTOMOBILE, BIRD -> AIRPLANE, DEER -> HORSE, CAT -> DOG, and DOG -> CAT

    :param eta: The likelihood of true label switching from one of the specified classes to nearest class.
                In other words, likelihood of introducing a class-dependent label noise
    """

    # Generate a noise transition matrix.
    assert (0.0 <= eta) and (eta <= 1.0)

    eps = 1e-12
    num_classes = 10
    conf_mat = np.eye(N=num_classes)
    indices = [[2, 0], [9, 1], [5, 3], [3, 5], [4, 7]]
    for ind in indices:
        conf_mat[ind[0], ind[1]] = eta / (1.0 - eta + eps)
    return conf_mat / np.sum(conf_mat, axis=1, keepdims=True)


def get_cifar10_sym_noise_model(eta: float = 0.3) -> np.ndarray:
    """
    Symmetric LABEL NOISE
    :param eta: The likelihood of true label switching from true class to rest of the classes.
    """
    # Generate a noise transition matrix.
    assert (0.0 <= eta) and (eta <= 1.0)
    assert isinstance(eta, float)

    num_classes = 10
    conf_mat = np.eye(N=num_classes)
    for ind in range(num_classes):
        conf_mat[ind, ind] -= eta
        other_classes = np.setdiff1d(range(num_classes), ind)
        for o_c in other_classes:
            conf_mat[ind, o_c] += eta / other_classes.size

    assert np.all(np.abs(np.sum(conf_mat, axis=1) - 1.0) < 1e-9)

    return conf_mat
