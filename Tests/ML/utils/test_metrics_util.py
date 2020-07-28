#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import numpy as np
import pytest

from InnerEye.ML.utils.metrics_util import get_label_overlap_stats, get_label_volume, get_number_of_voxels_per_class


@pytest.mark.parametrize("labels", [None, np.zeros(shape=[3]), np.zeros(shape=[3] * 6)])
def test_get_number_of_voxels_per_class_invalid(labels: Any) -> None:
    with pytest.raises(Exception):
        get_number_of_voxels_per_class(labels)


def test_get_number_of_voxels_per_class() -> None:
    non_batched_labels = np.zeros(shape=[3] * 4)
    non_batched_labels[0, 0, 0, 0] = 1
    non_batched_labels[0, 0, 0, 1] = 1
    non_batched_labels[1, 0, 0, 2] = 1
    non_batched_labels[2, 0, 1, 0] = 1

    number_batches = 5
    batched_labels = np.zeros(shape=[number_batches] + [3] * 4)
    for i in range(len(batched_labels)):
        batched_labels[i] = non_batched_labels
    count_non_batched = get_number_of_voxels_per_class(non_batched_labels)
    assert isinstance(count_non_batched, list)
    assert isinstance(count_non_batched[0], int)
    assert count_non_batched == [2, 1, 1]
    count_batched = get_number_of_voxels_per_class(batched_labels)
    assert count_batched == np.sum([[2, 1, 1]] * number_batches, axis=0).tolist()


def test_get_label_overlap_stats() -> None:
    """
    Test of the computed label overlap stats are computed
    as expected.
    """
    test_data = {'spleen': np.array([[0, 1], [1, 1]]),
                 'liver': np.array([[0, 1], [1, 0]]),
                 'prostate': np.array([[0, 0], [0, 1]]),
                 'bladder': np.array([[1, 0], [0, 0]])}

    label_overlap_stats = get_label_overlap_stats(labels=np.stack(test_data.values()),
                                                  label_names=list(test_data.keys()))
    assert label_overlap_stats['spleen'] == 3
    assert label_overlap_stats['liver'] == 2
    assert label_overlap_stats['prostate'] == 1
    assert label_overlap_stats['bladder'] == 0


def test_get_label_volume() -> None:
    """
    Tests computation of label volumes.
    """
    test_data = {'spleen': np.array([[0, 1, 0], [1, 1, 0]]),
                 'liver': np.array([[0, 0, 0], [0, 0, 0]])}
    label_overlap_stats = get_label_volume(labels=np.stack(test_data.values()),
                                           label_names=list(test_data.keys()),
                                           label_spacing=(1.00, 2.00, 3.00))

    assert label_overlap_stats['spleen'] == pytest.approx(3 * 6 / 1000.0, 1e-6)
    assert label_overlap_stats['liver'] == pytest.approx(0.0, 1e-6)
