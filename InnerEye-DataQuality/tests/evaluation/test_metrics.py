#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path

import numpy as np

from InnerEyeDataQuality.evaluation.metrics import compute_label_entropy, compute_accuracy
from InnerEyeDataQuality.evaluation.plot_stats import plot_stats_scores


def test_label_entropy() -> None:
    labels = np.array([[0, 10, 0], [1, 0, 1], [5, 0, 0]])
    target_entropies = np.array([0.0, 0.6309, 0.0])
    computed_entropies = compute_label_entropy(label_counts=labels)

    assert np.allclose(computed_entropies, target_entropies, rtol=1e-3, atol=1e-8)


def test_compute_accuracy() -> None:
    source = [[0.0, 1.0], [0.51, 0.49], [1.0, 0.0], [0.0, 1.0]]
    target = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    acc = compute_accuracy(source=source, target=target)

    assert acc == 50.0


def test_plot_stats_scores() -> None:
    scores = np.random.uniform(0, 1, 100)
    labels = np.random.choice([0, 1], 100)
    plot_stats_scores('name', scores, labels)
    scores_ambiguous = np.random.uniform(0, 1, 100)
    labels_ambiguous = np.random.choice([0, 1], 100)
    plot_stats_scores('name', scores, labels, scores_ambiguous, labels_ambiguous, save_path=Path())
    assert Path("name_stats_scoring.png").exists()
