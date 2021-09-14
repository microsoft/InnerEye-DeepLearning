#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np
from InnerEyeDataQuality.datasets.cifar10h import CIFAR10H


# make sure two consecutive draws yield the same label when use_fixed_labels is true
def test_cifar10h_dataset() -> None:
    dataset = CIFAR10H('~/.torch/datasets/CIFAR10', seed=1234)
    _, target_1 = dataset.__getitem__(1)
    _, target_2 = dataset.__getitem__(1)
    assert isinstance(target_1, int) and isinstance(target_1, int) and target_1 == target_2


def test_subset_selection() -> None:
    dataset = CIFAR10H('~/.torch/datasets/CIFAR10', seed=1234,
                       num_samples=2500, noise_temperature=2.0)
    assert dataset.num_samples == 2500
    assert len(dataset.indices) == 2500

    mislabelled_amb_rate = len(dataset.ambiguous_mislabelled_cases) / dataset.num_samples
    ratio_of_clear_label_noise = len(dataset.clear_mislabeled_cases) / dataset.num_samples
    assert mislabelled_amb_rate > 0.05
    assert ratio_of_clear_label_noise > 0.05

    # Check the reproducibility of selected indices
    dataset2 = CIFAR10H('~/.torch/datasets/CIFAR10', seed=1234,
                        num_samples=2500, noise_temperature=2.0)
    assert np.all(dataset.indices == dataset2.indices)
