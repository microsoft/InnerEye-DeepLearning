#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, List, Union

import numpy as np
import torch
from math import inf
from scipy import stats
import torch.nn.functional as F


def get_instance_noise_model(n: float, dataset: Any, labels: Union[List, torch.Tensor], num_classes: int,
                             feature_size: int, norm_std: float, seed: int) -> np.ndarray:
    """
    :param n: noise_rate
    :param dataset: cifar10 # not train_loader
    :param labels: labels (targets)
    :param num_classes: class number
    :param feature_size: the size of input images (e.g. 28*28)
    :param norm_std: default 0.1
    :param seed: random_seed
    """
    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    P = []
    random_state = np.random.RandomState(seed)
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0], random_state=seed)

    W = random_state.randn(num_classes, feature_size, num_classes)
    W = torch.FloatTensor(W)

    for i, (x, y) in enumerate(dataset):
        # (1 x M) * (M x 10) = (1 x 10)
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).numpy()

    return P
