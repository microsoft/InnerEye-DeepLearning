#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import List, Tuple

import numpy as np
import pytest
import torch
from torch.optim.optimizer import Optimizer

from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.deep_learning_config import LRSchedulerType
from InnerEye.ML.utils.lr_scheduler import LRScheduler
from Tests.ML.configs.DummyModel import DummyModel


def test_create_lr_scheduler_last_epoch() -> None:
    """
    Test to check if the lr scheduler is initialized to the correct epoch
    """
    expected_lrs_per_epoch = [0.001, 0.0005358867312681466]
    config = DummyModel()
    # create lr scheduler
    lr_scheduler, optimizer = _create_lr_scheduler_and_optimizer(config)
    # check lr scheduler initialization step
    assert np.isclose(lr_scheduler.get_last_lr(), expected_lrs_per_epoch[:1])
    # create lr scheduler for recovery checkpoint
    config.start_epoch = 1
    lr_scheduler, _ = _create_lr_scheduler_and_optimizer(config, optimizer)
    # check lr scheduler initialization matches the checkpoint epoch
    # as training will start for start_epoch + 1 in this case
    lr = lr_scheduler.get_last_lr()
    assert np.isclose(lr, expected_lrs_per_epoch[1:])


@pytest.mark.parametrize("lr_scheduler_type", [x for x in LRSchedulerType])
def test_min_and_initial_lr(lr_scheduler_type: LRSchedulerType) -> None:
    """
    Test if minimum learning rate threshold is applied as expected
    """
    config = DummyModel()
    config.l_rate = 1e-3
    config.min_l_rate = 0.0009
    config.l_rate_decay = lr_scheduler_type
    # create lr scheduler
    lr_scheduler, _ = _create_lr_scheduler_and_optimizer(config)
    assert lr_scheduler.get_last_lr()[0] == config.l_rate
    lr_scheduler.step(2)
    assert lr_scheduler.get_last_lr()[0] == config.min_l_rate


@pytest.mark.parametrize("lr_scheduler_type", [x for x in LRSchedulerType])
def test_lr_monotonically_decreasing_function(lr_scheduler_type: LRSchedulerType) -> None:
    """
    Tests if LR scheduler is a monotonically decreasing function
    """
    config = DummyModel()
    config.l_rate_decay = lr_scheduler_type
    config.num_epochs = 10

    def strictly_decreasing(L: List) -> bool:
        return all(x > y for x, y in zip(L, L[1:]))

    # create lr scheduler
    lr_scheduler, _ = _create_lr_scheduler_and_optimizer(config)
    lr_list = list()
    for _ in range(config.num_epochs):
        lr_scheduler.step()
        lr_list.append(lr_scheduler.get_last_lr()[0])

    assert strictly_decreasing(lr_list)


def test_cosine_decay_function() -> None:
    """
    Tests Cosine lr decay function at (pi/2) and verifies if the value is correct.
    """
    config = DummyModel()
    config.l_rate_decay = LRSchedulerType.Cosine
    config.num_epochs = 10
    config.min_l_rate = 0.0

    # create lr scheduler
    test_epoch = 5
    lr_scheduler, _ = _create_lr_scheduler_and_optimizer(config)
    lr_scheduler.step(test_epoch)
    assert lr_scheduler.get_last_lr()[0] == 0.5 * config.l_rate


def _create_lr_scheduler_and_optimizer(config: SegmentationModelBase, optimizer: Optimizer = None) \
        -> Tuple[LRScheduler, Optimizer]:
    # create dummy optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam([torch.ones(2, 2, requires_grad=True)], lr=config.l_rate)
    # create lr scheduler
    lr_scheduler = LRScheduler(config, optimizer)
    return lr_scheduler, optimizer
