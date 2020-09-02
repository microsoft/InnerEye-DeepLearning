#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import List, Tuple, Any, Optional

import numpy as np
import pytest
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import ExponentialLR, StepLR, MultiStepLR, LambdaLR, CosineAnnealingLR, _LRScheduler

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
    config = DummyModel(num_epochs=3, l_rate=1e-3, min_l_rate=0.0009,
                        l_rate_decay=lr_scheduler_type, l_rate_milestones=[1, 2])
    # create lr scheduler
    lr_scheduler, _ = _create_lr_scheduler_and_optimizer(config)
    assert lr_scheduler.get_last_lr()[0] == config.l_rate
    lr_scheduler.step(3)
    assert lr_scheduler.get_last_lr()[0] == config.min_l_rate


@pytest.mark.parametrize("lr_scheduler_type", [x for x in LRSchedulerType])
def test_lr_monotonically_decreasing_function(lr_scheduler_type: LRSchedulerType) -> None:
    """
    Tests if LR scheduler is a monotonically decreasing function
    """
    config = DummyModel(l_rate_decay=lr_scheduler_type, num_epochs=10, l_rate_milestones=[3, 5, 7])

    def strictly_decreasing(L: List) -> bool:
        return all(x >= y for x, y in zip(L, L[1:]))

    # create lr scheduler
    lr_scheduler, _ = _create_lr_scheduler_and_optimizer(config)
    lr_list = []
    for _ in range(config.num_epochs):
        lr_scheduler.step()
        lr_list.append(lr_scheduler.get_last_lr()[0])

    assert strictly_decreasing(lr_list)


@pytest.mark.parametrize("lr_scheduler_type", [x for x in LRSchedulerType])
@pytest.mark.parametrize("warmup_epochs", [0, 4, 5])
def test_warmup_against_original_schedule(lr_scheduler_type: LRSchedulerType, warmup_epochs: int) -> None:
    """
    Tests if LR scheduler with warmup matches the Pytorch implementation after the warmup stage is completed.
    """
    config = DummyModel(l_rate_decay=lr_scheduler_type, num_epochs=10, warmup_epochs=warmup_epochs,
                        l_rate_step_size=2, l_rate_milestones=[3, 5, 7])
    # create lr scheduler
    lr_scheduler, optimizer = _create_lr_scheduler_and_optimizer(config)

    original_scheduler: Optional[_LRScheduler] = None
    if lr_scheduler_type == LRSchedulerType.Exponential:
        original_scheduler = ExponentialLR(optimizer=optimizer, gamma=config.l_rate_gamma)
    elif lr_scheduler_type == LRSchedulerType.Step:
        original_scheduler = StepLR(optimizer=optimizer, step_size=config.l_rate_step_size,
                                    gamma=config.l_rate_gamma)
    elif lr_scheduler_type == LRSchedulerType.Cosine:
        original_scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=config.min_l_rate)
    elif lr_scheduler_type == LRSchedulerType.MultiStep:
        assert config.l_rate_milestones is not None  # for mypy
        original_scheduler = MultiStepLR(optimizer=optimizer, milestones=config.l_rate_milestones,
                                         gamma=config.l_rate_gamma)
    elif lr_scheduler_type == LRSchedulerType.Polynomial:
        x = config.min_l_rate / config.l_rate
        polynomial_decay: Any = lambda epoch: (1 - x) * (
                (1. - float(epoch) / config.num_epochs) ** config.l_rate_gamma) + x
        original_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=polynomial_decay)
    else:
        raise ValueError("Scheduler has not been added to this test.")

    result_lr_list = []
    for _ in range(config.num_epochs):
        result_lr_list.append(lr_scheduler.get_last_lr()[0])
        lr_scheduler.step()

    expected_lr_list = []
    for i in range(warmup_epochs):
        expected_lr_list.append(config.l_rate * i / warmup_epochs)
    for _ in range(config.num_epochs - warmup_epochs):
        # For pytorch version 1.6:
        # expected_lr_list.append(original_scheduler.get_last_lr())
        expected_lr_list.append(original_scheduler.get_lr()[0])  # type: ignore
        original_scheduler.step()  # type: ignore

    assert result_lr_list == expected_lr_list


def _create_dummy_optimizer(config: SegmentationModelBase) -> Optimizer:
    return torch.optim.Adam([torch.ones(2, 2, requires_grad=True)], lr=config.l_rate)


def _create_lr_scheduler_and_optimizer(config: SegmentationModelBase, optimizer: Optimizer = None) \
        -> Tuple[LRScheduler, Optimizer]:
    # create dummy optimizer
    if optimizer is None:
        optimizer = _create_dummy_optimizer(config)
    # create lr scheduler
    lr_scheduler = LRScheduler(config, optimizer)
    return lr_scheduler, optimizer


@pytest.mark.parametrize("lr_scheduler_type", [x for x in LRSchedulerType])
@pytest.mark.parametrize("warmup_epochs", [0, 4, 5])
@pytest.mark.parametrize("restart_from_epoch", [4])
def test_resume_from_saved_state(lr_scheduler_type: LRSchedulerType,
                                 warmup_epochs: int, restart_from_epoch: int) -> None:
    """
    Tests if LR scheduler when reloaded from a state dict continues as expected.
    """
    config = DummyModel(l_rate_decay=lr_scheduler_type, num_epochs=10, warmup_epochs=warmup_epochs,
                        l_rate_step_size=2, l_rate_milestones=[3, 5, 7])
    # create two lr schedulers
    lr_scheduler_1, optimizer_1 = _create_lr_scheduler_and_optimizer(config)
    lr_scheduler_2, optimizer_2 = _create_lr_scheduler_and_optimizer(config)

    expected_lr_list = []
    for _ in range(config.num_epochs):
        expected_lr_list.append(lr_scheduler_2.get_last_lr()[0])
        lr_scheduler_2.step()

    result_lr_list = []
    for _ in range(restart_from_epoch):
        result_lr_list.append(lr_scheduler_1.get_last_lr()[0])
        lr_scheduler_1.step()

    # resume state: This just means setting start_epoch in the config
    config.start_epoch = restart_from_epoch
    lr_scheduler_resume, _ = _create_lr_scheduler_and_optimizer(config, optimizer_1)
    for _ in range(config.num_epochs - restart_from_epoch):
        result_lr_list.append(lr_scheduler_resume.get_last_lr()[0])
        lr_scheduler_resume.step()

    assert result_lr_list == expected_lr_list


def test_cosine_decay_function() -> None:
    """
    Tests Cosine lr decay function at (pi/2) and verifies if the value is correct.
    """
    config = DummyModel(l_rate_decay=LRSchedulerType.Cosine,
                        num_epochs=10,
                        min_l_rate=0.0)

    # create lr scheduler
    test_epoch = 5
    lr_scheduler, _ = _create_lr_scheduler_and_optimizer(config)
    lr_scheduler.step(test_epoch)
    assert lr_scheduler.get_last_lr()[0] == 0.5 * config.l_rate


@pytest.mark.parametrize("warmup_epochs, expected_lrs",
                         [(0, np.array([1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6, 1e-6])),
                          (5, np.array([0, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]))])
def test_multistep_lr(warmup_epochs: int, expected_lrs: np.ndarray) -> None:
    """
    Creates a MultiStep LR and check values are returned as expected
    """

    num_epochs = 10
    config = DummyModel(l_rate_decay=LRSchedulerType.MultiStep,
                        l_rate_gamma=0.1,
                        num_epochs=num_epochs,
                        l_rate_milestones=[2, 5, 7],
                        warmup_epochs=warmup_epochs)

    # create lr scheduler
    lr_scheduler, optimizer = _create_lr_scheduler_and_optimizer(config)

    lrs = []
    for _ in range(num_epochs):
        lrs.append(lr_scheduler.get_last_lr()[0])
        lr_scheduler.step()

    assert np.allclose(expected_lrs, lrs)
