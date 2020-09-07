#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LambdaLR, MultiStepLR, MultiplicativeLR, \
    StepLR, _LRScheduler
from torch.optim.optimizer import Optimizer

from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.deep_learning_config import DeepLearningConfig, LRSchedulerType, LRWarmUpType
from InnerEye.ML.utils.lr_scheduler import SchedulerWithWarmUp
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
    config = DummyModel(num_epochs=2, l_rate=1e-3, min_l_rate=0.0009,
                        l_rate_scheduler=lr_scheduler_type,
                        l_rate_exponential_gamma=0.9,
                        l_rate_step_gamma=0.9,
                        l_rate_step_step_size=1,
                        l_rate_multi_step_gamma=0.7,
                        l_rate_multi_step_milestones=[1],
                        l_rate_polynomial_gamma=0.9)
    # create lr scheduler
    lr_scheduler, _ = _create_lr_scheduler_and_optimizer(config)
    assert lr_scheduler.get_last_lr()[0] == config.l_rate
    lr_scheduler.step()
    lr_scheduler.step()
    assert lr_scheduler.get_last_lr()[0] == config.min_l_rate


@pytest.mark.parametrize("lr_scheduler_type", [x for x in LRSchedulerType])
def test_lr_monotonically_decreasing_function(lr_scheduler_type: LRSchedulerType) -> None:
    """
    Tests if LR scheduler is a monotonically decreasing function
    """
    config = DummyModel(num_epochs=10,
                        l_rate_scheduler=lr_scheduler_type,
                        l_rate_exponential_gamma=0.9,
                        l_rate_step_gamma=0.9,
                        l_rate_step_step_size=1,
                        l_rate_multi_step_gamma=0.9,
                        l_rate_multi_step_milestones=[3, 5, 7],
                        l_rate_polynomial_gamma=0.9)

    def non_increasing(L: List) -> bool:
        return all(x >= y for x, y in zip(L, L[1:]))

    # create lr scheduler
    lr_scheduler, _ = _create_lr_scheduler_and_optimizer(config)
    lr_list = []
    for _ in range(config.num_epochs):
        lr_scheduler.step()
        lr_list.append(lr_scheduler.get_last_lr()[0])

    assert non_increasing(lr_list)


@pytest.mark.parametrize("lr_scheduler_type", [x for x in LRSchedulerType])
@pytest.mark.parametrize("warmup_epochs", [0, 4, 5])
def test_warmup_against_original_schedule(lr_scheduler_type: LRSchedulerType, warmup_epochs: int) -> None:
    """
    Tests if LR scheduler with warmup matches the Pytorch implementation after the warmup stage is completed.
    """
    config = DummyModel(num_epochs=10,
                        l_rate_scheduler=lr_scheduler_type,
                        l_rate_exponential_gamma=0.9,
                        l_rate_step_gamma=0.9,
                        l_rate_step_step_size=2,
                        l_rate_multi_step_gamma=0.9,
                        l_rate_multi_step_milestones=[3, 5, 7],
                        l_rate_polynomial_gamma=0.9,
                        l_rate_warmup=LRWarmUpType.Linear if warmup_epochs > 0 else LRWarmUpType.NoWarmUp,
                        l_rate_warmup_epochs=warmup_epochs)
    # create lr scheduler
    lr_scheduler, optimizer = _create_lr_scheduler_and_optimizer(config)

    original_scheduler: Optional[_LRScheduler] = None
    if lr_scheduler_type == LRSchedulerType.Exponential:
        original_scheduler = ExponentialLR(optimizer=optimizer, gamma=config.l_rate_exponential_gamma)
    elif lr_scheduler_type == LRSchedulerType.Step:
        original_scheduler = StepLR(optimizer=optimizer, step_size=config.l_rate_step_step_size,
                                    gamma=config.l_rate_step_gamma)
    elif lr_scheduler_type == LRSchedulerType.Cosine:
        original_scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=config.min_l_rate)
    elif lr_scheduler_type == LRSchedulerType.MultiStep:
        assert config.l_rate_multi_step_milestones is not None  # for mypy
        original_scheduler = MultiStepLR(optimizer=optimizer, milestones=config.l_rate_multi_step_milestones,
                                         gamma=config.l_rate_multi_step_gamma)
    elif lr_scheduler_type == LRSchedulerType.Polynomial:
        x = config.min_l_rate / config.l_rate
        polynomial_decay: Any = lambda epoch: (1 - x) * (
                (1. - float(epoch) / config.num_epochs) ** config.l_rate_polynomial_gamma) + x
        original_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=polynomial_decay)
    else:
        raise ValueError("Scheduler has not been added to this test.")

    result_lr_list = []
    for _ in range(config.num_epochs):
        result_lr_list.append(lr_scheduler.get_last_lr()[0])
        lr_scheduler.step()

    expected_lr_list = []
    for i in range(warmup_epochs):
        expected_lr_list.append(config.l_rate * (i + 1) / warmup_epochs)
    for _ in range(config.num_epochs - warmup_epochs):
        expected_lr_list.append(original_scheduler.get_last_lr())
        original_scheduler.step()  # type: ignore

    assert np.allclose(result_lr_list, expected_lr_list)


def _create_dummy_optimizer(config: SegmentationModelBase) -> Optimizer:
    return torch.optim.Adam([torch.ones(2, 2, requires_grad=True)], lr=config.l_rate)


def _create_lr_scheduler_and_optimizer(config: SegmentationModelBase, optimizer: Optimizer = None) \
        -> Tuple[SchedulerWithWarmUp, Optimizer]:
    # create dummy optimizer
    if optimizer is None:
        optimizer = _create_dummy_optimizer(config)
    # create lr scheduler
    lr_scheduler = SchedulerWithWarmUp(config, optimizer)
    return lr_scheduler, optimizer


@pytest.mark.parametrize("scheduler_func, expected_values",
                         # A scheduler that reduces learning rate by a factor of 0.5 in each epoch
                         [(lambda optimizer: MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.5),
                           [1, 0.5, 0.25, 0.125, 0.0625]),
                          # A scheduler that reduces learning rate by a factor of 0.5 at epochs 2 and 4
                          (lambda optimizer: MultiStepLR(optimizer, [2, 4], gamma=0.5),
                           [1, 1, 0.5, 0.5, 0.25]),
                          (lambda optimizer: MultiStepLR(optimizer, [1, 2, 3, 4, 5], gamma=0.5),
                           [1, 0.5, 0.25, 0.125, 0.0625])
                          ])
def test_built_in_lr_scheduler(scheduler_func: Callable[[Optimizer], _LRScheduler],
                               expected_values: List[float]) -> None:
    """
    A test to check that the behaviour of the built-in learning rate schedulers is still what we think it is.
    """
    initial_lr = 1
    optimizer = torch.optim.Adam([torch.ones(2, 2, requires_grad=True)], lr=initial_lr)
    scheduler = scheduler_func(optimizer)
    lrs = []
    for _ in range(5):
        last_lr = scheduler.get_last_lr()
        lrs.append(last_lr)
        # get_last_lr should not change the state when called twice
        assert scheduler.get_last_lr() == scheduler.get_last_lr()
        scheduler.step()
    # Expected behaviour: First LR should be the initial LR set in the optimizers.
    expected_values = [[v] for v in expected_values]
    assert lrs == expected_values


@pytest.mark.parametrize("min_l_rate, expected_values",
                         [(0.4, [1, 0.5, 0.4, 0.4]),
                          (0.0, [1, 0.5, 0.25, 0.125])])
def test_lr_scheduler_with_min_l_rate(min_l_rate: float, expected_values: List[float]) -> None:
    """
    Check that the minimum learning rate parameter is respected.
    """
    initial_lr = 1
    optimizer = torch.optim.Adam([torch.ones(2, 2, requires_grad=True)], lr=initial_lr)
    config = DeepLearningConfig(l_rate=initial_lr,
                                l_rate_scheduler=LRSchedulerType.MultiStep,
                                l_rate_multi_step_milestones=[1, 2, 3, 4],
                                l_rate_multi_step_gamma=0.5,
                                should_validate=False)
    config._min_l_rate = min_l_rate
    scheduler = SchedulerWithWarmUp(config, optimizer)
    lrs = []
    for _ in range(4):
        last_lr = scheduler.get_last_lr()
        lrs.append(last_lr)
        scheduler.step()
    expected_values = [[v] for v in expected_values]
    assert lrs == expected_values


@pytest.mark.parametrize("warmup_epochs, expected_values",
                         [(0, [1, 1, 0.5, 0.5]),
                          (1, [0.5, 1, 0.5, 0.5]),
                          (2, [1/3, 2/3, 0.25, 0.25])])
def test_lr_scheduler_with_warmup(warmup_epochs: int, expected_values: List[float]) -> None:
    """
    Check that warmup is applied correctly to a multistep scheduler
    """
    initial_lr = 1
    optimizer = torch.optim.Adam([torch.ones(2, 2, requires_grad=True)], lr=initial_lr)
    config = DeepLearningConfig(l_rate=initial_lr,
                                l_rate_scheduler=LRSchedulerType.MultiStep,
                                l_rate_multi_step_milestones=[2, 4],
                                l_rate_multi_step_gamma=0.5,
                                l_rate_warmup_epochs=warmup_epochs,
                                l_rate_warmup=LRWarmUpType.Linear,
                                should_validate=False)
    scheduler = SchedulerWithWarmUp(config, optimizer)
    lrs = []
    for _ in range(4):
        last_lr = scheduler.get_last_lr()
        lrs.append(last_lr)
        scheduler.step()
    expected_values = [[v] for v in expected_values]
    assert lrs == expected_values


@pytest.mark.parametrize("lr_scheduler_type", [x for x in LRSchedulerType])
@pytest.mark.parametrize("warmup_epochs", [0, 3, 4, 5])
@pytest.mark.parametrize("restart_from_epoch", [4])
def test_resume_from_saved_state(lr_scheduler_type: LRSchedulerType,
                                 warmup_epochs: int, restart_from_epoch: int) -> None:
    """
    Tests if LR scheduler when restarted from an epoch continues as expected.
    """
    config = DummyModel(num_epochs=10,
                        l_rate_scheduler=lr_scheduler_type,
                        l_rate_exponential_gamma=0.9,
                        l_rate_step_gamma=0.9,
                        l_rate_step_step_size=2,
                        l_rate_multi_step_gamma=0.9,
                        l_rate_multi_step_milestones=[3, 5, 7],
                        l_rate_polynomial_gamma=0.9,
                        l_rate_warmup=LRWarmUpType.Linear if warmup_epochs > 0 else LRWarmUpType.NoWarmUp,
                        l_rate_warmup_epochs=warmup_epochs)
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


@pytest.mark.parametrize("lr_scheduler_type", [x for x in LRSchedulerType])
def test_save_and_load_state_dict(lr_scheduler_type: LRSchedulerType) -> None:
    def object_dict_same(lr1: SchedulerWithWarmUp, lr2: SchedulerWithWarmUp) -> bool:
        """
        Tests to see if two LRScheduler objects are the same.
        This ignores lambdas if one of the schedulers is LambdaLR, since lambdas are not stored to the state dict.
        """
        # dict of object LR scheduler
        # ignore the _scheduler attribute, which is of type SchedulerWithWarmUp and compare it separately
        dict1 = {key: val for key, val in lr1.__dict__.items() if key != "_scheduler"}
        dict2 = {key: val for key, val in lr2.__dict__.items() if key != "_scheduler"}

        # see if the SchedulerWithWarmUp object is the same
        warmup_and_scheduler1 = lr1.__dict__["_scheduler"]
        warmup_and_scheduler2 = lr2.__dict__["_scheduler"]

        # scheduler object
        scheduler1 = warmup_and_scheduler1.__dict__["_scheduler"]
        scheduler2 = warmup_and_scheduler2.__dict__["_scheduler"]
        # remove lambdas from scheduler dict
        scheduler1_dict = {key: val for key, val in scheduler1.__dict__.items() if key != "lr_lambdas"}
        scheduler2_dict = {key: val for key, val in scheduler2.__dict__.items() if key != "lr_lambdas"}

        # warmup object
        warmup1 = warmup_and_scheduler1.__dict__["_warmup_scheduler"]
        warmup2 = warmup_and_scheduler2.__dict__["_warmup_scheduler"]

        # Other variables in the object SchedulerWithWarmUp
        other_variables1 = {key: val for key, val in warmup_and_scheduler1.__dict__.items()
                            if key != "_scheduler" and key != "_warmup_scheduler"}
        other_variables2 = {key: val for key, val in warmup_and_scheduler2.__dict__.items()
                            if key != "_scheduler" and key != "_warmup_scheduler"}

        return dict1 == dict2 and other_variables1 == other_variables2 and \
               scheduler1_dict == scheduler2_dict and \
               warmup1.__dict__ == warmup2.__dict__

    config = DummyModel(num_epochs=10,
                        l_rate_scheduler=lr_scheduler_type,
                        l_rate_exponential_gamma=0.9,
                        l_rate_step_gamma=0.9,
                        l_rate_step_step_size=2,
                        l_rate_multi_step_gamma=0.9,
                        l_rate_multi_step_milestones=[3, 5, 7],
                        l_rate_polynomial_gamma=0.9,
                        l_rate_warmup=LRWarmUpType.Linear,
                        l_rate_warmup_epochs=4)
    lr_scheduler_1, optimizer = _create_lr_scheduler_and_optimizer(config)

    lr_scheduler_1.step()
    # This is not supported functionality - we are doing this just to change _scheduler from its default state
    lr_scheduler_1.scheduler.step()
    lr_scheduler_1.scheduler.step()

    state_dict = lr_scheduler_1.state_dict()

    lr_scheduler_2, _ = _create_lr_scheduler_and_optimizer(config, optimizer)

    assert not object_dict_same(lr_scheduler_1, lr_scheduler_2)

    lr_scheduler_2.load_state_dict(state_dict)

    assert object_dict_same(lr_scheduler_1, lr_scheduler_2)


def test_cosine_decay_function() -> None:
    """
    Tests Cosine lr decay function at (pi/2) and verifies if the value is correct.
    """
    config = DummyModel(l_rate_scheduler=LRSchedulerType.Cosine,
                        num_epochs=10,
                        min_l_rate=0.0)

    # create lr scheduler
    test_epoch = 5
    lr_scheduler, _ = _create_lr_scheduler_and_optimizer(config)
    lr_scheduler.step(test_epoch)
    assert lr_scheduler.get_last_lr()[0] == 0.5 * config.l_rate


@pytest.mark.parametrize("warmup_epochs, expected_lrs",
                         [(0, np.array([1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6, 1e-6])),
                          (5, np.array([2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]))])
def test_multistep_lr(warmup_epochs: int, expected_lrs: np.ndarray) -> None:
    """
    Creates a MultiStep LR and check values are returned as expected
    """
    num_epochs = 10
    initial_l_rate = 1e-3
    config = DummyModel(l_rate_scheduler=LRSchedulerType.MultiStep,
                        l_rate=initial_l_rate,
                        l_rate_multi_step_gamma=0.1,
                        num_epochs=num_epochs,
                        l_rate_multi_step_milestones=[2, 5, 7],
                        l_rate_warmup=LRWarmUpType.Linear if warmup_epochs > 0 else LRWarmUpType.NoWarmUp,
                        l_rate_warmup_epochs=warmup_epochs)

    # create lr scheduler
    lr_scheduler, optimizer = _create_lr_scheduler_and_optimizer(config)

    lrs = []
    for _ in range(num_epochs):
        last_lr = lr_scheduler.get_last_lr()
        assert isinstance(last_lr, list)
        lrs.append(last_lr[0])
        lr_scheduler.step()
    print(f"Actual learning rate schedule: {lrs}")
    if warmup_epochs == 0:
        assert lrs[0] == initial_l_rate
    else:
        assert lrs[warmup_epochs - 1] == initial_l_rate
    assert np.allclose(lrs, expected_lrs)
