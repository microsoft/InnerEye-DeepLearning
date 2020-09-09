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


def enumerate_scheduler(scheduler: _LRScheduler, steps: int) -> List[float]:
    """
    Reads the current learning rate via get_last_lr, run 1 scheduler step, and repeat. Returns the LR values.
    """
    lrs = []
    for _ in range(steps):
        lr = scheduler.get_last_lr()  # type: ignore
        assert isinstance(lr, list)
        assert len(lr) == 1
        lrs.append(lr[0])
        scheduler.step()
    return lrs


def test_create_lr_scheduler_last_epoch() -> None:
    """
    Test to check if the lr scheduler is initialized to the correct epoch
    """
    l_rate = 1e-3
    gamma = 0.5
    total_epochs = 5
    expected_lrs_per_epoch = [l_rate * (gamma ** i) for i in range(total_epochs)]
    config = DummyModel()
    config.l_rate = l_rate
    config.l_rate_scheduler = LRSchedulerType.Step
    config.l_rate_step_step_size = 1
    config.l_rate_step_gamma = gamma
    # create lr scheduler
    initial_scheduler, initial_optimizer = _create_lr_scheduler_and_optimizer(config)
    # check lr scheduler initialization step
    initial_epochs = 3
    assert np.allclose(enumerate_scheduler(initial_scheduler, initial_epochs), expected_lrs_per_epoch[:initial_epochs])
    # create lr scheduler for recovery checkpoint
    config.start_epoch = initial_epochs
    recovery_scheduler, recovery_optimizer = _create_lr_scheduler_and_optimizer(config)
    # Both the scheduler and the optimizer need to be loaded from the checkpoint.
    recovery_scheduler.load_state_dict(initial_scheduler.state_dict())
    recovery_optimizer.load_state_dict(initial_optimizer.state_dict())
    assert recovery_scheduler.last_epoch == config.start_epoch
    # check lr scheduler initialization matches the checkpoint epoch
    # as training will start for start_epoch + 1 in this case
    assert np.allclose(enumerate_scheduler(recovery_scheduler, 2), expected_lrs_per_epoch[initial_epochs:])


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
    lr_list = enumerate_scheduler(lr_scheduler, config.num_epochs)
    assert non_increasing(lr_list)


@pytest.mark.parametrize("lr_scheduler_type", [x for x in LRSchedulerType])
@pytest.mark.parametrize("warmup_epochs", [0, 3])
def test_warmup_against_original_schedule(lr_scheduler_type: LRSchedulerType, warmup_epochs: int) -> None:
    """
    Tests if LR scheduler with warmup matches the Pytorch implementation after the warmup stage is completed.
    """
    config = DummyModel(num_epochs=6,
                        l_rate=1e-2,
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
    lr_scheduler, optimizer1 = _create_lr_scheduler_and_optimizer(config)

    original_scheduler: Optional[_LRScheduler] = None
    optimizer2 = _create_dummy_optimizer(config)
    # This mimics the code in SchedulerWithWarmUp.get_scheduler and must be in sync
    if lr_scheduler_type == LRSchedulerType.Exponential:
        original_scheduler = ExponentialLR(optimizer=optimizer2, gamma=config.l_rate_exponential_gamma)
    elif lr_scheduler_type == LRSchedulerType.Step:
        original_scheduler = StepLR(optimizer=optimizer2, step_size=config.l_rate_step_step_size,
                                    gamma=config.l_rate_step_gamma)
    elif lr_scheduler_type == LRSchedulerType.Cosine:
        original_scheduler = CosineAnnealingLR(optimizer2, T_max=config.num_epochs, eta_min=config.min_l_rate)
    elif lr_scheduler_type == LRSchedulerType.MultiStep:
        assert config.l_rate_multi_step_milestones is not None  # for mypy
        original_scheduler = MultiStepLR(optimizer=optimizer2, milestones=config.l_rate_multi_step_milestones,
                                         gamma=config.l_rate_multi_step_gamma)
    elif lr_scheduler_type == LRSchedulerType.Polynomial:
        x = config.min_l_rate / config.l_rate
        polynomial_decay: Any = lambda epoch: (1 - x) * (
                (1. - float(epoch) / config.num_epochs) ** config.l_rate_polynomial_gamma) + x
        original_scheduler = LambdaLR(optimizer=optimizer2, lr_lambda=polynomial_decay)
    else:
        raise ValueError("Scheduler has not been added to this test.")

    expected_lr_list = []
    if warmup_epochs == 0:
        pass
    elif warmup_epochs == 3:
        # For the first config.l_rate_warmup_epochs, the learning rate is lower than the initial learning rate by a
        # linear factor
        expected_lr_list.extend([f * config.l_rate for f in [0.25, 0.5, 0.75]])
    else:
        raise NotImplementedError()
    expected_lr_list.extend(enumerate_scheduler(original_scheduler, config.num_epochs - warmup_epochs))
    print(f"Expected schedule with warmup: {expected_lr_list}")

    lr_with_warmup_scheduler = enumerate_scheduler(lr_scheduler, config.num_epochs)
    print(f"Actual schedule: {lr_with_warmup_scheduler}")

    if ((lr_scheduler_type == LRSchedulerType.Polynomial or lr_scheduler_type == LRSchedulerType.Cosine)
            and warmup_epochs > 0):
        # Polynomial and Cosine scheduler will be squashed in time because the number of epochs is reduced
        # (both schedulers take a "length of training" argument, and that is now shorter). Skip comparing those.
        pass
    else:
        assert np.allclose(lr_with_warmup_scheduler, expected_lr_list, rtol=1e-5)


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
        last_lr = scheduler.get_last_lr()  # type: ignore
        lrs.append(last_lr)
        # get_last_lr should not change the state when called twice
        assert scheduler.get_last_lr() == last_lr  # type: ignore
        scheduler.step()
    # Expected behaviour: First LR should be the initial LR set in the optimizers.
    assert lrs == [[v] for v in expected_values]


@pytest.mark.parametrize("warmup_epochs, expected_values",
                         [(0, [1, 1, 0.5, 0.5]),
                          (1, [0.5, 1, 1, 0.5]),
                          (2, [1 / 3, 2 / 3, 1, 1])])
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
    lrs = enumerate_scheduler(scheduler, 4)
    assert lrs == expected_values


# Exclude Polynomial scheduler because that uses lambdas, which we can't save to a state dict
@pytest.mark.parametrize("lr_scheduler_type", [x for x in LRSchedulerType if x != LRSchedulerType.Polynomial])
@pytest.mark.parametrize("warmup_epochs", [0, 3, 4, 5])
def test_resume_from_saved_state(lr_scheduler_type: LRSchedulerType, warmup_epochs: int) -> None:
    """
    Tests if LR scheduler when restarted from an epoch continues as expected.
    """
    restart_from_epoch = 4
    config = DummyModel(num_epochs=7,
                        l_rate_scheduler=lr_scheduler_type,
                        l_rate_exponential_gamma=0.9,
                        l_rate_step_gamma=0.9,
                        l_rate_step_step_size=2,
                        l_rate_multi_step_gamma=0.9,
                        l_rate_multi_step_milestones=[3, 5, 7],
                        l_rate_polynomial_gamma=0.9,
                        l_rate_warmup=LRWarmUpType.Linear if warmup_epochs > 0 else LRWarmUpType.NoWarmUp,
                        l_rate_warmup_epochs=warmup_epochs)
    # This scheduler mimics what happens if we train for the full set of epochs
    scheduler_all_epochs, _ = _create_lr_scheduler_and_optimizer(config)
    expected_lr_list = enumerate_scheduler(scheduler_all_epochs, config.num_epochs)

    # Create a scheduler where training will be recovered
    scheduler1, optimizer1 = _create_lr_scheduler_and_optimizer(config)
    # Scheduler 1 is only run for 4 epochs, and then "restarted" to train the rest of the epochs.
    result_lr_list = enumerate_scheduler(scheduler1, restart_from_epoch)
    # resume state: This just means setting start_epoch in the config
    config.start_epoch = restart_from_epoch
    scheduler_resume, optimizer_resume = _create_lr_scheduler_and_optimizer(config)
    # Load a "checkpoint" for both scheduler and optimizer
    scheduler_resume.load_state_dict(scheduler1.state_dict())
    optimizer_resume.load_state_dict(optimizer1.state_dict())
    result_lr_list.extend(enumerate_scheduler(scheduler_resume, config.num_epochs - restart_from_epoch))
    print(f"Actual   schedule: {result_lr_list}")
    print(f"Expected schedule: {expected_lr_list}")
    assert len(result_lr_list) == len(expected_lr_list)
    assert np.allclose(result_lr_list, expected_lr_list)


@pytest.mark.parametrize("lr_scheduler_type", [x for x in LRSchedulerType])
def test_save_and_load_state_dict(lr_scheduler_type: LRSchedulerType) -> None:
    def object_dict_same(lr1: SchedulerWithWarmUp, lr2: SchedulerWithWarmUp) -> bool:
        """
        Tests to see if two LRScheduler objects are the same.
        This ignores lambdas if one of the schedulers is LambdaLR, since lambdas are not stored to the state dict.
        """
        # ignore the _scheduler and _warmup objects, compare those separately
        dict1 = {key: val for key, val in lr1.__dict__.items() if key != "_scheduler" and key != "_warmup"}
        dict2 = {key: val for key, val in lr2.__dict__.items() if key != "_scheduler" and key != "_warmup"}

        # see if the underlying scheduler object is the same
        scheduler1_dict = {key: val for key, val in lr1._scheduler.__dict__.items() if key != "lr_lambdas"}
        scheduler2_dict = {key: val for key, val in lr2._scheduler.__dict__.items() if key != "lr_lambdas"}
        warmup1_dict = lr1._warmup.__dict__
        warmup2_dict = lr2._warmup.__dict__
        return dict1 == dict2 and scheduler1_dict == scheduler2_dict and warmup1_dict == warmup2_dict

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
    lr_scheduler_1._scheduler.step()
    lr_scheduler_1._scheduler.step()

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
    for _ in range(test_epoch):
        lr_scheduler.step()
    assert lr_scheduler.get_last_lr()[0] == 0.5 * config.l_rate


def test_multistep_lr() -> None:
    l_rate = 0.3
    config = DummyModel(l_rate_scheduler=LRSchedulerType.MultiStep,
                        l_rate=l_rate,
                        l_rate_multi_step_gamma=0.1,
                        num_epochs=10,
                        l_rate_multi_step_milestones=[2],
                        l_rate_warmup=LRWarmUpType.Linear,
                        l_rate_warmup_epochs=5)

    def check_warmup(expected: List[float]) -> None:
        scheduler, _ = _create_lr_scheduler_and_optimizer(config)
        actual = enumerate_scheduler(scheduler, 4)
        assert actual == expected

    # No warmup: multi-step LR with milestone after 2 epochs
    original_schedule = [l_rate, l_rate, l_rate * 0.1, l_rate * 0.1]
    config.l_rate_warmup = LRWarmUpType.Linear
    config.l_rate_warmup_epochs = 0
    check_warmup(original_schedule)

    # 1 epoch warmup: linear function up to the initial learning rate gives a warmup value of half the initial LR
    config.l_rate_warmup_epochs = 1
    check_warmup([l_rate * 0.5] + original_schedule[:3])

    # 2 epochs warmup
    config.l_rate_warmup_epochs = 2
    check_warmup([l_rate / 3, l_rate * 2 / 3] + original_schedule[:2])
