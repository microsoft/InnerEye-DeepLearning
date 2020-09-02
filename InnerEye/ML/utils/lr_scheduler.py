#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, List, Optional
from typing_extensions import Protocol
from collections import Iterable

from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, StepLR, MultiStepLR, CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from InnerEye.ML.deep_learning_config import DeepLearningConfig, LRSchedulerType


class WarmUpMixin(Protocol):
    """
    This class exists to tell mypy about the attributes the mixin expects
    """
    last_epoch: int
    base_lrs: Iterable
    warmup_epochs: int


class SchedulerWithWarmUpMixin(object):
    """
    Inherit this class as the FIRST parent, and a scheduler from torch.optim.lr_scheduler as the second parent
    to add a warmup step before the scheduler starts the learning rate decay.

    Ex.
    from torch.optim.lr_scheduler import SomeScheduler
    class SomeSchedulerWithWarmup(SchedulerWithWarmUpMixin, SomeScheduler)
        pass

    scheduler = SomeSchedulerWithWarmup(warmup_epochs=..., <keyword args for scheduler>)
    """
    def __init__(self, *, warmup_epochs: int, optimizer: Optimizer, last_epoch: int, **kwargs: Any):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer=optimizer, last_epoch=last_epoch, **kwargs)  # type: ignore

    def get_lr(self: WarmUpMixin) -> Iterable:
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * self.last_epoch / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            self.last_epoch -= self.warmup_epochs
            lrs = super().get_lr()  # type: ignore
            self.last_epoch += self.warmup_epochs
            return lrs


class ExponentialLRWithWarmUp(SchedulerWithWarmUpMixin, ExponentialLR):  # type: ignore
    """
    ExponentialLR with added warmup steps.
    Usage is the same as ExponentialLR, with an added keyword param "warmup_epochs"
    """
    pass


class StepLRWithWarmUp(SchedulerWithWarmUpMixin, StepLR):  # type: ignore
    """
    StepLR with added warmup steps.
    Usage is the same as StepLR, with an added keyword param "warmup_epochs"
    """
    pass


class LambdaLRWithWarmUp(SchedulerWithWarmUpMixin, LambdaLR):  # type: ignore
    """
    LambdaLR with added warmup steps.
    Usage is the same as LambdaLR, with an added keyword param "warmup_epochs"
    """
    pass


class CosineAnnealingLRWithWarmUp(SchedulerWithWarmUpMixin, CosineAnnealingLR):  # type: ignore
    """
    CosineAnnealingLR with added warmup steps.
    Usage is the same as CosineAnnealingLR, with an added keyword param "warmup_epochs"
    """
    pass


class MultiStepLRWithWarmUp(SchedulerWithWarmUpMixin, MultiStepLR):  # type: ignore
    """
    MultiStepLR with added warmup steps.
    Usage is the same as MultiStepLR, with an added keyword param "warmup_epochs"
    """
    pass


class LRScheduler:
    """
    Wrapper around Torch LRScheduler functions with added functionality to restrict learning rate to a
    minimum value based on the provided configurations.
    """
    _scheduler: _LRScheduler
    _min_lr: float = 0
    _max_epochs: int = 0

    def __init__(self, args: DeepLearningConfig, optimizer: Optimizer):
        """

        :param args: the config defining the model
        :param optimizer: the optimizer to use for model training
        """
        self._min_lr = args.min_l_rate
        self._max_epochs = args.num_epochs

        # if loading from a checkpoint, then last epoch will be the checkpoint epoch
        # otherwise -1 as no epochs have been trained.
        # For pytorch version 1.3:
        last_epoch = args.start_epoch if args.should_load_checkpoint_for_training() else -1
        # For pytorch version 1.6:
        # last_epoch = args.start_epoch - 1 if args.should_load_checkpoint_for_training() else -1

        if args.l_rate_decay == LRSchedulerType.Exponential:
            self._scheduler = ExponentialLRWithWarmUp(warmup_epochs=args.warmup_epochs,
                                                      optimizer=optimizer,
                                                      gamma=args.l_rate_gamma,
                                                      last_epoch=last_epoch)
        elif args.l_rate_decay == LRSchedulerType.Step:
            self._scheduler = StepLRWithWarmUp(warmup_epochs=args.warmup_epochs,
                                               optimizer=optimizer,
                                               step_size=args.l_rate_step_size,
                                               gamma=args.l_rate_gamma,
                                               last_epoch=last_epoch)
        elif args.l_rate_decay == LRSchedulerType.MultiStep:
            self._scheduler = MultiStepLRWithWarmUp(warmup_epochs=args.warmup_epochs,
                                                    optimizer=optimizer,
                                                    milestones=args.l_rate_milestones,
                                                    gamma=args.l_rate_gamma,
                                                    last_epoch=last_epoch)
        elif args.l_rate_decay == LRSchedulerType.Polynomial:
            x = args.min_l_rate / args.l_rate
            polynomial_decay: Any = lambda epoch: (1 - x) * (
                    (1. - float(epoch) / self._max_epochs) ** args.l_rate_gamma) + x
            self._scheduler = LambdaLRWithWarmUp(warmup_epochs=args.warmup_epochs,
                                                 optimizer=optimizer,
                                                 lr_lambda=polynomial_decay,
                                                 last_epoch=last_epoch)
        elif args.l_rate_decay == LRSchedulerType.Cosine:
            self._scheduler = CosineAnnealingLRWithWarmUp(warmup_epochs=args.warmup_epochs,
                                                          optimizer=optimizer,
                                                          T_max=self._max_epochs,
                                                          eta_min=args.min_l_rate,
                                                          last_epoch=last_epoch)
        else:
            raise ValueError("Unknown learning rate scheduler {}".format(args.l_rate_decay))

    def get_last_lr(self) -> List[float]:
        """
        Get the current learning rate (making sure it is >= min_l_rate if provided in the config)
        """
        # For pytorch version 1.3:
        lr = self._scheduler.get_lr()  # type: ignore
        # For pytorch version 1.6:
        # lr = self._scheduler.get_last_lr()  # type: ignore
        lrs: List[float] = [lr] if isinstance(lr, float) else lr
        return [max(self._min_lr, x) for x in lrs]

    def state_dict(self) -> dict:
        """
        Get the current lr scheduler state
        """
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the given state into the lr scheduler
        """
        self._scheduler.load_state_dict(state_dict)

    def step(self, epoch: Optional[int] = None) -> None:
        """
        Move the lr scheduler to the state corresponding to the provided epoch or next epoch.
        """
        if epoch is not None and epoch > self._max_epochs:
            raise ValueError("epoch must be <= {}".format(self._max_epochs))
        else:
            # noinspection PyTypeChecker
            self._scheduler.step(epoch)
