#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, List, Optional, Dict
from typing_extensions import Protocol

from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, StepLR, MultiStepLR, CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from InnerEye.ML.deep_learning_config import DeepLearningConfig, LRSchedulerType, LRWarmUpType


class LRWarmUp(_LRScheduler):
    """
    Base class for schedulers that implement learning rate warmup.
    """

    def __init__(self, optimizer: Optimizer, warmup_epochs: int, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * min(self.warmup_function(), 1) for base_lr in self.base_lrs]

    def warmup_function(self):
        raise NotImplementedError


class NoLRWarmUp(LRWarmUp):
    """
    Identity class when there is no warmup step.
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        warmup_epochs = 0
        super().__init__(optimizer, warmup_epochs, last_epoch)

    def warmup_function(self):
        return 1


class LinearLRWarmUp(LRWarmUp):
    """
    Implements linear warmup.
    """
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, last_epoch: int = -1):
        if warmup_epochs < 1:
            raise ValueError("The number of warmup epochs must be a positive integer.")
        super().__init__(optimizer, warmup_epochs, last_epoch)

    def warmup_function(self):
        return min((self.last_epoch + 1) / self.warmup_epochs, 1)


class SchedulerWithWarmUp(_LRScheduler):
    """
    LR Scheduler which runs first a warmup step and then a standard scheduler for LR decay.
    """
    def __init__(self, args: DeepLearningConfig, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self._warmup_scheduler = self.get_warmup(args)
        self._scheduler = self.get_scheduler(args)
        super().__init__(optimizer, last_epoch)

    def get_scheduler(self, args: DeepLearningConfig) -> _LRScheduler:
        """
        Create a LR scheduler from the config params.
        """

        last_epoch = max(-1, self.last_epoch - args.l_rate_warmup_epochs)

        if args.l_rate_scheduler == LRSchedulerType.Exponential:
            scheduler = ExponentialLR(optimizer=self.optimizer,
                                      gamma=args.l_rate_exponential_gamma,
                                      last_epoch=last_epoch)
        elif args.l_rate_scheduler == LRSchedulerType.Step:
            scheduler = StepLR(optimizer=self.optimizer,
                               step_size=args.l_rate_step_step_size,
                               gamma=args.l_rate_step_gamma,
                               last_epoch=last_epoch)
        elif args.l_rate_scheduler == LRSchedulerType.MultiStep:
            scheduler = MultiStepLR(optimizer=self.optimizer,
                                    milestones=args.l_rate_multi_step_milestones,
                                    gamma=args.l_rate_multi_step_gamma,
                                    last_epoch=last_epoch)
        elif args.l_rate_scheduler == LRSchedulerType.Polynomial:
            x = args.min_l_rate / args.l_rate
            polynomial_decay: Any = lambda epoch: (1 - x) * (
                    (1. - float(epoch) / args.num_epochs) ** args.l_rate_polynomial_gamma) + x
            scheduler = LambdaLR(optimizer=self.optimizer,
                                 lr_lambda=polynomial_decay,
                                 last_epoch=last_epoch)
        elif args.l_rate_scheduler == LRSchedulerType.Cosine:
            scheduler = CosineAnnealingLR(optimizer=self.optimizer,
                                          T_max=args.num_epochs,
                                          eta_min=args.min_l_rate,
                                          last_epoch=last_epoch)
        else:
            raise ValueError("Unknown learning rate scheduler {}".format(args.l_rate_scheduler))
        return scheduler

    def get_warmup(self, args: DeepLearningConfig) -> LRWarmUp:
        """
        Create a scheduler for warmup steps from the config params.
        """

        if args.l_rate_warmup == LRWarmUpType.NoWarmUp:
            warmup = NoLRWarmUp(optimizer=self.optimizer,
                                last_epoch=self.last_epoch)
        elif args.l_rate_warmup == LRWarmUpType.Linear:
            warmup = LinearLRWarmUp(optimizer=self.optimizer,
                                    warmup_epochs=args.l_rate_warmup_epochs,
                                    last_epoch=self.last_epoch)
        else:
            raise ValueError("Unknown learning rate warmup {}".format(args.l_rate_warmup))
        return warmup

    def state_dict(self) -> Dict:
        state_dict = {key: val for key, val in self.__dict__.items()
                                            if key != "_scheduler" and key != "_warmup_scheduler"
                                                                   and key != "optimizer"}
        state_dict['_scheduler'] = self._scheduler.state_dict()
        state_dict['_warmup_scheduler'] = self._warmup_scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        top_level = {key: val for key, val in state_dict.items()
                     if key != "_scheduler" and key != "_warmup_scheduler"}
        self.__dict__.update(top_level)
        self._scheduler.__dict__.update(state_dict["_scheduler"])
        self._warmup_scheduler.__dict__.update(state_dict["_warmup_scheduler"])

    def get_lr(self):
        if self.last_epoch < self._warmup_scheduler.warmup_epochs:
            return self._warmup_scheduler.get_lr()
        else:
            return self._scheduler.get_lr()

    def step(self, epoch=None):
        target_epoch = epoch if epoch is not None else self.last_epoch + 1

        if target_epoch < self._warmup_scheduler.warmup_epochs:
            self._warmup_scheduler.step(epoch)
        elif target_epoch == self._warmup_scheduler.warmup_epochs:
            # don't step here, or we will miss the first value from the scheduler
            pass
        else:
            scheduler_epoch = epoch - self._warmup_scheduler.warmup_epochs if epoch else None
            self._scheduler.step(scheduler_epoch)

        self.last_epoch = target_epoch


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

        self._scheduler = SchedulerWithWarmUp(args, optimizer, last_epoch)

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
