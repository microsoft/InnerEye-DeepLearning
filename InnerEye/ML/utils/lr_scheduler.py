#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Dict, List

from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LambdaLR, MultiStepLR, StepLR, _LRScheduler
from torch.optim.optimizer import Optimizer

from InnerEye.ML.deep_learning_config import DeepLearningConfig, LRSchedulerType, LRWarmUpType


def get_current_learning_rates(optimizer: Optimizer) -> List[float]:
    """
    Reads the current values of the learning rate(s) for all parameter groups from the optimizer.
    """
    return [group['lr'] for group in optimizer.param_groups]


class LinearWarmUp(_LRScheduler):
    """
    Implements linear warmup up to a given initial learning rate.
    """
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, final_lr: float, last_epoch: int = -1):
        if warmup_epochs < 0:
            raise ValueError("The number of warmup epochs must be >= 0.")
        self.warmup_epochs = warmup_epochs
        self.final_lr = final_lr
        self.last_epoch = last_epoch
        super().__init__(optimizer, last_epoch)

    def warmup_multiplier(self) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        if self.last_epoch >= self.warmup_epochs:
            return 1.0
        return (self.last_epoch + 1) / (self.warmup_epochs + 1)

    def get_lr(self) -> List[float]:  # type: ignore
        return [self.final_lr * self.warmup_multiplier()]


class PolynomialLR:
    def __init__(self, gamma: float, l_rate: float, min_l_rate: float, epochs_after_warmup: int) -> None:
        self.gamma = gamma
        self.l_rate = l_rate
        self.min_l_rate = min_l_rate
        self.epochs_after_warmup = epochs_after_warmup

    def get_lr(self, epoch: int) -> float:
        x = self.min_l_rate / self.l_rate
        return (1 - x) * ((1. - float(epoch) / self.epochs_after_warmup) ** self.gamma) + x


class SchedulerWithWarmUp(_LRScheduler):
    """
    LR Scheduler which runs a warmup schedule (linear ramp-up) for a few iterations, and then switches to one
    of the normal schedulers.
    """

    def __init__(self, args: DeepLearningConfig, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.warmup_epochs = 0 if args.l_rate_warmup == LRWarmUpType.NoWarmUp else args.l_rate_warmup_epochs
        self._scheduler = self.get_scheduler(args)
        # This must be called after self.get_scheduler, because we want the optimizer to have the learning rate
        # guided by the warmup schedule
        self._warmup = LinearWarmUp(optimizer,
                                    warmup_epochs=self.warmup_epochs,
                                    final_lr=args.l_rate,
                                    last_epoch=last_epoch)
        self._last_lr = get_current_learning_rates(optimizer)
        self.min_l_rate = args.min_l_rate
        super().__init__(optimizer, last_epoch)

    def get_scheduler(self, args: DeepLearningConfig) -> _LRScheduler:
        """
        Create the LR scheduler that will be used after warmup, based on the config params.
        """
        scheduler: _LRScheduler
        epochs_after_warmup = args.num_epochs - self.warmup_epochs
        if args.l_rate_scheduler == LRSchedulerType.Exponential:
            scheduler = ExponentialLR(optimizer=self.optimizer,
                                      gamma=args.l_rate_exponential_gamma,
                                      last_epoch=self.last_epoch)
        elif args.l_rate_scheduler == LRSchedulerType.Step:
            scheduler = StepLR(optimizer=self.optimizer,
                               step_size=args.l_rate_step_step_size,
                               gamma=args.l_rate_step_gamma,
                               last_epoch=self.last_epoch)
        elif args.l_rate_scheduler == LRSchedulerType.MultiStep:
            assert args.l_rate_multi_step_milestones is not None
            scheduler = MultiStepLR(optimizer=self.optimizer,
                                    milestones=args.l_rate_multi_step_milestones,
                                    gamma=args.l_rate_multi_step_gamma,
                                    last_epoch=self.last_epoch)
        elif args.l_rate_scheduler == LRSchedulerType.Polynomial:
            polynomial_lr = PolynomialLR(gamma=args.l_rate_polynomial_gamma,
                                         l_rate = args.l_rate,
                                         min_l_rate=args.min_l_rate,
                                         epochs_after_warmup=epochs_after_warmup)
            scheduler = LambdaLR(optimizer=self.optimizer,
                                 lr_lambda=polynomial_lr.get_lr,
                                 last_epoch=self.last_epoch)
        elif args.l_rate_scheduler == LRSchedulerType.Cosine:
            scheduler = CosineAnnealingLR(optimizer=self.optimizer,
                                          T_max=epochs_after_warmup,
                                          eta_min=args.min_l_rate,
                                          last_epoch=self.last_epoch)
        else:
            raise ValueError("Unknown learning rate scheduler {}".format(args.l_rate_scheduler))
        return scheduler

    def state_dict(self) -> Dict:
        """
        Added for completeness, since base class _LRScheduler implements this.
        Returns a dictionary with all the values in this objects __dict__.
        It creates the dictionary entry for variables "_scheduler" and "_warmup_scheduler" separately, by calling
        state_dict for these variables.
        The state dict does not include the state of the optimizer.
        """
        state_dict = {key: val for key, val in self.__dict__.items()
                      if key != "_scheduler" and key != "optimizer" and key != "_warmup"}
        state_dict['_scheduler'] = self._scheduler.state_dict()
        state_dict['_warmup'] = self._warmup.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Added for completeness, since base class _LRScheduler implements this.
        Initializes the current object with values from state_dict.
        Initializes variables "_scheduler" and "_warmup_scheduler" separately, by calling load_state_dict
        for these variables.
        """
        top_level = {key: val for key, val in state_dict.items() if key != "_scheduler" and key != "_warmup"}
        self.__dict__.update(top_level)
        self._scheduler.__dict__.update(state_dict["_scheduler"])
        self._warmup.__dict__.update(state_dict["_warmup"])

    def step(self, epoch: int = None) -> None:
        # self.step() is called in the _LRScheduler.__init__, as the very last operation, when self.last_epoch == -1
        # Inside of the default implementation of self.step, it calls
        # self.last_epoch += 1
        # values = self.get_lr()
        # The values are then set in the optimizer, and stored in self._last_lr
        if epoch is not None:
            raise ValueError("Calling scheduler.step with an epoch argument will be deprecated.")
        # self.step is called from within the base class constructor, _LRScheduler.__init__
        # The scheduler itself has already been initialized, and scheduler.step has also been called already in
        # the respective constructor. Avoid calling it again here.
        if self.last_epoch != -1:
            if self.last_epoch < self._warmup.warmup_epochs:
                self._warmup.step()
            else:
                self._scheduler.step()
        self.last_epoch += 1
        self._last_lr = get_current_learning_rates(self.optimizer)

    def get_last_lr(self) -> List[float]:
        return self._last_lr
