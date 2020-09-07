#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Dict, List

from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LambdaLR, MultiStepLR, StepLR, _LRScheduler
from torch.optim.optimizer import Optimizer

from InnerEye.ML.deep_learning_config import DeepLearningConfig, LRSchedulerType, LRWarmUpType


class LRWarmUp(_LRScheduler):
    """
    Base class for schedulers that implement learning rate warmup.
    """

    def __init__(self, optimizer: Optimizer, warmup_epochs: int, last_epoch: int = -1):
        if warmup_epochs < 0:
            raise ValueError("The number of warmup epochs must be >= 0")
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore
        # Base_lrs is set in the constructor of _LRScheduler to be the initial learning rates of the optimizer.
        if self.last_epoch > self.warmup_epochs:
            raise ValueError(f"The warmup scheduler should be called for at most {self.warmup_epochs} epochs. "
                             f"Current self.last_epoch == {self.last_epoch}")
        warmup_factor = min(self.warmup_function(self.last_epoch), 1)
        return [base_lr * warmup_factor for base_lr in self.base_lrs]  # type: ignore

    def warmup_function(self, last_epoch: int) -> float:
        return 1.0


class LinearLRWarmUp(LRWarmUp):
    """
    Implements linear warmup.
    """
    def warmup_function(self, last_epoch: int) -> float:
        if self.warmup_epochs == 0:
            return 1.0
        return (last_epoch + 1) / self.warmup_epochs


class SchedulerWithWarmUp(_LRScheduler):
    """
    LR Scheduler which runs first a warmup step and then a standard scheduler for LR decay.
    """

    def __init__(self, args: DeepLearningConfig, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.warmup_scheduler = self.get_warmup(args)
        self.scheduler = self.get_scheduler(args)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        self.min_l_rate = args.min_l_rate
        super().__init__(optimizer, last_epoch)

    def get_scheduler(self, args: DeepLearningConfig) -> _LRScheduler:
        """
        Create the LR scheduler that will be used after warmup, based on the config params.
        """
        last_epoch = max(-1, self.last_epoch - args.l_rate_warmup_epochs)

        scheduler: _LRScheduler
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
            assert args.l_rate_multi_step_milestones is not None  # for mypy, we have done this check elsewhere
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
        warmup: LRWarmUp
        if args.l_rate_warmup == LRWarmUpType.NoWarmUp:
            warmup_epochs = 0
        elif args.l_rate_warmup == LRWarmUpType.Linear:
            warmup_epochs = args.l_rate_warmup_epochs
        else:
            raise ValueError("Unknown learning rate warmup {}".format(args.l_rate_warmup))
        return LinearLRWarmUp(optimizer=self.optimizer,
                              warmup_epochs=warmup_epochs,
                              last_epoch=self.last_epoch)

    def state_dict(self) -> Dict:
        """
        Added for completeness, since base class _LRScheduler implements this.
        Returns a dictionary with all the values in this objects __dict__.
        It creates the dictionary entry for variables "_scheduler" and "_warmup_scheduler" separately, by calling
        state_dict for these variables.
        The state dict does not include the state of the optimizer.
        """
        state_dict = {key: val for key, val in self.__dict__.items()
                      if key != "_scheduler" and key != "_warmup_scheduler"
                      and key != "optimizer"}
        state_dict['_scheduler'] = self.scheduler.state_dict()
        state_dict['_warmup_scheduler'] = self.warmup_scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Added for completeness, since base class _LRScheduler implements this.
        Initializes the current object with values from state_dict.
        Initializes variables "_scheduler" and "_warmup_scheduler" separately, by calling load_state_dict
        for these variables.
        """
        top_level = {key: val for key, val in state_dict.items()
                     if key != "_scheduler" and key != "_warmup_scheduler"}
        self.__dict__.update(top_level)
        self.scheduler.__dict__.update(state_dict["_scheduler"])
        self.warmup_scheduler.__dict__.update(state_dict["_warmup_scheduler"])

    def get_lr(self) -> List[float]:  # type: ignore
        if self.last_epoch < self.warmup_scheduler.warmup_epochs:
            lr = self.warmup_scheduler.get_lr()
        else:
            lr = self.scheduler.get_lr()
        lrs: List[float] = [lr] if isinstance(lr, float) else lr
        lrs = [max(self.min_l_rate, lr) for lr in lrs]
        self._last_lr = lrs
        return lrs

    def step(self, epoch: int = None) -> None:
        # self.step() is called in the _LRScheduler.__init__, as the very last operation, when self.last_epoch == -1
        # Inside of the default implementation of self.step, it calls
        # self.last_epoch += 1
        # values = self.get_lr()
        # The values are then set in the optimizer, and stored in self._last_lr
        if epoch is not None:
            raise ValueError("Calling scheduler.step with an epoch argument will be deprecated.")
        epoch = self.last_epoch + 1
        if epoch < self.warmup_scheduler.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.scheduler.step()
        self.last_epoch = epoch

    def get_last_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_scheduler.warmup_epochs:
            return self.warmup_scheduler.get_last_lr()
        return self.scheduler.get_last_lr()
