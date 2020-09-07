#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Dict, List

from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LambdaLR, MultiStepLR, StepLR, _LRScheduler
from torch.optim.optimizer import Optimizer

from InnerEye.ML.deep_learning_config import DeepLearningConfig, LRSchedulerType, LRWarmUpType


class SchedulerWithWarmUp(_LRScheduler):
    """
    LR Scheduler which runs first a warmup step and then a standard scheduler for LR decay.
    """

    def __init__(self, args: DeepLearningConfig, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.warmup_epochs = args.l_rate_warmup_epochs
        self.warmup_method = args.l_rate_warmup
        self.scheduler = self.get_scheduler(args)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        self.min_l_rate = args.min_l_rate
        super().__init__(optimizer, last_epoch)

    def warmup_function(self, last_epoch: int) -> float:
        if self.warmup_epochs <= 0 or self.warmup_method == LRWarmUpType.NoWarmUp:
            return 1.0
        if self.warmup_method == LRWarmUpType.Linear:
            return (last_epoch + 1) / (self.warmup_epochs + 1)

    def get_scheduler(self, args: DeepLearningConfig) -> _LRScheduler:
        """
        Create the LR scheduler that will be used after warmup, based on the config params.
        """
        scheduler: _LRScheduler
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
            assert args.l_rate_multi_step_milestones is not None  # for mypy, we have done this check elsewhere
            scheduler = MultiStepLR(optimizer=self.optimizer,
                                    milestones=args.l_rate_multi_step_milestones,
                                    gamma=args.l_rate_multi_step_gamma,
                                    last_epoch=self.last_epoch)
        elif args.l_rate_scheduler == LRSchedulerType.Polynomial:
            x = args.min_l_rate / args.l_rate
            polynomial_decay: Any = lambda epoch: (1 - x) * (
                    (1. - float(epoch) / args.num_epochs) ** args.l_rate_polynomial_gamma) + x
            scheduler = LambdaLR(optimizer=self.optimizer,
                                 lr_lambda=polynomial_decay,
                                 last_epoch=self.last_epoch)
        elif args.l_rate_scheduler == LRSchedulerType.Cosine:
            scheduler = CosineAnnealingLR(optimizer=self.optimizer,
                                          T_max=args.num_epochs,
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
                      if key != "_scheduler" and key != "optimizer"}
        state_dict['_scheduler'] = self.scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Added for completeness, since base class _LRScheduler implements this.
        Initializes the current object with values from state_dict.
        Initializes variables "_scheduler" and "_warmup_scheduler" separately, by calling load_state_dict
        for these variables.
        """
        top_level = {key: val for key, val in state_dict.items() if key != "_scheduler"}
        self.__dict__.update(top_level)
        self.scheduler.__dict__.update(state_dict["_scheduler"])

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
            self.scheduler.step()
        self.last_epoch += 1
        warmup_factor = min(self.warmup_function(self.last_epoch), 1)
        lrs = []
        for group in self.optimizer.param_groups:
            lr = max(group['lr'] * warmup_factor, self.min_l_rate)
            group['lr'] = lr
            lrs.append(lr)
        self._last_lr = lrs
