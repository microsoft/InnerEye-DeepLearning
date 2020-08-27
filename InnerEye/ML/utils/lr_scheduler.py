#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, List, Optional

import math
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, StepLR, _LRScheduler
from torch.optim.optimizer import Optimizer

from InnerEye.ML.deep_learning_config import DeepLearningConfig, LRSchedulerType


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
            self._scheduler = ExponentialLR(optimizer, args.l_rate_gamma, last_epoch=last_epoch)
        elif args.l_rate_decay == LRSchedulerType.Step:
            self._scheduler = StepLR(optimizer, args.l_rate_step_size, args.l_rate_gamma, last_epoch=last_epoch)
        elif args.l_rate_decay == LRSchedulerType.Polynomial:
            x = args.min_l_rate / args.l_rate
            polynomial_decay: Any = lambda epoch: (1 - x) * (
                    (1. - float(epoch) / self._max_epochs) ** args.l_rate_gamma) + x
            self._scheduler = LambdaLR(optimizer, polynomial_decay, last_epoch=last_epoch)
        elif args.l_rate_decay == LRSchedulerType.Cosine:
            def cosine_decay(epoch: int) -> float:
                min_lr = args.min_l_rate / args.l_rate
                epoch_ratio = float(epoch) / self._max_epochs
                return (1.0 - min_lr) * 0.5 * (1 + math.cos(epoch_ratio * math.pi)) + min_lr

            # noinspection PyTypeChecker
            self._scheduler = LambdaLR(optimizer, cosine_decay, last_epoch=last_epoch)  # type: ignore
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
