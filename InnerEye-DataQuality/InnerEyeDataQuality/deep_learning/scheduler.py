#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging

import numpy as np

class ForgetRateScheduler(object):
    """
    Forget rate scheduler for the co-teaching model
    """

    def __init__(self,
                 num_epochs: int,
                 forget_rate: float = 0.0,
                 num_gradual: int = 10,
                 num_warmup_epochs: int = 25,
                 start_epoch: int = 0):
        """

        :param num_epochs: The total number of training epochs
        :param forget_rate: The base forget rate
        :param num_gradual: The number of epochs to gradual increase the forget_rate to its base value
        :param start_epoch: Allows manually set the start epoch if training is resumed.
        """
        logging.info(f"No samples will be excluded in co-teaching for the first {num_warmup_epochs} epochs.")
        if num_gradual <= num_warmup_epochs:
            logging.warning(f"Num gradual {num_gradual} <= num warm up epochs. This argument will be ignored.")
        assert 0 <= forget_rate < 1.
        self.forget_rate_schedule = np.ones(num_epochs) * forget_rate
        self.forget_rate_schedule[:num_gradual] = np.linspace(0, forget_rate, num_gradual)
        self.forget_rate_schedule[:num_warmup_epochs] = np.zeros(num_warmup_epochs)
        self.current_epoch = start_epoch

    def step(self) -> None:
        """
        Step the current epoch by one
        :return:
        """
        self.current_epoch += 1

    @property
    def get_forget_rate(self) -> float:
        """

        :return: The current forget rate
        """
        current_epoch = min(self.current_epoch, len(self.forget_rate_schedule) - 1)
        return float(self.forget_rate_schedule[current_epoch])
