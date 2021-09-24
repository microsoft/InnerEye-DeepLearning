#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Callable

import torch
import torch.nn as nn


class Lambda(nn.Module):
    """
    Lambda torch nn module that can be used to modularise nn.functional methods.
    """

    def __init__(self, fn: Callable) -> None:
        super(Lambda, self).__init__()
        self.lambda_func = fn

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.lambda_func(input)
