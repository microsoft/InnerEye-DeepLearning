#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn


class Identity(nn.Module):
    """
    Implements an identity torch module where input is passed as it is to output.
    There are no parameters in the module.
    """

    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return input
