#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from collections import Callable

import torch
from torchvision.models import densenet121


class DenseNet121Encoder(torch.nn.Module):
    """
    This module creates a Densenet121 encoder i.e. Densenet121 model without
    its classification head.
    """

    def __init__(self) -> None:
        super().__init__()
        self.densenet_features = densenet121().features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.densenet_features(x)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)


class Lambda(torch.nn.Module):
    """
    Lambda torch nn module that can be used to modularise nn.functional methods.
    It can be integrated in nn.Sequential constructs to simply chain functional methods and default nn modules
    """

    def __init__(self, fn: Callable) -> None:
        super(Lambda, self).__init__()
        self.lambda_func = fn

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.lambda_func(input)
