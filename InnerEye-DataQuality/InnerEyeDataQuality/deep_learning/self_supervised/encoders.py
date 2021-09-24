#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch
from torchvision.models import densenet121


class DenseNet121Encoder(torch.nn.Module):
    """
    This module creates a Densenet121 encoder i.e. Densenet121 model without
    its classification head.
    """

    def __init__(self) -> None:
        super().__init__()
        self.densenet121 = densenet121()
        self.cnn_model = self.densenet121.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn_model(x)
