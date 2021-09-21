#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torchvision
from torch import nn

from InnerEyeDataQuality.configs.config_node import ConfigNode


class Network(nn.Module):
    """
    Resnet50 as implement in torchvision
    """

    def __init__(self, config: ConfigNode) -> None:
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=config.train.pretrained, progress=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, config.dataset.n_classes)
        self.projection = self.resnet.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        return x
