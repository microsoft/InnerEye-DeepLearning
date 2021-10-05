#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from InnerEyeDataQuality.deep_learning.self_supervised.byol.byol_models import SSLEncoder
from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR


class _Projection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLRInnerEye(SimCLR):
    def __init__(self, encoder_name: str, dataset_name: str, **kwargs: Any) -> None:
        """
        Args:
            encoder_name [str]: Image encoder name (predefined models)
            dataset_name [str]: Image dataset name (e.g. cifar10, kaggle, etc.)
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.encoder = SSLEncoder(encoder_name, dataset_name, use_output_pooling=True)
        self.projection = _Projection(input_dim=self.encoder.get_output_feature_dim(), hidden_dim=2048, output_dim=128)
