#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Optional, Union

import pytorch_lightning as pl
import torch
from torch import Tensor as T, nn
from torchvision.models import densenet121

from InnerEye.ML.SSL.utils import SSLDataModuleType, create_ssl_encoder


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


class SSLEncoder(nn.Module):
    """
    CNN image encoder that generates fixed size BYOL image embeddings.
    Feature responses are pooled to generate a 1-D embedding vector.
    """

    def __init__(self, encoder_name: str, use_7x7_first_conv_in_resnet: bool = True):
        """
        :param encoder_name: Type of the image encoder: {'resnet18', 'resnet50', 'resnet101', 'densenet121'}.
        :param use_7x7_first_conv_in_resnet: If True, use a 7x7 kernel (default) in the first layer of resnet.
            If False, replace first layer by a 3x3 kernel. This is required for small CIFAR 32x32 images to not
            shrink them.
        """

        super().__init__()
        self.cnn_model = create_ssl_encoder(
            encoder_name=encoder_name,
            use_7x7_first_conv_in_resnet=use_7x7_first_conv_in_resnet,
        )

    def forward(self, x: T) -> T:
        x = self.cnn_model(x)
        return x[-1] if isinstance(x, list) else x

    def get_output_feature_dim(self) -> int:
        return get_encoder_output_dim(self)


def get_encoder_output_dim(
    pl_module: Union[pl.LightningModule, torch.nn.Module],
    dm: Optional[pl.LightningDataModule] = None,
) -> int:
    """
    Calculates the output dimension of ssl encoder by making a single forward pass.
    :param pl_module: pl encoder module
    :param dm: pl datamodule
    """
    # Target device
    device = (
        pl_module.device
        if isinstance(pl_module, pl.LightningDataModule)
        else next(pl_module.parameters()).device
    )  # type: ignore
    assert isinstance(device, torch.device)

    # Create a dummy input image
    if dm is not None:
        from InnerEye.ML.SSL.lightning_modules.ssl_online_evaluator import (
            SSLOnlineEvaluatorInnerEye,
        )

        batch = next(iter(dm.train_dataloader()))
        batch = batch[SSLDataModuleType.LINEAR_HEAD] if isinstance(batch, dict) else batch  # type: ignore
        x, _ = SSLOnlineEvaluatorInnerEye.to_device(batch, device)
    else:
        x = torch.rand((1, 3, 256, 256)).to(device)

    # Extract the number of output feature dimensions
    with torch.no_grad():
        representations = pl_module(x)

    return representations.shape[1]
