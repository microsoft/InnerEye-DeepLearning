#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pl_bolts.models.self_supervised import SimCLR
from torch import Tensor as T, nn
from torchvision.models import densenet121, resnet18
from torchvision.transforms import Compose

from health_ml.utils.histopathology.layer_utils import (get_imagenet_preprocessing,
                                                        load_weights_to_model,
                                                        setup_feature_extractor)
from InnerEye.ML.SSL.utils import create_ssl_encoder, create_ssl_image_classifier, SSLDataModuleType



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


class TileEncoder(nn.Module):
    """Base tile encoder class for use in dataset transforms or as part of a bigger model"""

    def __init__(self, tile_size: int = 0, n_channels: int = 3,
                 input_dim: Optional[Sequence[int]] = None) -> None:
        """The `TileEncoder` constructor should be called after setting any attributes needed in
        `_get_preprocessing()` or `_get_encoder()`.

        :param tile_size: Tile width/height, in pixels.
        :param n_channels: Number of channels in the tile (default=3).
        :param input_dim: Input shape, to override default of `(n_channels, tile_size, tile_size)`.
        """
        super().__init__()
        if input_dim is None:
            if tile_size == 0:
                raise ValueError("Either input_dim or tile_size must be specified")
            input_dim = (n_channels, tile_size, tile_size)
        self.input_dim = tuple(input_dim)

        self.preprocessing_fn = self._get_preprocessing()
        self.feature_extractor_fn, self.num_encoding = self._get_encoder()

    def _get_preprocessing(self) -> Callable:
        return Compose([])

    def _get_encoder(self) -> Tuple[Callable, int]:
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        prep_images = self.preprocessing_fn(images)
        return self.feature_extractor_fn(prep_images)


class IdentityEncoder(TileEncoder):
    """Dummy encoder that just flattens the input"""

    def _get_encoder(self) -> Tuple[Callable, int]:
        return nn.Flatten(), np.prod(self.input_dim)


class ImageNetEncoder(TileEncoder):
    """Feature extractor pretrained for classification on ImageNet"""

    def __init__(self, feature_extraction_model: Callable[..., nn.Module],
                 tile_size: int, n_channels: int = 3) -> None:
        """
        :param feature_extraction_model: A function accepting a `pretrained` keyword argument that
        returns a classifier pretrained on ImageNet, such as the ones from `torchvision.models.*`.
        :param tile_size: Tile width/height, in pixels.
        :param n_channels: Number of channels in the tile (default=3).
        """
        self.create_feature_extractor_fn = feature_extraction_model
        super().__init__(tile_size=tile_size, n_channels=n_channels)

    def _get_preprocessing(self) -> Callable:
        return get_imagenet_preprocessing()

    def _get_encoder(self) -> Tuple[Callable, int]:
        pretrained_model = self.create_feature_extractor_fn(pretrained=True)
        return setup_feature_extractor(pretrained_model, self.input_dim)  # type: ignore


class ImageNetSimCLREncoder(TileEncoder):
    """SimCLR encoder pretrained on ImageNet"""

    WEIGHTS_URL = ("https://pl-bolts-weights.s3.us-east-2.amazonaws.com/"
                   "simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt")
    EMBEDDING_DIM = 2048

    def _get_preprocessing(self) -> Callable:
        return get_imagenet_preprocessing()

    def _get_encoder(self) -> Tuple[SimCLR, int]:
        simclr = SimCLR.load_from_checkpoint(self.WEIGHTS_URL, strict=False)
        simclr.freeze()
        return simclr, self.EMBEDDING_DIM


class InnerEyeSSLEncoder(TileEncoder):
    """SSL encoder trained on Azure ML using InnerEye"""

    def __init__(self, pl_checkpoint_path: Path, tile_size: int, n_channels: int = 3) -> None:
        """
        :param pl_checkpoint_path: The path of the downloaded checkpoint file.
        :param tile_size: Tile width/height, in pixels.
        :param n_channels: Number of channels in the tile (default=3).
        """
        self.pl_checkpoint_path = pl_checkpoint_path
        super().__init__(tile_size=tile_size, n_channels=n_channels)

    def _get_encoder(self) -> Tuple[SSLEncoder, int]:
        model: Any = create_ssl_image_classifier(num_classes=1,  # dummy value
                                                           freeze_encoder=True,
                                                           pl_checkpoint_path=self.pl_checkpoint_path)
        encoder = model.encoder
        for param in encoder.parameters():
            param.requires_grad = False  # freeze_encoder does not disable gradients

        embedding_dim = model.classifier_head.n_input

        return encoder, embedding_dim


class HistoSSLEncoder(TileEncoder):
    """HistoSSL encoder pretrained on multiple histological datasets

    Reference:
    - Ciga, Xu, Martel (2021). Self supervised contrastive learning for digital histopathology.
    arXiv:2011.13971
    """

    WEIGHTS_URL = ("https://github.com/ozanciga/self-supervised-histopathology/releases/"
                   "download/tenpercent/tenpercent_resnet18.ckpt")

    def _get_encoder(self) -> Tuple[Callable, int]:
        resnet18_model = resnet18(pretrained=False)
        histossl_encoder = load_weights_to_model(self.WEIGHTS_URL, resnet18_model)
        return setup_feature_extractor(histossl_encoder, self.input_dim)  # type: ignore


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

        dataloader = dm.train_dataloader()
        dataloader = dataloader[SSLDataModuleType.LINEAR_HEAD] if isinstance(dataloader, dict) else dataloader  # type: ignore
        batch = iter(dataloader).next()  # type: ignore
        x, _ = SSLOnlineEvaluatorInnerEye.to_device(batch, device)
    else:
        x = torch.rand((1, 3, 256, 256)).to(device)

    # Extract the number of output feature dimensions
    with torch.no_grad():
        representations = pl_module(x)

    return representations.shape[1]
