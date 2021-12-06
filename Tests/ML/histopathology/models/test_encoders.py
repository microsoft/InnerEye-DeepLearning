from typing import Callable

import pytest
from torch import Tensor, float32, nn, rand
from torchvision.models import resnet18

from InnerEye.ML.Histopathology.models.encoders import (TileEncoder, HistoSSLEncoder, ImageNetEncoder,
                                                           ImageNetSimCLREncoder)


def get_supervised_imagenet_encoder() -> TileEncoder:
    return ImageNetEncoder(feature_extraction_model=resnet18, tile_size=224)


def get_simclr_imagenet_encoder() -> TileEncoder:
    return ImageNetSimCLREncoder(tile_size=224)


def get_histo_ssl_encoder() -> TileEncoder:
    return HistoSSLEncoder(tile_size=224)


@pytest.mark.parametrize("create_encoder_fn", [get_supervised_imagenet_encoder,
                                               get_simclr_imagenet_encoder,
                                               get_histo_ssl_encoder])
def test_encoder(create_encoder_fn: Callable[[], TileEncoder]) -> None:
    batch_size = 10

    encoder = create_encoder_fn()

    if isinstance(encoder, nn.Module):
        for param_name, param in encoder.named_parameters():
            assert not param.requires_grad, \
                f"Feature extractor has unfrozen parameters: {param_name}"

    images = rand(batch_size, *encoder.input_dim, dtype=float32)

    features = encoder(images)
    assert isinstance(features, Tensor)
    assert features.shape == (batch_size, encoder.num_encoding)
