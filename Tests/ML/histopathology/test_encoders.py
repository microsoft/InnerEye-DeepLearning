import os
from pathlib import Path
from typing import Callable

import pytest
from torch import Tensor, float32, nn, rand
from torchvision.models import resnet18

from health_azure.utils import CheckpointDownloader, get_workspace
from InnerEye.Common import fixed_paths
from InnerEye.ML.Histopathology.models.encoders import (TileEncoder, HistoSSLEncoder, ImageNetEncoder,
                                                           ImageNetSimCLREncoder, InnerEyeSSLEncoder)


def get_supervised_imagenet_encoder() -> TileEncoder:
    return ImageNetEncoder(feature_extraction_model=resnet18, tile_size=224)


def get_simclr_imagenet_encoder() -> TileEncoder:
    return ImageNetSimCLREncoder(tile_size=224)


def get_simclr_crck_encoder(tmp_path: Path) -> TileEncoder:
    aml_workspace = get_workspace()
    download_dir = tmp_path / "downlads"
    download_dir.mkdir(exist_ok=True)
    downloader = CheckpointDownloader(aml_workspace=aml_workspace,
                                      run_id="vsalva_ssl_crck:vsalva_ssl_crck_1630691119_af10db8a",
                                      checkpoint_filename="best_checkpoint.ckpt",
                                      download_dir=download_dir)
    os.chdir(fixed_paths.repository_root_directory())
    _ = downloader.download_checkpoint_if_necessary()

    return InnerEyeSSLEncoder(pl_checkpoint_path=downloader.local_checkpoint_path,
                          tile_size=224)


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

@pytest.mark.skip(reason="This checkpoint has extra keys with respect to the latest SimCLRInnerEye class, cannot be loaded with strict=True")
def test_simclr_crck_encoder() -> None:
    batch_size = 10

    encoder = get_simclr_crck_encoder()

    if isinstance(encoder, nn.Module):
        for param_name, param in encoder.named_parameters():
            assert not param.requires_grad, \
                f"Feature extractor has unfrozen parameters: {param_name}"

    images = rand(batch_size, *encoder.input_dim, dtype=float32)

    features = encoder(images)
    assert isinstance(features, Tensor)
    assert features.shape == (batch_size, encoder.num_encoding)
