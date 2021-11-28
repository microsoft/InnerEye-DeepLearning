"""BaseMIL is an abstract container defining basic functionality for running MIL experiments.
It is responsible for instantiating the encoder and full DeepMIL model. Subclasses should define
their datamodules and configure experiment-specific parameters.
"""
import os
from pathlib import Path
from typing import Type
from health_azure.utils import get_workspace

import param
from torch import nn
from torchvision.models.resnet import resnet18

from health_ml.data.histopathology.datamodules.base_module import CacheMode, TilesDataModule
from health_azure.utils import CheckpointDownloader

from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.Common import fixed_paths

from InnerEyePrivate.Histopathology.models.attention_layers import AttentionLayer, GatedAttentionLayer
from InnerEyePrivate.Histopathology.models.deepmil import DeepMILModule
from InnerEyePrivate.Histopathology.models.encoders import (HistoSSLEncoder, IdentityEncoder,
                                                            ImageNetEncoder, ImageNetSimCLREncoder,
                                                            InnerEyeSSLEncoder, TileEncoder)


class BaseMIL(LightningContainer):
    # Model parameters:
    pooling_type: str = param.String(doc="Name of the pooling layer class to use.")
    # l_rate, weight_decay, adam_betas are already declared in OptimizerParams superclass

    # Encoder parameters:
    encoder_type: str = param.String(doc="Name of the encoder class to use.")
    tile_size: int = param.Integer(224, bounds=(1, None), doc="Tile width/height, in pixels.")
    n_channels: int = param.Integer(3, bounds=(1, None), doc="Number of channels in the tile.")

    # Data module parameters:
    batch_size: int = param.Integer(16, bounds=(1, None), doc="Number of slides to load per batch.")
    max_bag_size: int = param.Integer(1000, bounds=(0, None),
                                      doc="Upper bound on number of tiles in each loaded bag. "
                                          "If 0 (default), will return all samples in each bag. "
                                          "If > 0, bags larger than `max_bag_size` will yield "
                                          "random subsets of instances.")
    cache_mode: CacheMode = param.ClassSelector(default=CacheMode.MEMORY, class_=CacheMode,
                                                doc="The type of caching to perform: "
                                                    "'memory' (default), 'disk', or 'none'.")
    save_precache: bool = param.Boolean(True, doc="Whether to pre-cache the entire transformed "
                                                  "dataset upfront and save it to disk.")
    # local_dataset (used as data module root_path) is declared in DatasetParams superclass

    @property
    def cache_dir(self) -> Path:
        raise NotImplementedError

    def setup(self) -> None:
        if self.encoder_type == InnerEyeSSLEncoder.__name__:
            self.downloader = CheckpointDownloader(
                aml_workspace=get_workspace(),
                run_recovery_id="vsalva_ssl_crck:vsalva_ssl_crck_1630691119_af10db8a",
                checkpoint_filename="best_checkpoint.ckpt",
                download_dir='outputs/'
            )
            os.chdir(fixed_paths.repository_root_directory())
            self.downloader.download_checkpoint_if_necessary()

        self.encoder = self.get_encoder()
        self.encoder.cuda()
        self.encoder.eval()

    def get_encoder(self) -> TileEncoder:
        if self.encoder_type == ImageNetEncoder.__name__:
            return ImageNetEncoder(feature_extraction_model=resnet18,
                                   tile_size=self.tile_size, n_channels=self.n_channels)

        elif self.encoder_type == ImageNetSimCLREncoder.__name__:
            return ImageNetSimCLREncoder(tile_size=self.tile_size, n_channels=self.n_channels)

        elif self.encoder_type == HistoSSLEncoder.__name__:
            return HistoSSLEncoder(tile_size=self.tile_size, n_channels=self.n_channels)

        elif self.encoder_type == InnerEyeSSLEncoder.__name__:
            return InnerEyeSSLEncoder(pl_checkpoint_path=self.downloader.local_checkpoint_path,
                                      tile_size=self.tile_size, n_channels=self.n_channels)

        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def get_pooling_layer(self) -> Type[nn.Module]:
        if self.pooling_type == AttentionLayer.__name__:
            return AttentionLayer
        elif self.pooling_type == GatedAttentionLayer.__name__:
            return GatedAttentionLayer
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

    def create_model(self) -> DeepMILModule:
        self.data_module = self.get_data_module()
        # Encoding is done in the datamodule, so here we provide instead a dummy
        # no-op IdentityEncoder to be used inside the model
        return DeepMILModule(encoder=IdentityEncoder(input_dim=(self.encoder.num_encoding,)),
                             label_column=self.data_module.train_dataset.LABEL_COLUMN,
                             n_classes=self.data_module.train_dataset.N_CLASSES,
                             pooling_layer=self.get_pooling_layer(),
                             class_weights=self.data_module.class_weights,
                             l_rate=self.l_rate,
                             weight_decay=self.weight_decay,
                             adam_betas=self.adam_betas)

    def get_data_module(self) -> TilesDataModule:
        raise NotImplementedError
