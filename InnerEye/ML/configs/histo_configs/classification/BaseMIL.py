#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""BaseMIL is an abstract container defining basic functionality for running MIL experiments.
It is responsible for instantiating the encoder and full DeepMIL model. Subclasses should define
their datamodules and configure experiment-specific parameters.
"""
from pathlib import Path
from typing import Optional, Type  # noqa

import param
from torch import nn
from torchvision.models.resnet import resnet18

from health_ml.networks.layers.attention_layers import AttentionLayer, GatedAttentionLayer, MeanPoolingLayer, TransformerPooling
from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.ML.Histopathology.datasets.base_dataset import SlidesDataset
from InnerEye.ML.Histopathology.datamodules.base_module import CacheMode, CacheLocation, TilesDataModule
from InnerEye.ML.Histopathology.models.deepmil import DeepMILModule
from InnerEye.ML.Histopathology.models.encoders import (HistoSSLEncoder, IdentityEncoder,
                                                        ImageNetEncoder, ImageNetSimCLREncoder,
                                                        InnerEyeSSLEncoder, TileEncoder)


class BaseMIL(LightningContainer):
    # Model parameters:
    pool_type: str = param.String(doc="Name of the pooling layer class to use.")
    pool_hidden_dim: str = param.Integer(128, doc="If pooling has a learnable part, this defines the number of the hidden dimensions.")
    pool_out_dim: str = param.Integer(1, doc="Dimension of the pooled representation.")
    num_transformer_pool_layers: int = param.Integer(4, doc="If transformer pooling is chosen, this defines the number of encoding layers.")
    num_transformer_pool_heads: int = param.Integer(4, doc="If transformer pooling is chosen, this defines the number of attention heads.")

    is_finetune: bool = param.Boolean(doc="Whether to fine-tune the encoder. Options:"
                                      "`False` (default), or `True`.")
    dropout_rate: Optional[float] = param.Number(None, bounds=(0, 1), doc="Pre-classifier dropout rate.")
    # l_rate, weight_decay, adam_betas are already declared in OptimizerParams superclass

    # Encoder parameters:
    encoder_type: str = param.String(doc="Name of the encoder class to use.")
    tile_size: int = param.Integer(224, bounds=(1, None), doc="Tile width/height, in pixels.")
    n_channels: int = param.Integer(3, bounds=(1, None), doc="Number of channels in the tile.")

    # Data module parameters:
    batch_size: int = param.Integer(16, bounds=(1, None), doc="Number of slides to load per batch.")
    max_bag_size: int = param.Integer(2, bounds=(0, None),
                                      doc="Upper bound on number of tiles in each loaded bag. "
                                          "If 0 (default), will return all samples in each bag. "
                                          "If > 0, bags larger than `max_bag_size` will yield "
                                          "random subsets of instances.")
    cache_mode: CacheMode = param.ClassSelector(default=CacheMode.MEMORY, class_=CacheMode,
                                                doc="The type of caching to perform: "
                                                    "'memory' (default), 'disk', or 'none'.")
    precache_location: str = param.ClassSelector(default=CacheLocation.NONE, class_=CacheLocation,
                                                 doc="Whether to pre-cache the entire transformed dataset upfront "
                                                 "and save it to disk and if re-load in cpu or gpu. Options:"
                                                 "`none` (default),`cpu`, `gpu`")
    encoding_chunk_size: int = param.Integer(0, doc="If > 0 performs encoding in chunks, by loading"
                                                     "enconding_chunk_size tiles per chunk")
    # local_dataset (used as data module root_path) is declared in DatasetParams superclass

    @property
    def cache_dir(self) -> Path:
        raise NotImplementedError

    def setup(self) -> None:
        if self.encoder_type == InnerEyeSSLEncoder.__name__:
            raise NotImplementedError("InnerEyeSSLEncoder requires a pre-trained checkpoint.")

        self.encoder = self.get_encoder()
        if not self.is_finetune:
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
        num_encoding = self.encoder.num_encoding

        if self.pool_type == AttentionLayer.__name__:
            pooling_layer = AttentionLayer(num_encoding,
                                  self.pool_hidden_dim,
                                  self.pool_out_dim)
        elif self.pool_type == GatedAttentionLayer.__name__:
            pooling_layer = GatedAttentionLayer(num_encoding,
                                  self.pool_hidden_dim,
                                  self.pool_out_dim)
        elif self.pool_type == MeanPoolingLayer.__name__:
            pooling_layer = MeanPoolingLayer(num_encoding,
                                  self.pool_hidden_dim,
                                  self.pool_out_dim)
        elif self.pool_type == TransformerPooling.__name__:
            pooling_layer = TransformerPooling(self.num_transformer_pool_layers,
                                               self.num_transformer_pool_heads,
                                               num_encoding)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        num_features = num_encoding*self.pool_out_dim
        return pooling_layer, num_features

    def create_model(self) -> DeepMILModule:
        self.data_module = self.get_data_module()
        # Encoding is done in the datamodule, so here we provide instead a dummy
        # no-op IdentityEncoder to be used inside the model
        if self.is_finetune:
            self.model_encoder = self.encoder
            for params in self.model_encoder.parameters():
                params.requires_grad = True
        else:
            self.model_encoder = IdentityEncoder(input_dim=(self.encoder.num_encoding,))

        # Construct pooling layer
        pooling_layer, num_features = self.get_pooling_layer()

        return DeepMILModule(encoder=self.model_encoder,
                            label_column=self.data_module.train_dataset.LABEL_COLUMN,
                            n_classes=self.data_module.train_dataset.N_CLASSES,
                            pooling_layer=pooling_layer,
                            num_features=num_features,
                            dropout_rate=self.dropout_rate,
                            class_weights=self.data_module.class_weights,
                            l_rate=self.l_rate,
                            weight_decay=self.weight_decay,
                            adam_betas=self.adam_betas)

    def get_data_module(self) -> TilesDataModule:
        raise NotImplementedError

    def get_slide_dataset(self) -> SlidesDataset:
        raise NotImplementedError
