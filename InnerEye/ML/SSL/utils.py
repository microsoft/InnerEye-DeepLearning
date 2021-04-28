#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import torch

from InnerEye.ML.SSL import ssl_augmentation_config
from InnerEye.ML.SSL.config_node import ConfigNode
from InnerEye.ML.SSL.encoders import DenseNet121Encoder
from InnerEye.ML.lightning_container import LightningModuleWithOptimizer


class SSLModule(Enum):
    ENCODER = 'encoder'
    LINEAR_HEAD = 'linear_head'


class SSLType(Enum):
    SimCLR = "SimCLR"
    BYOL = "BYOL"


def load_ssl_model_config(config_path: Path) -> ConfigNode:
    """
    Loads configs required for self supervised learning. Does not setup cudann as this is being
    taken care of by lightning.
    """
    config = ssl_augmentation_config.get_default_model_config()
    config.merge_from_file(config_path)
    config.freeze()
    return config


def create_ssl_encoder(encoder_name: str, use_7x7_first_conv_in_resnet: bool = True) -> torch.nn.Module:
    """
    Creates SSL encoder.
    :param encoder_name: available choices: resnet18, resnet50, resnet101 and densenet121.
    :param use_7x7_first_conv_in_resnet: If True, use a 7x7 kernel (default) in the first layer of resnet. 
    If False, replace first layer by a 3x3 kernel. This is required for small CIFAR 32x32 images to not shrink them.
    """
    from pl_bolts.models.self_supervised.resnets import resnet18, resnet50, resnet101
    if encoder_name == 'resnet18':
        encoder = resnet18(return_all_feature_maps=False, first_conv=use_7x7_first_conv_in_resnet)
    elif encoder_name == 'resnet50':
        encoder = resnet50(return_all_feature_maps=False, first_conv=use_7x7_first_conv_in_resnet)
    elif encoder_name == 'resnet101':
        encoder = resnet101(return_all_feature_maps=False, first_conv=use_7x7_first_conv_in_resnet)
    elif encoder_name == 'densenet121':
        encoder = DenseNet121Encoder()
    else:
        raise ValueError("Unknown model type")
    return encoder


def create_ssl_image_classifier(num_classes: int, freeze_encoder: bool, pl_checkpoint_path: str,
                                class_weights: Optional[torch.Tensor] = None) -> LightningModuleWithOptimizer:
    """
    Creates a SSL image classifier from a frozen encoder trained on in an unsupervised manner.
    """
    from InnerEye.ML.SSL.byol.byol_module import BYOLInnerEye
    from InnerEye.ML.SSL.simclr_module import SimCLRInnerEye
    from InnerEye.ML.SSL.ssl_online_evaluator import WrapSSL
    from InnerEye.ML.SSL.lightning_containers.ssl_image_classifier import SSLClassifier

    logging.info(f"Size of ckpt {Path(pl_checkpoint_path).stat().st_size}")
    loaded_params = torch.load(pl_checkpoint_path, map_location=lambda storage, loc: storage)["hyper_parameters"]
    ssl_type = loaded_params["ssl_type"]

    logging.info(f"Creating a {ssl_type} based image classifier")
    logging.info(f"Loading pretrained {ssl_type} weights from:\n {pl_checkpoint_path}")

    if ssl_type == SSLType.BYOL.value or ssl_type == SSLType.BYOL:
        # Here we need to indicate how many classes where used for linear evaluator at training time, to load the
        # checkpoint (incl. linear evaluator) with strict = True
        byol_module = WrapSSL(BYOLInnerEye, loaded_params["num_classes"]).load_from_checkpoint(pl_checkpoint_path)
        encoder = byol_module.target_network.encoder
    elif ssl_type == SSLType.SimCLR.value or ssl_type == SSLType.SimCLR:
        simclr_module = WrapSSL(SimCLRInnerEye, loaded_params["num_classes"]).load_from_checkpoint(pl_checkpoint_path)
        encoder = simclr_module.encoder
    else:
        raise NotImplementedError(f"Unknown unsupervised model: {ssl_type}")

    model = SSLClassifier(num_classes=num_classes,
                          encoder=encoder,
                          freeze_encoder=freeze_encoder,
                          class_weights=class_weights)

    return model
