#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Optional
import logging

import torch
from InnerEyeDataQuality.deep_learning.self_supervised.ssl_classifier_module import PretrainedClassifier, SSLClassifier, \
    WrapSSL
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50_bn, resnet101


def create_ssl_encoder(encoder_name: str, dataset_name: Optional[str] = None) -> torch.nn.Module:
    """
    """
    if encoder_name == 'resnet18':
        encoder = resnet18(return_all_feature_maps=False)
    elif encoder_name == 'resnet50':
        encoder = resnet50_bn(return_all_feature_maps=False)
    elif encoder_name == 'resnet101':
        encoder = resnet101(return_all_feature_maps=False)
    else:
        raise ValueError("Unknown model type")

    if dataset_name is not None:
        if dataset_name in ["CIFAR10", "CIFAR10H"]:
            logging.info("Updating the initial convolution in order not to shrink CIFAR10 images")
            encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    return encoder


def create_ssl_image_classifier(num_classes: int, pl_checkpoint_path: str, freeze_encoder: bool = True) -> torch.nn.Module:
    from InnerEyeDataQuality.deep_learning.self_supervised.byol.byol_module import BYOLInnerEye
    from InnerEyeDataQuality.deep_learning.self_supervised.simclr_module import SimCLRInnerEye
    """
    """
    ssl_type = torch.load(pl_checkpoint_path, map_location=lambda storage, loc: storage)["hyper_parameters"]["ssl_type"]
    logging.info(f"Creating a {ssl_type} based image classifier")
    logging.info(f"Loading pretrained {ssl_type} weights from:\n {pl_checkpoint_path}")

    if ssl_type == "byol":
        byol_module = WrapSSL(BYOLInnerEye, num_classes).load_from_checkpoint(pl_checkpoint_path, strict=False)
        if freeze_encoder:
            model = SSLClassifier(num_classes=num_classes, encoder=byol_module.target_network.encoder,
                                  projection=byol_module.target_network.projector_normalised)
        else:
            model = PretrainedClassifier(num_classes=num_classes,  # type: ignore
                                         encoder=byol_module.target_network.encoder)
    elif ssl_type == "simclr":
        simclr_module = WrapSSL(SimCLRInnerEye, num_classes).load_from_checkpoint(pl_checkpoint_path, strict=False)
        if freeze_encoder:
            model = SSLClassifier(num_classes=num_classes, encoder=simclr_module.encoder,
                                  projection=simclr_module.projection)
        else:
            model = PretrainedClassifier(num_classes=num_classes,  # type: ignore
                                         encoder=simclr_module.encoder)
    else:
        raise NotImplementedError(f"Unknown unsupervised model: {ssl_type}")

    return model
