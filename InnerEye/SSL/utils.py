import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import torch

from InnerEye.ML.lightning_container import LightningWithInference
from InnerEye.SSL import ssl_augmentation_config
from InnerEye.SSL.config_node import ConfigNode

from InnerEye.SSL.encoders import DenseNet121Encoder


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


def create_ssl_encoder(encoder_name: str, dataset_name: Optional[str] = None) -> torch.nn.Module:
    """
    Creates SSL encoder.
    :param encoder_name: available choices: resnet18, resnet50, resnet101 and densenet121.
    :param dataset_name: optional, if "CIFAR10" the initial convolution layer with be adapted to not shrink the
    images. Else if None or other the argument is ignored.
    """
    from pl_bolts.models.self_supervised.resnets import resnet18, resnet50, resnet101
    if encoder_name == 'resnet18':
        encoder = resnet18(return_all_feature_maps=False)
    elif encoder_name == 'resnet50':
        encoder = resnet50(return_all_feature_maps=False)
    elif encoder_name == 'resnet101':
        encoder = resnet101(return_all_feature_maps=False)
    elif encoder_name == 'densenet121':
        encoder = DenseNet121Encoder()
    else:
        raise ValueError("Unknown model type")

    if dataset_name is not None:
        if dataset_name.startswith("CIFAR"):
            logging.info("Updating the initial convolution in order not to shrink CIFAR10 images")
            encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    return encoder


def create_ssl_image_classifier(num_classes: int, freeze_encoder: bool, pl_checkpoint_path: Union[str, Path],
                                class_weights: Optional[torch.Tensor]) -> LightningWithInference:
    """
    Creates a SSL image classifier from a frozen encoder trained on in an unsupervised manner.
    """
    from InnerEye.SSL.byol.byol_module import BYOLInnerEye
    from InnerEye.SSL.simclr_module import SimCLRInnerEye
    from InnerEye.SSL.ssl_online_evaluator import WrapSSL
    from InnerEye.SSL.lightning_containers.ssl_image_classifier import SSLClassifier

    ssl_type = torch.load(str(pl_checkpoint_path), map_location=lambda storage, loc: storage)["hyper_parameters"]["ssl_type"]
    logging.info(f"Creating a {ssl_type} based image classifier")
    logging.info(f"Loading pretrained {ssl_type} weights from:\n {pl_checkpoint_path}")

    if ssl_type == SSLType.BYOL:
        byol_module = WrapSSL(BYOLInnerEye, num_classes).load_from_checkpoint(pl_checkpoint_path)
        encoder = byol_module.target_network.encoder
    elif ssl_type == SSLType.SimCLR:
        simclr_module = WrapSSL(SimCLRInnerEye, num_classes).load_from_checkpoint(pl_checkpoint_path)
        encoder = simclr_module.encoder
    else:
        raise NotImplementedError(f"Unknown unsupervised model: {ssl_type}")

    model = SSLClassifier(num_classes=num_classes, encoder=encoder,
                          freeze_encoder=freeze_encoder, class_weights=class_weights)

    return model
