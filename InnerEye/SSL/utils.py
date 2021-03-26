import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import torch

from InnerEye.SSL.configs import ssl_model_config
from InnerEye.SSL.configs.config_node import ConfigNode
from InnerEye.SSL.encoders import DenseNet121Encoder


class SSLModule(Enum):
    ENCODER = 'encoder'
    LINEAR_HEAD = 'linear_head'


def load_ssl_model_config(config_path: Path) -> ConfigNode:
    """
    Loads configs required for self supervised learning. Does not setup cudann as this is being
    taken care of by lightning.
    """
    config = ssl_model_config.get_default_model_config()
    config.merge_from_file(config_path)
    update_model_config(config)

    # Freeze config entries
    config.freeze()

    return config


def update_model_config(config: ConfigNode) -> ConfigNode:
    """
    Adds dataset specific parameters in model config for CIFAR10 and CIFAR100. For other datasets simply return
    the config.
    """
    if config.dataset.name in ['CIFAR10', 'CIFAR100']:
        dataset_dir = f'~/.torch/datasets/{config.dataset.name}'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 32
        config.dataset.n_channels = 3
        config.dataset.n_classes = int(config.dataset.name[5:])

    if not torch.cuda.is_available():
        config.device = 'cpu'

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
        if dataset_name == ["CIFAR10"]:
            logging.info("Updating the initial convolution in order not to shrink CIFAR10 images")
            encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    return encoder


def create_ssl_image_classifier(num_classes: int, pl_checkpoint_path: str) -> torch.nn.Module:
    """
    Creates a SSL image classifier from a frozen encoder trained on in an unsupervised manner.
    """
    from InnerEye.SSL.byol.byol_module import BYOLInnerEye
    from InnerEye.SSL.simclr_module import SimCLRInnerEye
    from InnerEye.SSL.ssl_classifier_module import SSLClassifier, WrapSSL
    ssl_type = torch.load(pl_checkpoint_path, map_location=lambda storage, loc: storage)["hyper_parameters"]["ssl_type"]
    logging.info(f"Creating a {ssl_type} based image classifier")
    logging.info(f"Loading pretrained {ssl_type} weights from:\n {pl_checkpoint_path}")

    if ssl_type == "byol":
        byol_module = WrapSSL(BYOLInnerEye, num_classes).load_from_checkpoint(pl_checkpoint_path)
        model = SSLClassifier(num_classes=num_classes, encoder=byol_module.target_network.encoder,
                              projection=byol_module.target_network.projector_normalised)
    elif ssl_type == "simclr":
        simclr_module = WrapSSL(SimCLRInnerEye, num_classes).load_from_checkpoint(pl_checkpoint_path)
        model = SSLClassifier(num_classes=num_classes, encoder=simclr_module.encoder,
                              projection=simclr_module.projection)
    else:
        raise NotImplementedError(f"Unknown unsupervised model: {ssl_type}")

    return model


