#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import torch
from yacs.config import CfgNode

from InnerEye.ML.SSL import ssl_augmentation_config
from InnerEye.ML.lightning_container import LightningModuleWithOptimizer


class SSLDataModuleType(Enum):
    ENCODER = "encoder"
    LINEAR_HEAD = "linear_head"


class SSLTrainingType(Enum):
    SimCLR = "SimCLR"
    BYOL = "BYOL"


def load_yaml_augmentation_config(config_path: Path) -> CfgNode:
    """
    Loads augmentations configs defined as yaml files.
    """
    config = ssl_augmentation_config.get_default_model_config()
    config.merge_from_file(config_path)
    config.freeze()
    return config


def create_ssl_encoder(
    encoder_name: str, use_7x7_first_conv_in_resnet: bool = True
) -> torch.nn.Module:
    """
    Creates SSL encoder.
    :param encoder_name: available choices: resnet18, resnet50, resnet101 and densenet121.
    :param use_7x7_first_conv_in_resnet: If True, use a 7x7 kernel (default) in the first layer of resnet.
    If False, replace first layer by a 3x3 kernel. This is required for small CIFAR 32x32 images to not shrink them.
    """
    from pl_bolts.models.self_supervised.resnets import resnet18, resnet50, resnet101
    from InnerEye.ML.SSL.encoders import DenseNet121Encoder

    if encoder_name == "resnet18":
        encoder = resnet18(
            return_all_feature_maps=False, first_conv=use_7x7_first_conv_in_resnet
        )
    elif encoder_name == "resnet50":
        encoder = resnet50(
            return_all_feature_maps=False, first_conv=use_7x7_first_conv_in_resnet
        )
    elif encoder_name == "resnet101":
        encoder = resnet101(
            return_all_feature_maps=False, first_conv=use_7x7_first_conv_in_resnet
        )
    elif encoder_name == "densenet121":
        if not use_7x7_first_conv_in_resnet:
            raise ValueError(
                "You set use_7x7_first_conv_in_resnet to False (non-default) but you requested a "
                "DenseNet121 encoder."
            )
        encoder = DenseNet121Encoder()
    else:
        raise ValueError("Unknown model type")
    return encoder


def create_ssl_image_classifier(
    num_classes: int,
    freeze_encoder: bool,
    pl_checkpoint_path: str,
    class_weights: Optional[torch.Tensor] = None,
) -> LightningModuleWithOptimizer:
    """
    Creates a SSL image classifier from a frozen encoder trained on in an unsupervised manner.
    """

    # Use local imports to avoid circular imports
    from InnerEye.ML.SSL.lightning_modules.byol.byol_module import BYOLInnerEye
    from InnerEye.ML.SSL.lightning_modules.simclr_module import SimCLRInnerEye
    from InnerEye.ML.SSL.lightning_modules.ssl_classifier_module import SSLClassifier

    logging.info(f"Size of ckpt {Path(pl_checkpoint_path).stat().st_size}")
    loaded_params = torch.load(
        pl_checkpoint_path, map_location=lambda storage, loc: storage
    )["hyper_parameters"]
    ssl_type = loaded_params["ssl_type"]

    logging.info(f"Creating a {ssl_type} based image classifier")
    logging.info(f"Loading pretrained {ssl_type} weights from:\n {pl_checkpoint_path}")

    if ssl_type == SSLTrainingType.BYOL.value or ssl_type == SSLTrainingType.BYOL:
        byol_module = BYOLInnerEye.load_from_checkpoint(pl_checkpoint_path)
        encoder = byol_module.target_network.encoder
    elif ssl_type == SSLTrainingType.SimCLR.value or ssl_type == SSLTrainingType.SimCLR:
        simclr_module = SimCLRInnerEye.load_from_checkpoint(
            pl_checkpoint_path, strict=False
        )
        encoder = simclr_module.encoder
    else:
        raise NotImplementedError(f"Unknown unsupervised model: {ssl_type}")

    model = SSLClassifier(
        num_classes=num_classes,
        encoder=encoder,
        freeze_encoder=freeze_encoder,
        class_weights=class_weights,
    )

    return model


def SSLModelLoader(ssl_class: Any, num_classes: int) -> Any:
    """
    This class is a helper class for SSL model loading from checkpoints with strict=True.
    We cannot simply load the class directly via  do BYOLInnerEye().load_from_checkpoint("ckpt") with strict loading
    because the checkpoint will contain the weights of the linear evaluator, but this one is defined outside of the
    BYOLInnerEye class (as it is defined as a callback), hence we can only load the checkpoint if we manually re-add
    the linear evaluator prior to loading.

    :param ssl_class:   SSL object either BYOL or SimCLR.
    :param num_classes: Number of target classes for the linear head.
    """
    from pl_bolts.models.self_supervised import SSLEvaluator
    from InnerEye.ML.SSL.encoders import get_encoder_output_dim

    class _wrap(ssl_class):  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.non_linear_evaluator = SSLEvaluator(
                n_input=get_encoder_output_dim(self),
                n_classes=num_classes,
                n_hidden=None,
            )

    return _wrap
