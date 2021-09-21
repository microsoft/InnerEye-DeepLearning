#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule
from yacs.config import CfgNode

from InnerEye.ML.SSL import ssl_augmentation_config
from InnerEye.ML.lightning_container import LightningModuleWithOptimizer


class SSLDataModuleType(Enum):
    ENCODER = 'encoder'
    LINEAR_HEAD = 'linear_head'


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


def create_ssl_encoder(encoder_name: str, use_7x7_first_conv_in_resnet: bool = True) -> torch.nn.Module:
    """
    Creates SSL encoder.
    :param encoder_name: available choices: resnet18, resnet50, resnet101 and densenet121.
    :param use_7x7_first_conv_in_resnet: If True, use a 7x7 kernel (default) in the first layer of resnet. 
    If False, replace first layer by a 3x3 kernel. This is required for small CIFAR 32x32 images to not shrink them.
    """
    from pl_bolts.models.self_supervised.resnets import resnet18, resnet50, resnet101
    from InnerEye.ML.SSL.encoders import DenseNet121Encoder
    if encoder_name == 'resnet18':
        encoder = resnet18(return_all_feature_maps=False, first_conv=use_7x7_first_conv_in_resnet)
    elif encoder_name == 'resnet50':
        encoder = resnet50(return_all_feature_maps=False, first_conv=use_7x7_first_conv_in_resnet)
    elif encoder_name == 'resnet101':
        encoder = resnet101(return_all_feature_maps=False, first_conv=use_7x7_first_conv_in_resnet)
    elif encoder_name == 'densenet121':
        if not use_7x7_first_conv_in_resnet:
            raise ValueError("You set use_7x7_first_conv_in_resnet to False (non-default) but you requested a "
                             "DenseNet121 encoder.")
        encoder = DenseNet121Encoder()
    else:
        raise ValueError("Unknown model type")
    return encoder


def create_ssl_image_classifier(num_classes: int,
                                freeze_encoder: bool,
                                pl_checkpoint_path: str,
                                class_weights: Optional[torch.Tensor] = None) -> LightningModuleWithOptimizer:
    """
    Creates a SSL image classifier from a frozen encoder trained on in an unsupervised manner.
    """

    # Use local imports to avoid circular imports
    from InnerEye.ML.SSL.lightning_modules.byol.byol_module import BYOLInnerEye
    from InnerEye.ML.SSL.lightning_modules.simclr_module import SimCLRInnerEye
    from InnerEye.ML.SSL.lightning_modules.ssl_classifier_module import SSLClassifier

    logging.info(f"Size of ckpt {Path(pl_checkpoint_path).stat().st_size}")
    loaded_params = torch.load(pl_checkpoint_path, map_location=lambda storage, loc: storage)["hyper_parameters"]
    ssl_type = loaded_params["ssl_type"]

    logging.info(f"Creating a {ssl_type} based image classifier")
    logging.info(f"Loading pretrained {ssl_type} weights from:\n {pl_checkpoint_path}")

    if ssl_type == SSLTrainingType.BYOL.value or ssl_type == SSLTrainingType.BYOL:
        # Here we need to indicate how many classes where used for linear evaluator at training time, to load the
        # checkpoint (incl. linear evaluator) with strict = True
        byol_module = SSLModelLoader(BYOLInnerEye, loaded_params["num_classes"]).load_from_checkpoint(
            pl_checkpoint_path)
        encoder = byol_module.target_network.encoder
    elif ssl_type == SSLTrainingType.SimCLR.value or ssl_type == SSLTrainingType.SimCLR:
        simclr_module = SSLModelLoader(SimCLRInnerEye, loaded_params["num_classes"]).load_from_checkpoint(
            pl_checkpoint_path)
        encoder = simclr_module.encoder
    else:
        raise NotImplementedError(f"Unknown unsupervised model: {ssl_type}")

    model = SSLClassifier(num_classes=num_classes,
                          encoder=encoder,
                          freeze_encoder=freeze_encoder,
                          class_weights=class_weights)

    return model


def manual_optimization_step(pl: LightningModule, loss: torch.Tensor, optimizer_idx: int = 0) -> None:
    """
    Execute a manual optimization step in the given PL module, with the provided loss value. This will ONLY update
    the optimizer with the given index.

    :param pl: The module on which the optimization step should be run.
    :param loss: The loss tensor.
    :param optimizer_idx: The index of the optimizer where the optimization step should be taken.
    """

    def get_from_list_or_singleton(items: Any, message: str) -> Any:
        """
        Get an item with index optimizer_idx from the given list. If `items` is not a list, it is possible to retrieve
        that very element with index 0. This is due to PL's handling of optimizers: self.optimizers() is a single
        Optimizer object if only one is used, but a list if multiple are provided.
        """
        if not isinstance(items, list):
            items = [items]
        if optimizer_idx >= len(items):
            raise ValueError(f"Requested to optimize for index {optimizer_idx}, but there are only {len(items)} "
                             f"{message} available.")
        return items[optimizer_idx]

    optimizer = get_from_list_or_singleton(pl.optimizers(), "optimizers")
    optimizer.zero_grad()
    pl.manual_backward(loss)
    optimizer.step()
    if pl.trainer.is_last_batch:
        scheduler = get_from_list_or_singleton(pl.lr_schedulers(), "LR schedulers")
        scheduler.step()


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
            self.non_linear_evaluator = SSLEvaluator(n_input=get_encoder_output_dim(self),
                                                     n_classes=num_classes,
                                                     n_hidden=None)

    return _wrap
