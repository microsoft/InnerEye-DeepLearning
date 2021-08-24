#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Callable, Tuple

import PIL
import torch

from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform
from yacs.config import CfgNode

from InnerEye.ML.augmentations.transform_pipeline import create_cxr_transforms_from_config


def get_cxr_ssl_transforms(config: CfgNode,
                           return_two_views_per_sample: bool,
                           use_training_augmentations_for_validation: bool = False,
                           expand_channels: bool = True) -> Tuple[Any, Any]:
    """
    Returns training and validation transforms for CXR.
    Transformations are constructed in the following way:
    1. Construct the pipeline of augmentations in create_chest_xray_transform (e.g. resize, flip, affine) as defined
    by the config.
    2. If we just want to construct the transformation pipeline for a classification model or for the linear evaluator
    of the SSL module, return this pipeline.
    2. If we are constructing transforms for the SSL training, we have to return two versions of each image, hence
    apply DualViewTransformWrapper a wrapper around the obtained transformation pipeline so that we return two augmented
    version of each sample.

    :param config: configuration defining which augmentations to apply as well as their intensities.
    :param return_two_views_per_sample: if True the resulting transforms will return two versions of each sample they
    are called on. If False, simply return one transformed version of the sample.
    :param use_training_augmentations_for_validation: If True, use augmentation at validation time too.
    This is required for SSL validation loss to be meaningful. If False, only apply basic processing step
    (no augmentations)
    :param expand_channels: if True the expand channel transformation from InnerEye.ML.augmentations.image_transforms
    will be added to the transformation passed through the config. This is needed for single channel images as CXR.
    """
    train_transforms = create_cxr_transforms_from_config(config, apply_augmentations=True,
                                                         expand_channels=expand_channels)
    val_transforms = create_cxr_transforms_from_config(config,
                                                       apply_augmentations=use_training_augmentations_for_validation,
                                                       expand_channels=expand_channels)
    if return_two_views_per_sample:
        train_transforms = DualViewTransformWrapper(train_transforms)  # type: ignore
        val_transforms = DualViewTransformWrapper(val_transforms)  # type: ignore
    return train_transforms, val_transforms


class InnerEyeCIFARTrainTransform(SimCLRTrainDataTransform):
    """
    Overload lightning-bolts SimCLRTrainDataTransform, to avoid return unused eval transform. Used for
    training and
    val of SSL models.
    """

    def __call__(self, sample: Any) -> Tuple[Any, Any]:
        transform = self.train_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class InnerEyeCIFARLinearHeadTransform(SimCLRTrainDataTransform):
    """
    Overload lightning-bolts SimCLRTrainDataTransform, to only return linear head eval transform.
    """

    def __call__(self, sample: Any) -> Any:
        return self.online_transform(sample)


class DualViewTransformWrapper:
    """
    Returns two versions of one image, given a random transformation function.
    """

    def __init__(self, transform: Callable):
        self.transform = transform

    def __call__(self, sample: PIL.Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
