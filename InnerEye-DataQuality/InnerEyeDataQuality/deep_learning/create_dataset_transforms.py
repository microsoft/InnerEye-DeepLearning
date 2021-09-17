#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, Callable, List, Tuple

import numpy as np
import torchvision
from torchvision.transforms import ToTensor

from InnerEyeDataQuality.configs.config_node import ConfigNode
from InnerEyeDataQuality.deep_learning.transforms import AddGaussianNoise, CenterCrop, ElasticTransform, \
    ExpandChannels, RandomAffine, RandomColorJitter, RandomErasing, RandomGamma, \
    RandomHorizontalFlip, RandomResizeCrop, Resize


def _get_dataset_stats(
        config: ConfigNode) -> Tuple[np.ndarray, np.ndarray]:
    name = config.dataset.name
    if name == 'CIFAR10':
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
    else:
        raise ValueError()
    return mean, std


def create_transform(config: ConfigNode, is_train: bool) -> Callable:
    if config.dataset.name in ["NoisyChestXray", "Kaggle"]:
        return create_chest_xray_transform(config, is_train)
    elif config.dataset.name in ["CIFAR10", "CIFAR10H", "CIFAR10IDN", "CIFAR10H_TRAIN_VAL", "CIFAR10SYM"]:
        return create_cifar_transform(config, is_train)
    else:
        raise ValueError


def create_cifar_transform(config: ConfigNode,
                           is_train: bool) -> Callable:
    transforms: List[Any] = list()
    if is_train:
        if config.augmentation.use_random_affine:
            transforms.append(RandomAffine(config))
        if config.augmentation.use_random_crop:
            transforms.append(RandomResizeCrop(config))
        if config.augmentation.use_random_horizontal_flip:
            transforms.append(RandomHorizontalFlip(config))
        if config.augmentation.use_random_color:
            transforms.append(RandomColorJitter(config))
    transforms += [ToTensor()]
    return torchvision.transforms.Compose(transforms)


def create_chest_xray_transform(config: ConfigNode,
                                is_train: bool) -> Callable:
    """
    Defines the image transformations pipeline for Chest-Xray datasets.
    """
    transforms: List[Any] = []
    if is_train:
        if config.augmentation.use_random_affine:
            transforms.append(RandomAffine(config))
        if config.augmentation.use_random_crop:
            transforms.append(RandomResizeCrop(config))
        else:
            transforms.append(Resize(config))
        if config.augmentation.use_random_horizontal_flip:
            transforms.append(RandomHorizontalFlip(config))
        if config.augmentation.use_gamma_transform:
            transforms.append(RandomGamma(config))
        if config.augmentation.use_random_color:
            transforms.append(RandomColorJitter(config))
        if config.augmentation.use_elastic_transform:
            transforms.append(ElasticTransform(config))
        transforms += [CenterCrop(config), ToTensor()]
        if config.augmentation.use_random_erasing:
            transforms.append(RandomErasing(config))
        if config.augmentation.add_gaussian_noise:
            transforms.append(AddGaussianNoise(config))
    else:
        transforms += [Resize(config), CenterCrop(config), ToTensor()]
    transforms.append(ExpandChannels())
    return torchvision.transforms.Compose(transforms)
