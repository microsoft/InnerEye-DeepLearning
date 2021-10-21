#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from typing import Any, Tuple, Type

import PIL.Image
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.datasets.vision import VisionDataset

from default_paths import CIFAR10_ROOT_DIR
from InnerEyeDataQuality.configs.config_node import ConfigNode
from InnerEyeDataQuality.datasets.cifar10_idn import CIFAR10IDN
from InnerEyeDataQuality.datasets.cifar10_asym_noise import CIFAR10AsymNoise
from InnerEyeDataQuality.datasets.cifar10h import CIFAR10H
from InnerEyeDataQuality.datasets.kaggle_cxr import KaggleCXR
from InnerEyeDataQuality.datasets.noisy_kaggle_cxr import NoisyKaggleSubsetCXR
from InnerEyeDataQuality.deep_learning.create_dataset_transforms import create_transform
from InnerEyeDataQuality.deep_learning.utils import load_selector_config
from InnerEyeDataQuality.utils.generic import convert_labels_to_one_hot


def dataset_with_indices(cls: Type[VisionDataset]) -> Type:
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self: VisionDataset, index: int) -> Tuple[int, PIL.Image.Image, int]:
        data, target = cls.__getitem__(self, index)
        return index, data, target

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


def load_dataset_and_initial_labels_for_simulation(path_config: str, on_val_set: bool) -> Tuple[Any, np.ndarray]:
    config = load_selector_config(path_config)
    _train_dataset, _val_dataset = get_datasets(config, use_augmentation=True, use_noisy_labels_for_validation=True)
    dataset = _val_dataset if on_val_set else _train_dataset
    initial_labels = convert_labels_to_one_hot(dataset.targets, n_classes=dataset.num_classes)  # type: ignore

    # Make sure that the assigned label does not have probability 0
    if config.dataset.name not in ["NoisyChestXray", "Kaggle", "CIFAR10IDN", "CIFAR10SYM"] and \
       config.dataset.noise_offset == 0.0:
        assert all(dataset.label_distribution.distribution[initial_labels.astype(bool)] > 0.0)  # type: ignore
    dataset.name = config.dataset.name  # type: ignore
    return dataset, initial_labels


def get_datasets(config: ConfigNode,
                 use_augmentation: bool = True,
                 use_noisy_labels_for_validation: bool = False,
                 use_fixed_labels: bool = True) -> Tuple[torch.utils.data.Dataset,
                                                         torch.utils.data.Dataset]:
    """
    :param config: Config object required to create datasets
    :param use_augmentation: Bool flag to indicate whether loaded images should be augmented or not.
    :param use_noisy_labels_for_validation: whether to report validation scores on clean or noisy labels (only for
                                            Noisy Kaggle dataset at the moment).
    :param use_fixed_labels: If set to True, target image labels are fixed.
                             Otherwise, a new label is fetched from the distribution at each get item call.
    """
    num_samples = config.dataset.num_samples

    # The CIFAR10H training set is the CIFAR10 test-set, and so the validation-set is the CIFAR10 training-set
    if config.dataset.name == 'CIFAR10H':
        train_dataset = dataset_with_indices(CIFAR10H)(root=str(CIFAR10_ROOT_DIR),
                                                       transform=create_transform(config, use_augmentation),
                                                       noise_temperature=config.dataset.noise_temperature,
                                                       noise_offset=config.dataset.noise_offset,
                                                       num_samples=num_samples)
        val_dataset = dataset_with_indices(CIFAR10)(root=str(CIFAR10_ROOT_DIR),
                                                    train=True,
                                                    transform=create_transform(config, is_train=False))

    elif config.dataset.name == "CIFAR10IDN":
        train_dataset = dataset_with_indices(CIFAR10IDN)(root=str(CIFAR10_ROOT_DIR),
                                                         train=False,
                                                         transform=create_transform(config, is_train=True),
                                                         noise_rate=config.dataset.noise_rate,
                                                         use_fixed_labels=use_fixed_labels)
        val_dataset = dataset_with_indices(CIFAR10IDN)(root=str(CIFAR10_ROOT_DIR),
                                                       train=True,
                                                       transform=create_transform(config, is_train=False),
                                                       noise_rate=config.dataset.noise_rate,
                                                       use_fixed_labels=use_fixed_labels)

    elif config.dataset.name == "CIFAR10SYM":
        train_dataset = dataset_with_indices(CIFAR10AsymNoise)(root=str(CIFAR10_ROOT_DIR),
                                                               train=False,
                                                               transform=create_transform(config, is_train=True),
                                                               use_fixed_labels=use_fixed_labels)
        val_dataset = dataset_with_indices(CIFAR10AsymNoise)(root=str(CIFAR10_ROOT_DIR),
                                                             train=True,
                                                             transform=create_transform(config, is_train=False),
                                                             use_fixed_labels=use_fixed_labels)

    # Default CIFAR10 training and validation sets
    elif config.dataset.name == 'CIFAR10':
        if num_samples is not None:
            raise ValueError("Dataset subset selection is not implemented for default CIFAR10.")

        train_dataset = dataset_with_indices(CIFAR10)(root=str(CIFAR10_ROOT_DIR),
                                                      train=True,
                                                      transform=create_transform(config, use_augmentation))
        val_dataset = dataset_with_indices(CIFAR10)(root=str(CIFAR10_ROOT_DIR),
                                                    train=False,
                                                    transform=create_transform(config, is_train=False))

    # Noisy subset of Kaggle dataset
    elif config.dataset.name == "NoisyChestXray":
        train_dataset = NoisyKaggleSubsetCXR(config.dataset.dataset_dir,
                                             use_training_split=True,
                                             transform=create_transform(config, use_augmentation),
                                             num_samples=num_samples,
                                             use_noisy_fixed_labels=True,
                                             consolidation_noise_rate=config.dataset.cxr_consolidation_noise_rate)
        val_dataset = NoisyKaggleSubsetCXR(config.dataset.dataset_dir,
                                           use_training_split=False,
                                           transform=create_transform(config, is_train=False),
                                           use_noisy_fixed_labels=use_noisy_labels_for_validation,
                                           consolidation_noise_rate=config.dataset.cxr_consolidation_noise_rate)

    elif config.dataset.name == "Kaggle":
        train_dataset = KaggleCXR(config.dataset.dataset_dir,
                                  use_training_split=True,
                                  transform=create_transform(config, is_train=use_augmentation))
        val_dataset = KaggleCXR(config.dataset.dataset_dir,
                                use_training_split=False,
                                transform=create_transform(config, is_train=False))
    else:
        raise ValueError('Unsupported dataset choice')

    logging.info(f"Training dataset size N={len(train_dataset)}")
    logging.info(f"Validation dataset size N={len(val_dataset)}")

    if num_samples is not None:
        assert 0 < num_samples <= len(train_dataset)
        assert num_samples == len(train_dataset)
        if use_fixed_labels:
            assert train_dataset.targets is not None

    return train_dataset, val_dataset
