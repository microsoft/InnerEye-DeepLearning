#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import os
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, Dataset

from InnerEye.ML.SSL.utils import SSLDataModuleType


class InnerEyeVisionDataModule(VisionDataModule):

    def __init__(self,
                 dataset_cls: type,
                 return_index: bool,
                 train_transforms: Optional[Callable],
                 val_transforms: Optional[Callable],
                 data_dir: Optional[str] = None,
                 val_split: Union[int, float] = 0.2,
                 num_workers: int = 6,
                 batch_size: int = 32,
                 seed: int = 42,
                 *args: Any, **kwargs: Any) -> None:
        """
        Wrapper around VisionDatamodule to load torchvision dataset into a pytorch-lightning module.

        :param dataset_cls: class to load the dataset. Expected to inherit from InnerEyeDataClassBaseWithReturnIndex and
         VisionDataset. See InnerEyeCIFAR10 for an example.
         BEWARE VisionDataModule expects the first positional argument of your class to be the data directory.
        :param return_index: whether the return the index in __get_item__, the dataset_cls is expected to implement
        this logic.
        :param train_transforms: transforms to use at training time
        :param val_transforms: transforms to use at validation time
        :param data_dir: data directory where to find the data
        :param val_split: proportion of training dataset to use for validation
        :param num_workers: number of processes for dataloaders.
        :param batch_size: batch size for training & validation.
        :param seed: random seed for dataset splitting
        """
        data_dir = data_dir if data_dir is not None else os.getcwd()
        super().__init__(data_dir=data_dir,
                         val_split=val_split,
                         num_workers=num_workers,
                         batch_size=batch_size,
                         drop_last=True,
                         train_transforms=train_transforms,
                         val_transforms=val_transforms,
                         seed=seed,
                         *args,
                         **kwargs)
        self.dataset_cls = dataset_cls
        self.class_weights: Optional[torch.Tensor] = None
        # In setup() VisionDataModule expects the extra arguments to be passed to the dataset class init
        # via the self.EXTRA_ARGS attribute
        self.EXTRA_ARGS = {"return_index": return_index}

    def prepare_data(self) -> None:
        """
        Initializes the dataset class, and in the case of CIFAR dataset, optionally downloads the data from URL to
        local data_dir.
        """
        self.dataset_cls(self.data_dir, train=True, download=True, **self.EXTRA_ARGS)
        self.dataset_cls(self.data_dir, train=False, download=True, **self.EXTRA_ARGS)

    def _split_dataset(self, dataset: Dataset, train: bool = True) -> Dataset:
        """
        Splits the dataset into train and validation set
        """
        if hasattr(dataset, "_split_dataset"):
            # If the dataset implements a more complex logic than just splitting randomly by index.
            # The dataset class can implements its own _split_dataset function.
            dataset_train, dataset_val = dataset._split_dataset(val_split=self.val_split,  # type: ignore
                                                                seed=self.seed)
            return dataset_train if train else dataset_val
        else:
            return super()._split_dataset(dataset, train)

    def compute_class_weights(self) -> Optional[torch.Tensor]:
        dataset = self.dataset_train.dataset
        class_weights = None
        if hasattr(dataset, "targets"):
            class_weights = len(dataset.targets) / np.bincount(dataset.targets)
            # Normalized class weights
            class_weights /= class_weights.sum()
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        return class_weights


class CombinedDataModule(LightningDataModule):

    def __init__(self,
                 encoder_module: InnerEyeVisionDataModule,
                 linear_head_module: InnerEyeVisionDataModule,
                 use_balanced_loss_linear_head: bool,
                 *args: Any,
                 **kwargs: Any) -> None:
        """
        Combined data module to use different datamodules for training SSL encoder and finetuning the linear head.
        Each batch returned by this data module, will be a dictionary of type Dict[SSLDataModuleType, Any]. If one
        dataloader is shorter than the other, combined dataloader will start looping again from the start of the shorter
        one until we looped through the longest one.

        :param encoder_module: datamodule to use for training of SSL.
        :param linear_head_module: datamodule to use for training of linear head on top of frozen encoder. Can use a
        different batch size than the encoder module. CombinedDataModule logic will take care of aggregation.
        """
        super().__init__(*args, **kwargs)
        self.encoder_module = encoder_module
        self.linear_head_module = linear_head_module
        self.class_weights = None
        if use_balanced_loss_linear_head:
            self.class_weights = self.linear_head_module.compute_class_weights()
        self.batch_size = self.encoder_module.batch_size

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Saves files to data_dir
        """
        self.encoder_module.prepare_data()
        self.linear_head_module.prepare_data()
        logging.info(f"Len encoder train dataloader {len(self.encoder_module.train_dataloader())}")
        logging.info(f"Len total train dataloader {len(self.train_dataloader())}")

    def train_dataloader(self, *args: Any, **kwargs: Any) -> Dict[SSLDataModuleType, DataLoader]:  # type: ignore
        """
        The train dataloaders
        """
        dataloaders = {
            SSLDataModuleType.ENCODER: self.encoder_module.train_dataloader(),
            SSLDataModuleType.LINEAR_HEAD: self.linear_head_module.train_dataloader()}
        return dataloaders

    def val_dataloader(self, *args: Any, **kwargs: Any) -> CombinedLoader:  # type: ignore
        """
        The val dataloader
        """
        dataloaders = {
            SSLDataModuleType.ENCODER: self.encoder_module.val_dataloader(),
            SSLDataModuleType.LINEAR_HEAD: self.linear_head_module.val_dataloader()}

        return CombinedLoader(dataloaders, mode="max_size_cycle")

    @property
    def num_samples(self) -> int:
        """
        Returns number of samples in training set
        """
        return len(self.encoder_module.dataset_train)

    @property
    def num_classes(self) -> int:
        return self.linear_head_module.dataset_train.dataset.num_classes

    def setup(self, stage: Optional[str] = None) -> None:
        self.encoder_module.setup(stage)
        self.linear_head_module.setup(stage)
