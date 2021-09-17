#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, List

import torch
from InnerEyeDataQuality.datasets.cifar10h import TOTAL_CIFAR10H_DATASET_SIZE
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transform_lib


class CIFAR10HDataModule(CIFAR10DataModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        """
        super().__init__(*args, **kwargs)
        self.num_samples = TOTAL_CIFAR10H_DATASET_SIZE
        self.class_weights = None

    def train_dataloader(self) -> DataLoader:
        """
        CIFAR train set removes a subset to use for validation
        """
        transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
        dataset = self.DATASET(self.data_dir, train=False, download=True, transform=transforms, **self.extra_args)
        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            drop_last=True,
                            pin_memory=True)
        assert len(dataset) == TOTAL_CIFAR10H_DATASET_SIZE

        return loader

    def val_dataloader(self) -> DataLoader:
        """
        CIFAR10 val set uses a subset of the training set for validation
        """
        transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False, transform=transforms, **self.extra_args)
        num_samples = len(dataset)
        _, dataset_val = random_split(dataset,
                                      [num_samples - self.val_split, self.val_split],
                                      generator=torch.Generator().manual_seed(self.seed))
        assert len(dataset_val) == self.val_split
        loader = DataLoader(dataset_val,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=True,
                            drop_last=True)
        return loader

    def test_dataloader(self) -> DataLoader:
        """
        CIFAR10 test set uses the test split
        """
        transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False, transform=transforms, **self.extra_args)
        num_samples = len(dataset)
        dataset_test, _ = random_split(dataset,
                                       [num_samples - self.val_split, self.val_split],
                                       generator=torch.Generator().manual_seed(self.seed))
        loader = DataLoader(dataset_test,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            drop_last=True,
                            pin_memory=True)
        return loader

    def default_transforms(self) -> List[object]:
        cf10_transforms = transform_lib.Compose([transform_lib.ToTensor(), cifar10_normalization()])
        return cf10_transforms
