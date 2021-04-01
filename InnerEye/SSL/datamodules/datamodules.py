import logging
from typing import Any, Callable, Dict, Optional, Union

import torch
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader

from InnerEye.SSL.utils import SSLModule

import numpy as np


class InnerEyeVisionDataModule(VisionDataModule):

    def __init__(self,
                 dataset_cls: type,
                 return_index: bool,
                 train_transforms: Optional[Callable],
                 val_transforms: Optional[Callable],
                 data_dir: Optional[str] = None,
                 val_split: Union[int, float] = 0.2,
                 num_workers: int = 16,
                 batch_size: int = 32,
                 *args: Any, **kwargs: Any) -> None:
        """
        Wrapper around VisionDatamodule to load torchvision dataset into a pytorch-lightning module.

        :param dataset_cls: class to load the dataset. Expected to inherit from VisionDataset as well
        as taking return_index as __init__ argument. See InnerEyeCXRDatasetBase for an example.
        :param return_index: whether the return the index in __get_item__, the dataset_cls is expected to implement
        this logic.
        :param train_transforms: transforms to use at training time
        :param val_transforms: transforms to use at validation time
        :param data_dir: data directory where to find the data
        :param val_split: proportion of training dataset to use for validation
        :param num_workers: number of processes for dataloaders.
        :param batch_size: batch size for training & validation.
        """
        super().__init__(data_dir=data_dir,
                         val_split=val_split,
                         num_workers=num_workers,
                         batch_size=batch_size,
                         drop_last=True,
                         train_transforms=train_transforms,
                         val_transforms=val_transforms,
                         *args,
                         **kwargs)
        self.dataset_cls = dataset_cls
        self.EXTRA_ARGS = {"return_index": return_index}


class CombinedDataModule(LightningDataModule):

    def __init__(self,
                 encoder_module: InnerEyeVisionDataModule,
                 linear_head_module: InnerEyeVisionDataModule,
                 use_balanced_loss_linear_head: bool,
                 *args: Any,
                 **kwargs: Any) -> None:
        """
        Combined data module to use different datasets for training SSL encoder and finetuning the linear head.

        :param encoder_module: datamodule to use for training of SSL.
        :param linear_head_module: datamodule to use for training of linear head on top of frozen encoder.
        """
        super().__init__(*args, **kwargs)
        self.encoder_module = encoder_module
        self.linear_head_module = linear_head_module
        self.class_weights = None
        dataset = self.linear_head_module.dataset_train.dataset
        if use_balanced_loss_linear_head and hasattr(dataset, "targets"):
            # Weight = inverse class proportion.
            class_weights = len(dataset.targets) / np.bincount(dataset.targets)
            # Normalized class weights
            class_weights /= class_weights.sum()
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.batch_size = self.encoder_module.batch_size

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Saves files to data_dir
        """
        self.encoder_module.prepare_data()
        self.linear_head_module.prepare_data()
        logging.info(f"Len encoder train dataloader {len(self.encoder_module.train_dataloader())}")
        logging.info(f"Len total train dataloader {len(self.train_dataloader())}")

    def train_dataloader(self, *args: Any, **kwargs: Any) -> Dict[SSLModule, DataLoader]:
        """
        The train dataloaders
        """
        dataloaders = {
            SSLModule.ENCODER: self.encoder_module.train_dataloader(),
            SSLModule.LINEAR_HEAD: self.linear_head_module.train_dataloader()}
        return dataloaders

    def val_dataloader(self, *args: Any, **kwargs: Any) -> CombinedLoader:
        """
        The val dataloader
        """
        dataloaders = {
            SSLModule.ENCODER: self.encoder_module.val_dataloader(),
            SSLModule.LINEAR_HEAD: self.linear_head_module.val_dataloader()}

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
