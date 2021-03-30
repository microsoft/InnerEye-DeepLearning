from pathlib import Path
from typing import Any, Callable, Optional

import torch
from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader

import numpy as np

from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.SSL.config_node import ConfigNode
from InnerEye.SSL.datamodules.rsna_cxr_dataset import RSNAKaggleCXR, WorkerInitFunc
from InnerEye.SSL.datamodules.transforms_utils import DualViewTransformWrapper, create_chest_xray_transform


class RSNAKaggleDataModule(LightningDataModule):
    def __init__(self,
                 augmentation_config: ConfigNode,
                 model_config: LightningContainer,
                 dataset_path: Path,
                 num_devices: int,
                 dataset_class: Any = RSNAKaggleCXR,
                 *args: Any, **kwargs: Any) -> None:
        """
        This is the data module to load and prepare the Kaggle RSNA Pneumonia detection challenge dataset.
        :param augmentation_config: the config parametrizing the experiment. In particular, used for augmentation
        strength parameters
        (cf. config doc).
        :param num_devices: The number of GPUs to use. The total batch size specified in the config will be divided
        by the number of GPUs.
        :param num_workers: The number of cpu dataloader workers.
        """
        super().__init__(*args, **kwargs)
        self.seed = model_config.random_seed
        self._dataset_class = dataset_class
        self.augmentation_config = augmentation_config
        self.dataset_path = dataset_path
        self.batch_size = model_config.batch_size // num_devices
        self.num_workers = model_config.num_workers
        self.train_transforms = DualViewTransformWrapper(create_chest_xray_transform(self.augmentation_config, is_train=True))
        self.train_dataset = self._dataset_class(self.dataset_path,
                                                 use_training_split=True,
                                                 transform=self.train_transforms)
        self.class_weights: Optional[torch.Tensor] = None
        if model_config.use_balanced_binary_loss_for_linear_head and hasattr(self.train_dataset,
                                                                                              "targets"):
            # Weight = inverse class proportion.
            class_weights = len(self.train_dataset.targets) / np.bincount(self.train_dataset.targets)
            # Normalized class weights
            class_weights /= class_weights.sum()
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.num_samples = len(self.train_dataset.indices)

    @property
    def num_classes(self) -> int:
        return self.train_dataset.num_classes

    def train_dataloader(self) -> DataLoader:  # type: ignore
        """
        Returns Kaggle training set (80% of total dataset)
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            worker_init_fn=WorkerInitFunc(self.seed),
            drop_last=True)

    def val_dataloader(self) -> DataLoader:  # type: ignore
        """
        Returns Kaggle validation set (20% of total dataset)
        """
        val_transforms = DualViewTransformWrapper(create_chest_xray_transform(self.augmentation_config, is_train=False))
        val_dataset = self._dataset_class(self.dataset_path, use_training_split=False,
                                          transform=val_transforms)
        loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            worker_init_fn=WorkerInitFunc(self.seed),
            drop_last=True)
        return loader

    def test_dataloader(self) -> DataLoader:  # type: ignore
        """
        No Kaggle test split implemented
        """
        pass

    def default_transforms(self) -> Callable:
        transform = create_chest_xray_transform(self.augmentation_config, is_train=False)
        return transform
