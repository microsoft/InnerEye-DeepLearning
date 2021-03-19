from typing import Any, Callable, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import numpy as np

from InnerEye.SSL.configs.config_node import ConfigNode
from InnerEye.SSL.datamodules.rsna_cxr_dataset import RSNA_KAGGLE_TOTAL_SIZE, RSNAKaggleCXR, WorkerInitFunc
from InnerEye.SSL.datamodules.transforms_utils import DualViewTransformWrapper, create_chest_xray_transform


class RSNAKaggleDataModule(LightningDataModule):

    def __init__(self, config: ConfigNode, num_devices: int, num_workers: int,
                 *args: Any, **kwargs: Any) -> None:
        """
        This is the data module to load and prepare the Kaggle RSNA Pneumonia detection challenge dataset.
        :param config: the config parametrizing the experiment. In particular, used for augmentation strength parameters
        (cf. config doc).
        :param num_devices: The number of GPUs to use. The total batch size specified in the config will be divided
        by the number of GPUs.
        :param num_workers: The number of cpu dataloader workers.
        """
        super().__init__(*args, **kwargs)
        self.config = config
        self.num_samples = RSNA_KAGGLE_TOTAL_SIZE
        self.batch_size = config.train.batch_size // num_devices
        self.num_workers = num_workers
        self.train_transforms = DualViewTransformWrapper(create_chest_xray_transform(self.config, is_train=True))
        self.val_transforms = DualViewTransformWrapper(create_chest_xray_transform(self.config, is_train=False))
        self.train_dataset = RSNAKaggleCXR(self.config.dataset.dataset_dir, use_training_split=True,
                                           transform=self.train_transforms, return_index=False)
        self.class_weights: Optional[torch.Tensor] = None
        if config.train.self_supervision.use_balanced_binary_loss_for_linear_head:
            # Weight = inverse class proportion.
            class_weights = len(self.train_dataset.targets) / np.bincount(self.train_dataset.targets)
            # Normalized class weights
            class_weights /= class_weights.sum()
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    @property
    def num_classes(self) -> int:
        return 2

    def train_dataloader(self) -> DataLoader:  # type: ignore
        """
        Returns Kaggle training set (80% of total dataset)
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            worker_init_fn=WorkerInitFunc(self.config.train.seed),
            drop_last=True)

    def val_dataloader(self) -> DataLoader:  # type: ignore
        """
        Returns Kaggle validation set (20% of total dataset)
        """
        val_dataset = RSNAKaggleCXR(self.config.dataset.dataset_dir, use_training_split=False,
                                    transform=self.val_transforms, return_index=False)
        loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            worker_init_fn=WorkerInitFunc(self.config.train.seed),
            drop_last=True)
        return loader

    def test_dataloader(self) -> DataLoader:  # type: ignore
        """
        No Kaggle test split implemented
        """
        raise NotImplementedError

    def default_transforms(self) -> Callable:
        transform = create_chest_xray_transform(self.config, is_train=False)
        return transform
