#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import multiprocessing
import torch
import pytorch_lightning as pl
from InnerEyeDataQuality.configs.config_node import ConfigNode
from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform

from .chestxray_datamodule import KaggleDataModule, NIHDataModule
from .cifar10h_datamodule import CIFAR10HDataModule


num_gpus = torch.cuda.device_count()
num_devices = num_gpus if num_gpus > 0 else 1

def create_ssl_data_modules(config: ConfigNode) -> pl.LightningDataModule:
    """
    Returns torch lightining data module.
    """
    num_workers = config.dataset.num_workers if config.dataset.num_workers else multiprocessing.cpu_count()

    if config.dataset.name == "Kaggle":
        dm = KaggleDataModule(config, num_devices=num_devices, num_workers=num_workers)  # type: ignore
    elif config.dataset.name == "NIH":
        dm = NIHDataModule(config, num_devices=num_devices, num_workers=num_workers)  # type: ignore
    elif config.dataset.name == "CIFAR10H":
        dm = CIFAR10HDataModule(num_workers=num_workers,
                                batch_size=config.train.batch_size // num_devices,
                                seed=1234)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)
    else:
        raise NotImplementedError(f"No pytorch data module implemented for dataset type: {config.dataset.name}")
    return dm
