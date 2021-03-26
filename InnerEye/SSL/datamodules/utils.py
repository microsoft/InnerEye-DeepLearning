import multiprocessing
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule
from .chestxray_datamodule import RSNAKaggleDataModule
from .cifar_ie_datamodule import CIFARIEDataModule
from ..configs.config_node import ConfigNode

num_gpus = torch.cuda.device_count()
num_devices = num_gpus if num_gpus > 0 else 1


def create_ssl_data_modules(config: ConfigNode, dataset_path: Optional[Path]) -> pl.LightningDataModule:
    """
    Returns torch lightning data module.
    """
    num_workers = config.dataset.num_workers if config.dataset.num_workers else multiprocessing.cpu_count()

    if config.dataset.name == "RSNAKaggle":
        assert dataset_path is not None
        dm = RSNAKaggleDataModule(config,
                                  dataset_path=dataset_path,
                                  num_devices=num_devices,
                                  num_workers=num_workers)  # type: ignore
    elif config.dataset.name == "CIFAR10":
        dm = CIFARIEDataModule(data_dir=dataset_path,
                               num_workers=num_workers,
                               batch_size=config.train.batch_size // num_devices,
                               seed=1234,
                               val_split=5000)
        dm.prepare_data()
        dm.setup('fit')
    else:
        raise NotImplementedError(f"No pytorch data module implemented for dataset type: {config.dataset.name}")

    return dm
