import multiprocessing
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl

from .chestxray_datamodule import RSNAKaggleDataModule
from .cifar_ie_datamodule import CIFARIEDataModule
from ..configs.config_node import ConfigNode


def create_ssl_data_modules(config: ConfigNode, dataset_path: Optional[Path],
                            num_devices: int) -> pl.LightningDataModule:
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
                               seed=1234)
        dm.prepare_data()
        dm.setup('fit')
    else:
        raise NotImplementedError(f"No pytorch data module implemented for dataset type: {config.dataset.name}")

    return dm
