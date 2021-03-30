import multiprocessing
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl

from .chestxray_datamodule import RSNAKaggleDataModule
from .cifar_ie_datamodule import CIFARIEDataModule
from ..config_node import ConfigNode
from ...ML.lightning_container import LightningContainer


def create_ssl_data_modules(dataset_path: Optional[Path],
                            model_config: LightningContainer,
                            num_devices: int,
                            augmentation_config: Optional[ConfigNode]) -> pl.LightningDataModule:
    """
    Returns torch lightning data module.
    """
    if model_config.dataset_name == "RSNAKaggle":
        assert dataset_path is not None
        dm = RSNAKaggleDataModule(augmentation_config,
                                  model_config,
                                  dataset_path=dataset_path,
                                  num_devices=num_devices)  # type: ignore
    elif model_config.dataset_name == "CIFAR10":
        dm = CIFARIEDataModule(num_workers=model_config.num_workers,
                               batch_size= model_config.batch_size // num_devices,
                               seed=1234,
                               val_split=5000)
        dm.prepare_data()
        dm.setup('fit')
    else:
        raise NotImplementedError(f"No pytorch data module implemented for dataset type: {augmentation_config.dataset.name}")

    return dm
