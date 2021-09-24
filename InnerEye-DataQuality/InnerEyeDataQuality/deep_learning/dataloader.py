#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any

import numpy as np
import torch
import yacs.config
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset


def get_number_of_samples_per_epoch(dataloader: torch.utils.data.DataLoader) -> int:
    """
    Returns the expected number of samples for a single epoch
    """
    total_num_samples = len(dataloader.dataset)  # type: ignore
    batch_size = dataloader.batch_size
    drop_last = dataloader.drop_last
    num_samples = int(total_num_samples / batch_size) * batch_size if drop_last else total_num_samples  # type:ignore
    return num_samples


def get_train_dataloader(train_dataset: VisionDataset,
                         config: yacs.config.CfgNode,
                         seed: int,
                         **kwargs: Any) -> DataLoader:
    if config.train.use_balanced_sampler:
        counts = np.bincount(train_dataset.targets)
        class_weights = counts.sum() / counts
        sample_weights = class_weights[train_dataset.targets]
        sample = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset))
        kwargs.pop("shuffle", None)
        kwargs.update({"sampler": sample})
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.dataloader.num_workers,
        pin_memory=config.train.dataloader.pin_memory,
        worker_init_fn=WorkerInitFunc(seed),
        **kwargs)


class WorkerInitFunc:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def __call__(self, worker_id: int) -> None:
        return np.random.seed(self.seed + worker_id)


def get_val_dataloader(val_dataset: VisionDataset, config: yacs.config.CfgNode, seed: int) -> DataLoader:
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.validation.batch_size,
        shuffle=False,
        num_workers=config.validation.dataloader.num_workers,
        pin_memory=config.validation.dataloader.pin_memory,
        worker_init_fn=WorkerInitFunc(seed))
