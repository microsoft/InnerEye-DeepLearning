#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from random import randint
from typing import Any
from unittest import mock

import numpy as np
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from InnerEye.ML.SSL.datamodules_and_datasets.cxr_datasets import RSNAKaggleCXR
from InnerEye.ML.SSL.lightning_modules.byol.byol_module import BYOLInnerEye
from InnerEye.ML.SSL.lightning_modules.byol.byol_moving_average import ByolMovingAverageWeightUpdate
from Tests.SSL.test_ssl_containers import create_cxr_test_dataset, path_to_cxr_test_dataset


def test_update_tau() -> None:
    class DummyRSNADataset(RSNAKaggleCXR):
        def __getitem__(self, item: Any) -> Any:
            return (torch.rand([3, 224, 224], dtype=torch.float32),
                    torch.rand([3, 224, 224], dtype=torch.float32)), \
                   randint(0, 1)

    create_cxr_test_dataset(path_to_cxr_test_dataset)
    dataset_dir = str(path_to_cxr_test_dataset)
    dummy_rsna_train_dataloader: DataLoader = torch.utils.data.DataLoader(
        DummyRSNADataset(root=dataset_dir, return_index=False, train=True),
        batch_size=20,
        num_workers=0,
        drop_last=True)

    byol_weight_update = ByolMovingAverageWeightUpdate(initial_tau=0.99)
    trainer = Trainer(max_epochs=5)
    trainer.train_dataloader = dummy_rsna_train_dataloader  # type: ignore
    n_steps_per_epoch = len(trainer.train_dataloader)  # type: ignore
    total_steps = n_steps_per_epoch * trainer.max_epochs  # type: ignore
    byol_module = BYOLInnerEye(num_samples=16,
                               learning_rate=1e-3,
                               batch_size=4,
                               encoder_name="resnet50",
                               warmup_epochs=10,
                               max_epochs=100)
    with mock.patch("InnerEye.ML.SSL.lightning_modules.byol.byol_module.BYOLInnerEye.global_step", 15):
        new_tau = byol_weight_update.update_tau(pl_module=byol_module, trainer=trainer)
    assert new_tau == 1 - 0.01 * (math.cos(math.pi * 15 / total_steps) + 1) / 2


def test_update_weights() -> None:
    online_network = torch.nn.Linear(in_features=3, out_features=1, bias=False)
    target_network = torch.nn.Linear(in_features=3, out_features=1, bias=False)
    byol_weight_update = ByolMovingAverageWeightUpdate(initial_tau=0.9)
    old_target_net_weight = target_network.weight.data.numpy().copy()
    byol_weight_update.update_weights(online_network, target_network)
    assert np.isclose(target_network.weight.data.numpy(),
                      0.9 * old_target_net_weight + 0.1 * online_network.weight.data.numpy()).all()
