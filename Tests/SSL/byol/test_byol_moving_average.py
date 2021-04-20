#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from pathlib import Path
from random import randint
from typing import Any
from unittest import mock

import numpy as np
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from InnerEye.SSL.byol.byol_module import BYOLInnerEye
from InnerEye.SSL.byol.byol_moving_average import BYOLMAWeightUpdate
from InnerEye.SSL.datamodules.cxr_datasets import RSNAKaggleCXR


def test_update_tau() -> None:
    class DummyRSNADataset(RSNAKaggleCXR):
        def __getitem__(self, item: Any) -> Any:
            return (torch.rand([3, 224, 224], dtype=torch.float32),
                    torch.rand([3, 224, 224], dtype=torch.float32)), \
                   randint(0, 1)

    dataset_dir = str(Path(__file__).parent.parent / "test_dataset")
    dummy_rsna_train_dataloader: DataLoader = torch.utils.data.DataLoader(
        DummyRSNADataset(dataset_dir, True),
        batch_size=20,
        num_workers=0,
        drop_last=True)

    byol_weight_update = BYOLMAWeightUpdate(initial_tau=0.99)
    trainer = Trainer(max_epochs=5)
    trainer.train_dataloader = dummy_rsna_train_dataloader
    n_steps_per_epoch = len(trainer.train_dataloader)
    total_steps = n_steps_per_epoch * trainer.max_epochs  # type: ignore
    byol_module = BYOLInnerEye(num_samples=16,
                               learning_rate=1e-3,
                               batch_size=4,
                               encoder_name="resnet50",
                               warmup_epochs=10)
    with mock.patch("InnerEye.SSL.byol.byol_module.BYOLInnerEye.global_step", 15):
        new_tau = byol_weight_update.update_tau(pl_module=byol_module, trainer=trainer)
    assert new_tau == 1 - 0.01 * (math.cos(math.pi * 15 / total_steps) + 1) / 2


def test_update_weights() -> None:
    online_network = torch.nn.Linear(in_features=3, out_features=1, bias=False)
    target_network = torch.nn.Linear(in_features=3, out_features=1, bias=False)
    byol_weight_update = BYOLMAWeightUpdate(initial_tau=0.9)
    old_target_net_weight = target_network.weight.data.numpy().copy()
    byol_weight_update.update_weights(online_network, target_network)
    assert np.isclose(target_network.weight.data.numpy(),
                      0.9 * old_target_net_weight + 0.1 * online_network.weight.data.numpy()).all()
