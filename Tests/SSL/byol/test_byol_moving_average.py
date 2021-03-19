#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from unittest import mock

from pytorch_lightning import Trainer

from InnerEye.SSL.byol.byol_module import BYOLInnerEye
from InnerEye.SSL.byol.byol_moving_average import BYOLMAWeightUpdate
from Tests.SSL.test_main import _get_dummy_val_train_rsna_dataloaders


def test_update_tau():
    byol_weight_update = BYOLMAWeightUpdate(initial_tau=0.99)
    trainer = Trainer()
    trainer.train_dataloader = _get_dummy_val_train_rsna_dataloaders()
    trainer.max_epochs = 5
    n_steps_per_epoch = len(trainer.train_dataloader)
    total_steps = n_steps_per_epoch * trainer.max_epochs
    byol_module = BYOLInnerEye(num_samples=16,
                               learning_rate=1e-3,
                               batch_size=4,
                               encoder_name="resnet50",
                               warmup_epochs=10)
    with mock.patch("InnerEye.SSL.byol.byol_module.BYOLInnerEye.global_step", 15):
        new_tau = byol_weight_update.update_tau(pl_module=byol_module, trainer=trainer)
    assert new_tau == 1 - 0.01 * (math.cos(math.pi * 15 / total_steps) + 1) / 2

