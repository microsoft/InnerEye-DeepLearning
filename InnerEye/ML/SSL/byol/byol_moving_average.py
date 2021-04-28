#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback


class ByolMovingAverageWeightUpdate(Callback):
    """
    Weight updates for BYOL moving average encoder (e.g. teacher). Pl_module is expected to contain three attributes:
        - ``pl_module.online_network``
        - ``pl_module.target_network``
        - ``pl_module.global_step``

    Updates the target_network params using an exponential moving average update rule weighted by tau.
    Tau parameter is increased from its base value to 1.0 with every training step scheduled with a cosine function.
    global_step correspond to the total number of sgd updates expected to happen throughout the BYOL training.

    Target network is updated at the end of each SGD update on training batch.
    """

    def __init__(self, initial_tau: float = 0.99):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def on_before_zero_grad(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore
        # get networks
        online_net = pl_module.online_network
        target_net = pl_module.target_network
        assert (isinstance(online_net, torch.nn.Module))
        assert (isinstance(target_net, torch.nn.Module))

        # update weights
        self.update_weights(online_net, target_net)

        # update tau after
        self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> float:
        """
        Update tau parameter (controlling update of teacher model) for BYOL according to current step (cosine schedule).
        """
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs  # type: ignore
        tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * pl_module.global_step / max_steps) + 1) / 2
        return tau

    def update_weights(self, online_net: torch.nn.Module, target_net: torch.nn.Module) -> None:
        """
        Update target network weights with new online network weights.
        """
        # apply MA weight update
        for current_params, ma_params in zip(online_net.parameters(), target_net.parameters()):
            up_weight, old_weight = current_params.data, ma_params.data
            ma_params.data = old_weight * self.current_tau + (1 - self.current_tau) * up_weight
