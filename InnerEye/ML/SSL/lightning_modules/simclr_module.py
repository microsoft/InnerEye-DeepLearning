#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR
from torch import Tensor as T
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from InnerEye.ML.SSL.encoders import SSLEncoder
from InnerEye.ML.SSL.utils import SSLDataModuleType, manual_optimization_step
from InnerEye.ML.lightning_loggers import log_learning_rate, log_on_epoch

SingleBatchType = Tuple[List, T]
BatchType = Union[Dict[SSLDataModuleType, SingleBatchType], SingleBatchType]


class _Projection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLRInnerEye(SimCLR):
    def __init__(self, encoder_name: str, dataset_name: str, use_7x7_first_conv_in_resnet: bool = True,
                 **kwargs: Any) -> None:
        """
        Returns SimCLR pytorch-lightning module, based on lightning-bolts implementation.
        :param encoder_name: Image encoder name (predefined models)
        :param dataset_name: Dataset name (e.g. cifar10, kaggle, etc.)
        :param use_7x7_first_conv_in_resnet: If True, use a 7x7 kernel (default) in the first layer of resnet.
            If False, replace first layer by a 3x3 kernel. This is required for small CIFAR 32x32 images to not
            shrink them.
        """
        if "dataset" not in kwargs:  # needed for the new version of lightning-bolts
            kwargs.update({"dataset": dataset_name})
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.encoder = SSLEncoder(encoder_name, use_7x7_first_conv_in_resnet)
        self.projection = _Projection(input_dim=self.encoder.get_output_feature_dim(), hidden_dim=2048, output_dim=128)
        # The optimizer for the linear head is managed by this module, so that its state can be stored in a checkpoint
        # automatically by Lightning. The training for the linear head is outside of this module though.
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def training_step(self, batch: BatchType, batch_idx: int, optimizer_idx: int) -> None:  # type: ignore
        if optimizer_idx != 0:
            return
        loss = self.shared_step(batch)
        log_on_epoch(self, "simclr/train/loss", loss)
        log_learning_rate(self, name="simclr/learning_rate")
        manual_optimization_step(self, loss)

    def validation_step(self, batch: BatchType, batch_idx: int) -> T:  # type: ignore
        loss = self.shared_step(batch)
        log_on_epoch(self, "simclr/val/loss", loss)
        return loss

    def shared_step(self, batch: BatchType) -> T:
        batch = batch[SSLDataModuleType.ENCODER] if isinstance(batch, dict) else batch

        (img1, img2), y = batch

        # get h representations, bolts resnet returns a list
        h1, h2 = self(img1), self(img2)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.temperature)

        return loss

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        base_optim, base_scheduler = super().configure_optimizers()
        base_optim.append(self.online_eval_optimizer)
        return base_optim, base_scheduler


    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        out_1_dist = out_1
        out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e**(1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss
