#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from health_ml.utils import log_learning_rate, log_on_epoch
from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR

from InnerEye.ML.SSL.encoders import SSLEncoder
from InnerEye.ML.SSL.utils import SSLDataModuleType

SingleBatchType = Tuple[List, torch.Tensor]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def training_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        # print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), 'training', 'batch_idx', batch_idx, 'rank',
        #       self.global_rank)

        loss = self.shared_step(batch)
        log_on_epoch(self, "simclr/train/loss", loss, sync_dist=False)
        log_learning_rate(self, name="simclr/learning_rate")
        return loss

    def validation_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:  # type: ignore
        # print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), 'validation', 'batch_idx', batch_idx, 'rank',
        #       self.global_rank)

        loss = self.shared_step(batch)
        log_on_epoch(self, "simclr/val/loss", loss, sync_dist=False)
        return loss

    # def on_train_epoch_end(self, *args, **kwargs) -> None:
    #     print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), "train epoch end", 'rank', self.global_rank, 'epoch',
    #           self.trainer.current_epoch)
    #     super().on_train_epoch_end(*args, **kwargs)

    # def on_train_epoch_start(self, *args, **kwargs) -> None:
    #     print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), "train epoch start", 'rank', self.global_rank, 'epoch',
    #           self.trainer.current_epoch)
    #     super().on_train_epoch_start(*args, **kwargs)

    # def on_validation_epoch_end(self, *args, **kwargs) -> None:
    #     print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), "val epoch end", 'rank', self.global_rank, 'epoch',
    #           self.trainer.current_epoch)
    #     super().on_validation_epoch_end(*args, **kwargs)

    # def on_validation_epoch_start(self, *args, **kwargs) -> None:
    #     print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), "val epoch start", 'rank', self.global_rank, 'epoch',
    #           self.trainer.current_epoch)
    #     super().on_validation_epoch_start(*args, **kwargs)

    def shared_step(self, batch: BatchType) -> torch.Tensor:
        batch = batch[SSLDataModuleType.ENCODER] if isinstance(batch, dict) else batch

        (img1, img2), y = batch

        # get h representations, bolts resnet returns a list
        # print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), 'before encoder call', 'rank', self.global_rank)
        h1, h2 = self(img1), self(img2)

        # get z representations
        # print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), 'before projection call', 'rank', self.global_rank)
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.temperature)

        return loss

    # def on_fit_start(self) -> None:
    #     print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), "on_fit_start", 'rank', self.global_rank)

    # def on_train_start(self) -> None:
    #     print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), "on_train_start", 'rank', self.global_rank)

    # def on_pretrain_routine_start(self) -> None:
    #     print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), "on_pretrain_routine_start", 'rank', self.global_rank)

    # def on_pretrain_routine_end(self) -> None:
    #     print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), "on_pretrain_routine_end", 'rank', self.global_rank)

    # def on_train_batch_start(self, batch: Any, batch_idx: int, *args, **kwargs) -> None:
    #     print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), "on_train_batch_start", 'rank', self.global_rank, 'epoch',
    #           self.trainer.current_epoch, "batch_idx", batch_idx)
    #     super().on_train_batch_start(batch, batch_idx, *args, **kwargs)

    # def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, *args, **kwargs) -> None:
    #     print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), "on_train_batch_start", 'rank', self.global_rank, 'epoch',
    #           self.trainer.current_epoch, "batch_idx", batch_idx)
    #     super().on_train_batch_end(outputs, batch, batch_idx, *args, **kwargs)

    # def on_epoch_start(self) -> None:
    #     print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), "on_epoch_start", 'rank', self.global_rank)

    # def on_epoch_end(self) -> None:
    #     print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), "on_epoch_end", 'rank', self.global_rank)

    # def on_before_zero_grad(self, optimizer) -> None:
    #     print(datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ "), "on_before_zero_grad", 'rank', self.global_rank)
    #     super().on_before_zero_grad(optimizer)
