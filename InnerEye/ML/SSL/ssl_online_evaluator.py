#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, Dict, Optional, Set, Tuple, Union

import pytorch_lightning as pl
import torch
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
from torch import Tensor as T
from torch.nn import functional as F

from InnerEye.ML.SSL.utils import SSLModule
from InnerEye.ML.lightning_metrics import Accuracy05, AreaUnderPrecisionRecallCurve, AreaUnderRocCurve

BatchType = Union[Dict[SSLModule, Any], Any]


class SSLOnlineEvaluatorInnerEye(SSLOnlineEvaluator):
    def __init__(self, learning_rate: float,
                 class_weights: Optional[torch.Tensor] = None,
                 **kwargs: Any) -> None:
        """
        Creates a hook to evaluate a linear model on top of an SSL embedding.

        :param class_weights: The class weights to use when computing the cross entropy loss. If set to None,
                              no weighting will be done.
        :param length_linear_head_loader: The maximum number of batches in the dataloader for the linear head.
        """

        super().__init__(**kwargs)
        self.training_step = int(0)
        self.weight_decay = 1e-4
        self.learning_rate = learning_rate

        self.train_metrics = [AreaUnderRocCurve(), AreaUnderPrecisionRecallCurve(), Accuracy05()] \
            if self.num_classes == 2 else [Accuracy05()]
        self.val_metrics = [AreaUnderRocCurve(), AreaUnderPrecisionRecallCurve(), Accuracy05()] \
            if self.num_classes == 2 else [Accuracy05()]
        self.class_weights = class_weights

    def on_pretrain_routine_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Initializes modules and moves metrics and class weights to module device
        """
        for metric in [*self.train_metrics, *self.val_metrics]:
            metric.to(device=pl_module.device)  # type: ignore

        pl_module.non_linear_evaluator = SSLEvaluator(n_input=self.z_dim,
                                                      n_classes=self.num_classes,
                                                      p=self.drop_p,
                                                      n_hidden=self.hidden_dim).to(pl_module.device)
        assert isinstance(pl_module.non_linear_evaluator, torch.nn.Module)
        self.optimizer = torch.optim.Adam(pl_module.non_linear_evaluator.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.weight_decay)

    @staticmethod
    def to_device(batch: Any, device: Union[str, torch.device]) -> Tuple[T, T]:
        """
        Moves batch to device.
        :param device: device to move the batch to.
        """
        _, x, y = batch
        return x.to(device), y.to(device)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.visited_ids: Set[Any] = set()

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.visited_ids = set()

    def shared_step(self, batch: BatchType, pl_module: pl.LightningModule, is_training: bool) -> T:
        """
        Forward pass and MLP loss computation for the linear head only. Representations from the encoder are frozen and
        detach from computation graph for this loss computation.
        Returns cross-entropy loss for the input batch.
        """
        batch = batch[SSLModule.LINEAR_HEAD] if isinstance(batch, dict) else batch
        x, y = self.to_device(batch, pl_module.device)
        with torch.no_grad():
            representations = self.get_representations(pl_module, x)
        representations = representations.detach()
        assert isinstance(pl_module.non_linear_evaluator, torch.nn.Module)

        # Run the linear-head with SSL embeddings.
        mlp_preds = pl_module.non_linear_evaluator(representations)
        weights = None if self.class_weights is None else self.class_weights.to(device=pl_module.device)
        mlp_loss = F.cross_entropy(mlp_preds, y, weight=weights)

        with torch.no_grad():
            posteriors = F.softmax(mlp_preds, dim=-1)
            for metric in (self.train_metrics if is_training else self.val_metrics):
                metric(posteriors, y)  # type: ignore

        return mlp_loss

    def on_validation_batch_end(self, trainer: pl.Trainer,
                                pl_module: pl.LightningModule,
                                outputs: Any,
                                batch: BatchType,
                                batch_idx: int,
                                dataloader_idx: int) -> None:  # type: ignore
        """
        Get and log validation metrics.
        """
        ids_linear_head = tuple(batch[SSLModule.LINEAR_HEAD][0].tolist())
        if ids_linear_head not in self.visited_ids:
            self.visited_ids.add(ids_linear_head)
            loss = self.shared_step(batch, pl_module, is_training=False)
            pl_module.log('ssl/online_val_loss', loss, on_step=False, on_epoch=True, sync_dist=False)
            for metric in self.val_metrics:
                pl_module.log(f"ssl/online_val_{metric.name}", metric, on_epoch=True, on_step=False)  # type: ignore

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:  # type: ignore
        """
        Get and log training metrics, perform network update.
        """
        ids_linear_head = tuple(batch[SSLModule.LINEAR_HEAD][0].tolist())
        if ids_linear_head not in self.visited_ids:
            self.visited_ids.add(ids_linear_head)
            loss = self.shared_step(batch, pl_module, is_training=True)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.training_step += 1

            # log metrics
            pl_module.log('ssl/online_train_loss', loss)
            for metric in self.train_metrics:
                pl_module.log(f"ssl/online_train_{metric.name}", metric, on_epoch=True, on_step=False)  # type: ignore


def get_encoder_output_dim(pl_module: Union[pl.LightningModule, torch.nn.Module],
                           dm: Optional[pl.LightningDataModule] = None) -> int:
    """
    Calculates the output dimension of ssl encoder by making a single forward pass.
    :param pl_module: pl encoder module
    :param dm: pl datamodule
    """
    # Target device
    device = pl_module.device if isinstance(pl_module, pl.LightningDataModule) else \
        next(pl_module.parameters()).device  # type: ignore
    assert (isinstance(device, torch.device))

    # Create a dummy input image
    if dm is not None:
        dataloader = dm.train_dataloader()
        dataloader = dataloader[SSLModule.LINEAR_HEAD] if isinstance(dataloader, dict) else dataloader
        batch = iter(dataloader).next()  # type: ignore 
        x, _ = SSLOnlineEvaluatorInnerEye.to_device(batch, device)
    else:
        x = torch.rand((1, 3, 256, 256)).to(device)

    # Extract the number of output feature dimensions
    with torch.no_grad():
        representations = pl_module(x)

    return representations.shape[1]


def WrapSSL(ssl_class: Any, num_classes: int) -> Any:
    """
    Wraps a given SSL encoder and adds a non-linear evaluator to it. This is done to load pre-trained SSL checkpoints.
    PL requires non_linear_evaluator to be included in pl_module at SSL training time.
    :param num_classes: Number of target classes for the linear head.
    :param ssl_class:   SSL object either BYOL or SimCLR.
    """
    class _wrap(ssl_class):  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.non_linear_evaluator = SSLEvaluator(n_input=get_encoder_output_dim(self),
                                                     n_classes=num_classes,
                                                     n_hidden=None)

    return _wrap
