#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pytorch_lightning as pl
import torch
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
from torch import Tensor as T
from torch.nn import functional as F
from torchmetrics import Metric

from InnerEye.ML.SSL.utils import SSLDataModuleType, add_submodules_to_same_device
from InnerEye.ML.lightning_metrics import Accuracy05, AreaUnderPrecisionRecallCurve, AreaUnderRocCurve

BatchType = Union[Dict[SSLDataModuleType, Any], Any]


class SSLOnlineEvaluatorInnerEye(SSLOnlineEvaluator):
    def __init__(self,
                 learning_rate: float,
                 class_weights: Optional[torch.Tensor] = None,
                 **kwargs: Any) -> None:
        """
        Creates a hook to evaluate a linear model on top of an SSL embedding.

        :param class_weights: The class weights to use when computing the cross entropy loss. If set to None,
                              no weighting will be done.
        :param length_linear_head_loader: The maximum number of batches in the dataloader for the linear head.
        """

        super().__init__(**kwargs)
        self.weight_decay = 1e-4
        self.learning_rate = learning_rate

        self.train_metrics: List[Metric] = [AreaUnderRocCurve(), AreaUnderPrecisionRecallCurve(),
                                            Accuracy05()] \
            if self.num_classes == 2 else [Accuracy05()]
        self.val_metrics: List[Metric] = [AreaUnderRocCurve(), AreaUnderPrecisionRecallCurve(),
                                          Accuracy05()] \
            if self.num_classes == 2 else [Accuracy05()]
        self.class_weights = class_weights

    def on_pretrain_routine_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Initializes modules and moves metrics and class weights to module device
        """
        for prefix, metrics in [("train", self.train_metrics), ("val", self.val_metrics)]:
            add_submodules_to_same_device(pl_module, metrics, prefix=prefix)

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
        batch = batch[SSLDataModuleType.LINEAR_HEAD] if isinstance(batch, dict) else batch
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
        ids_linear_head = tuple(batch[SSLDataModuleType.LINEAR_HEAD][0].tolist())
        if ids_linear_head not in self.visited_ids:
            self.visited_ids.add(ids_linear_head)
            loss = self.shared_step(batch, pl_module, is_training=False)
            pl_module.log('ssl_online_evaluator/val/loss', loss, on_step=False, on_epoch=True, sync_dist=False)
            for metric in self.val_metrics:
                pl_module.log(f"ssl_online_evaluator/val/{metric.name}", metric, on_epoch=True,
                              on_step=False)  # type: ignore

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # type: ignore
        """
        Get and log training metrics, perform network update.
        """
        ids_linear_head = tuple(batch[SSLDataModuleType.LINEAR_HEAD][0].tolist())
        if ids_linear_head not in self.visited_ids:
            self.visited_ids.add(ids_linear_head)
            loss = self.shared_step(batch, pl_module, is_training=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log metrics
            pl_module.log('ssl_online_evaluator/train/loss', loss)
            for metric in self.train_metrics:
                pl_module.log(f"ssl_online_evaluator/train/online_{metric.name}", metric, on_epoch=True,
                              on_step=False)  # type: ignore
