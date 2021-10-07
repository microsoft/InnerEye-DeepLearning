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

from InnerEye.ML.SSL.utils import SSLDataModuleType
from InnerEye.ML.lightning_loggers import log_on_epoch
from InnerEye.ML.lightning_metrics import Accuracy05, AreaUnderPrecisionRecallCurve, AreaUnderRocCurve

BatchType = Union[Dict[SSLDataModuleType, Any], Any]

OPTIMIZER_STATE_NAME = "evaluator_optimizer"
EVALUATOR_STATE_NAME = "evaluator_weights"


class SSLOnlineEvaluatorInnerEye(SSLOnlineEvaluator):
    def __init__(self,
                 learning_rate: float,
                 class_weights: Optional[torch.Tensor] = None,
                 **kwargs: Any) -> None:
        """
        Creates a hook to evaluate a linear model on top of an SSL embedding.

        :param class_weights: The class weights to use when computing the cross entropy loss. If set to None,
                              no weighting will be done.
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
        self.non_linear_evaluator = SSLEvaluator(n_input=self.z_dim,
                                                 n_classes=self.num_classes,
                                                 p=self.drop_p,
                                                 n_hidden=self.hidden_dim)
        self.optimizer = torch.optim.Adam(self.non_linear_evaluator.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.weight_decay)

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                           checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        # Each callback gets its own state dictionary, that are fed back in during load
        return {
            OPTIMIZER_STATE_NAME: self.optimizer.state_dict(),
            EVALUATOR_STATE_NAME: self.non_linear_evaluator.state_dict()
        }

    def on_load_checkpoint(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           callback_state: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(callback_state[OPTIMIZER_STATE_NAME])
        self.non_linear_evaluator.load_state_dict(callback_state[EVALUATOR_STATE_NAME])

    def on_pretrain_routine_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Initializes modules and moves metrics and class weights to module device
        """
        for metric in [*self.train_metrics, *self.val_metrics]:
            metric.to(device=pl_module.device)  # type: ignore
        self.non_linear_evaluator = self.non_linear_evaluator.to(pl_module.device)

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

        # Run the linear-head with SSL embeddings.
        mlp_preds = self.non_linear_evaluator(representations)
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
            old_mode = self.non_linear_evaluator.training
            self.non_linear_evaluator.eval()
            loss = self.shared_step(batch, pl_module, is_training=False)
            log_on_epoch(pl_module, 'ssl_online_evaluator/val/loss', loss)
            for metric in self.val_metrics:
                log_on_epoch(pl_module, f"ssl_online_evaluator/val/{metric.name}", metric)
            self.non_linear_evaluator.train(old_mode)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:  # type: ignore
        """
        Get and log training metrics, perform network update.
        """
        # Similar code should also live in the encoder training.
        # There is a silent assumption here that SSL data is larger than linear head data
        ids_linear_head = tuple(batch[SSLDataModuleType.LINEAR_HEAD][0].tolist())
        if ids_linear_head not in self.visited_ids:
            self.visited_ids.add(ids_linear_head)
            loss = self.shared_step(batch, pl_module, is_training=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log metrics
            log_on_epoch(pl_module, 'ssl_online_evaluator/train/loss', loss)
            for metric in self.train_metrics:
                log_on_epoch(pl_module, f"ssl_online_evaluator/train/online_{metric.name}", metric)
