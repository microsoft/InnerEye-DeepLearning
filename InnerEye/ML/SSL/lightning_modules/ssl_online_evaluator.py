#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pytorch_lightning as pl
import torch
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
from pytorch_lightning.utilities import rank_zero_warn
from torch import Tensor as T
from torch.nn import SyncBatchNorm, functional as F
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Metric
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from InnerEye.ML.SSL.utils import SSLDataModuleType, add_submodules_to_same_device
from InnerEye.ML.lightning_metrics import Accuracy05, AreaUnderPrecisionRecallCurve, AreaUnderRocCurve
from InnerEye.ML.utils.layer_util import set_model_to_eval_mode
from health_ml.utils import log_on_epoch

BatchType = Union[Dict[SSLDataModuleType, Any], Any]


class SSLOnlineEvaluatorInnerEye(SSLOnlineEvaluator):
    OPTIMIZER_STATE_NAME = "evaluator_optimizer"
    EVALUATOR_STATE_NAME = "evaluator_weights"

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
        self.evaluator = SSLEvaluator(n_input=self.z_dim,
                                      n_classes=self.num_classes,
                                      p=self.drop_p,
                                      n_hidden=self.hidden_dim)
        self.optimizer = torch.optim.Adam(self.evaluator.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.weight_decay)

    def _wrapped_evaluator(self) -> torch.nn.Module:
        """
        Gets the evaluator model that is wrapped in DDP, or the evaluator model itself.
        """
        if isinstance(self.evaluator, DistributedDataParallel):
            return self.evaluator.module
        else:
            return self.evaluator

    def on_save_checkpoint(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        # Each callback gets its own state dictionary, that are fed back in during load
        # When saving the evaluator, use the wrapped DDP module (otherwise the resulting checkpoint will depend
        # on use of DDP or not).
        return {
            self.OPTIMIZER_STATE_NAME: self.optimizer.state_dict(),
            self.EVALUATOR_STATE_NAME: self._wrapped_evaluator().state_dict()
        }

    def on_load_checkpoint(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           callback_state: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(callback_state[self.OPTIMIZER_STATE_NAME])
        # on_load_checkpoint is called before we wrap the evaluator with DDP
        self._wrapped_evaluator().load_state_dict(callback_state[self.EVALUATOR_STATE_NAME])

    def on_pretrain_routine_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Moves metrics and the online evaluator to the correct GPU.
        If training happens via DDP, SyncBatchNorm is enabled for the online evaluator, and it is converted to
        a DDP module.
        """
        for prefix, metrics in [("train", self.train_metrics), ("val", self.val_metrics)]:
            add_submodules_to_same_device(pl_module, metrics, prefix=prefix)
        self.evaluator.to(pl_module.device)
        if hasattr(trainer, "accelerator_connector"):
            # This works with Lightning 1.3.8
            accelerator = trainer.accelerator_connector
        elif hasattr(trainer, "_accelerator_connector"):
            # This works with Lightning 1.5.5
            accelerator = trainer._accelerator_connector
        else:
            raise ValueError("Unable to retrieve the accelerator information")
        if accelerator.is_distributed:
            if accelerator.use_ddp:
                self.evaluator = SyncBatchNorm.convert_sync_batchnorm(self.evaluator)
                self.evaluator = DistributedDataParallel(self.evaluator, device_ids=[pl_module.device])  # type: ignore
            else:
                rank_zero_warn("This type of distributed accelerator is not supported. "
                               "The online evaluator will not synchronize across GPUs.")

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
        mlp_preds = self.evaluator(representations)
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
        Metrics are computed only if the sample IDs in the batch have not yet been seen in this epoch (linear head
        data may be repeated if the SSL data is longer than the linear head data).
        """
        ids_linear_head = tuple(batch[SSLDataModuleType.LINEAR_HEAD][0].tolist())
        if ids_linear_head not in self.visited_ids:
            self.visited_ids.add(ids_linear_head)
            with set_model_to_eval_mode(self.evaluator):
                loss = self.shared_step(batch, pl_module, is_training=False)
                log_on_epoch(pl_module, 'ssl_online_evaluator/val/loss', loss)
                for metric in self.val_metrics:
                    log_on_epoch(pl_module, f"ssl_online_evaluator/val/{metric.name}", metric)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # type: ignore
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
