#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, List

import param
import torch
from pl_bolts.models.self_supervised import SSLEvaluator
from torch.nn import functional as F

from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.lightning_container import LightningModuleWithOptimizer
from InnerEye.ML.lightning_metrics import Accuracy05, AreaUnderPrecisionRecallCurve, AreaUnderRocCurve
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.SSL.datamodules.datamodules import InnerEyeVisionDataModule
from InnerEye.SSL.lightning_containers.ssl_container import SSLContainer
from InnerEye.SSL.ssl_online_evaluator import get_encoder_output_dim
from InnerEye.SSL.utils import create_ssl_image_classifier


class SSLClassifier(LightningModuleWithOptimizer, DeviceAwareModule):
    """
    SSL Image classifier that combines pre-trained SSL encoder with a trainable linear-head.
    """

    def __init__(self, num_classes: int, encoder: torch.nn.Module, freeze_encoder: bool, class_weights: torch.Tensor):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.class_weights = class_weights
        self.encoder.eval()
        self.classifier_head = SSLEvaluator(n_input=get_encoder_output_dim(self.encoder),
                                            n_hidden=None,
                                            n_classes=num_classes,
                                            p=0.20)
        self.train_metrics = [AreaUnderRocCurve(), AreaUnderPrecisionRecallCurve(), Accuracy05()] \
            if self.num_classes == 2 else [Accuracy05()]
        self.val_metrics = [AreaUnderRocCurve(), AreaUnderPrecisionRecallCurve(), Accuracy05()] \
            if self.num_classes == 2 else [Accuracy05()]

    def on_train_start(self) -> None:
        for metric in [*self.train_metrics, *self.val_metrics]:
            metric.to(device=self.device)  # type: ignore

    def train(self, mode: bool = True) -> Any:
        self.classifier_head.train(mode)
        if self.freeze_encoder:
            return self
        self.encoder.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.encoder.avgpool, torch.nn.Module)
        if self.freeze_encoder:
            with torch.no_grad():
                repr = self.encoder(x)
                agg_repr = self.encoder.avgpool(repr) if repr.ndim > 2 else repr
                agg_repr = agg_repr.reshape(agg_repr.size(0), -1).detach()
        else:
            repr = self.encoder(x)
            agg_repr = self.encoder.avgpool(repr) if repr.ndim > 2 else repr
            agg_repr = agg_repr.reshape(agg_repr.size(0), -1)
        return self.classifier_head(agg_repr)

    def shared_step(self, batch: Any, is_training: bool) -> Any:
        _, x, y = batch
        mlp_preds = self.forward(x)
        weights = None if self.class_weights is None else self.class_weights.to(device=self.device)
        mlp_loss = F.cross_entropy(mlp_preds, y, weight=weights)

        with torch.no_grad():
            posteriors = F.softmax(mlp_preds, dim=-1)
            for metric in (self.train_metrics if is_training else self.val_metrics):
                metric(posteriors, y)
        return mlp_loss

    def training_step(self, batch, batch_id, *args: Any, **kwargs: Any) -> Any:
        loss = self.shared_step(batch, True)
        self.log("loss/train", loss)
        for metric in self.train_metrics:
            self.log(f"train_{metric.name}", metric, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_id, *args, **kwargs):
        loss = self.shared_step(batch, is_training=False)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=False)
        for metric in self.val_metrics:
            self.log(f"val_{metric.name}", metric, on_epoch=True, on_step=False)

    def get_input_tensors(self, item: ScalarItem) -> List[torch.Tensor]:
        """
        Not used for CXRImageClassifier container. This is just need if we use this model within a InnerEyeContainer.
        """
        return [item.images]


class SSLClassifierContainer(SSLContainer):
    """
    This module is usef to train a linear classifier on top of a frozen (or not) encoder.

    You need to specify:
        todo
    """
    freeze_encoder = param.Boolean(default=True, doc="Whether to freeze the pretrained encoder or not.")
    local_ssl_weights_path = param.ClassSelector(class_=Path, default=None, doc="Local path to SSL weights")

    def create_model(self) -> LightningModuleWithOptimizer:
        """
        This method must create the actual Lightning model that will be trained.
        """
        if self.local_ssl_weights_path is None:
            assert self.extra_downloaded_run_id is not None
            try:
                path_to_checkpoint = self.extra_downloaded_run_id.get_best_checkpoint_paths()
            except FileNotFoundError:
                path_to_checkpoint = self.extra_downloaded_run_id.get_recovery_checkpoint_paths()
            path_to_checkpoint = path_to_checkpoint[0]
        else:
            path_to_checkpoint = self.local_ssl_weights_path

        model = create_ssl_image_classifier(num_classes=self.data_module.dataset_train.dataset.num_classes,
                                            pl_checkpoint_path=str(path_to_checkpoint),
                                            freeze_encoder=self.freeze_encoder,
                                            class_weights=self.data_module.class_weights)

        return model

    def get_data_module(self) -> InnerEyeVisionDataModule:
        """
        Gets the data that is used for the training and validation steps.
        Here we use different data loader for training of linear head and training of SSL model.
        """
        if hasattr(self, "data_module"):
            return self.data_module
        self.data_module = self._create_ssl_data_modules(linear_head_module=True)
        if self.use_balanced_binary_loss_for_linear_head:
            self.data_module.class_weights = self.data_module.compute_class_weights()
        return self.data_module

    def get_trainer_arguments(self):
        trained_kwargs = {}
        if self.debug:
            trained_kwargs.update({"limit_train_batches": 2, "limit_val_batches": 2})
        return trained_kwargs
