#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List, Optional

import torch
from pl_bolts.models.self_supervised import SSLEvaluator
from torch.nn import functional as F

from InnerEye.ML.SSL.encoders import get_encoder_output_dim
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.lightning_container import LightningModuleWithOptimizer
from InnerEye.ML.lightning_metrics import Accuracy05, AreaUnderPrecisionRecallCurve, AreaUnderRocCurve, \
    ScalarMetricsBase
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule


class SSLClassifier(LightningModuleWithOptimizer, DeviceAwareModule):
    """
    SSL Image classifier that combines pre-trained SSL encoder with a trainable linear-head.
    """

    def __init__(self,
                 num_classes: int,
                 encoder: torch.nn.Module,
                 freeze_encoder: bool,
                 class_weights: Optional[torch.Tensor]):
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
        if self.num_classes == 2:
            self.train_metrics: List[ScalarMetricsBase] = \
                [AreaUnderRocCurve(), AreaUnderPrecisionRecallCurve(), Accuracy05()]
            self.val_metrics: List[ScalarMetricsBase] = \
                [AreaUnderRocCurve(), AreaUnderPrecisionRecallCurve(), Accuracy05()]
        else:
            # Note that for multi-class, Accuracy05 is the standard multi-class accuracy.
            self.train_metrics = [Accuracy05()]
            self.val_metrics = [Accuracy05()]

    def train(self, mode: bool = True) -> Any:
        self.classifier_head.train(mode)
        if self.freeze_encoder:
            return self
        self.encoder.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        if self.freeze_encoder:
            with torch.no_grad():
                agg_repr = self.encoder(x).flatten(1).detach()
        else:
            agg_repr = self.encoder(x).flatten(1)
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

    def training_step(self, batch: Any, batch_id: int, *args: Any, **kwargs: Any) -> None:  # type: ignore
        loss = self.shared_step(batch, True)
        self.log("loss/train", loss)
        for metric in self.train_metrics:
            self.log(f"train_{metric.name}", metric, on_epoch=True, on_step=False)

    def validation_step(self, batch: Any, batch_id: int, *args: Any, **kwargs: Any) -> None:  # type: ignore
        loss = self.shared_step(batch, is_training=False)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=False)
        for metric in self.val_metrics:
            self.log(f"val_{metric.name}", metric, on_epoch=True, on_step=False)

    def get_input_tensors(self, item: ScalarItem) -> List[torch.Tensor]:
        """
        Not used for CXRImageClassifier container. This is just need if we use this model within a InnerEyeContainer.
        """
        return [item.images]
