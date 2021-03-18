from typing import Any, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
from torch import Tensor as T
from torch.nn import functional as F

from InnerEye.SSL.metrics import AreaUnderRocCurve
from pytorch_lightning.metrics import Accuracy

BatchType = Tuple[List, T]


class SSLOnlineEvaluatorInnerEye(SSLOnlineEvaluator):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs: Any) -> None:
        """
        Creates a hook to evaluate a linear model on top of an SSL embedding.

        :param class_weights: The class weights to use when computing the cross entropy loss. If set to None,
                              no weighting will be done.
        """

        super().__init__(**kwargs)
        self.training_step = int(0)
        self.weight_decay = 1e-4
        self.learning_rate = 1e-4

        self.train_metrics = [AreaUnderRocCurve(), Accuracy()] if self.num_classes == 2 else [Accuracy()]
        self.val_metrics = [AreaUnderRocCurve(), Accuracy()] if self.num_classes == 2 else [Accuracy()]
        self.class_weights = class_weights

    def on_pretrain_routine_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Move metrics and class weights to module device
        for metric in [*self.train_metrics, *self.val_metrics]:
            metric.to(device=pl_module.device)  # type: ignore
        if self.class_weights:
            self.class_weights.to(device=pl_module.device)

        pl_module.non_linear_evaluator = SSLEvaluator(n_input=self.z_dim,
                                                      n_classes=self.num_classes,
                                                      p=self.drop_p,
                                                      n_hidden=self.hidden_dim).to(pl_module.device)
        assert isinstance(pl_module.non_linear_evaluator, torch.nn.Module)
        self.optimizer = torch.optim.Adam(pl_module.non_linear_evaluator.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.weight_decay)

    # Use only one of the transformed images
    @staticmethod
    def to_device(batch: BatchType, device: Union[str, torch.device]) -> Tuple[T, T]:
        (x1, x2, _), y = batch
        x1 = x1.to(device)
        y = y.to(device)
        return x1, y

    def shared_step(self, batch: BatchType, pl_module: pl.LightningModule, is_training: bool) -> T:
        x, y = self.to_device(batch, pl_module.device)
        with torch.no_grad():
            representations = self.get_representations(pl_module, x)
        representations = representations.detach()
        assert isinstance(pl_module.non_linear_evaluator, torch.nn.Module)

        # Run the linear-head with SSL embeddings.
        mlp_preds = pl_module.non_linear_evaluator(representations)
        mlp_loss = F.cross_entropy(mlp_preds, y, weight=self.class_weights)

        with torch.no_grad():
            posteriors = F.softmax(mlp_preds, dim=-1)
            for metric in (self.train_metrics if is_training else self.val_metrics):
                metric(posteriors, y)

        return mlp_loss

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):  # type: ignore
        loss = self.shared_step(batch, pl_module, is_training=False)
        # Log classification metrics
        pl_module.log('ssl/online_val_loss', loss, on_step=False, on_epoch=True, sync_dist=False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:  # type: ignore
        logger = trainer.logger.experiment  # type: ignore 
        loss = self.shared_step(batch, pl_module, is_training=True)

        # update finetune weights
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.training_step += 1

        # log metrics
        logger.add_scalar('ssl/online_train_loss', loss, global_step=self.training_step)


class SSLClassifier(torch.nn.Module):
    """
    SSL Image classifier that combines pre-trained SSL encoder with a trainable linear-head.
    """

    def __init__(self, num_classes: int, encoder: torch.nn.Module, projection: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.encoder.eval(), self.projection.eval()
        self.classifier_head = SSLEvaluator(n_input=get_encoder_output_dim(self.encoder),
                                            n_hidden=None,
                                            n_classes=num_classes,
                                            p=0.20)

    def train(self, mode: bool = True) -> Any:
        self.classifier_head.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.encoder.avgpool, torch.nn.Module)
        with torch.no_grad():
            # Generate representations
            repr = self.encoder(x)

            # Generate image embeddings
            self.projection(repr)

            # Generate class logits
            agg_repr = self.encoder.avgpool(repr) if repr.ndim > 2 else repr
            agg_repr = agg_repr.reshape(agg_repr.size(0), -1).detach()

        return self.classifier_head(agg_repr)


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
        batch = iter(dm.train_dataloader()).next()  # type: ignore 
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
