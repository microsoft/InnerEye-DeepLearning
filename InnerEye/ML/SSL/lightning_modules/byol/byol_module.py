#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from copy import deepcopy
from typing import Any, Dict, Iterator, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch import Tensor as T
from torch.optim import Adam

from InnerEye.ML.SSL.lightning_modules.byol.byol_models import SiameseArm
from InnerEye.ML.SSL.lightning_modules.byol.byol_moving_average import ByolMovingAverageWeightUpdate
from InnerEye.ML.SSL.utils import SSLModule

SingleBatchType = Tuple[List, T]
BatchType = Union[Dict[SSLModule, SingleBatchType], SingleBatchType]


class BYOLInnerEye(pl.LightningModule):
    """
    Implementation of `Bootstrap Your Own Latent (BYOL) <https://arxiv.org/pdf/2006.07733.pdf>`
    """

    def __init__(self,
                 num_samples: int,
                 learning_rate: float,
                 batch_size: int,
                 encoder_name: str,
                 warmup_epochs: int,
                 use_7x7_first_conv_in_resnet: bool = True,
                 weight_decay: float = 1e-6,
                 **kwargs: Any) -> None:
        """
        Args:
            :param num_samples: Number of samples present in training dataset / dataloader.
            :param learning_rate: Optimizer learning rate.
            :param batch_size: Sample batch size used in gradient updates.
            :param encoder_name: Type of CNN encoder used to extract image embeddings. The options are:
                          {'resnet18', 'resnet50', 'resnet101', 'densenet121'}.
            :param warmup_epochs: Number of epochs for scheduler warm up (linear increase from 0 to base_lr).
            :param use_7x7_first_conv_in_resnet: If True, use a 7x7 kernel (default) in the first layer of resnet.
            If False, replace first layer by a 3x3 kernel. This is required for small CIFAR 32x32 images to not
            shrink them.
            :param weight_decay: L2-norm weight decay.
        """
        super().__init__()
        self.save_hyperparameters()

        self.min_learning_rate = 1e-4
        self.online_network = SiameseArm(encoder_name, use_7x7_first_conv_in_resnet)
        self.target_network = deepcopy(self.online_network)
        self.weight_callback = ByolMovingAverageWeightUpdate()

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        # Add callback for user automatically since it's key to BYOL weight update
        self.weight_callback.on_before_zero_grad(self.trainer, self)

    def forward(self, x: T) -> T:  # type: ignore
        return self.target_network.encoder(x)

    @staticmethod
    def cosine_loss(a: T, b: T) -> T:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        neg_cos_sim = -(a * b).sum(dim=-1).mean()
        return neg_cos_sim

    def shared_step(self, batch: BatchType, batch_idx: int) -> T:
        """
        Returns the BYOL loss for a given batch of images, used in validation
        and training step.
        :param batch: assumed to be a batch a Tuple(List[tensor, tensor, tensor], tensor) to match lightning-bolts
        SimCLRTrainDataTransform API; the first tuple element contains a list of three tensor where the two first
        elements contain two are two strong augmented versions  of the original images in the batch and the last
        is a milder augmentation (ignored here).
        :param batch_idx: index of the batch
        :return: BYOL loss
        """
        batch = batch[SSLModule.ENCODER] if isinstance(batch, dict) else batch
        (img_1, img_2), _ = batch

        # Image 1 to image 2 loss
        _, _, h_img1 = self.online_network(img_1)
        _, _, h_img2 = self.online_network(img_2)
        with torch.no_grad():
            _, z_img1, _ = self.target_network(img_1)
            _, z_img2, _ = self.target_network(img_2)
        loss = 0.5 * (self.cosine_loss(h_img1, z_img2.detach())
                      + self.cosine_loss(h_img2, z_img1.detach()))

        return loss

    def training_step(self, batch: BatchType, batch_idx: int, **kwargs: Any) -> T:  # type: ignore
        loss = self.shared_step(batch, batch_idx)
        self.log_dict({'byol/train_loss': loss, 'byol/tau': self.weight_callback.current_tau})
        return loss

    def validation_step(self, batch: BatchType, batch_idx: int, **kwargs: Any) -> T:  # type: ignore
        loss = self.shared_step(batch, batch_idx)
        self.log_dict({'byol/validation_loss': loss})
        return loss

    def setup(self, *args: Any, **kwargs: Any) -> None:
        global_batch_size = self.trainer.world_size * self.hparams.batch_size  # type: ignore
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size  # type: ignore

    def configure_optimizers(self) -> Any:
        """
        Configures the optimizer to use for training: Adam optimizer with Lars scheduling, excluding certain parameters
        (batch norm and bias of convolution) from weight decay. Apply Linear Cosine Annealing schedule of learning
        rate with warm-up.
        """
        # TRICK 1 (Use lars + filter weights)
        # exclude certain parameters
        parameters = self.exclude_from_wt_decay(self.online_network.named_parameters(),
                                                weight_decay=self.hparams.weight_decay)  # type: ignore
        optimizer = LARSWrapper(Adam(parameters, lr=self.hparams.learning_rate))  # type: ignore

        # Trick 2 (after each step)
        self.hparams.warmup_epochs = self.hparams.warmup_epochs * self.train_iters_per_epoch  # type: ignore
        max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

        linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,  # type: ignore
            max_epochs=max_epochs,
            warmup_start_lr=0,
            eta_min=self.min_learning_rate,
        )

        scheduler = {'scheduler': linear_warmup_cosine_decay, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]

    def exclude_from_wt_decay(self,
                              named_params: Iterator[Tuple[str, T]],
                              weight_decay: float,
                              skip_list: List[str] = ['bias', 'bn']) -> List[Dict[str, Any]]:
        """
        Convolution-Linear bias-terms and batch-norm parameters are excluded from l2-norm weight decay regularisation.
        https://arxiv.org/pdf/2006.07733.pdf Section 3.3 Optimisation and Section F.5.
        """
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.}
        ]

