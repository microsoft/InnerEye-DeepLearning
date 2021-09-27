#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from torch.types import Device
from torch.utils.data import DataLoader

from InnerEyeDataQuality.configs.config_node import ConfigNode
from InnerEyeDataQuality.deep_learning.collect_embeddings import get_all_embeddings
from InnerEyeDataQuality.deep_learning.loss import CrossEntropyLoss
from InnerEyeDataQuality.deep_learning.trainers.model_trainer_base import Loss
from InnerEyeDataQuality.deep_learning.trainers.vanilla_trainer import VanillaTrainer


class ELRTrainer(VanillaTrainer):
    def __init__(self, config: ConfigNode):
        super().__init__(config)
        self.num_classes = config.dataset.n_classes
        self.loss_fn = {"TRAIN": ELRLoss(num_examples=self.train_trackers[0].num_samples_total,
                                         num_classes=config.dataset.n_classes,
                                         device=self.device),
                        "VAL": CrossEntropyLoss(config)}

    def compute_loss(self, is_train: bool, outputs: List[torch.Tensor], labels: torch.Tensor,
                     indices: torch.Tensor = None) -> Loss:
        """
        Implements the standard cross-entropy loss using one model
        :param outputs:  A list of logits outputed by each model
        :param labels: The target labels
        :return: A list of Loss object, each element contains the loss that is fed to the optimizer and a
        tensor of per sample losses
        """
        logits = outputs[0]
        if is_train:
            per_sample_loss = self.loss_fn["TRAIN"](predictions=logits, targets=labels, indices=indices)  # type: ignore
        else:
            per_sample_loss = self.loss_fn["VAL"](predictions=logits, targets=labels, reduction='none')  # type: ignore
        loss = torch.mean(per_sample_loss)
        return Loss(per_sample_loss, loss)

    def run_epoch(self, dataloader: DataLoader, epoch: int, is_train: bool = False) -> None:
        """
        Run a training or validation epoch of the base model trainer but also step the forget rate scheduler
        :param dataloader: A dataloader object
        :param epoch: Current epoch id.
        :param is_train: Whether this is a training epoch or not
        :param run_inference_on_training_set: If True, record all metrics using the train_trackers
        (even if is_train = False)
        :return:
        """

        for model in self.models:
            model.train() if is_train else model.eval()

        for indices, images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.forward(images, requires_grad=is_train)
            embeddings = get_all_embeddings(self.all_model_cnn_embeddings)[0]
            losses = self.compute_loss(is_train, outputs, labels, indices)
            if is_train:
                self.step_optimizers([losses])

            # Log training and validation stats in metric tracker
            tracker = self.train_trackers[0] if is_train else self.val_trackers[0]
            tracker.sample_metrics.append_batch(epoch, outputs[0].detach(), labels.detach(), losses.loss.item(),
                                                indices.cpu().tolist(), losses.per_sample_loss.detach(), embeddings)


class ELRLoss(nn.Module):
    """
    Adapted from https://github.com/shengliu66/ELR.
    """

    def __init__(self, num_examples: int, num_classes: int, device: Device, beta: float = 0.9, _lambda: float = 3):
        super().__init__()
        self.num_classes = num_classes
        self.targets = torch.zeros(num_examples, self.num_classes, device=device)
        self.beta = beta
        self._lambda = _lambda

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        y_pred = F.softmax(predictions, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        self.targets[indices] = self.beta * self.targets[indices] + (1 - self.beta) * (
                (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        ce_loss = F.cross_entropy(predictions, targets, reduction="none")
        elr_reg = (1 - (self.targets[indices] * y_pred).sum(dim=1)).log()
        per_sample_loss = ce_loss + self._lambda * elr_reg
        return per_sample_loss
