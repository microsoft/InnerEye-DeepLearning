#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import List, Optional, Tuple

import torch

from InnerEyeDataQuality.configs.config_node import ConfigNode
from InnerEyeDataQuality.deep_learning.collect_embeddings import get_all_embeddings
from InnerEyeDataQuality.deep_learning.trainers.model_trainer_base import IndexContainer, Loss, ModelTrainer
from InnerEyeDataQuality.deep_learning.utils import create_model
from InnerEyeDataQuality.deep_learning.loss import tanh_loss
from torch.utils.data.dataloader import DataLoader
from InnerEyeDataQuality.deep_learning.loss import CrossEntropyLoss


class VanillaTrainer(ModelTrainer):
    """
    Implements vanilla cross entropy training with one model
    """

    def __init__(self, config: ConfigNode):
        super().__init__(config)
        self.tanh_regularisation = config.train.tanh_regularisation
        self.loss_fn = CrossEntropyLoss(config)

    def get_models(self, config: ConfigNode) -> List[torch.nn.Module]:
        """
        :param config: The job config
        :return: A list with one model to be trained
        """
        return [create_model(config, model_id=0)]

    def compute_loss(self, outputs: List[torch.Tensor], labels: torch.Tensor,
                     indices: Optional[Tuple[IndexContainer, IndexContainer]] = None) -> Loss:
        """
        Implements the standard cross-entropy loss using one model
        :param outputs:  A list of logits outputed by each model
        :param labels: The target labels
        :return: A list of Loss object, each element contains the loss that is fed to the optimizer and a
        tensor of per sample losses
        """
        logits = outputs[0]
        per_sample_loss = self.loss_fn(predictions=logits, targets=labels, reduction='none')
        if self.weight is not None:
            per_sample_loss *= self.weight[labels]
        loss = torch.mean(per_sample_loss)

        if self.tanh_regularisation != 0.0:
            loss += self.tanh_regularisation * tanh_loss(logits)

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
            losses = self.compute_loss(outputs, labels)
            if is_train:
                self.step_optimizers([losses])

            # Log training and validation stats in metric tracker
            tracker = self.train_trackers[0] if is_train else self.val_trackers[0]
            tracker.sample_metrics.append_batch(epoch, outputs[0].detach(), labels.detach(), losses.loss.item(),
                                                indices.cpu().tolist(), losses.per_sample_loss.detach(), embeddings)
