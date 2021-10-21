#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Union

import yacs
import torch
from torch import Tensor as T
import torch.nn.functional as F


def tanh_loss(logits: T) -> T:
    """
    Penalises sparse logits with large values
    "Bootstrap Your Own Latent A New Approach to Self-Supervised Learning" Supplementary C.1
    """
    alpha = 10
    tclip = alpha * torch.tanh(logits / alpha)
    return torch.mean(torch.pow(tclip, 2.0))

# Computes consistency loss between unlabelled or excluded points
def consistency_loss(logits_source: T, logits_target: T) -> Union[T, float]:
    """
    Class probability consistency loss based on total variation. 
    """
    if (logits_source.numel() > 0) & (logits_source.shape == logits_target.shape):
        _prob_source = torch.softmax(logits_source, dim=-1)
        _prob_target = torch.softmax(logits_target, dim=-1)
        loss = torch.mean(torch.norm(_prob_source - _prob_target.detach(), p=2, dim=-1))
        return loss
    else:
        return 0.0

def early_regularisation_loss(student_logits: torch.Tensor, ema_logits: torch.Tensor) -> torch.Tensor:
    """
    Implements early regularisation loss term proposed in:
    https://arxiv.org/abs/2007.00151
    """
    posteriors_teacher = torch.softmax(ema_logits, dim=-1).detach()
    posteriors_student = torch.softmax(student_logits, dim=-1)
    inner_prod = torch.sum(posteriors_teacher * posteriors_student, dim=-1)
    early_regularisation = torch.mean(torch.log(1 - inner_prod))

    return early_regularisation

def onehot_encoding(label: torch.Tensor, n_classes: int) -> torch.Tensor:
    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(1, label.view(-1, 1), 1)

class CrossEntropyLoss:
    """
    Cross entropy loss - implements label smoothing 
    """

    def __init__(self, config: yacs.config.CfgNode):
        self.n_classes = config.dataset.n_classes
        self.use_label_smoothing = config.augmentation.use_label_smoothing
        self.epsilon = config.augmentation.label_smoothing.epsilon

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        if self.use_label_smoothing:
            device = predictions.device
            onehot = onehot_encoding(targets, self.n_classes).type_as(predictions).to(device)
            targets = onehot * (1 - self.epsilon) + torch.ones_like(onehot).to(device) * self.epsilon / self.n_classes
            logp = F.log_softmax(predictions, dim=1)
            loss_per_sample = torch.sum(-logp * targets, dim=1)
            if reduction == 'none':
                return loss_per_sample
            elif reduction == 'mean':
                return loss_per_sample.mean()
            else:
                raise NotImplementedError
        else:
            return F.cross_entropy(predictions, targets, reduction=reduction)
