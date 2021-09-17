#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from InnerEyeDataQuality.deep_learning.metrics.joint_metrics import JointMetrics
from InnerEyeDataQuality.deep_learning.metrics.sample_metrics import SampleMetrics
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


class MetricTracker(object):
    """
    """

    def __init__(self,
                 output_dir: str,
                 num_epochs: int,
                 num_samples_total: int,
                 num_samples_per_epoch: int,
                 num_classes: int,
                 save_tf_events: bool,
                 dataset: Optional[Dataset] = None,
                 name: str = "default_metric",
                 **sample_info_kwargs: Any):
        """
        Class to track model training metrics.
        If a co-teaching model is trained, joint model metrics are stored such as disagreement rate and kl divergence.
        Similarly, it stores loss and logits values on a per sample basis for each epoch for post-training analysis.
        This stored data can be utilised in data selection simulation.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.output_dir = output_dir
        self.name = name
        self.num_classes = num_classes
        self.num_samples_total = num_samples_total
        self.num_samples_per_epoch = num_samples_per_epoch
        clean_targets = dataset.clean_targets if hasattr(dataset, "clean_targets") else None  # type: ignore

        self.joint_model_metrics = JointMetrics(num_samples_total, num_epochs, dataset, **sample_info_kwargs)
        self.sample_metrics = SampleMetrics(name, num_epochs, num_samples_total, num_classes,
                                            clear_labels=clean_targets,  
                                            embeddings_size=None, **sample_info_kwargs)
        self.writer = SummaryWriter(log_dir=output_dir) if save_tf_events else None

    def reset(self) -> None:
        self.sample_metrics.reset()
        self.joint_model_metrics.reset()

    def log_epoch_and_reset(self, epoch: int) -> None:
        # assert np.count_nonzero(~np.isnan(self.sample_metrics.loss_per_sample[:, epoch])) == self.num_samples_per_epoch
        self.sample_metrics.log_results(epoch=epoch, name=self.name, writer=self.writer)
        if self.writer:
            self.joint_model_metrics.log_results(self.writer, epoch, self.sample_metrics)
        # Reset epoch metrics
        self.reset() 

    def append_batch_aggregate(self, epoch: int, logits_x: torch.Tensor, logits_y: torch.Tensor,
                               dropped_cases: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Stores the disagreement stats for co-teaching models
        """
        post_x = torch.softmax(logits_x, dim=-1)
        post_y = torch.softmax(logits_y, dim=-1)
        sym_kl_per_sample = torch.sum(post_x * torch.log(post_x / post_y) + post_y * torch.log(post_y / post_x), dim=-1)

        pred_x = torch.argmax(logits_x, dim=-1)
        pred_y = torch.argmax(logits_y, dim=-1)
        class_pred_disagreement = pred_x != pred_y
        self.joint_model_metrics.kl_divergence_symmetric[indices] = sym_kl_per_sample.cpu().numpy()
        self.joint_model_metrics.prediction_disagreement[indices, epoch] = class_pred_disagreement.cpu().numpy()
        self.joint_model_metrics.case_drop_histogram[dropped_cases.cpu().numpy(), epoch] = True
        self.joint_model_metrics.active = True

    def save_loss(self) -> None:
        output_path = os.path.join(self.output_dir, f'{self.name}_training_stats.npz')
        if hasattr(self.joint_model_metrics, "case_drop_histogram"):
            np.savez(output_path,
                     loss_per_sample=self.sample_metrics.loss_per_sample,
                     logits_per_sample=self.sample_metrics.logits_per_sample,
                     dropped_cases=self.joint_model_metrics.case_drop_histogram[:, :-1])
        else:
            np.savez(output_path,
                     loss_per_sample=self.sample_metrics.loss_per_sample,
                     logits_per_sample=self.sample_metrics.logits_per_sample)
