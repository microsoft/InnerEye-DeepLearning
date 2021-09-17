#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch

from InnerEyeDataQuality.deep_learning.dataloader import get_train_dataloader, get_val_dataloader
from InnerEyeDataQuality.deep_learning.trainers.model_trainer_base import ModelTrainer
from InnerEyeDataQuality.evaluation.metrics import cross_entropy, max_prediction_error
from InnerEyeDataQuality.selection.selectors.base import SampleSelector
from PyTorchImageClassification.optim import create_optimizer
from InnerEyeDataQuality.deep_learning.scheduler import ForgetRateScheduler


class LabelBasedDecisionRule(Enum):
    CROSS_ENTROPY = 1
    INV = 2
    JOINT = 3


class LabelDistributionBasedSampler(SampleSelector):
    """
    TBD
    """

    def __init__(self,
                 num_samples: int,
                 decision_rule: LabelBasedDecisionRule,
                 allow_repeat_samples: bool = False,
                 embeddings: Optional[np.ndarray] = None,
                 trainer: Optional[ModelTrainer] = None,
                 **kwargs: Any):
        super().__init__(num_samples=num_samples, allow_repeat_samples=allow_repeat_samples, embeddings=embeddings,
                         **kwargs)

        self.trainer = trainer
        self.decision_rule = decision_rule

    def get_predicted_label_distribution(self, current_labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_relabelling_scores(self, current_labels: np.ndarray) -> np.ndarray:
        predicted_label_distribution = self.get_predicted_label_distribution(current_labels)
        current_distribution = current_labels / np.sum(current_labels, axis=-1, keepdims=True)
        return cross_entropy(predicted_label_distribution, current_distribution)


class PosteriorBasedSelector(LabelDistributionBasedSampler):
    """
    Selects samples based on the label posteriors computed by a model
    """

    def __init__(self,
                 predicted_label_distribution: np.ndarray,
                 num_samples: int,
                 embeddings: Optional[np.ndarray] = None,
                 allow_repeat_samples: bool = False,
                 **kwargs: Any):
        super().__init__(num_samples=num_samples, allow_repeat_samples=allow_repeat_samples,
                         embeddings=embeddings, **kwargs)
        assert (len(predicted_label_distribution) == self.num_samples)
        self.predicted_label_distribution = predicted_label_distribution
        self.exp_num_relabels = np.zeros(predicted_label_distribution.shape[0])
        self.current_labels = np.zeros_like(predicted_label_distribution)

        # Active relabelling
        self.update_milestones = [1000 * ii for ii in range(1, 20)]
        self.already_added_data_points = set()  # type: ignore
        self.total_updated_cases = 0

    def get_predicted_label_distribution(self, current_labels: np.ndarray) -> np.ndarray:
        return self.predicted_label_distribution

    def get_relabelling_scores(self, current_labels: np.ndarray) -> np.ndarray:
        current_distribution = current_labels / np.sum(current_labels, axis=-1, keepdims=True)

        # Formulation based on cross-entropy and relabelling calculation
        if self.decision_rule == LabelBasedDecisionRule.INV:
            relabel_scores = self.get_exp_num_relabels(current_labels)
            current_distribution = current_labels / np.sum(current_labels, axis=-1, keepdims=True)
            mislabelled_scores = max_prediction_error(self.predicted_label_distribution, current_distribution)
            relabelling_score = mislabelled_scores / relabel_scores

        # Formulation based on cross-entropy and self-entropy
        elif self.decision_rule == LabelBasedDecisionRule.JOINT:
            ce_qp = cross_entropy(self.predicted_label_distribution, current_distribution)
            h_p = cross_entropy(self.predicted_label_distribution, self.predicted_label_distribution)
            gamma = 0.30
            h_p = -1 * np.clip(gamma - h_p, np.min(gamma - h_p), 0)
            relabelling_score = ce_qp - h_p
            self.stats['ambiguity'] = h_p
            self.stats['label_correctness'] = ce_qp

        # Cross-entropy only
        else:
            relabelling_score = cross_entropy(self.predicted_label_distribution, current_distribution)

        # Keep a copy of current labels and scoring stats
        self.current_labels = current_labels.copy()  # keep a copy of current state of the labels
        self.stats['relabelling_score'] = np.mean(relabelling_score)

        return relabelling_score

    def get_exp_num_relabels(self, current_labels: np.ndarray) -> np.ndarray:
        all_classes = np.arange(current_labels.shape[1])
        majority_label = np.argmax(current_labels, axis=1)
        # Only compute if necessary i.e. if current_labels is different from last round for a given sample
        rows = np.unique(np.where(current_labels != self.current_labels)[0])
        logging.info(f"Update {rows} subjects")
        for ix in rows:
            # get the chosen label and the predicted distribution for each image
            chosen = majority_label[ix]
            dist = self.predicted_label_distribution[ix, :]

            # if the distribution has a unit mass, use this info and don't use recursion
            if np.max(dist) == 1:
                self.exp_num_relabels[ix] = 1 if dist[chosen] == 1 else 2
            else:
                # then get the expected number of relabels
                all_classes_curr = all_classes[~np.isin(all_classes, chosen)]
                exp_relabel = 1 + self.exp_num_fun3(all_classes_curr, dist, ctr=0, max_levels=3)
                self.exp_num_relabels[ix] = exp_relabel
        return self.exp_num_relabels

    def exp_num_fun3(self, remaining_classes: np.ndarray, dist: np.ndarray, ctr: int, max_levels: int = 2) -> float:
        # keep track of how many recursive calls were made
        ctr = ctr + 1
        if len(remaining_classes) == 0:
            return 0
        else:
            if ctr >= max_levels:  # if you reached the max depth in recursion
                return float(np.sum(dist[remaining_classes]))
            else:  # if you have not reached max depth yet
                sumval = 0
                for j in remaining_classes:
                    rem_classes_curr = remaining_classes[~np.isin(remaining_classes, j)]
                    sumval += dist[j] * (1 + self.exp_num_fun3(rem_classes_curr, dist, ctr, max_levels))
                return sumval

    def update_beliefs(self, iteration_id: int, *args: Any, **kwargs: Any) -> None:
        """
        """
        if (self.trainer is None) or (iteration_id < self.update_milestones[0]):
            return
        self.update_milestones.pop(0)

        # Inputs
        dataset = self.trainer.train_loader.dataset

        learning_rate = 0.0025  # for CXR experiments: 1e-5 for SSL, 1e-6 for vanilla
        num_epochs = 20
        logging.info(f"Learning rate: {learning_rate} - num finetuning epochs: {num_epochs}")

        # Create a dataset and dataloader object.
        logging.info(f"Updating posteriors, number of relabels count: {iteration_id}")
        assert hasattr(dataset, "targets")

        old = dataset.targets.copy()  # type: ignore
        dataset.targets = np.argmax(self.current_labels, axis=1)  # type: ignore
        # Changed since last update beliefs
        number_changed = (old != dataset.targets).sum()  # type: ignore
        # Total changed
        self.total_updated_cases += number_changed
        logging.info(f"Total updated labels {number_changed}")
        # Assign a new optimiser and scheduler for fine-tuning.
        cfg = self.trainer.config.clone()
        cfg.defrost()
        cfg.train.base_lr = learning_rate
        if cfg.train.use_co_teaching:
            gain_accuracy = float(self.total_updated_cases) / len(dataset)  # type: ignore
            logging.info(f"Total gain accuracy since start {gain_accuracy}")
            # Update our coteaching belief
            cfg.train.co_teaching_forget_rate -= gain_accuracy
            logging.info(f"New co-teaching drop rate {cfg.train.co_teaching_forget_rate}")
            self.trainer.forget_rate_scheduler = ForgetRateScheduler(  # type: ignore
                cfg.scheduler.epochs,
                forget_rate=cfg.train.co_teaching_forget_rate,
                num_gradual=0,
                start_epoch=0,
                num_warmup_epochs=0)
        self.trainer.schedulers = [torch.optim.lr_scheduler.LambdaLR(
            create_optimizer(cfg, model), lambda x: 1) for model in self.trainer.models]

        train_dataloader = get_train_dataloader(dataset, self.trainer.config, seed=self.trainer.config.train.seed,
                                                drop_last=False, shuffle=True)
        for epoch in range(num_epochs):
            self.trainer.run_epoch(dataloader=train_dataloader, epoch=epoch, is_train=True)
            for _t in self.trainer.train_trackers:
                _t.log_epoch_and_reset(epoch=epoch)
            logging.info(f"Epoch {epoch} done")

        # Run inference and update `predicted_label_distribution`.
        inference_dataloader = get_val_dataloader(dataset, self.trainer.config, seed=self.trainer.config.train.seed)
        trackers = self.trainer.run_inference(dataloader=inference_dataloader)
        probs = [metric_tracker.sample_metrics.probabilities for metric_tracker in trackers]
        self.predicted_label_distribution = np.mean(probs, axis=0)

        # Collect the performance on a test set:
        val_tracker = self.trainer.run_inference(dataloader=self.trainer.val_loader, use_mc_sampling=False)[0]
        logging.info(f"{self.name} - Test avg acc: {val_tracker.sample_metrics.get_average_accuracy()}")
        logging.info(f"{self.name} - Test avg loss: {val_tracker.sample_metrics.get_average_loss()}")
