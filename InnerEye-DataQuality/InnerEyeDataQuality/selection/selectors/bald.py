#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from typing import Any, Optional

import numpy as np
import torch
from InnerEyeDataQuality.deep_learning.dataloader import get_train_dataloader, get_val_dataloader
from InnerEyeDataQuality.deep_learning.model_inference import NUM_MULTIPLE_RUNS
from InnerEyeDataQuality.deep_learning.trainers.model_trainer_base import ModelTrainer
from InnerEyeDataQuality.deep_learning.trainers.vanilla_trainer import VanillaTrainer
from InnerEyeDataQuality.evaluation.metrics import compute_model_disagreement_score
from InnerEyeDataQuality.selection.selectors.base import SampleSelector
from pytorch_image_classification import create_optimizer


class BaldSelector(SampleSelector):
    """
    Selects samples based on the BALD criterion.
    """

    def __init__(self, posteriors: np.ndarray, num_samples: int, trainer: Optional[ModelTrainer] = None, **kwargs: Any):
        super().__init__(num_samples=num_samples, allow_repeat_samples=False, **kwargs)

        assert posteriors.shape[2] == self.num_samples
        assert posteriors.shape[1] == 1
        posteriors = np.squeeze(posteriors, axis=1)

        self.posteriors = posteriors
        self.num_samples = num_samples
        self.get_bald_scores = lambda p: compute_model_disagreement_score(p)
        self.bald_scores = self.get_bald_scores(posteriors)

        # Active cleaning parameters - model updates:
        self.current_labels = np.zeros_like(posteriors[0])
        self.update_milestones = [1000 * ii for ii in range(1, 20)]
        self.trainer = trainer

    def get_relabelling_scores(self, current_labels: np.ndarray) -> np.ndarray:
        relabelling_score = np.copy(self.bald_scores)  # large bald values correspond to large model disagreement.
        self.current_labels = current_labels.copy()  # keep a copy of current state of the labels

        return relabelling_score

    def update_beliefs(self, iteration_id: int, *args: Any, **kwargs: Any) -> None:

        if (self.trainer is None) or (iteration_id < self.update_milestones[0]):
            return
        self.update_milestones.pop(0)

        # Inputs
        dataset = self.trainer.train_loader.dataset
        learning_rate = 0.0025
        num_epochs = 20

        # Create a dataset and dataloader object.
        logging.info(f"Updating posteriors, number of relabels count: {iteration_id}")
        assert hasattr(dataset, "targets")
        assert isinstance(self.trainer, VanillaTrainer)
        dataset.targets = np.argmax(self.current_labels, axis=1)  # type: ignore

        # Assign a new optimiser and scheduler for fine-tuning.
        cfg = self.trainer.config.clone()
        cfg.defrost()
        cfg.train.base_lr = learning_rate
        self.trainer.schedulers = [torch.optim.lr_scheduler.LambdaLR(
            create_optimizer(cfg, model), lambda x: 1) for model in self.trainer.models] 

        # Run the trainer for one epoch.
        train_dataloader = get_train_dataloader(dataset, self.trainer.config, seed=self.trainer.config.train.seed,
                                                drop_last=False, shuffle=True)
        for epoch in range(num_epochs):
            self.trainer.run_epoch(dataloader=train_dataloader, epoch=epoch, is_train=True)
            for _t in self.trainer.train_trackers:
                _t.log_epoch_and_reset(epoch=epoch)

        # Run inference and update the posteriors and bald scores:
        all_posteriors = list()
        inference_dataloader = get_val_dataloader(dataset, self.trainer.config, seed=self.trainer.config.train.seed)
        for run_ind in range(NUM_MULTIPLE_RUNS):
            trackers = self.trainer.run_inference(dataloader=inference_dataloader, use_mc_sampling=True)
            all_posteriors.append(trackers[0].sample_metrics.probabilities)

        logging.info("Model fine-tuned - Updating model posteriors and bald scores")
        self.posteriors = np.stack(all_posteriors, axis=0)
        self.bald_scores = self.get_bald_scores(self.posteriors)

        # Collect the performance on a test set:
        val_tracker = self.trainer.run_inference(dataloader=self.trainer.val_loader, use_mc_sampling=False)[0]
        logging.info(f"Test avg acc: {val_tracker.sample_metrics.get_average_accuracy()}")
        logging.info(f"Test avg loss: {val_tracker.sample_metrics.get_average_loss()}")
