#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
from torch.utils.data.dataloader import DataLoader

from InnerEyeDataQuality.configs.config_node import ConfigNode
from InnerEyeDataQuality.deep_learning.architectures.ema import EMA
from InnerEyeDataQuality.deep_learning.collect_embeddings import register_embeddings_collector
from InnerEyeDataQuality.deep_learning.dataloader import (get_number_of_samples_per_epoch, get_train_dataloader,
                                                          get_val_dataloader)
from InnerEyeDataQuality.deep_learning.metrics.tracker import MetricTracker
from InnerEyeDataQuality.utils.dataset_utils import get_datasets
from PyTorchImageClassification.optim import create_optimizer
from PyTorchImageClassification.scheduler import create_scheduler


@dataclass(frozen=True)
class IndexContainer:
    keep: torch.Tensor
    exclude: torch.Tensor

@dataclass
class Loss:
    per_sample_loss: torch.Tensor  # the loss value computed for each sample without any modifications
    loss: torch.Tensor  # the loss that is fed to the optimizer; can be derived from per_sample loss given some rule


class ModelTrainer(object):
    """
    ModelTrainer class handles the training of models, logging, saving checkpoints and validation. This class trains one
    or more similar models at a time each having its own optimizer but which can interact in the loss function.
    This is an abstract class for which some methods are left not implemented;
    child classes must implement these methods
    """

    def __init__(self, config: ConfigNode) -> None:
        self.config = config
        self.checkpoint_dir = Path(self.config.train.output_dir) / 'checkpoints'
        self.log_dir = Path(self.config.train.output_dir) / 'logs'
        self.seed = config.train.seed
        self.device = torch.device(config.device)
        train_dataset, val_dataset = get_datasets(config)
        self.weight = torch.tensor([train_dataset.weight, (1-train_dataset.weight)],  # type: ignore
                                   device=self.device, dtype=torch.float) if hasattr(train_dataset, "weight") else None
        self.train_loader = get_train_dataloader(train_dataset, config, seed=self.seed,
                                                 drop_last=config.train.dataloader.drop_last, shuffle=True)
        self.val_loader = get_val_dataloader(val_dataset, config, seed=self.seed)
        self.models = self.get_models(config)
        self.ema_models: Optional[List[EMA]] = None
        self.schedulers = [create_scheduler(config, create_optimizer(config, model), len(self.train_loader))
                           for model in self.models]
        self.train_trackers, self.val_trackers = self._create_metric_trackers(config)
        self.all_trackers = self.train_trackers + self.val_trackers
        self.all_model_cnn_embeddings = register_embeddings_collector(self.models, use_only_in_train=True)

    def _create_metric_trackers(self, config: ConfigNode) -> Tuple[List[MetricTracker], List[MetricTracker]]:
        """
        Creates metric trackers used at model training and validation.
        """
        train_loader = self.train_loader
        val_loader = self.val_loader
        num_models = len(self.models)

        if hasattr(train_loader.dataset, "ambiguity_metric_args"):
            ambiguity_metric_args = train_loader.dataset.ambiguity_metric_args  # type: ignore
        else:
            ambiguity_metric_args = dict()

        save_tf_events = config.tensorboard.save_events if hasattr(config.tensorboard, 'save_events') else True
        metric_kwargs = {"num_epochs": config.scheduler.epochs,
                         "num_classes": config.dataset.n_classes,
                         "save_tf_events": save_tf_events}
        # A dataset without augmentations and not normalized
        dataset_train, dataset_val = get_datasets(config)
        train_trackers = [MetricTracker(dataset=dataset_train,
                                        output_dir=str(self.log_dir / f'model_{i:d}_train'),
                                        num_samples_total=len(train_loader.dataset),  # type: ignore
                                        num_samples_per_epoch=get_number_of_samples_per_epoch(train_loader),
                                        name=f"model_{i}_train",
                                        **{**metric_kwargs, **ambiguity_metric_args})
                                        for i in range(num_models)]
        val_trackers = [MetricTracker(dataset=dataset_val,
                                      output_dir=str(self.log_dir / f'model_{i:d}_val'),
                                      num_samples_total=len(val_loader.dataset),  # type: ignore
                                      num_samples_per_epoch=get_number_of_samples_per_epoch(val_loader),
                                      name=f"model_{i}_valid",
                                      **metric_kwargs) for i in range(num_models)]

        return train_trackers, val_trackers

    def get_models(self, config: ConfigNode) -> List[torch.nn.Module]:
        """
        :param config: The job config
        :return: A list of models to be trained
        """
        raise NotImplementedError

    def forward(self, images: torch.Tensor, requires_grad: bool = True) -> List[torch.Tensor]:
        """
        Performs the forward pass for all models in the ModelTrainer class
        :param images: The input images
        :param requires_grad: Flag to indicate if forward pass required .grad attributes
        :return: A list of the logits for each model
        """
        def _forward(inputs: Union[List[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
            if isinstance(inputs, list):
                return [_model(_input) for _model, _input in zip(self.models, inputs)]
            else:
                return [_model(inputs) for _model in self.models]

        @torch.no_grad()
        def _forward_inference_only(inputs: Union[List[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
            return _forward(inputs)

        if requires_grad:
            return _forward(images)
        else:
            return _forward_inference_only(images)

    def compute_loss(self, outputs: List[torch.Tensor], labels: torch.Tensor,
                     indices: Optional[Tuple[IndexContainer, IndexContainer]] = None) -> Union[List[Loss], Loss]:
        """
        Compute the losses that will be optimized
        :param outputs:  A list of logits outputed by each model
        :param labels: The target labels
        :return: A list of Loss object, each element contains the loss that is fed to the optimizer and a
        tensor of per sample losses
        """
        raise NotImplementedError

    def step_optimizers(self, losses: List[Loss]) -> None:
        """
        Take an optimizer step for every model's optimizer
        :param losses: A list of Loss objects
        :return:
        """
        for loss, scheduler in zip(losses, self.schedulers):
            scheduler.optimizer.zero_grad()
            loss.loss.backward()
            scheduler.optimizer.step()
            scheduler.step()

    def run_epoch(self, dataloader: DataLoader, epoch: int, is_train: bool = False) -> None:
        """
        Run a training or validation epoch
        :param dataloader: A dataloader object
        :param epoch: Current training epoch id
        :param is_train: Whether this is a training epoch or not
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, epoch: int) -> bool:
        """
        Save checkpoints for the models for the current epoch
        :param epoch: The current epoch
        :return: Save success
        """
        is_last_epoch = epoch == self.config.scheduler.epochs - 1
        is_save = is_last_epoch or (epoch > 0 and epoch % self.config.train.checkpoint_period == 0)

        if is_save:
            logging.info(f"Saving model checkpoints, epoch {epoch}")
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for ii in range(len(self.models)):
                path = str(self.checkpoint_dir / f'checkpoint_model_{ii:d}')
                full_save_name = path + '_last_epoch.pt' if is_last_epoch else path + f'_epoch_{epoch:d}.pt'
                state = {'epoch': epoch,
                         'model': self.models[ii].state_dict(),
                         'scheduler': self.schedulers[ii].state_dict(),
                         'optimizer': self.schedulers[ii].optimizer.state_dict()}
                torch.save(state, full_save_name)

        return is_save

    def load_checkpoints(self, restore_scheduler: bool, epoch: Optional[int] = None) -> None:
        """
        If epoch is not specified, latest checkpoint files are loaded to restore the state
        of model weights and optimisers
        :param restore_scheduler (bool): Restores the state of optimiser and scheduler from checkpoint
        :param epoch (int): Training epoch id.
        """
        suffix = f'_epoch_{epoch:d}.pt' if epoch else '_last_epoch.pt'

        for ii in range(len(self.models)):
            path = str(self.checkpoint_dir / f'checkpoint_model_{ii:d}') + suffix
            logging.info(f"Loading model-{ii} from checkpoint:\n {path}")
            state = torch.load(str(path))

            self.models[ii].load_state_dict(state['model'])
            if restore_scheduler:
                self.schedulers[ii].load_state_dict(state['scheduler'])
                self.schedulers[ii].optimizer.load_state_dict(state['optimizer'])
            if epoch is not None:
                logging.info(f"Model is loaded from epoch: {epoch}")
                assert state['epoch'] == epoch

        if self.ema_models:
            for ii in range(len(self.ema_models)):
                path = str(self.checkpoint_dir / f'checkpoint_ema_model_{ii:d}') + suffix
                logging.info(f"Loading ema teacher model-{ii} from checkpoint:\n {path}")
                self.ema_models[ii].restore_from_checkpoint(path)

    def run_training(self) -> None:
        """
        Perform model training.
        Model/s specified in config are trained for `num_epoch` epochs and results are stored in tf events.
        """
        num_epochs = self.config.scheduler.epochs
        epoch_range = range(num_epochs)
        if self.config.train.resume_epoch > 0:
            resume_epoch = self.config.train.resume_epoch
            epoch_range = range(resume_epoch + 1, num_epochs)
            self.load_checkpoints(restore_scheduler=self.config.train.restore_scheduler, epoch=resume_epoch)

        # Model evaluation - startup
        logging.info("Running evaluation on the validation set before training ...")
        self.run_epoch(self.val_loader, is_train=False, epoch=0)
        self.val_trackers[0].log_epoch_and_reset(epoch=0)

        # Model training loop
        for epoch in epoch_range:
            epoch_start = time.time()
            logging.info('\n' + f'Epoch {epoch:d}')
            self.run_epoch(self.train_loader, is_train=True, epoch=epoch)
            self.run_epoch(self.val_loader, is_train=False, epoch=epoch)
            self.save_checkpoint(epoch)

            for _t in self.all_trackers:
                _t.log_epoch_and_reset(epoch)
            logging.info('Epoch time: {0:.2f} secs'.format(time.time() - epoch_start))

        # Store loss values for post-training analysis
        for _t in self.all_trackers:
            _t.save_loss()

    def run_inference(self, dataloader: Any, use_mc_sampling: bool = False) -> List[MetricTracker]:
        """
        Deployment of pre-trained model.
        """
        dataset = dataloader.dataset
        trackers = [MetricTracker(os.path.join(self.config.train.output_dir, f'model_{i:d}_inference'),
                                  num_samples_total=len(dataset),
                                  num_samples_per_epoch=len(dataset),
                                  name=f"model_{i}_inference",
                                  num_epochs=1,
                                  num_classes=self.config.dataset.n_classes,
                                  save_tf_events=False) for i in range(len(self.models))]

        for model in self.models:
            model.eval()
            if use_mc_sampling:
                logging.info("Applying MC sampling at inference time.")
                dropout_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Dropout)]
                for _layer in dropout_layers:
                    _layer.training = True

        with torch.no_grad():
            for indices, images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.forward(images, requires_grad=False)
                losses = self.compute_loss(outputs, labels, indices=None)
                if not isinstance(losses, List):
                    losses = [losses]

                # Log training and validation stats in metric tracker
                for i, (logits, loss) in enumerate(zip(outputs, losses)):
                    trackers[i].sample_metrics.append_batch(epoch=0,
                                                            logits=logits.detach(),
                                                            labels=labels.detach(),
                                                            loss=loss.loss.item(),
                                                            indices=indices.cpu().tolist(),
                                                            per_sample_loss=loss.per_sample_loss.detach())
        return trackers
