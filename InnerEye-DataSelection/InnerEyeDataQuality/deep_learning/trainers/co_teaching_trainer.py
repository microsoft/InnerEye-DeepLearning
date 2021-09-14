#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data.dataloader import DataLoader

from InnerEyeDataQuality.configs.config_node import ConfigNode
from InnerEyeDataQuality.deep_learning.architectures.ema import EMA
from InnerEyeDataQuality.deep_learning.collect_embeddings import get_all_embeddings
from InnerEyeDataQuality.deep_learning.graph.classifier import GraphClassifier
from InnerEyeDataQuality.deep_learning.loss import CrossEntropyLoss, consistency_loss
from InnerEyeDataQuality.deep_learning.trainers.model_trainer_base import IndexContainer, Loss, ModelTrainer
from InnerEyeDataQuality.deep_learning.scheduler import ForgetRateScheduler
from InnerEyeDataQuality.deep_learning.utils import create_model
from InnerEyeDataQuality.utils.generic import find_union_set_torch, map_to_device


class CoTeachingTrainer(ModelTrainer):
    """
    Implements co-teaching training using two models
    """

    def __init__(self, config: ConfigNode):
        self.use_teacher_model = config.train.use_teacher_model
        super().__init__(config)

        self.forget_rate_scheduler = ForgetRateScheduler(
            config.scheduler.epochs,
            forget_rate=config.train.co_teaching_forget_rate,
            num_gradual=config.train.co_teaching_num_gradual,
            start_epoch=config.train.resume_epoch if config.train.resume_epoch > 0 else 0,
            num_warmup_epochs=config.train.co_teaching_num_warmup)

        self.joint_metric_tracker = self.train_trackers[0]
        self.use_consistency_loss = config.train.co_teaching_consistency_loss
        self.use_graph = config.train.co_teaching_use_graph
        self.consistency_loss_weight = 0.10
        self.num_models = len(self.models)
        self.loss_fn = CrossEntropyLoss(config)
        self.ema_models = [EMA(self.models[0]), EMA(self.models[1])] if config.train.use_teacher_model else None
        self.graph_classifiers = [GraphClassifier(num_samples=len(self.train_loader.dataset),  # type: ignore
                                                  num_classes=config.dataset.n_classes,
                                                  labels=self.train_loader.dataset.targets,  # type: ignore
                                                  device=config.device)
                                  for _ in range(2)]

    # Create two models for co-teaching
    def get_models(self, config: ConfigNode) -> List[torch.nn.Module]:
        """
        :param config: The job config
        :return: A list of two models to be trained
        """
        return [create_model(config, model_id=0), create_model(config, model_id=1)]

    def update_teacher_models(self) -> None:
        if self.ema_models:
            for i in range(len(self.models)):
                self.ema_models[i].update()

    def deploy_teacher_models(self, inputs: torch.Tensor) -> Optional[List[torch.Tensor]]:
        if not self.ema_models:
            return None
        elif isinstance(inputs, list):
            return [_m.inference(_i) for _m, _i in zip(self.ema_models, inputs)]
        else:
            return [_m.inference(inputs) for _m in self.ema_models]

    @torch.no_grad()
    def _get_samples_for_update(self,
                                outputs: List[torch.Tensor],
                                labels: torch.Tensor,
                                global_indices: torch.Tensor,
                                teacher_logits: Optional[List[torch.Tensor]]) -> Tuple[IndexContainer, IndexContainer]:
        """
        Return a list of indices that should be kept for gradient updates.
        """

        def _get_small_loss_sample_ids(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            num_samples = labels.shape[0]
            num_remember = int((1. - self.forget_rate_scheduler.get_forget_rate) * num_samples)
            per_sample_loss = self.loss_fn(logits, labels, reduction='none')
            ind_sorted = torch.argsort(per_sample_loss)
            return ind_sorted[:num_remember], ind_sorted[num_remember:]

        judge = lambda i: teacher_logits[i] if self.use_teacher_model and teacher_logits else outputs[i]
        ind_1_keep, ind_1_exc = _get_small_loss_sample_ids(judge(0))
        ind_0_keep, ind_0_exc = _get_small_loss_sample_ids(judge(1))

        # Use graph based classifier
        if self.graph_classifiers[0].graph is not None:
            ind_0_keep, ind_0_exc = self.graph_classifiers[0].filter_cases(ind_0_keep, ind_0_exc, global_indices)
            ind_1_keep, ind_1_exc = self.graph_classifiers[1].filter_cases(ind_1_keep, ind_1_exc, global_indices)

        return IndexContainer(ind_0_keep, ind_0_exc), IndexContainer(ind_1_keep, ind_1_exc)

    # Co-teaching loss
    def compute_loss(self,
                     outputs: List[torch.Tensor],
                     labels: torch.Tensor,
                     indices: Optional[Tuple[IndexContainer, IndexContainer]] = None,
                     **kwargs: Any) -> List[Loss]:
        """
        Implements the co-teaching loss using the outputs of two different models
        :param outputs:  A list of logits outputed by each model
        :param labels: The target labels
        :param indices: Sample indices that should be kept and excluded in loss computation (for both models).
        :return: A list of Loss object, each element contains the loss that is fed to the optimizer and a
        tensor of per sample losses
        """
        options = {'ema_logits': None}
        options.update(kwargs)
        ema_logits: Optional[List[torch.Tensor]] = options['ema_logits']

        loss_obj = list()
        indices: Tuple[IndexContainer, IndexContainer] = [  # type: ignore
            IndexContainer(keep=torch.arange(labels.shape[0]), exclude=torch.tensor([], dtype=torch.long)) for _ in
            range(len(outputs))] if indices is None else indices
        assert indices is not None  # for mypy
        assert len(outputs) == len(indices) == 2
        for _output, _index in zip(outputs, indices):
            # The indices to keep for each model is determined by the loss of the other.
            per_sample_loss = self.loss_fn(_output, labels, reduction='none')
            if self.weight is not None:
                per_sample_loss *= self.weight[labels]
            loss_update = torch.mean(per_sample_loss[_index.keep])
            loss_obj.append(Loss(per_sample_loss, loss_update))

        # Consistency loss between predictions on noisy samples
        if self.use_consistency_loss and ema_logits:
            joint_excluded = find_union_set_torch(indices[0].exclude, indices[1].exclude)
            c_loss0 = consistency_loss(outputs[0][joint_excluded], ema_logits[0][joint_excluded])
            c_loss1 = consistency_loss(outputs[1][joint_excluded], ema_logits[1][joint_excluded])
            loss_obj[0].loss += self.consistency_loss_weight * c_loss0
            loss_obj[1].loss += self.consistency_loss_weight * c_loss1

        return loss_obj

    def run_epoch(self, dataloader: DataLoader, epoch: int, is_train: bool = False) -> None:
        """
        Run a training or validation epoch of the base model trainer but also step the forget rate scheduler
        :param dataloader: A dataloader object.
        :param epoch: Current epoch id.
        :param is_train: Whether this is a training epoch or not.
        :param run_inference_on_training_set: If True, record all metrics using the train_trackers
        (even if is_train = False)
        :return:
        """
        for model in self.models:
            model.train() if is_train else model.eval()
        trackers = self.train_trackers if is_train else self.val_trackers
        # Consume input dataloader and update model
        for indices, images, labels in dataloader:
            images, labels = map_to_device(images, self.device), labels.to(self.device)
            outputs = self.forward(images, requires_grad=is_train)
            ema_logits = self.deploy_teacher_models(images)
            selected_ind = self._get_samples_for_update(outputs, labels, indices, ema_logits) if is_train else None
            losses = self.compute_loss(outputs, labels, selected_ind, ema_logits=ema_logits)
            assert (len(outputs) == len(losses)) & (len(outputs) == 2)

            if is_train:
                assert selected_ind is not None
                self.step_optimizers(losses)
                self.update_teacher_models()
                self.joint_metric_tracker.append_batch_aggregate(epoch=epoch,
                                                                 logits_x=outputs[0].detach(),
                                                                 logits_y=outputs[1].detach(),
                                                                 dropped_cases=indices[selected_ind[0].exclude],
                                                                 indices=indices)

            # Collect model embeddings
            embeddings = get_all_embeddings(self.all_model_cnn_embeddings)
            # Log training and validation stats in metric tracker
            for i, (logits, loss) in enumerate(zip(outputs, losses)):
                teacher_logits = ema_logits[i].detach() if self.ema_models else None  # type: ignore
                trackers[i].sample_metrics.append_batch(epoch=epoch,
                                                        logits=logits.detach(),
                                                        labels=labels.detach(),
                                                        loss=loss.loss.item(),
                                                        indices=indices.tolist(),
                                                        per_sample_loss=loss.per_sample_loss.detach(),
                                                        embeddings=embeddings[i],
                                                        teacher_logits=teacher_logits)

        # Adjust forget rate for co-teaching
        if is_train:
            self.forget_rate_scheduler.step()
            if self.use_graph:
                self.graph_classifiers[0].build_graph(embeddings=trackers[1].sample_metrics.embeddings_per_sample)
                self.graph_classifiers[1].build_graph(embeddings=trackers[0].sample_metrics.embeddings_per_sample)

    def save_checkpoint(self, epoch: int) -> bool:
        is_save = super().save_checkpoint(epoch=epoch)
        if is_save and self.ema_models:
            is_last_epoch = epoch == self.config.scheduler.epochs - 1
            suffix = '_last_epoch.pt' if is_last_epoch else f'_epoch_{epoch:d}.pt'
            for i, ema_model in enumerate(self.ema_models):
                save_path = str(self.checkpoint_dir / f'checkpoint_ema_model_{i:d}') + suffix
                ema_model.save_model(save_path)
        return is_save
