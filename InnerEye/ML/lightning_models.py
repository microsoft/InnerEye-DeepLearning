#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import time
from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningModule

from InnerEye.Common.metrics_dict import MetricType, MetricsDict
from InnerEye.ML import metrics
from InnerEye.ML.config import BACKGROUND_CLASS_NAME, SegmentationModelBase
from InnerEye.ML.dataset.sample import CroppedSample
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.utils import image_util, metrics_util, model_util
from InnerEye.ML.utils.lr_scheduler import SchedulerWithWarmUp


MAX_ITEM_LOAD_TIME_SEC = 0.5
MAX_LOAD_TIME_WARNINGS = 3


class InnerEyeLightning(LightningModule):
    def __init__(self, config: DeepLearningConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        # Timers for monitoring data loading time
        self.epoch_start_time = 0
        self.item_start_time = 0
        self.num_load_time_warnings = 0
        self.num_load_time_exceeded = 0
        self.num_batches = 0
        self.total_extra_load_time = 0.0
        self.total_load_time = 0.0
        # Metrics for all epochs
        self.train_metrics_per_epoch: List[MetricsDict] = []
        self.validation_metrics_per_epoch: List[MetricsDict] = []
        # This will be initialized correctly in epoch_start
        self.metrics = MetricsDict()

    def configure_optimizers(self):
        # TODO: This will be the same for all types of models, can this be in base class?
        optimizer = model_util.create_optimizer(self.config, self.model.parameters())
        l_rate_scheduler = SchedulerWithWarmUp(self.config, optimizer)
        return [optimizer], [l_rate_scheduler]

    def on_train_epoch_end(self, outputs) -> None:
        self.epoch_end(is_training=True)

    def on_validation_epoch_end(self) -> None:
        self.epoch_end(is_training=False)

    def epoch_end(self, is_training: bool) -> None:
        epoch_time_seconds = time.time() - self.epoch_start_time
        status = "training" if is_training else "validation"
        logging.info(f"Epoch {self.current_epoch} {status} took {epoch_time_seconds:0.2f}sec, from which waiting for "
                     f"data took {self.total_load_time:0.2f} sec total. {self.num_batches} minibatches in total.")
        if self.num_load_time_exceeded > 0:
            logging.warning("The dataloaders were not fast enough to always supply the next batch in less than "
                            f"{MAX_ITEM_LOAD_TIME_SEC}sec.")
            logging.warning(
                f"In this epoch, {self.num_load_time_exceeded} out of {self.num_batches} batches exceeded the load "
                f"time threshold. Total loading time for the slow batches was {self.total_extra_load_time:0.2f}sec.")
        # TODO antonsc: Make that safer
        learning_rate = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
        result = self.aggregate_metrics()
        result.add_metric(MetricType.LEARNING_RATE, learning_rate)
        result.add_metric(MetricType.SECONDS_PER_EPOCH, epoch_time_seconds)
        logger = self.config.azure_loggers_train if is_training else self.config.azure_loggers_val
        df_logger = self.config.data_frame_loggers.train_epoch_metrics if is_training \
            else self.config.data_frame_loggers.val_epoch_metrics
        metrics.store_epoch_metrics(logger,
                                    df_logger,
                                    self.current_epoch,
                                    result,
                                    self.config)
        if is_training:
            self.train_metrics_per_epoch.append(result)
        else:
            self.validation_metrics_per_epoch.append(result)

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.batch_start(batch_idx=batch_idx, is_training=True)

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.batch_start(batch_idx=batch_idx, is_training=False)

    def batch_start(self, batch_idx: int, is_training: bool) -> None:
        # TODO antonsc: Do we want that only on rank zero?
        item_finish_time = time.time()
        item_load_time = item_finish_time - self.item_start_time
        self.total_load_time += item_load_time
        # Having slow minibatch loading is OK in the very first batch of the every epoch, where processes
        # are spawned. Later, the load time should be zero.
        status_string = "training" if is_training else "validation"
        if batch_idx == 0:
            logging.info(f"Loaded the first minibatch of {status_string} data in {item_load_time:0.2f} sec.")
        elif item_load_time > MAX_ITEM_LOAD_TIME_SEC:
            self.num_load_time_exceeded += 1
            self.total_extra_load_time += item_load_time
            if self.num_load_time_warnings < MAX_LOAD_TIME_WARNINGS:
                logging.warning(f"Loading {status_string} minibatch {batch_idx} took {item_load_time:0.2f} sec. "
                                "This can mean that there are not enough data loader worker processes, or that there "
                                "is a performance problem in loading. This warning will be printed at most "
                                f"{MAX_LOAD_TIME_WARNINGS} times.")
                self.num_load_time_warnings += 1

    def on_batch_end(self):
        self.item_start_time = time.time()
        self.num_batches += 1

    def reset_timers(self) -> None:
        self.epoch_start_time = time.time()
        self.item_start_time = time.time()
        self.num_load_time_warnings = 0
        self.num_load_time_exceeded = 0
        self.total_extra_load_time = 0.0
        self.total_load_time = 0.0
        self.num_batches = 0


class SegmentationModel(InnerEyeLightning):
    def __init__(self, config: SegmentationModelBase, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.model = config.create_model()
        self.loss_fn = model_util.create_segmentation_loss_function(config)

    def forward(self, patches) -> torch.Tensor:
        return self.logits_to_posterior(self.model(patches))

    def logits_to_posterior(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply Softmax on dimension 1 (Class) to map model output into a posterior probability distribution [0,1]
        """
        return torch.nn.functional.softmax(logits, dim=1)

    def on_train_epoch_start(self) -> None:
        self.epoch_start()

    def on_validation_epoch_start(self) -> None:
        self.epoch_start()

    def epoch_start(self) -> None:
        self.reset_timers()
        self.metrics = MetricsDict(hues=[BACKGROUND_CLASS_NAME] + self.config.ground_truth_ids)

    def training_step(self,
                      sample: Dict[str, Any],
                      batch_index: int):
        return self.training_or_validation_step(sample, batch_index, is_training=True)

    def validation_step(self,
                        sample: Dict[str, Any],
                        batch_index: int):
        return self.training_or_validation_step(sample, batch_index, is_training=False)

    def training_or_validation_step(self,
                                    sample: Dict[str, Any],
                                    batch_index: int,
                                    is_training: bool):
        """
        Runs training for a single minibatch of training data, and computes all metrics.
        :param sample: The batched sample on which the model should be trained.
        :param batch_index: The index of the present batch (supplied only for diagnostics).
        :param epoch: The number of the present epoch.
        """
        cropped_sample: CroppedSample = CroppedSample.from_dict(sample=sample)
        labels = cropped_sample.labels_center_crop

        mask = cropped_sample.mask_center_crop if is_training else None
        logits = self.model(cropped_sample.image)
        loss = self.loss_fn(logits, labels)

        # apply Softmax on dimension 1 (Class) to map model output into a posterior probability distribution [0,1]
        posteriors = self.logits_to_posterior(logits)

        # apply mask if required
        if mask is not None:
            posteriors = image_util.apply_mask_to_posteriors(posteriors=posteriors, mask=mask)

        # post process posteriors to compute result
        segmentations = image_util.posteriors_to_segmentation(posteriors=posteriors).data.cpu().numpy()

        dice_for_all_classes = metrics.compute_dice_across_patches(
            segmentation=torch.tensor(segmentations).long(),
            ground_truth=labels,
            use_cuda=self.config.use_gpu,
            allow_multiple_classes_for_each_pixel=True).cpu().numpy()
        foreground_voxels = metrics_util.get_number_of_voxels_per_class(cropped_sample.labels)
        # loss is a scalar, also when running the forward pass over multiple crops.
        # dice_for_all_structures has one row per crop.

        # store metrics per batch
        self.metrics.add_metric(MetricType.LOSS, loss.item())
        for i, ground_truth_id in enumerate(self.metrics.get_hue_names(include_default=False)):
            for b in range(dice_for_all_classes.shape[0]):
                self.metrics.add_metric(MetricType.DICE, dice_for_all_classes[b, i].item(),
                                        hue=ground_truth_id, skip_nan_when_averaging=True)
            self.metrics.add_metric(MetricType.VOXEL_COUNT, foreground_voxels[i], hue=ground_truth_id)
        # store diagnostics per batch
        center_indices = cropped_sample.center_indices
        if isinstance(center_indices, torch.Tensor):
            center_indices = center_indices.cpu().numpy()
        self.metrics.add_diagnostics(MetricType.PATCH_CENTER.value, center_indices.copy())
        # if self.train_val_params.in_training_mode:
        #     # store the sample train patch from this epoch for visualization
        #     if batch_index == self.example_to_save and self.config.store_dataset_sample:
        #         _store_dataset_sample(self.config, self.train_val_params.epoch, forward_pass_result,
        #                               cropped_sample)
        metric_name = 'train_loss' if is_training else "val_loss"
        self.log(metric_name, loss)
        return loss

    def aggregate_metrics(self) -> MetricsDict:
        return metrics.aggregate_segmentation_metrics(self.metrics)
