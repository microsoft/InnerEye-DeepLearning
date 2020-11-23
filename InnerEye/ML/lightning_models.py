#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningDataModule, LightningModule

from InnerEye.Common.metrics_dict import MetricType, MetricsDict, ScalarMetricsDict, \
    create_metrics_dict_for_scalar_models
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import BACKGROUND_CLASS_NAME, SegmentationModelBase
from InnerEye.ML.dataset.sample import CroppedSample
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.metrics import aggregate_segmentation_metrics, compute_dice_across_patches, compute_scalar_metrics, \
    store_epoch_metrics
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.utils import image_util, metrics_util, model_util
from InnerEye.ML.utils.lr_scheduler import SchedulerWithWarmUp
from InnerEye.ML.utils.ml_util import RandomStateSnapshot, set_random_seed
from InnerEye.ML.utils.model_util import get_scalar_model_inputs_and_labels
from InnerEye.ML.visualizers.regression_visualization import plot_variation_error_prediction

MAX_ITEM_LOAD_TIME_SEC = 0.5
MAX_LOAD_TIME_WARNINGS = 3


class TrainingAndValidationDataLightning(LightningDataModule):
    def _init__(self, config: ModelConfigBase):
        super().__init__()
        self.config = config
        self.data_loaders = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.data_loaders = self.config.create_data_loaders()

    def train_dataloader(self):
        return self.data_loaders[ModelExecutionMode.TRAIN]

    def val_dataloader(self):
        return self.data_loaders[ModelExecutionMode.VAL]

    def test_dataloader(self):
        raise NotImplementedError("For segmentation models, the test dataset should not be evaluated patch-wise.")


class InnerEyeLightning(LightningModule):
    def __init__(self, config: DeepLearningConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.model = torch.nn.Module()
        # Timers for monitoring data loading time
        self.epoch_start_time = 0
        self.item_start_time = 0
        self.num_load_time_warnings = 0
        self.num_load_time_exceeded = 0
        self.num_batches = 0
        self.total_extra_load_time = 0.0
        self.total_load_time = 0.0
        # Metrics for all epochs
        self.training_metrics_per_epoch: List[MetricsDict] = []
        self.validation_metrics_per_epoch: List[MetricsDict] = []
        # This will be initialized correctly in epoch_start
        self.training_metrics = MetricsDict()
        self.validation_metrics = MetricsDict()
        self.random_state: Optional[RandomStateSnapshot] = None

    def configure_optimizers(self):
        optimizer = model_util.create_optimizer(self.config, self.model.parameters())
        l_rate_scheduler = SchedulerWithWarmUp(self.config, optimizer)
        return [optimizer], [l_rate_scheduler]

    def create_empty_metrics_dict(self) -> MetricsDict:
        """
        Returns a new, empty MetricsDict object. This can be overwritten in derived classes to support model-specific
        initialization.
        """
        return MetricsDict()

    def on_train_epoch_end(self, outputs) -> None:
        self.epoch_end(is_training=True)

    def on_validation_epoch_end(self) -> None:
        # reset the random state for training, so that we get continue from where we were before the validation step.
        self.random_state.restore_random_state()
        self.epoch_end(is_training=False)

    def on_train_epoch_start(self) -> None:
        self.training_metrics = self.create_empty_metrics_dict()
        self.reset_timers()

    def on_validation_epoch_start(self) -> None:
        self.reset_timers()
        self.validation_metrics = self.create_empty_metrics_dict()
        # Store the random number generator state, so that the next training epoch starts from here.
        self.random_state = RandomStateSnapshot.snapshot_random_state()
        # reset the random state for validation, so that we get consistent behaviour when drawing random patches
        # when validating segmentation models.
        seed = self.config.get_effective_random_seed()
        set_random_seed(seed, "Validation")

    def epoch_end(self, is_training: bool) -> MetricsDict:
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
        # Aggregate the metrics in a way that is specific to individual types of models.
        result = self.aggregate_metrics(self.current_metrics(is_training))
        result.add_metric(MetricType.LEARNING_RATE, learning_rate)
        result.add_metric(MetricType.SECONDS_PER_EPOCH, epoch_time_seconds)
        logger = self.config.azure_loggers_train if is_training else self.config.azure_loggers_val
        df_logger = self.config.data_frame_loggers.train_epoch_metrics if is_training \
            else self.config.data_frame_loggers.val_epoch_metrics
        store_epoch_metrics(logger,
                            df_logger,
                            self.current_epoch,
                            result,
                            self.config)
        if is_training:
            self.training_metrics_per_epoch.append(result)
        else:
            self.validation_metrics_per_epoch.append(result)
        return result

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.batch_start(batch_idx=batch_idx, is_training=True)

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.batch_start(batch_idx=batch_idx, is_training=False)

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
                                    is_training: bool) -> Any:
        raise NotImplementedError("This method must be overwritten in a derived class.")

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

    def aggregate_metrics(self, epoch_metrics: MetricsDict) -> MetricsDict:
        raise NotImplementedError("This method must be overwritten in a derived class.")

    def current_metrics(self, is_training: bool) -> MetricsDict:
        """
        Returns either the training or validation metrics that are stored.
        :param is_training: If True, return the training metrics, otherwise the validation metrics.
        """
        return self.training_metrics if is_training else self.validation_metrics

    def write_metric(self,
                     is_training: bool,
                     metric_type: MetricType,
                     metric_value: float,
                     hue: str = MetricsDict.DEFAULT_HUE_KEY) -> None:
        self.current_metrics(is_training).add_metric(metric_type, metric_value, hue=hue)

    def write_loss(self, is_training: bool, loss: Any) -> None:
        """
        Writes the given loss value to Lightning, labelled either "val_loss" or "train_loss".
        :param is_training: If True, the logged metric will be called "train_loss". If False, "val_loss"
=        """
        metric_name = 'train_loss' if is_training else "val_loss"
        self.log(metric_name, loss)
        loss_scalar = loss.float().item() if torch.is_tensor(loss) else loss
        self.write_metric(is_training, MetricType.LOSS, loss_scalar)


class SegmentationLightning(InnerEyeLightning):
    def __init__(self, config: SegmentationModelBase, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.model = config.create_model()
        self.loss_fn = model_util.create_segmentation_loss_function(config)

    def create_empty_metrics_dict(self) -> MetricsDict:
        return MetricsDict(hues=[BACKGROUND_CLASS_NAME] + self.config.ground_truth_ids)

    def forward(self, patches) -> torch.Tensor:
        return self.logits_to_posterior(self.model(patches))

    def logits_to_posterior(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply Softmax on dimension 1 (Class) to map model output into a posterior probability distribution [0,1]
        """
        return torch.nn.functional.softmax(logits, dim=1)

    def training_or_validation_step(self,
                                    sample: Dict[str, Any],
                                    batch_index: int,
                                    is_training: bool):
        """
        Runs training for a single minibatch of training or validation data, and computes all metrics.
        :param sample: The batched sample on which the model should be trained.
        :param batch_index: The index of the present batch (supplied only for diagnostics).
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

        epoch_metrics = self.current_metrics(is_training)
        dice_for_all_classes = compute_dice_across_patches(
            segmentation=torch.tensor(segmentations).long(),
            ground_truth=labels,
            use_cuda=self.config.use_gpu,
            allow_multiple_classes_for_each_pixel=True).cpu().numpy()
        foreground_voxels = metrics_util.get_number_of_voxels_per_class(cropped_sample.labels)
        # loss is a scalar, also when running the forward pass over multiple crops.
        # dice_for_all_structures has one row per crop.

        # store metrics per batch
        epoch_metrics.add_metric(MetricType.LOSS, loss.item())
        for i, ground_truth_id in enumerate(epoch_metrics.get_hue_names(include_default=False)):
            for b in range(dice_for_all_classes.shape[0]):
                epoch_metrics.add_metric(MetricType.DICE, dice_for_all_classes[b, i].item(),
                                         hue=ground_truth_id, skip_nan_when_averaging=True)
            epoch_metrics.add_metric(MetricType.VOXEL_COUNT, foreground_voxels[i], hue=ground_truth_id)
        # store diagnostics per batch
        center_indices = cropped_sample.center_indices
        if isinstance(center_indices, torch.Tensor):
            center_indices = center_indices.cpu().numpy()
        epoch_metrics.add_diagnostics(MetricType.PATCH_CENTER.value, center_indices.copy())
        # if self.train_val_params.in_training_mode:
        #     # store the sample train patch from this epoch for visualization
        #     if batch_index == self.example_to_save and self.config.store_dataset_sample:
        #         _store_dataset_sample(self.config, self.train_val_params.epoch, forward_pass_result,
        #                               cropped_sample)
        self.write_loss(is_training, loss)
        return loss

    def aggregate_metrics(self, epoch_metrics: MetricsDict) -> MetricsDict:
        return aggregate_segmentation_metrics(epoch_metrics)


class ScalarLightning(InnerEyeLightning):
    def __init__(self, config: ScalarModelBase, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.model = config.create_model()
        # TODO antonsc: The old code also changed the datatype for the loss tensor, depending on the
        # loss function
        self.loss_fn = model_util.create_scalar_loss_function(config)
        self.use_mean_teacher_model = self.config.compute_mean_teacher_model
        self.logits_to_posterior_fn = config.get_post_loss_logits_normalization_function()
        # TODO antonsc: Work out how we handle mean teacher model
        # if config.compute_grad_cam:
        #     model_to_evaluate = self.train_val_params.mean_teacher_model if \
        #         config.compute_mean_teacher_model else self.train_val_params.model
        #     self.guided_grad_cam = VisualizationMaps(model_to_evaluate, config)
        #     config.visualization_folder.mkdir(exist_ok=True)

    def create_empty_metrics_dict(self) -> MetricsDict:
        return create_metrics_dict_for_scalar_models(self.config)

    def aggregate_metrics(self, epoch_metrics: MetricsDict) -> MetricsDict:
        return epoch_metrics.average(across_hues=False)

    def forward(self, *model_inputs: torch.Tensor) -> torch.Tensor:
        return self.logits_to_posterior(self.model(*model_inputs))

    def logits_to_posterior(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the model-specific normalization to go from logits (model outputs) to posteriors.
        """
        return self.logits_to_posterior_fn(logits)

    def training_or_validation_step(self,
                                    sample: Dict[str, Any],
                                    batch_index: int,
                                    is_training: bool):
        model_inputs_and_labels = get_scalar_model_inputs_and_labels(self.config, self.model, sample)  # type: ignore
        labels = model_inputs_and_labels.labels
        logits = self.model(*model_inputs_and_labels.model_inputs)
        loss = self.loss_fn(logits, labels)
        compute_scalar_metrics(self.current_metrics(is_training),
                               subject_ids=model_inputs_and_labels.subject_ids,
                               model_output=self.logits_to_posterior(logits),
                               labels=labels,
                               loss_type=self.config.loss_type)

        self.write_loss(is_training, loss)
        return loss

    def epoch_end(self, is_training: bool) -> None:
        super().epoch_end(is_training)
        epoch_metrics = self.current_metrics(is_training)
        assert isinstance(epoch_metrics, ScalarMetricsDict)
        # Store subject level metrics
        subject_logger = self.config.data_frame_loggers.train_subject_metrics if is_training \
            else self.config.data_frame_loggers.val_subject_metrics
        epoch_metrics.store_metrics_per_subject(
            epoch=self.current_epoch,
            df_logger=subject_logger,
            mode=ModelExecutionMode.TRAIN if is_training else ModelExecutionMode.VAL,
            cross_validation_split_index=self.config.cross_validation_split_index)
        # if self._should_save_regression_error_plot(self.current_epoch):
        #     error_plot_name = f"error_plot_{self.train_val_params.epoch}"
        #     path = str(self.config.outputs_folder / f"{error_plot_name}.png")
        #     plot_variation_error_prediction(epoch_metrics.get_labels(), epoch_metrics.get_predictions(), path)
        #     logger = self.config.azure_loggers_train if is_training else self.config.azure_loggers_val
        #     logger.log_image(error_plot_name, path)


def create_lightning_model(config: ModelConfigBase) -> LightningModule:
    if config.is_segmentation_model:
        return SegmentationLightning(config)
    elif config.is_scalar_model:
        return ScalarLightning(config)
    else:
        raise NotImplementedError(f"Don't know how to handle config of type {type(config)}")


def create_model_from_lightning_checkpoint(config: ModelConfigBase, checkpoint_path: Path) -> torch.nn.Module:
    lightning_model = create_lightning_model(config)
    # For model debugging, allow loading a GPU trained model onto the CPU. This will clearly only work
    # if the model is small.
    map_location = None if config.use_gpu else 'cpu'
    type(lightning_model).load_from_checkpoint(checkpoint_path=str(checkpoint_path),
                                               map_location=map_location,
                                               config=config)
    config.adjust_after_mixed_precision_and_parallel(lightning_model.model)
    return lightning_model
