#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.metrics import Metric
from pytorch_lightning.utilities import move_data_to_device, rank_zero_only
from torch.nn import ModuleDict, ModuleList

from InnerEye.Common.common_util import EPOCH_METRICS_FILE_NAME, SUBJECT_METRICS_FILE_NAME
from InnerEye.Common.metrics_dict import DataframeLogger, MetricType, MetricsDict, SequenceMetricsDict
from InnerEye.Common.type_annotations import DictStrFloat
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.dataset.sample import CroppedSample
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.metrics import Accuracy05, AccuracyAtOptimalThreshold, AreaUnderPrecisionRecallCurve, \
    AreaUnderRocCurve, \
    BinaryCrossEntropy, ExplainedVariance, FalseNegativeRateOptimalThreshold, \
    FalsePositiveRateOptimalThreshold, \
    MeanAbsoluteError, MeanSquaredError, OptimalThreshold, add_average_dice, \
    compute_dice_across_patches, \
    nanmean, store_epoch_metrics
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils import image_util, metrics_util, model_util
from InnerEye.ML.utils.lr_scheduler import SchedulerWithWarmUp
from InnerEye.ML.utils.metrics_constants import LoggingColumns
from InnerEye.ML.utils.ml_util import RandomStateSnapshot, set_random_seed
from InnerEye.ML.utils.model_util import get_scalar_model_inputs_and_labels
from InnerEye.ML.utils.sequence_utils import apply_sequence_model_loss, get_masked_model_outputs_and_labels

MAX_ITEM_LOAD_TIME_SEC = 0.5
MAX_LOAD_TIME_WARNINGS = 3

TRAIN_PREFIX = "train/"
VALIDATION_PREFIX = "val/"


class StoringLogger(LightningLoggerBase):
    """
    A Pytorch Lightning logger that simply stores the metrics that are written to it.
    Used for diagnostic purposes in unit tests.
    """

    def __init__(self):
        super().__init__()
        self.results: List[Dict[str, float]] = []
        self.hyperparams = {}

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self.results.append(metrics)

    def log_hyperparams(self, params: Any) -> None:
        self.hyperparams = params

    def experiment(self) -> Any:
        return ""

    def name(self) -> Any:
        return ""

    def version(self) -> int:
        return 0

    def extract_by_prefix(self, metrics: Dict[str, float], prefix_filter: str = "") -> Tuple[int, DictStrFloat]:
        epoch_name = "epoch"
        epoch = metrics.get(epoch_name, None)
        if epoch is None:
            raise ValueError("Each of the logged metrics should have an 'epoch' key.")
        metrics_dict = {}
        for key, value in metrics.items():
            assert isinstance(key, str), f"All dictionary keys should be strings, but got: {type(key)}"
            # Add the metric if either there is no prefix filter (prefix does not matter), or if the prefix
            # filter is supplied and really matches the metric name
            if key != epoch_name and (not prefix_filter) or key.startswith(prefix_filter):
                stripped_key = key[len(prefix_filter):]
                metrics_dict[stripped_key] = value
        return epoch, metrics_dict

    def to_metrics_dicts(self, prefix_filter: str = "") -> Dict[int, DictStrFloat]:
        result = {}
        for metrics in self.results:
            epoch, metrics_dict = self.extract_by_prefix(metrics, prefix_filter)
            if len(metrics_dict) != 0:
                result[epoch] = metrics_dict
        return result


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
        self.outputs_folder = config.outputs_folder
        self.model = torch.nn.Module()
        # These two will be set later in set_optimizer_and_scheduler.
        # The ddp_spawn accelerator only works if the model configuration object is
        # not stored in here. Hence, need to do operations that require a full config
        # in a way that does not require storing the config.
        self.optimizer = None
        self.l_rate_scheduler = None
        self.cross_validation_split_index = config.cross_validation_split_index
        self.effective_random_seed = config.get_effective_random_seed()
        # Timers for monitoring data loading time
        self.epoch_start_time = 0
        self.item_start_time = 0
        self.batch_start_time = 0
        self.num_load_time_warnings = 0
        self.num_load_time_exceeded = 0
        self.num_batches = 0
        self.total_extra_load_time = 0.0
        self.total_load_time = 0.0
        # This should be re-assigned on the outside, to a logger that is hooked up with the Trainer object.
        self.storing_logger = StoringLogger()
        # This will be initialized correctly in epoch_start
        self.random_state: Optional[RandomStateSnapshot] = None
        # training loggers
        self.train_metrics_folder = self.outputs_folder / ModelExecutionMode.TRAIN.value
        self.val_metrics_folder = self.outputs_folder / ModelExecutionMode.VAL.value
        fixed_logger_columns = {LoggingColumns.CrossValidationSplitIndex.value: config.cross_validation_split_index}
        self.train_epoch_metrics_logger = DataframeLogger(self.train_metrics_folder / EPOCH_METRICS_FILE_NAME,
                                                          fixed_columns=fixed_logger_columns)
        self.val_epoch_metrics_logger = DataframeLogger(self.val_metrics_folder / EPOCH_METRICS_FILE_NAME,
                                                        fixed_columns=fixed_logger_columns)
        # Fields to store diagnostics for testing
        self.train_diagnostics = []
        self.val_diagnostics = []
        self.use_sync_dist = self.use_ddp

    def set_optimizer_and_scheduler(self, config: DeepLearningConfig) -> None:
        self.optimizer = model_util.create_optimizer(config, self.model.parameters())
        self.l_rate_scheduler = SchedulerWithWarmUp(config, self.optimizer)

    def configure_optimizers(self):
        return [self.optimizer], [self.l_rate_scheduler]

    def close_all_loggers(self) -> None:
        self.train_epoch_metrics_logger.flush()
        self.val_epoch_metrics_logger.flush()

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.training_or_validation_epoch_end(is_training=True)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # reset the random state for training, so that we get continue from where we were before the validation step.
        self.random_state.restore_random_state()
        self.training_or_validation_epoch_end(is_training=False)

    def on_train_epoch_start(self) -> None:
        self.reset_timers()

    def on_validation_epoch_start(self) -> None:
        self.reset_timers()
        # Store the random number generator state, so that the next training epoch starts from here.
        self.random_state = RandomStateSnapshot.snapshot_random_state()
        # reset the random state for validation, so that we get consistent behaviour when drawing random patches
        # when validating segmentation models.
        seed = self.effective_random_seed
        set_random_seed(seed, "Validation")

    def on_train_epoch_end(self, outputs) -> None:
        self.on_train_or_validation_epoch_end(is_training=True)

    def on_validation_epoch_end(self) -> None:
        self.on_train_or_validation_epoch_end(is_training=False)

    @rank_zero_only
    def on_train_or_validation_epoch_end(self, is_training: bool) -> None:
        """
        This is a hook called once all per-epoch computation is finished, and all metrics are written.
        It extracts the final set of metrics for the epoch from the `storing_logger` field, and writes them to a file.
        :param is_training: Set to True to read out the training set metrics, or False to read out the
        validation set metrics.
        """
        prefix_filter = TRAIN_PREFIX if is_training else VALIDATION_PREFIX
        # Get the last set of metrics that the logger stores. That set belongs to the current train/validation
        # epoch because it was written just before this hook is called.
        epoch, metrics = self.storing_logger.extract_by_prefix(self.storing_logger.results[-1], prefix_filter)
        # Sanity check: We should see metrics for the current epoch.
        assert epoch == self.current_epoch, f"Epochs don't match: logger has {epoch}, module has {self.current_epoch}"
        self.store_epoch_results(metrics, epoch, is_training)

    @rank_zero_only
    def training_or_validation_epoch_end(self, is_training: bool) -> None:
        """
        This is a hook called at the end of a training or validation epoch. In here, we can still write
        metrics to a logger.
        :param is_training: If True, this is called at the end of a training epoch. If False, this is at the
        end of a validation epoch.
        """
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
        self.log_on_epoch(MetricType.SECONDS_PER_EPOCH, epoch_time_seconds, is_training=is_training)

    def log_on_epoch(self,
                     name: Union[MetricType, str],
                     value: Union[float, Metric],
                     is_training: bool,
                     reduce_fx: Callable = torch.mean) -> None:
        """
        Logs a metrics to Pytorch Lightning with the on_epoch flag set. The metric will get a prefix indicating
        if it is a training or a validation metric. A custom reducer function can be provided.
        The method also ensures that the correct synchronization across nodes is used.
        :param name: The name of the metric to log
        :param value: The value of the metric
        :param is_training: If true, give the metric a "train/" prefix, otherwise a "val/" prefix.
        :param reduce_fx: The function that should be used to aggregate multiple logged values.
        """
        metric_name = name if isinstance(name, str) else name.value
        prefix = TRAIN_PREFIX if is_training else VALIDATION_PREFIX
        self.log(prefix + metric_name, value,
                 sync_dist=self.use_sync_dist, on_step=False, on_epoch=True, reduce_fx=reduce_fx)

    def store_epoch_results(self, metrics: DictStrFloat, epoch: int, is_training: bool) -> None:
        """
        Stores a set of metrics (key/value pairs) to a file logger. That file logger is either one that only holds
        training or only holds validation metrics.
        :param metrics: A dictionary with all the metrics to write, as key/value pairs.
        :param epoch: The epoch to which the metrics belong.
        :param is_training: If true, write the metrics to the logger for training metrics, if False, write to the logger
        for validation metrics.
        """
        file_logger = self.train_epoch_metrics_logger if is_training else self.val_epoch_metrics_logger
        store_epoch_metrics(metrics,
                            epoch,
                            file_logger=file_logger)

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.batch_start(batch_idx=batch_idx, is_training=True)

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.batch_start(batch_idx=batch_idx, is_training=False)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.batch_end(is_training=True)

    def on_validation_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.batch_end(is_training=False)

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
        # Print out data loading statistics only on local rank 0. This will print out stats about data loading
        # once on each individual machine in DDP
        if self.local_rank != 0:
            return
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
        self.batch_start_time = time.time()

    def batch_end(self, is_training: bool) -> None:
        self.log_on_epoch(MetricType.SECONDS_PER_BATCH, time.time() - self.batch_start_time, is_training=is_training)
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

    def write_loss(self, is_training: bool, loss: Any) -> None:
        """
        Writes the given loss value to Lightning, labelled either "val/loss" or "train/loss".
        If this comes from a training step, then also log the learning rate.
        :param is_training: If True, the logged metric will be called "train/Loss". If False, the metric will
        be called "val/Loss"
        """
        self.log_on_epoch(MetricType.LOSS, loss, is_training)
        learning_rate = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
        if is_training:
            self.log_on_epoch(MetricType.LEARNING_RATE, learning_rate, is_training)


class SegmentationLightning(InnerEyeLightning):
    def __init__(self, config: SegmentationModelBase, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.model = config.create_model()
        self.loss_fn = model_util.create_segmentation_loss_function(config)
        self.ground_truth_ids = config.ground_truth_ids

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
        # Forward propagation can lead to a model output that is smaller than the input image (crop).
        # labels_center_crop is the relevant part of the labels tensor that the model will actually produce.
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
        segmentations = image_util.posteriors_to_segmentation(posteriors=posteriors)

        dice_for_all_classes = compute_dice_across_patches(
            segmentation=segmentations,
            ground_truth=labels,
            allow_multiple_classes_for_each_pixel=True).cpu().numpy()
        foreground_voxels = metrics_util.get_number_of_voxels_per_class(cropped_sample.labels)
        # loss is a scalar, also when running the forward pass over multiple crops.
        # dice_for_all_structures has one row per crop.
        # store metrics per batch
        for i, ground_truth_id in enumerate(self.ground_truth_ids):
            dice_name = f"{MetricType.DICE.value}/{ground_truth_id}"
            for b in range(dice_for_all_classes.shape[0]):
                self.log_on_epoch(name=dice_name, value=dice_for_all_classes[b, i].item(),
                                  is_training=is_training, reduce_fx=nanmean)
            self.log_on_epoch(name=f"{MetricType.VOXEL_COUNT.value}/{ground_truth_id}",
                              value=foreground_voxels[i],
                              is_training=is_training)
        # store diagnostics per batch
        center_indices = cropped_sample.center_indices
        if isinstance(center_indices, torch.Tensor):
            center_indices = center_indices.cpu().numpy()
        if is_training:
            self.train_diagnostics.append(center_indices)
        else:
            self.val_diagnostics.append(center_indices)
        # if self.train_val_params.in_training_mode:
        #     # store the sample train patch from this epoch for visualization
        #     if batch_index == self.example_to_save and self.config.store_dataset_sample:
        #         _store_dataset_sample(self.config, self.train_val_params.epoch, forward_pass_result,
        #                               cropped_sample)
        num_subjects = cropped_sample.image.shape[0]
        self.log_on_epoch(name=MetricType.SUBJECT_COUNT,
                          value=num_subjects,
                          is_training=is_training,
                          reduce_fx=sum)
        self.write_loss(is_training, loss)
        return loss

    def store_epoch_results(self, metrics: DictStrFloat, epoch: int, is_training: bool):
        metrics = add_average_dice(metrics)
        super().store_epoch_results(metrics, epoch, is_training)


SUBJECT_OUTPUT_PER_RANK_PREFIX = f"{SUBJECT_METRICS_FILE_NAME}.rank"


def get_subject_output_file_per_rank(rank: int) -> str:
    """
    Gets the name of a file that will store the per-rank per-subject model outputs.
    :param rank: The rank of the current model in distributed training.
    :return: A string like "rank7_metrics.csv"
    """
    return f"{SUBJECT_OUTPUT_PER_RANK_PREFIX}{rank}"


class ScalarLightning(InnerEyeLightning):
    def __init__(self, config: ScalarModelBase, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.model = config.create_model()
        raw_loss = model_util.create_scalar_loss_function(config)
        if isinstance(config, SequenceModelBase):
            self.loss_fn = lambda model_output, loss: apply_sequence_model_loss(raw_loss, model_output, loss)
            self.target_indices = config.get_target_indices()
            self.target_names = [SequenceMetricsDict.get_hue_name_from_target_index(p)
                                 for p in config.sequence_target_positions]
        else:
            self.loss_fn = raw_loss
            self.target_indices = []
            self.target_names = [MetricsDict.DEFAULT_HUE_KEY]
        self.is_classification_model = config.is_classification_model
        self.use_mean_teacher_model = config.compute_mean_teacher_model
        self.logits_to_posterior_fn = config.get_post_loss_logits_normalization_function()
        self.loss_type = config.loss_type
        # These two fields store the PyTorch Lightning Metrics objects that will compute metrics on validation
        # and training set, in particular ones that are not possible to compute from a single minibatch (AUC and alike)
        self.train_metric_computers = self.create_metric_computers()
        self.val_metric_computers = self.create_metric_computers()

        # TODO antonsc: Work out how we handle mean teacher model
        # if config.compute_grad_cam:
        #     model_to_evaluate = self.train_val_params.mean_teacher_model if \
        #         config.compute_mean_teacher_model else self.train_val_params.model
        #     self.guided_grad_cam = VisualizationMaps(model_to_evaluate, config)
        #     config.visualization_folder.mkdir(exist_ok=True)

    def create_metric_computers(self) -> ModuleDict:
        # The metric computers should be stored in an object that derives from torch.Module,
        # so that they are picked up when moving the whole LightningModule to GPU.
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4713
        return ModuleDict([(p, self._get_metrics_classes()) for p in self.target_names])

    def _get_metrics_classes(self) -> ModuleList:
        if self.is_classification_model:
            return ModuleList([Accuracy05(),
                               AccuracyAtOptimalThreshold(),
                               OptimalThreshold(),
                               FalsePositiveRateOptimalThreshold(),
                               FalseNegativeRateOptimalThreshold(),
                               AreaUnderRocCurve(),
                               AreaUnderPrecisionRecallCurve(),
                               BinaryCrossEntropy()])
        else:
            return ModuleList([MeanAbsoluteError(), MeanSquaredError(), ExplainedVariance()])

    def forward(self, *model_inputs: torch.Tensor) -> torch.Tensor:
        return self.logits_to_posterior(self.model(*model_inputs))

    def logits_to_posterior(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the model-specific normalization to go from logits (model outputs) to posteriors.
        """
        return self.logits_to_posterior_fn(logits)

    def on_train_start(self) -> None:
        # These loggers store the per-subject model outputs. They cannot be initialized in the constructor because
        # the trainer object will not yet be set, and we need to get the rank from there.
        fixed_logger_columns = {LoggingColumns.CrossValidationSplitIndex.value: self.cross_validation_split_index}
        subject_output_file = get_subject_output_file_per_rank(self.trainer.global_rank)
        self.train_subject_outputs_logger = DataframeLogger(self.train_metrics_folder / subject_output_file,
                                                            fixed_columns=fixed_logger_columns)
        self.val_subject_outputs_logger = DataframeLogger(self.val_metrics_folder / subject_output_file,
                                                          fixed_columns=fixed_logger_columns)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.reset_metrics(is_training=True)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.reset_metrics(is_training=False)

    def reset_metrics(self, is_training: bool) -> None:
        metric_computers = self.train_metric_computers if is_training else self.val_metric_computers
        # Necessary to explicitly reset, otherwise the metrics keep accumulating over all epochs
        for metrics in metric_computers.values():
            for m in metrics:
                m.reset()

    def training_or_validation_step(self,
                                    sample: Dict[str, Any],
                                    batch_index: int,
                                    is_training: bool):
        model_inputs_and_labels = get_scalar_model_inputs_and_labels(self.model, self.target_indices, sample)
        labels = model_inputs_and_labels.labels
        logits = self.model(*model_inputs_and_labels.model_inputs)
        subject_ids = model_inputs_and_labels.subject_ids
        loss = self.loss_fn(logits, labels)
        self.write_loss(is_training, loss)
        self.compute_and_log_metrics(logits, labels, subject_ids, is_training)
        self.log_on_epoch(name=MetricType.SUBJECT_COUNT,
                          value=len(model_inputs_and_labels.subject_ids),
                          is_training=is_training,
                          reduce_fx=sum)
        return loss

    def compute_and_log_metrics(self,
                                logits: torch.Tensor,
                                targets: torch.Tensor,
                                subject_ids: List[str],
                                is_training: bool):
        metrics = self.train_metric_computers if is_training else self.val_metric_computers
        per_subject_outputs = list()
        for i, (prediction_target, metric_list) in enumerate(metrics.items()):
            # mask the model outputs and labels if required
            masked = get_masked_model_outputs_and_labels(
                logits[:, i, ...], targets[:, i, ...], subject_ids)
            # compute metrics on valid masked tensors only
            if masked is not None:
                _posteriors = self.logits_to_posterior(masked.model_outputs.data)
                # Image encoders already prepare images in float16, but the labels are not yet in that dtype
                _labels = masked.labels.data.to(dtype=_posteriors.dtype)
                _subject_ids = masked.subject_ids
                for metric in metric_list:
                    metric(_posteriors, _labels)
                per_subject_outputs.extend(
                    zip(_subject_ids, [prediction_target] * len(_subject_ids), _posteriors.tolist(), _labels.tolist()))
        # Write a full breakdown of per-subject predictions and labels to a file. These files are local to the current
        # rank in distributed training, and will be aggregated after training.
        logger = self.train_subject_outputs_logger if is_training else self.val_subject_outputs_logger
        data_split = ModelExecutionMode.TRAIN if is_training else ModelExecutionMode.VAL
        for subject, prediction_target, model_output, label in per_subject_outputs:
            logger.add_record({
                LoggingColumns.Epoch.value: self.current_epoch,
                LoggingColumns.Patient.value: subject,
                LoggingColumns.Hue.value: prediction_target,
                LoggingColumns.ModelOutput.value: model_output,
                LoggingColumns.Label.value: label,
                LoggingColumns.DataSplit.value: data_split.value
            })
        # TODO antonsc: Find a better place for this code. We can only draw plots once all results are aggregated,
        # maybe move to the report?
        # if self._should_save_regression_error_plot(self.current_epoch):
        #     error_plot_name = f"error_plot_{self.train_val_params.epoch}"
        #     path = str(self.config.outputs_folder / f"{error_plot_name}.png")
        #     plot_variation_error_prediction(epoch_metrics.get_labels(), epoch_metrics.get_predictions(), path)
        #     logger = self.config.azure_loggers_train if is_training else self.config.azure_loggers_val
        #     logger.log_image(error_plot_name, path)

    def training_or_validation_epoch_end(self, is_training: bool) -> None:
        metric_computers = self.train_metric_computers if is_training else self.val_metric_computers
        prefix = TRAIN_PREFIX if is_training else VALIDATION_PREFIX
        for prediction_target, metric_list in metric_computers.items():
            target_suffix = "" if prediction_target == MetricsDict.DEFAULT_HUE_KEY else f"/{prediction_target}"
            for metric in metric_list:
                if metric.has_predictions:
                    # Sequence models can have no predictions at all for particular positions, depending on the data.
                    # Hence, only log if anything really has been accumula
                    self.log(name=prefix + metric.name + target_suffix, value=metric.compute())
        logger = self.train_subject_outputs_logger if is_training else self.val_subject_outputs_logger
        logger.flush()
        super().training_or_validation_epoch_end(is_training)

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        """
        For sequence models, transfer the nested lists of items to the given GPU device.
        For all other models, this relies on the superclass to move the batch of data to the GPU.
        :param batch: A batch of data coming from the dataloader.
        :param device: The target CUDA device.
        :return: A modified batch of data, where all tensor now live on the given CUDA device.
        """
        return transfer_batch_to_device(batch, device)


def transfer_batch_to_device(batch: Any, device: torch.device) -> Any:
    """
    For sequence models, transfer the nested lists of items to the given GPU device.
    For all other models, this relies on Lightning's default code to move the batch of data to the GPU.
    :param batch: A batch of data coming from the dataloader.
    :param device: The target CUDA device.
    :return: A modified batch of data, where all tensor now live on the given CUDA device.
    """
    if not isinstance(batch, dict):
        raise ValueError(f"This function expects a dictionary input, but got: {type(batch)}")
    # For sequence models, this method receives a dictionary with "item": List[List[ScalarItem]]
    items = batch.get("items", None)
    if items is not None and isinstance(items, List) and isinstance(items[0], List) and \
            isinstance(items[0][0], ScalarItem):
        batch["items"] = [[j.to_device(device) for j in i] for i in items]
        return batch
    else:
        return move_data_to_device(batch, device)


def create_lightning_model(config: ModelConfigBase) -> InnerEyeLightning:
    if config.is_segmentation_model:
        model = SegmentationLightning(config)
    elif config.is_scalar_model:
        model = ScalarLightning(config)
    else:
        raise NotImplementedError(f"Don't know how to handle config of type {type(config)}")
    model.set_optimizer_and_scheduler(config)
    return model


def create_model_from_lightning_checkpoint(config: ModelConfigBase, checkpoint_path: Path) -> torch.nn.Module:
    # Create a Lighting model that matches the configuration, but keep only the type of it
    lightning_model_type = type(create_lightning_model(config))
    # For model debugging, allow loading a GPU trained model onto the CPU. This will clearly only work
    # if the model is small.
    map_location = None if config.use_gpu else 'cpu'
    lightning_model = lightning_model_type.load_from_checkpoint(checkpoint_path=str(checkpoint_path),
                                                                map_location=map_location,
                                                                config=config)
    config.adjust_after_mixed_precision_and_parallel(lightning_model.model)
    return lightning_model
