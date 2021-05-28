#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import numbers
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import param
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities import rank_zero_only
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

from InnerEye.Common.common_util import EPOCH_METRICS_FILE_NAME, logging_section
from InnerEye.Common.metrics_constants import LoggingColumns, MetricType, TRAIN_PREFIX, VALIDATION_PREFIX
from InnerEye.Common.type_annotations import DictStrFloat
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.deep_learning_config import DatasetParams, DeepLearningConfig, OutputParams, TrainerParams, \
    WorkflowParams
from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.ML.lightning_loggers import StoringLogger
from InnerEye.ML.metrics import EpochTimers, MAX_ITEM_LOAD_TIME_SEC, store_epoch_metrics
from InnerEye.ML.metrics_dict import DataframeLogger
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.utils import model_util
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.ML.utils.lr_scheduler import SchedulerWithWarmUp
from InnerEye.ML.utils.ml_util import RandomStateSnapshot, set_random_seed, validate_dataset_paths
from InnerEye.ML.utils.model_util import generate_and_print_model_summary
from InnerEye.ML.visualizers.patch_sampling import visualize_random_crops_for_dataset
from InnerEye.ML.utils.csv_util import CSV_SUBJECT_HEADER
from InnerEye.ML.dataset.full_image_dataset import convert_channels_to_file_paths

class TrainAndValDataLightning(LightningDataModule):
    """
    A class that wraps training and validation data from an InnerEye model configuration to a Lightning data module.
    When doing inference on the trained models, we use InferenceDataLightning. This is particularly important for
    segmentation models, where training and validation happens on equal sized patches, but inference is running on
    images of arbitrary size.
    """

    def __init__(self, config: ModelConfigBase) -> None:
        super().__init__()
        self.config = config
        self.data_loaders: Dict[ModelExecutionMode, DataLoader] = {}

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Writes the dataset files for later use in cross validation analysis. This is only executed once per
        distributed training run.
        """
        # Save the dataset files for later use in cross validation analysis
        self.config.write_dataset_files()

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Checks if the dataset folder is present, and the dataset file exists. This is execute on each node in
        distributed training.
        """
        # Check for existing dataset.csv file in the correct locations. Skip that if a dataset has already been
        # loaded (typically only during tests)
        if self.config.dataset_data_frame is None:
            assert self.config.local_dataset is not None
            validate_dataset_paths(self.config.local_dataset, self.config.dataset_csv)
        self.config.read_dataset_if_needed()
        self.data_loaders = self.config.create_data_loaders()

    def train_dataloader(self) -> DataLoader:  # type: ignore
        return self.data_loaders[ModelExecutionMode.TRAIN]

    def val_dataloader(self) -> DataLoader:  # type: ignore
        return self.data_loaders[ModelExecutionMode.VAL]

    def test_dataloader(self) -> DataLoader:  # type: ignore
        raise NotImplementedError("There is no test dataset stored here, because this object is only meant to be "
                                  "used for training and validation.")


class InferenceDataLightning(LightningDataModule):
    """
    A class that wraps data for running model inference on InnerEye models, as a Lightning data module.
    Note that training and validation data is handled by TrainAndValDataLightning.
    """

    def __init__(self, config: ModelConfigBase) -> None:
        super().__init__()
        self.config = config
        self.train_data: Dataset = Dataset()
        self.val_data: Dataset = Dataset()
        self.test_data: Dataset = Dataset()

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Initializes the datasets stored in the present object, by calling the config object to
        prepare the torch Dataset objects for train/val/test.
        """
        self.train_data = self.config.get_torch_dataset_for_inference(ModelExecutionMode.TRAIN)
        self.val_data = self.config.get_torch_dataset_for_inference(ModelExecutionMode.VAL)
        self.test_data = self.config.get_torch_dataset_for_inference(ModelExecutionMode.TEST)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.train_data)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.val_data)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.test_data)

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        pass


class InnerEyeContainer(LightningContainer):
    """
    A container that wraps the creation of Lightning datasets for the built-in InnerEye models.
    """

    def __init__(self, config: ModelConfigBase):
        super().__init__(should_validate=False)
        self.config = config
        self._model_name = config.model_name
        # Fields like cross validation index are defined at container level, but the InnerEye models define them
        # at model level. Copy everything over.
        for type_to_copy in [WorkflowParams, DatasetParams, TrainerParams, OutputParams]:
            assert issubclass(type_to_copy, param.Parameterized)
            self.apply_overrides({p: getattr(config, p) for p in type_to_copy.params()},  # type: ignore
                                 should_validate=False)
        self.validate()

    def setup(self) -> None:
        """
        This hook reads the dataset file, and possibly sets required pre-processing objects, like one-hot encoder
        for categorical features, that need to be available before creating the model.
        """
        # Following code validates segmentation training, validation and test data to ensure:
        # 1) Files exist,
        # 2) mask_id identifier is not empty,
        # 3) consistency for input channels, ground_truth and mask_id,
        # 4) ensures data is consistent with load_dataset_sources method prior running training, validation and testing.

        full_failed_channel_info: str = ''

        if self.config.is_segmentation_model:
            # Creates a list with all the channels of interest
            all_channels = self.config.image_channels + self.config.ground_truth_ids
            # Mask_id is an optional field. If non-empty in the config, will check dataframe.
            if self.config.mask_id:
                all_channels += [self.config.mask_id]
            # Root directory where data is stored
            if self.config.local_dataset is None:
                raise ValueError("Expecting that a dataset is available here.")
            local_dataset_root_folder = self.config.local_dataset
            # Iterate over train, validation and test dataset
            dataset_splits = self.config.get_dataset_splits()
            for split_data in [dataset_splits.train, dataset_splits.val, dataset_splits.test]:
                unique_ids = set(split_data[CSV_SUBJECT_HEADER])
                for patient_id in unique_ids:
                    rows = split_data.loc[split_data[CSV_SUBJECT_HEADER] == patient_id]
                    # Converts channels from data frame to file paths and gets errors if any
                    __, failed_channel_info = convert_channels_to_file_paths(all_channels,
                                                                              rows,
                                                                              local_dataset_root_folder,
                                                                              patient_id)
                    full_failed_channel_info += failed_channel_info

        if full_failed_channel_info:
            raise ValueError(full_failed_channel_info)

        self.config.read_dataset_if_needed()

    def create_model(self) -> LightningModule:  # type: ignore
        from InnerEye.ML.lightning_models import create_lightning_model
        return create_lightning_model(self.config)

    def get_data_module(self) -> LightningDataModule:
        return TrainAndValDataLightning(self.config)  # type: ignore

    def get_inference_data_module(self) -> LightningDataModule:
        return InferenceDataLightning(self.config)  # type: ignore

    def before_training_on_global_rank_zero(self) -> None:
        # Save the dataset files for later use in cross validation analysis
        self.config.write_dataset_files()
        if isinstance(self.config, SegmentationModelBase):
            with logging_section("Visualizing the effect of sampling random crops for training"):
                visualize_random_crops_for_dataset(self.config)

        # Print out a detailed breakdown of layers, memory consumption and time.
        assert isinstance(self.model, InnerEyeLightning)
        generate_and_print_model_summary(self.config, self.model.model)

    def load_checkpoint_and_modify(self, path_to_checkpoint: Path) -> Dict[str, Any]:
        return self.config.load_checkpoint_and_modify(path_to_checkpoint=path_to_checkpoint)


class InnerEyeLightning(LightningModule):
    """
    The base class for all InnerEye models for training in PyTorch Lightning. The base class handles all shared
    operations like choosing the optimizer and learning rate schedule, keeping track of IO performance (loading times),
    and IO to files.
    """

    def __init__(self, config: DeepLearningConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.outputs_folder = config.outputs_folder
        self.checkpoint_folder = config.checkpoint_folder
        self.model: DeviceAwareModule = DeviceAwareModule()
        # These two will be set later in set_optimizer_and_scheduler.
        # The ddp_spawn accelerator only works if the model configuration object is
        # not stored in here. Hence, need to do operations that require a full config
        # in a way that does not require storing the config.
        self.optimizer: Optional[Optimizer] = None
        self.l_rate_scheduler: Optional[_LRScheduler] = None
        self.cross_validation_split_index = config.cross_validation_split_index
        self.effective_random_seed = config.get_effective_random_seed()
        # Timers for monitoring data loading time
        self.train_timers = EpochTimers()
        self.val_timers = EpochTimers()
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
        # Stores information the checkpoint that created this model, if any.
        self.checkpoint_loading_message = ""

    def set_optimizer_and_scheduler(self, config: DeepLearningConfig) -> None:
        self.optimizer = model_util.create_optimizer(config, self.model.parameters())
        self.l_rate_scheduler = SchedulerWithWarmUp(config, self.optimizer, num_epochs=config.num_epochs)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        return [self.optimizer], [self.l_rate_scheduler]  # type: ignore

    def on_fit_end(self) -> None:
        """
        Flushes all logger objects that the present object holds.
        """
        self.train_epoch_metrics_logger.flush()
        self.val_epoch_metrics_logger.flush()

    @property
    def use_sync_dist(self) -> bool:
        """
        Returns True if metric logging should use sync_dist=True. This is read off from the use_ddp flag of the trainer.
        """
        # For PL from version 1.2.0 on: self.trainer.accelerator_connector.use_ddp
        return self.trainer.use_ddp

    def on_train_epoch_start(self) -> None:
        self.train_timers.reset()

    def training_epoch_end(self, outputs: List[Any]) -> None:
        # Write out all the metrics that have been accumulated in the StoringLogger in the previous epoch.
        # Metrics for the very last epoch are written in on_train_end
        self.read_epoch_results_from_logger_and_store(epoch=self.current_epoch - 1)
        self.training_or_validation_epoch_end(is_training=True)

    def on_validation_epoch_start(self) -> None:
        """
        Stores the state of all random number generators, and resets them all to a fixed seed. This is done to ensure
        that any randomization when loading validation data is consistent during training. In particular, this ensures
        that drawing random patches for segmentation model training is giving a validation set that does not fluctuate.
        """
        self.val_timers.reset()
        # In Lightning, the validation epoch is running "inside" the training. If we get here, it means that training
        # is done for this epoch, even though the on_training_epoch hook has not yet been called.
        self.train_timers.epoch_end()
        # Store the random number generator state, so that the next training epoch starts from here.
        self.random_state = RandomStateSnapshot.snapshot_random_state()
        # reset the random state for validation, so that we get consistent behaviour when drawing random patches
        # when validating segmentation models.
        seed = self.effective_random_seed
        set_random_seed(seed, "Validation")

    def on_validation_epoch_end(self) -> None:
        self.val_timers.epoch_end()

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """
        Resets the random number generator state to what it was before the current validation epoch started.
        :param outputs: The list of outputs from the individual validation minibatches.
        """
        # reset the random state for training, so that we get continue from where we were before the validation step.
        assert self.random_state is not None
        self.random_state.restore_random_state()
        self.training_or_validation_epoch_end(is_training=False)

    @rank_zero_only
    def on_train_end(self) -> None:
        """
        This hook is called at the very end of training. Use that to write the very last set of training and
        validation metrics from the StoringLogger to disk.
        """
        self.read_epoch_results_from_logger_and_store(epoch=self.current_epoch)

    @rank_zero_only
    def read_epoch_results_from_logger_and_store(self, epoch: int) -> None:
        """
        Reads the metrics for the previous epoch from the StoringLogger, and writes them to disk, broken down by
        Training and Validation metrics.
        """
        if epoch >= 0:
            if epoch in self.storing_logger.results:
                for is_training, prefix in [(True, TRAIN_PREFIX), (False, VALIDATION_PREFIX)]:
                    metrics = self.storing_logger.extract_by_prefix(epoch, prefix)
                    self.store_epoch_results(metrics, epoch, is_training)

    @rank_zero_only
    def training_or_validation_epoch_end(self, is_training: bool) -> None:
        """
        This is a hook called at the end of a training or validation epoch. In here, we can still write
        metrics to a logger.
        :param is_training: If True, this is called at the end of a training epoch. If False, this is at the
        end of a validation epoch.
        """
        if not is_training:
            # In validation epochs, mark that it has been completed. Training epochs are marked completed already
            # at the start of the validation epoch.
            self.val_timers.epoch_end()
            # Write all IO stats here, so that the order on the console is Train start, train end, val start, val end.
            self.write_and_log_epoch_time(is_training=True)
            self.write_and_log_epoch_time(is_training=False)

    def write_and_log_epoch_time(self, is_training: bool) -> None:
        """
        Reads the IO timers for either the training or validation epoch, writes them to the console, and logs the
        time per epoch.
        :param is_training: If True, show and log the data for the training epoch. If False, use the data for the
        validation epoch.
        """
        timers = self.get_timers(is_training=is_training)
        epoch_time_seconds = timers.total_epoch_time
        status = "training" if is_training else "validation"
        logging.info(f"Epoch {self.current_epoch} {status} took {epoch_time_seconds:0.2f}sec, of which waiting for "
                     f"data took {timers.total_load_time:0.2f} sec total.")
        if timers.num_load_time_exceeded > 0 and timers.should_warn_in_this_epoch:
            logging.warning("The dataloaders were not fast enough to always supply the next batch in less than "
                            f"{MAX_ITEM_LOAD_TIME_SEC}sec.")
            logging.warning(
                f"In this epoch, {timers.num_load_time_exceeded} out of {timers.num_batches} batches exceeded the load "
                f"time threshold. Total loading time for the slow batches was {timers.total_extra_load_time:0.2f}sec.")
        # This metric is only written at rank zero, and hence must no be synchronized across workers. If attempted,
        # training will get stuck.
        self.log_on_epoch(MetricType.SECONDS_PER_EPOCH, epoch_time_seconds, is_training=is_training,
                          sync_dist_override=False)

    def log_on_epoch(self,
                     name: Union[MetricType, str],
                     value: Any,
                     is_training: bool,
                     reduce_fx: Callable = torch.mean,
                     sync_dist_override: Optional[bool] = None,
                     sync_dist_op: Any = "mean") -> None:
        """
        Logs a metrics to Pytorch Lightning with the on_epoch flag set. The metric will get a prefix indicating
        if it is a training or a validation metric. A custom reducer function can be provided.
        The method also ensures that the correct synchronization across nodes is used. If the value to log is a
        floating point, it is converted to a Tensor on the current device to enable synchronization.
        :param sync_dist_override: If not None, use this value for the sync_dist argument to self.log. If None,
        set it automatically depending on the use of DDP.
        :param name: The name of the metric to log
        :param value: The value of the metric. This can be a tensor, floating point value, or a Metric class.
        :param is_training: If true, give the metric a "train/" prefix, otherwise a "val/" prefix.
        :param reduce_fx: The reduce function to apply after synchronizing the tensors across GPUs.
        :param sync_dist_op: The reduce operation to use when synchronizing the tensors across GPUs. This must be
        a value recognized by sync_ddp: Either 'None' to use 'sum' as aggregate, or 'mean' or 'avg'
        """
        metric_name = name if isinstance(name, str) else name.value
        if isinstance(value, numbers.Number):
            value = torch.tensor(value, dtype=torch.float, device=self.device)
        prefix = TRAIN_PREFIX if is_training else VALIDATION_PREFIX
        sync_dist = self.use_sync_dist if sync_dist_override is None else sync_dist_override
        self.log(prefix + metric_name, value,
                 sync_dist=sync_dist,
                 on_step=False, on_epoch=True,
                 reduce_fx=reduce_fx,
                 sync_dist_op=sync_dist_op)

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
        store_epoch_metrics(metrics, epoch, file_logger=file_logger)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        This hook is called when loading a model from a checkpoint. It just prints out diagnostics about which epoch
        created the present checkpoint.
        :param checkpoint: The checkpoint dictionary loaded from disk.
        """
        keys = ['epoch', 'global_step']
        present_keys = [f"{key} = {checkpoint[key]}" for key in keys if key in checkpoint]
        if present_keys:
            self.checkpoint_loading_message = f"Loading checkpoint that was created at ({', '.join(present_keys)})"
            logging.info(self.checkpoint_loading_message)

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.batch_start(batch_idx=batch_idx, is_training=True)

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.batch_start(batch_idx=batch_idx, is_training=False)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.batch_end(is_training=True)

    def on_validation_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.batch_end(is_training=False)

    def training_step(self,  # type: ignore
                      sample: Dict[str, Any],
                      batch_index: int) -> Any:
        return self.training_or_validation_step(sample, batch_index, is_training=True)

    def validation_step(self,  # type: ignore
                        sample: Dict[str, Any],
                        batch_index: int) -> Any:
        return self.training_or_validation_step(sample, batch_index, is_training=False)

    def training_or_validation_step(self,
                                    sample: Dict[str, Any],
                                    batch_index: int,
                                    is_training: bool) -> Any:
        """
        This is the shared method that handles the training (when `is_training==True`) and validation steps
        (when `is_training==False`)
        :param sample: The minibatch of data that should be processed.
        :param batch_index: The index of the current minibatch.
        :param is_training: If true, this has been called from `training_step`, otherwise it has been called from
        `validation_step`.
        """
        raise NotImplementedError("This method must be overwritten in a derived class.")

    @rank_zero_only
    def batch_start(self, batch_idx: int, is_training: bool) -> None:
        """
        Shared code to keep track of IO-related metrics when loading a minibatch. This is only done on rank zero.
        :param batch_idx: The index of the current minibatch.
        :param is_training: If true, this has been called from `on_train_batch_start`, otherwise it has been called from
        `on_validation_batch_start`.
        :return:
        """
        timers = self.get_timers(is_training=is_training)
        message_prefix = f"Epoch {self.current_epoch} {'training' if is_training else 'validation'}"
        timers.batch_start(batch_index=batch_idx, epoch=self.current_epoch, message_prefix=message_prefix)

    @rank_zero_only
    def batch_end(self, is_training: bool) -> None:
        """
        Shared code to keep track of IO-related metrics when loading a minibatch.
        :param is_training: If true, this has been called from `on_train_batch_end`, otherwise it has been called from
        `on_validation_batch_end`.
        """
        timers = self.get_timers(is_training=is_training)
        batch_time = timers.batch_end()
        # This metric is only written at rank 0, and hence must not be synchronized. Trying to synchronize will
        # block training.
        self.log_on_epoch(MetricType.SECONDS_PER_BATCH, batch_time, is_training=is_training, sync_dist_override=False)

    def get_timers(self, is_training: bool) -> EpochTimers:
        """
        Gets the object that holds all IO-related metrics and timers, for either the validation or the training epoch.
        """
        return self.train_timers if is_training else self.val_timers

    def reset_timers(self) -> None:
        """
        Resets all timers and counters for IO-related metrics, for both the validation and the training epoch.
        """
        self.train_timers.reset()
        self.val_timers.reset()

    def write_loss(self, is_training: bool, loss: torch.Tensor) -> None:
        """
        Writes the given loss value to Lightning, labelled either "val/loss" or "train/loss".
        If this comes from a training step, then also log the learning rate.
        :param loss: The loss value that should be logged.
        :param is_training: If True, the logged metric will be called "train/Loss". If False, the metric will
        be called "val/Loss"
        """
        self.log_on_epoch(MetricType.LOSS, loss, is_training)
        if is_training:
            learning_rate = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
            self.log_on_epoch(MetricType.LEARNING_RATE, learning_rate, is_training)
