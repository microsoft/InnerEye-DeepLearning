#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ProgressBarBase
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from InnerEye.Azure.azure_util import RUN_CONTEXT, is_offline_run_context
from InnerEye.Common.metrics_constants import TRAIN_PREFIX, VALIDATION_PREFIX
from InnerEye.Common.type_annotations import DictStrFloat


class StoringLogger(LightningLoggerBase):
    """
    A Pytorch Lightning logger that simply stores the metrics that are written to it.
    Used for diagnostic purposes in unit tests.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: Dict[int, DictStrFloat] = {}
        self.hyperparams: Any = None
        # Fields to store diagnostics for unit testing
        self.train_diagnostics: List[Any] = []
        self.val_diagnostics: List[Any] = []

    @rank_zero_only
    def log_metrics(self, metrics: DictStrFloat, step: Optional[int] = None) -> None:
        epoch_name = "epoch"
        if epoch_name not in metrics:
            raise ValueError("Each of the logged metrics should have an 'epoch' key.")
        epoch = int(metrics[epoch_name])
        del metrics[epoch_name]
        if epoch in self.results:
            current_results = self.results[epoch]
            overlapping_keys = set(metrics.keys()).intersection(current_results.keys())
            if len(overlapping_keys) > 0:
                raise ValueError(f"Unable to log metric with same name twice for epoch {epoch}: "
                                 f"{', '.join(overlapping_keys)}")
            current_results.update(metrics)
        else:
            self.results[epoch] = metrics

    @rank_zero_only
    def log_hyperparams(self, params: Any) -> None:
        self.hyperparams = params

    def experiment(self) -> Any:
        return None

    def name(self) -> Any:
        return ""

    def version(self) -> int:
        return 0

    @property
    def epochs(self) -> Iterable[int]:
        """
        Gets the epochs for which the present object holds any results.
        """
        return self.results.keys()

    def extract_by_prefix(self, epoch: int, prefix_filter: str = "") -> DictStrFloat:
        """
        Reads the set of metrics for a given epoch, filters them to retain only those that have the given prefix,
        and returns the filtered ones. This is used to break a set
        of results down into those for training data (prefix "Train/") or validation data (prefix "Val/").
        :param epoch: The epoch for which results should be read.
        :param prefix_filter: If empty string, return all metrics. If not empty, return only those metrics that
        have a name starting with `prefix`, and strip off the prefix.
        :return: A metrics dictionary.
        """
        epoch_results = self.results.get(epoch, None)
        if epoch_results is None:
            raise KeyError(f"No results are stored for epoch {epoch}")
        filtered = {}
        for key, value in epoch_results.items():
            assert isinstance(key, str), f"All dictionary keys should be strings, but got: {type(key)}"
            # Add the metric if either there is no prefix filter (prefix does not matter), or if the prefix
            # filter is supplied and really matches the metric name
            if (not prefix_filter) or key.startswith(prefix_filter):
                stripped_key = key[len(prefix_filter):]
                filtered[stripped_key] = value
        return filtered

    def to_metrics_dicts(self, prefix_filter: str = "") -> Dict[int, DictStrFloat]:
        """
        Converts the results stored in the present object into a two-level dictionary, mapping from epoch number to
        metric name to metric value. Only metrics where the name starts with the given prefix are retained, and the
        prefix is stripped off in the result.
        :param prefix_filter: If empty string, return all metrics. If not empty, return only those metrics that
        have a name starting with `prefix`, and strip off the prefix.
        :return: A dictionary mapping from epoch number to metric name to metric value.
        """
        return {epoch: self.extract_by_prefix(epoch, prefix_filter) for epoch in self.epochs}

    def get_metric(self, is_training: bool, metric_type: str) -> List[float]:
        """
        Gets a scalar metric out of either the list of training or the list of validation results. This returns
        the value that a specific metric attains in all of the epochs.
        :param is_training: If True, read metrics that have a "train/" prefix, otherwise those that have a "val/"
        prefix.
        :param metric_type: The metric to extract.
        :return: A list of floating point numbers, with one entry per entry in the the training or validation results.
        """
        full_metric_name = (TRAIN_PREFIX if is_training else VALIDATION_PREFIX) + metric_type
        return [self.results[epoch][full_metric_name] for epoch in self.epochs]

    def get_train_metric(self, metric_type: str) -> List[float]:
        """
        Gets a scalar metric from the list of training results. This returns
        the value that a specific metric attains in all of the epochs.
        :param metric_type: The metric to extract.
        :return: A list of floating point numbers, with one entry per entry in the the training results.
        """
        return self.get_metric(is_training=True, metric_type=metric_type)

    def get_val_metric(self, metric_type: str) -> List[float]:
        """
        Gets a scalar metric from the list of validation results. This returns
        the value that a specific metric attains in all of the epochs.
        :param metric_type: The metric to extract.
        :return: A list of floating point numbers, with one entry per entry in the the validation results.
        """
        return self.get_metric(is_training=False, metric_type=metric_type)

    def train_results_per_epoch(self) -> List[DictStrFloat]:
        """
        Gets the full set of training metrics that the logger stores, as a list of dictionaries per epoch.
        """
        return list(self.to_metrics_dicts(prefix_filter=TRAIN_PREFIX).values())

    def val_results_per_epoch(self) -> List[DictStrFloat]:
        """
        Gets the full set of validation metrics that the logger stores, as a list of dictionaries per epoch.
        """
        return list(self.to_metrics_dicts(prefix_filter=VALIDATION_PREFIX).values())


class AzureMLLogger(LightningLoggerBase):
    """
    A Pytorch Lightning logger that stores metrics in the current AzureML run. If the present run is not
    inside AzureML, nothing gets logged.
    """

    def __init__(self) -> None:
        super().__init__()
        self.is_azureml_run = not is_offline_run_context(RUN_CONTEXT)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        is_epoch_metric = "epoch" in metrics
        if self.is_azureml_run:
            for key, value in metrics.items():
                # Log all epoch-level metrics without the step information
                # All step-level metrics with step
                RUN_CONTEXT.log(key, value, step=None if is_epoch_metric else step)

    @rank_zero_only
    def log_hyperparams(self, params: Any) -> None:
        pass

    def experiment(self) -> Any:
        return None

    def name(self) -> Any:
        return ""

    def version(self) -> int:
        return 0


PROGRESS_STAGE_TRAIN = "Training"
PROGRESS_STAGE_VAL = "Validation"
PROGRESS_STAGE_TEST = "Testing"
PROGRESS_STAGE_PREDICT = "Prediction"


class AzureMLProgressBar(ProgressBarBase):
    """
    A PL progress bar that works better in AzureML. It prints timestamps for each message, and works well with a setup
    where there is no direct access to the console.
    """

    def __init__(self,
                 refresh_rate: int = 50,
                 write_to_logging_info: bool = False
                 ):
        """
        Creates a new AzureML progress bar.
        :param refresh_rate: The number of steps after which the progress should be printed out.
        :param write_to_logging_info: If True, the progress information will be printed via logging.info. If False,
        it will be printed to stdout via print.
        """
        super().__init__()
        self._refresh_rate = refresh_rate
        self._enabled = True
        self.stage = ""
        self.stage_start_time = 0.0
        self.max_batch_count = 0
        self.progress_print_fn = logging.info if write_to_logging_info else print
        self.flush_fn = None if write_to_logging_info else sys.stdout.flush

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.module = pl_module

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_start(trainer, pl_module)
        self.start_stage(PROGRESS_STAGE_TRAIN, self.total_train_batches)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.start_stage(PROGRESS_STAGE_VAL, self.total_val_batches)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_test_epoch_start(trainer, pl_module)
        self.start_stage(PROGRESS_STAGE_TEST, self.total_test_batches)

    def on_predict_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_predict_epoch_start(trainer, pl_module)
        self.start_stage(PROGRESS_STAGE_PREDICT, self.total_predict_batches)

    def start_stage(self, stage: str, max_batch_count: int) -> None:
        """
        Sets the information that a new stage of the PL loop is starting. The stage will be available in
        self.stage, max_batch_count in self.max_batch_count. The time when this method was called is recorded in
        self.stage_start_time
        :param stage: The string name of the stage that has just started.
        :param max_batch_count: The total number of batches that need to be processed in this stage.
        """
        self.stage = stage
        self.max_batch_count = max_batch_count
        self.stage_start_time = time.time()

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any,
                           batch_idx: int, dataloader_idx: int) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.update_progress(batches_processed=self.train_batch_idx)

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any,
                                batch_idx: int, dataloader_idx: int) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.update_progress(batches_processed=self.val_batch_idx)

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any,
                          batch_idx: int, dataloader_idx: int) -> None:
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.update_progress(batches_processed=self.test_batch_idx)

    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any,
                             batch_idx: int, dataloader_idx: int) -> None:
        super().on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.update_progress(batches_processed=self.predict_batch_idx)

    def update_progress(self, batches_processed: int):
        """
        Writes progress information once the refresh interval is full.
        :param batches_processed: The number of batches that have been processed for the current stage.
        """
        should_update = self.is_enabled and \
                        (batches_processed % self.refresh_rate == 0 or batches_processed == self.max_batch_count)
        if not should_update:
            return
        prefix = f"{self.stage}"
        if self.stage in [PROGRESS_STAGE_TRAIN, PROGRESS_STAGE_VAL]:
            prefix += f" epoch {self.module.current_epoch}"
        if self.stage == PROGRESS_STAGE_TRAIN:
            prefix += f" (step {self.module.global_step})"
        prefix += ": "
        if math.isinf(self.max_batch_count):
            # Can't print out per-cent progress or time estimates if the data is infinite
            message = f"{prefix}{batches_processed} batches completed"
        else:
            fraction_completed = batches_processed / self.max_batch_count
            percent_completed = int(fraction_completed * 100)
            time_elapsed = time.time() - self.stage_start_time
            estimated_epoch_duration = time_elapsed / fraction_completed

            def to_minutes(time_sec: float) -> str:
                minutes = int(time_sec / 60)
                seconds = int(time_sec % 60)
                return f"{minutes:02}:{seconds:02}"

            message = (f"{prefix}{batches_processed:4}/{self.max_batch_count} ({percent_completed:3}%) completed. "
                       f"{to_minutes(time_elapsed)} elapsed, total epoch time ~ {to_minutes(estimated_epoch_duration)}")
        self.progress_print_fn(message)
        if self.flush_fn:
            self.flush_fn()
