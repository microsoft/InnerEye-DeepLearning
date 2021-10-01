#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import numbers
import operator
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from InnerEye.Azure.azure_util import RUN_CONTEXT, is_offline_run_context
from InnerEye.Common.metrics_constants import TRAIN_PREFIX, VALIDATION_PREFIX
from InnerEye.Common.type_annotations import DictStrFloat, DictStrFloatOrFloatList


class StoringLogger(LightningLoggerBase):
    """
    A Pytorch Lightning logger that simply stores the metrics that are written to it, grouped by epoch.
    Used for diagnostic purposes in unit tests.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: Dict[int, DictStrFloatOrFloatList] = {}
        self.hyperparams: Any = None
        # Fields to store diagnostics for unit testing
        self.train_diagnostics: List[Any] = []
        self.val_diagnostics: List[Any] = []

    @rank_zero_only
    def log_metrics(self, metrics: DictStrFloat, step: Optional[int] = None) -> None:
        logging.debug(f"StoringLogger step={step}: {metrics}")
        epoch_name = "epoch"
        if epoch_name not in metrics:
            raise ValueError("Each of the logged metrics should have an 'epoch' key.")
        epoch = int(metrics[epoch_name])
        del metrics[epoch_name]
        for key, value in metrics.items():
            if isinstance(value, int):
                metrics[key] = float(value)
        if epoch in self.results:
            current_results = self.results[epoch]
            for key, value in metrics.items():
                if key in current_results:
                    logging.debug(f"StoringLogger: appending results for metric {key}")
                    current_metrics = current_results[key]
                    if isinstance(current_metrics, list):
                        current_metrics.append(value)
                    else:
                        current_results[key] = [current_metrics, value]
                else:
                    current_results[key] = value
        else:
            self.results[epoch] = metrics  # type: ignore

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
            assert isinstance(value, float), f"All metrics should be floats, but got: {type(value)}"
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
        result = []
        for epoch in self.epochs:
            value = self.results[epoch][full_metric_name]
            if not isinstance(value, float):
                raise ValueError(f"Expected a floating point value for metric {full_metric_name}, but got: "
                                 f"{value}")
            result.append(value)
        return result

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
        logging.debug(f"AzureMLLogger step={step}: {metrics}")
        if self.is_azureml_run:
            for key, value in metrics.items():
                RUN_CONTEXT.log(key, value)

    @rank_zero_only
    def log_hyperparams(self, params: Any) -> None:
        pass

    def experiment(self) -> Any:
        return None

    def name(self) -> Any:
        return ""

    def version(self) -> int:
        return 0


def log_on_epoch(module: LightningModule,
                 name: Optional[str] = None,
                 value: Optional[Any] = None,
                 metrics: Optional[Mapping[str, Any]] = None,
                 reduce_fx: Callable = torch.mean,
                 sync_dist: Optional[bool] = None,
                 sync_dist_op: Any = "mean") -> None:
    """
    Write a dictionary with metrics and/or an individual metric as a name/value pair to the loggers of the given module.
    Metrics are always logged upon epoch completion.
    The metrics in question first synchronized across GPUs if DDP with >1 node is used. Afterwards, they are aggregated
    across all steps via the reduce_fx (default: mean).
    Metrics that are fed in as plain numbers rather than tensors (for example, plain Python integers) are converted
    to tensors before logging.

    :param name: The name of the metric to log.
    :param value: The actual value of the metric to log.
    :param metrics: A dictionary with metrics to log.
    :param module: The PyTorch Lightning module where the metrics should be logged.
    :param sync_dist: If not None, use this value for the sync_dist argument to module.log. If None,
    set it automatically depending on the use of DDP. Set this to False if you want to log metrics that are only
    available on Rank 0 of a DDP job.
    :param reduce_fx: The reduce function to apply to the per-step values, after synchronizing the tensors across GPUs.
    Default: torch.mean
    :param sync_dist_op: The reduce operation to use when synchronizing the tensors across GPUs. This must be
    a value recognized by sync_ddp: Either 'None' to use 'sum' as aggregate, or 'mean' or 'avg'
    """
    assert module.trainer is not None, "No trainer is set for this module."
    if operator.xor(name is None, value is None):
        raise ValueError("Both or neither of 'name' and 'value' must be provided.")
    sync_dist = module.trainer.world_size > 1 if sync_dist is None else sync_dist
    metrics = metrics or {}
    if name is not None:
        metrics[name] = value
    metrics_as_tensors = {
        key: torch.tensor(value, dtype=torch.float, device=module.device)
        if isinstance(value, numbers.Number)
        else value
        for key, value in metrics.items()
    }
    module.log_dict(metrics_as_tensors,
                    on_epoch=True,
                    on_step=False,
                    sync_dist=sync_dist,
                    reduce_fx=reduce_fx,
                    sync_dist_op=sync_dist_op)
