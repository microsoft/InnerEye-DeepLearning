#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Dict, Iterable, List, Optional

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from InnerEye.Common.metrics_constants import TRAIN_PREFIX, VALIDATION_PREFIX
from InnerEye.Common.type_annotations import DictStrFloat


class StoringLogger(LightningLoggerBase):
    """
    A Pytorch Lightning logger that simply stores the metrics that are written to it.
    Used for diagnostic purposes in unit tests.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results_per_epoch: Dict[int, DictStrFloatOrFloatList] = {}
        self.hyperparams: Any = None
        # Fields to store diagnostics for unit testing
        self.train_diagnostics: List[Any] = []
        self.val_diagnostics: List[Any] = []
        self.results_without_epoch: List[DictStrFloat] = []

    @rank_zero_only
    def log_metrics(self, metrics: DictStrFloat, step: Optional[int] = None) -> None:
        logging.debug(f"StoringLogger step={step}: {metrics}")
        epoch_name = "epoch"
        if epoch_name not in metrics:
            # Metrics without an "epoch" key are logged during testing, for example
            self.results_without_epoch.append(metrics)
            return
        epoch = int(metrics[epoch_name])
        del metrics[epoch_name]
        for key, value in metrics.items():
            if isinstance(value, int):
                metrics[key] = float(value)
        if epoch in self.results_per_epoch:
            current_results = self.results_per_epoch[epoch]
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
            self.results_per_epoch[epoch] = metrics  # type: ignore

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
        return self.results_per_epoch.keys()

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
        epoch_results = self.results_per_epoch.get(epoch, None)
        if epoch_results is None:
            raise KeyError(f"No results are stored for epoch {epoch}")
        filtered = {}
        for key, value in epoch_results.items():
            assert isinstance(key, str), f"All dictionary keys should be strings, but got: {type(key)}"
            # Add the metric if either there is no prefix filter (prefix does not matter), or if the prefix
            # filter is supplied and really matches the metric name
            if (not prefix_filter) or key.startswith(prefix_filter):
                stripped_key = key[len(prefix_filter):]
                filtered[stripped_key] = value  # type: ignore
        return filtered  # type: ignore

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
            value = self.results_per_epoch[epoch][full_metric_name]
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
