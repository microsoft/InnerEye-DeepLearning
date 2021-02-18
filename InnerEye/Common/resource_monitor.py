#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Dict, List, Tuple

import GPUtil
import pandas as pd
import psutil
import tensorboardX
import torch
from GPUtil import GPU

from InnerEye.ML.utils.ml_util import is_gpu_available

COL_GPU = "gpu"
COL_METRIC = "metric"
COL_VALUE = "value"


def memory_in_gb(bytes: int) -> float:
    """
    Converts a memory amount in bytes to gigabytes.
    :param bytes:
    :return:
    """
    gb = 2 ** 30
    return bytes / gb


@dataclass
class GpuUtilization:
    # The numeric ID of the GPU
    id: int
    # GPU load, as a number between 0 and 1
    load: float
    # Memory utilization, as a number between 0 and 1
    mem_util: float
    # Allocated memory by pytorch
    mem_allocated_gb: float
    # Reserved memory by pytorch
    mem_reserved_gb: float
    # Number of observations that are stored in the present object
    count: int

    def __add__(self, other: GpuUtilization) -> GpuUtilization:
        return GpuUtilization(
            id=self.id,
            load=self.load + other.load,
            mem_util=self.mem_util + other.mem_util,
            mem_allocated_gb=self.mem_allocated_gb + other.mem_allocated_gb,
            mem_reserved_gb=self.mem_reserved_gb + other.mem_reserved_gb,
            count=self.count + other.count
        )

    def max(self, other: GpuUtilization) -> GpuUtilization:
        """
        Computes the metric-wise maximum of the two GpuUtilization objects.
        :param other:
        :return:
        """
        return GpuUtilization(
            # Effectively ignore ID. We could enforce consistent IDs, but then we could not compute overall max.
            id=self.id,
            load=max(self.load, other.load),
            mem_util=max(self.mem_util, other.mem_util),
            mem_allocated_gb=max(self.mem_allocated_gb, other.mem_allocated_gb),
            mem_reserved_gb=max(self.mem_reserved_gb, other.mem_reserved_gb),
            # Max does not make sense for the count field, hence just add up to see how many items we have done max for
            count=self.count + other.count
        )

    def average(self) -> GpuUtilization:
        """
        Returns a GPU utilization object that contains all metrics of the present object, divided by the number
        of observations.
        :return:
        """
        return GpuUtilization(
            id=self.id,
            load=self.load / self.count,
            mem_util=self.mem_util / self.count,
            mem_allocated_gb=self.mem_allocated_gb / self.count,
            mem_reserved_gb=self.mem_reserved_gb / self.count,
            count=1
        )

    @property
    def name(self) -> str:
        """
        Gets a string name for the GPU that the present objet describes, "GPU1" for GPU with id == 1.
        """
        return f"GPU{self.id}"

    def enumerate(self, prefix: str = "") -> List[Tuple[str, float]]:
        """
        Lists all metrics stored in the present object, as (metric_name, value) pairs suitable for logging in
        Tensorboard.
        :param prefix: If provided, this string as used as an additional prefix for the metric name itself. If prefix
        is "max", the metric would look like "maxLoad_Percent"
        :return: A list of (name, value) tuples.
        """
        return [
            (f'{prefix}MemUtil_Percent', round(self.mem_util * 100, 2)),
            (f'{prefix}Load_Percent', round(self.load * 100, 2)),
            (f'{prefix}MemReserved_GB', round(self.mem_reserved_gb, 4)),
            (f'{prefix}MemAllocated_GB', round(self.mem_allocated_gb, 4))
        ]

    @staticmethod
    def from_gpu(gpu: GPU) -> GpuUtilization:
        """
        Creates a GpuUtilization object from data coming from the gputil library.
        :param gpu: GPU diagnostic data from gputil.
        :return:
        """
        return GpuUtilization(
            id=gpu.id,
            load=gpu.load,
            mem_util=gpu.memoryUtil,
            mem_allocated_gb=memory_in_gb(torch.cuda.memory_allocated(int(gpu.id))),
            mem_reserved_gb=memory_in_gb(torch.cuda.memory_reserved(int(gpu.id))),
            count=1
        )


RESOURCE_MONITOR_AGGREGATE_METRICS = "aggregate_resource_usage.csv"


class ResourceMonitor(Process):
    """
    Monitor and log GPU and CPU stats in TensorBoard in a separate process.
    """

    def __init__(self,
                 interval_seconds: int,
                 tensorboard_folder: Path,
                 csv_results_folder: Path):
        """
        Creates a process that will monitor CPU and GPU utilization.
        :param interval_seconds: The interval in seconds at which usage statistics should be written.
        :param tensorboard_folder: The path in which to create a tensorboard logfile.
        :param csv_results_folder: The path in which the CSV file with aggregate metrics will be created.
        When running in AzureML, this should NOT reside inside the /logs folder.
        """
        super().__init__(name="Resource Monitor", daemon=True)
        self._interval_seconds = interval_seconds
        self.tensorboard_folder = tensorboard_folder
        self.gpu_aggregates: Dict[int, GpuUtilization] = dict()
        self.gpu_max: Dict[int, GpuUtilization] = dict()
        self.writer = tensorboardX.SummaryWriter(str(self.tensorboard_folder))
        self.step = 0
        self.aggregate_metrics: List[str] = []
        self.aggregate_metrics_file = csv_results_folder / RESOURCE_MONITOR_AGGREGATE_METRICS

    def log_to_tensorboard(self, label: str, value: float) -> None:
        """
        Write a scalar metric value to Tensorboard, marked with the present step.
        :param label: The name of the metric.
        :param value: The value.
        """
        self.writer.add_scalar(label, value, global_step=self.step)

    def update_metrics(self, gpus: List[GPU]) -> None:
        """
        Updates the stored GPU utilization metrics with the current status coming from gputil, and logs
        them to Tensorboard.
        :param gpus: The current utilization information, read from gputil, for all available GPUs.
        """
        for gpu in gpus:
            gpu_util = GpuUtilization.from_gpu(gpu)
            for (metric_name, value) in gpu_util.enumerate():
                self.log_to_tensorboard(f"{gpu_util.name}/{metric_name}", value)
            id = gpu_util.id
            # Update the total utilization
            if id in self.gpu_aggregates:
                self.gpu_aggregates[id] = self.gpu_aggregates[id] + gpu_util
            else:
                self.gpu_aggregates[id] = gpu_util
            # Update the maximum utilization
            if id in self.gpu_max:
                self.gpu_max[id] = self.gpu_max[id].max(gpu_util)
            else:
                self.gpu_max[id] = gpu_util

    def run(self) -> None:
        if self._interval_seconds <= 0:
            logging.warning("Resource monitoring requires an interval that is larger than 0 seconds, but "
                            f"got: {self._interval_seconds}. Exiting.")
            self.kill()
        logging.info(f"Process '{self.name}' started with pid: {self.pid}")
        gpu_available = is_gpu_available()
        while True:
            if gpu_available:
                self.update_metrics(GPUtil.getGPUs())
            # log the CPU utilization
            self.log_to_tensorboard('CPU/Load_Percent', psutil.cpu_percent(interval=None))
            self.log_to_tensorboard('CPU/MemUtil_Percent', psutil.virtual_memory()[2])
            self.step += 1
            self.store_to_file()
            # pause the thread for the requested delay
            time.sleep(self._interval_seconds)

    def store_to_file(self) -> None:
        """
        Writes the current aggregate metrics (average and maximum) to a file inside the csv_results_folder.
        """
        aggregate_metrics: List[str] = [f"{COL_GPU},{COL_METRIC},{COL_VALUE}"]
        for util in self.gpu_aggregates.values():
            for (metric, value) in util.average().enumerate():
                aggregate_metrics.append(f"{util.name},{metric},{value}")
        for util in self.gpu_max.values():
            for (metric, value) in util.enumerate(prefix="Max"):
                aggregate_metrics.append(f"{util.name},{metric},{value}")
        self.aggregate_metrics_file.write_text("\n".join(aggregate_metrics))

    def read_aggregate_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Reads the file containing aggregate metrics, and returns them parsed
        as nested dictionaries mapping from GPU name to metric name to value.
        """
        if not self.aggregate_metrics_file.is_file():
            return dict()
        df = pd.read_csv(self.aggregate_metrics_file)
        pivot = df.pivot(index=COL_GPU, columns=COL_METRIC, values=COL_VALUE)
        result = {index: series.to_dict() for index, series in pivot.iterrows()}
        return result
