#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Dict, List, Tuple

import GPUtil
import psutil
import tensorboardX
import torch
from GPUtil import GPU
from azureml.core import Run

from InnerEye.Azure.azure_util import is_offline_run_context
from InnerEye.ML.utils.ml_util import is_gpu_available


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
    mem_allocated: float
    # Reserved memory by pytorch
    mem_reserved: float
    # Number of observations that are stored in the present object
    count: int

    def __add__(self, other: GpuUtilization) -> GpuUtilization:
        return GpuUtilization(
            id=self.id,
            load=self.load + other.load,
            mem_util=self.mem_util + other.mem_util,
            mem_allocated=self.mem_allocated + other.mem_allocated,
            mem_reserved=self.mem_reserved + other.mem_reserved,
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
            mem_allocated=max(self.mem_allocated, other.mem_allocated),
            mem_reserved=max(self.mem_reserved, other.mem_reserved),
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
            mem_allocated=self.mem_allocated / self.count,
            mem_reserved=self.mem_reserved / self.count,
            count=1
        )

    def enumerate(self, prefix: str = "") -> List[Tuple[str, float]]:
        """
        Lists all metrics stored in the present object, as (metric_name, value) pairs suitable for logging in
        Tensorboard. All metrics have a prefix like "GPU2/" for GPU ID 2.
        :param prefix: If provided, this string as used as an additional prefix for the metric name itself. If prefix
        is "max", the metric would look like "GPU2/maxLoad_Percent"
        :return: A list of (name, value) tuples.
        """
        return [
            (f'GPU{self.id}/{prefix}MemUtil_Percent', self.mem_util * 100),
            (f'GPU{self.id}/{prefix}Load_Percent', self.load * 100),
            (f'GPU{self.id}/{prefix}MemReserved_GB', self.mem_reserved),
            (f'GPU{self.id}/{prefix}MemAllocated_GB', self.mem_allocated)
        ]

    @staticmethod
    def from_gpu(gpu: GPU):
        return GpuUtilization(
            id=gpu.id,
            load=gpu.load,
            mem_util=gpu.memoryUtil,
            mem_allocated=memory_in_gb(torch.cuda.memory_allocated(int(gpu.id))),
            mem_reserved=memory_in_gb(torch.cuda.memory_reserved(int(gpu.id))),
            count=1
        )


RESOURCE_MONITOR_AGGREGATE_METRICS = "aggregate_resource_usage.txt"


class ResourceMonitor(Process):
    """
    Monitor and log GPU and CPU stats in AzureML as well as TensorBoard in a separate process.
    """

    def __init__(self, interval_seconds: int, tensorboard_folder: Path):
        """
        Creates a process that will monitor CPU and GPU utilization.
        :param interval_seconds: The interval in seconds at which usage statistics should be written.
        :param tensorboard_folder: The path in which to create a tensorboard logfile.
        """
        super().__init__(name="Resource Monitor", daemon=True)
        self._interval_seconds = interval_seconds
        self.tensorboard_folder = tensorboard_folder
        self.gpu_aggregates: Dict[int, GpuUtilization] = dict()
        self.gpu_max: Dict[int, GpuUtilization] = dict()
        self.writer = tensorboardX.SummaryWriter(str(self.tensorboard_folder))
        self.step = 0
        self.aggregate_metrics: List[str] = []

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
            for (name, value) in gpu_util.enumerate():
                self.log_to_tensorboard(name, value)
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
            self.log_to_tensorboard(f'CPU/Load_Percent', psutil.cpu_percent(interval=None))
            self.log_to_tensorboard(f'CPU/MemUtil_Percent', psutil.virtual_memory()[2])
            self.step += 1
            # pause the thread for the requested delay
            time.sleep(self._interval_seconds)

    def flush(self) -> None:
        """
        Writes the aggregate GPU utilization metrics to AzureML, and flushes all writers.
        """
        run_context = Run.get_context()
        aggregate_metrics: List[str] = []
        for util in self.gpu_aggregates.values():
            for (name, value) in util.average().enumerate():
                aggregate_metrics.append(f"{name}: {value}")
                if not is_offline_run_context(run_context):
                    run_context.log(name, value)
        for util in self.gpu_max.values():
            for (name, value) in util.average().enumerate(prefix="Max"):
                aggregate_metrics.append(f"{name}: {value}")
                if not is_offline_run_context(run_context):
                    run_context.log(name, value)
        self.aggregate_metrics = aggregate_metrics
        aggregates_file = self.tensorboard_folder / RESOURCE_MONITOR_AGGREGATE_METRICS
        aggregates_file.write_text("\n".join(aggregate_metrics))
        if not is_offline_run_context(run_context):
            run_context.flush()
        self.writer.flush()

    def kill(self) -> None:
        super().kill()
