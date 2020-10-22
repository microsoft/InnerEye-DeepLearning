#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass
from multiprocessing import Process
from typing import List, Tuple

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
    Converts a memory amount in bytes to gigabytes, rounding to two decimal places.
    :param bytes:
    :return:
    """
    gb = 2 ** 30
    return round(bytes / gb, 2)


@dataclass
class GpuUtilization:
    id: int
    load: float
    mem_util: float
    mem_allocated: float
    mem_reserved: float
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

    def average(self) -> GpuUtilization:
        return GpuUtilization(
            id=self.id,
            load=self.load / self.count,
            mem_util=self.mem_util / self.count,
            mem_allocated=self.mem_allocated / self.count,
            mem_reserved=self.mem_reserved / self.count,
            count=0
        )

    def enumerate(self) -> List[Tuple[str, float]]:
        return [
            (f'GPU{self.id}/MemUtil_Percent', self.mem_util * 100),
            (f'GPU{self.id}/Load_Percent', self.load * 100),
            (f'GPU{self.id}/MemReserved_GB', self.mem_reserved),
            (f'GPU{self.id}/MemAllocated_GB', self.mem_allocated)
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


class ResourceMonitor(Process):
    """
    Monitor and log GPU and CPU stats in AzureML as well as TensorBoard in a separate process.
    """

    def __init__(self, interval_seconds: int, tb_log_file_path: str):
        """
        Creates a process that will monitor CPU and GPU utilization.
        :param interval_seconds: The interval in seconds at which usage statistics should be written.
        :param tb_log_file_path: The path to a tensorboard logfile.
        """
        super().__init__(name="Resource Monitor", daemon=True)
        self._interval_seconds = interval_seconds
        self._tb_log_file_path = tb_log_file_path
        self.gpu_aggregates: List[GpuUtilization] = []

    def run(self) -> None:
        if self._interval_seconds <= 0:
            logging.warning("Resource monitoring requires an interval that is larger than 0 seconds, but "
                            f"got: {self._interval_seconds}. Exiting.")
            self.kill()
        logging.info(f"Process '{self.name}' started with pid: {self.pid}")
        # create the TB writers and AML run context for this process
        writer = tensorboardX.SummaryWriter(self._tb_log_file_path)

        def log_to_tensorboard(label: str, value: float) -> None:
            writer.add_scalar(label, value)

        gpu_available = is_gpu_available()
        while True:
            if gpu_available:
                gpu_utils = []
                gpus: List[GPU] = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_util = GpuUtilization.from_gpu(gpu)
                    for (name, value) in gpu_util.enumerate():
                        log_to_tensorboard(name, value)
                    gpu_utils.append(gpu_util)
                if self.gpu_aggregates:
                    for i in range(len(self.gpu_aggregates)):
                        self.gpu_aggregates[i] = self.gpu_aggregates[i] + gpu_utils[i]
                else:
                    self.gpu_aggregates = gpu_utils
            # log the CPU utilization
            log_to_tensorboard(f'CPU/Load_Percent', psutil.cpu_percent(interval=None))
            log_to_tensorboard(f'CPU/MemUtil_Percent', psutil.virtual_memory()[2])
            # pause the thread for the requested delay
            time.sleep(self._interval_seconds)

    def kill(self) -> None:
        run_context = Run.get_context()
        if not is_offline_run_context(run_context):
            for util in self.gpu_aggregates:
                for (name, value) in util.average().enumerate():
                    run_context.log(name, value)
        super().kill()
