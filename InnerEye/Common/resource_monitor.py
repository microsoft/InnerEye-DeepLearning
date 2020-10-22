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
from typing import List

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
    load: float
    mem_util: float
    mem_allocated: float
    mem_reseverd: float
    count: int

    def __add__(self, other: GpuUtilization) -> GpuUtilization:
        return GpuUtilization(
            load=self.load + other.load,
            mem_util=self.mem_util + other.mem_util,
            mem_allocated=self.mem_allocated + other.mem_allocated,
            mem_reseverd=self.mem_reseverd + other.mem_reseverd,
            count=self.count + other.count
        )

    def average(self) -> GpuUtilization:
        return GpuUtilization(
            load=self.load/self.count,
            mem_util=self.mem_util/self.count,
            mem_allocated=self.mem_allocated/self.count,
            mem_reseverd=self.mem_reseverd/self.count,
            count=0
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

        prefix = "Diagnostics/"
        gpu_available = is_gpu_available()
        while True:
            if gpu_available:
                gpus: List[GPU] = GPUtil.getGPUs()
                if len(gpus) > 0:
                    for gpu in gpus:
                        log_to_tensorboard(f'{prefix}GPU{gpu.id}/MemUtil_Percent', gpu.memoryUtil * 100)
                        log_to_tensorboard(f'{prefix}GPU{gpu.id}/Load_Percent', gpu.load * 100)
                        log_to_tensorboard(f'{prefix}GPU{gpu.id}/MemReserved_GB',
                                           memory_in_gb(torch.cuda.memory_reserved(int(gpu.id))))
                        log_to_tensorboard(f'{prefix}GPU{gpu.id}/MemAllocated_GB',
                                           memory_in_gb(torch.cuda.memory_allocated(int(gpu.id))))
                    # log the average GPU usage
                    log_to_tensorboard(f'{prefix}GPU/Average_Load_Percent',
                                       statistics.mean(map(lambda x: x.load, gpus)) * 100)
                    log_to_tensorboard(f'{prefix}GPU/Average_MemUtil_Percent',
                                       statistics.mean(map(lambda x: x.memoryUtil, gpus)) * 100)

            # log the CPU util
            log_to_tensorboard(f'{prefix}CPU/Load_Percent', psutil.cpu_percent(interval=None))
            log_to_tensorboard(f'{prefix}CPU/MemUtil_Percent', psutil.virtual_memory()[2])
            # pause the thread for the requested delay
            time.sleep(self._interval_seconds)

    def kill(self) -> None:
        run_context = Run.get_context()
        if not is_offline_run_context(run_context):
            run_context.log("Diagnostics/Total", 0.5)
        super().kill()
