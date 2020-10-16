#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import statistics
import time
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
                            "got: {}. Exiting.".format(self._interval_seconds))
        logging.info("Process ({}) started with pid: {}".format(self.name, self.pid))
        # create the TB writers and AML run context for this process
        writer = tensorboardX.SummaryWriter(self._tb_log_file_path)
        run_context = Run.get_context()
        is_offline_run = is_offline_run_context(run_context)
        current_iteration = 0

        def log_to_azure_and_tb(label: str, value: float) -> None:
            writer.add_scalar(label, value, current_iteration)
            if not is_offline_run:
                run_context.log(label, value)

        gpu_available = is_gpu_available()
        while True:
            if gpu_available:
                gpus: List[GPU] = GPUtil.getGPUs()
                if len(gpus) > 0:
                    for gpu in gpus:
                        log_to_azure_and_tb('Diagnostics/GPU_{}_CUDA_Memory_Reserved'.format(gpu.id),
                                            torch.cuda.memory_reserved(int(gpu.id)))
                        log_to_azure_and_tb('Diagnostics/GPU_{}_CUDA_Memory_Allocated'.format(gpu.id),
                                            torch.cuda.memory_allocated(int(gpu.id)))
                        log_to_azure_and_tb('Diagnostics/GPU_{}_Load_Percent'.format(gpu.id),
                                            gpu.load * 100)
                        log_to_azure_and_tb('Diagnostics/GPU_{}_MemUtil_Percent'.format(gpu.id),
                                            gpu.memoryUtil * 100)
                    # log the average GPU usage
                    log_to_azure_and_tb('Diagnostics/Average_GPU_Load_Percent',
                                        statistics.mean(map(lambda x: x.load, gpus)) * 100)
                    log_to_azure_and_tb('Diagnostics/Average_GPU_MemUtil_Percent',
                                        statistics.mean(map(lambda x: x.memoryUtil, gpus)) * 100)

            # log the CPU util
            log_to_azure_and_tb('Diagnostics/CPU_Util_Percent', psutil.cpu_percent(interval=None))
            log_to_azure_and_tb('Diagnostics/CPU_MemUtil_Percent', psutil.virtual_memory()[2])

            current_iteration += 1
            # pause the thread for the requested delay
            time.sleep(self._interval_seconds)
