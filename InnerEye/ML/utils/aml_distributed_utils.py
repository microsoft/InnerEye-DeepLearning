#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
from typing import Optional

import torch

from InnerEye.ML.model_config_base import ModelConfigBase


def get_local_rank() -> int:
    """Returns the local rank of the current process for AML (online) runs."""
    rank = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK')
    assert isinstance(rank, str), "Expected env var 'OMPI_COMM_WORLD_LOCAL_RANK' - perhaps this isn't an MPI run?"
    return int(rank)


def get_global_rank() -> int:
    """Returns the global rank of the current process for AML (online) runs."""
    rank = os.environ.get("OMPI_COMM_WORLD_RANK")
    assert isinstance(rank, str), "Expected env var 'OMPI_COMM_WORLD_RANK' - perhaps this isn't an MPI run?"
    return int(rank)


def get_global_size(config: ModelConfigBase) -> int:
    """
    If running in AML, global size is the total number of devices across all machines. Otherwise,
    asssumes 1 machine only, and will set global size as all devices on current machine. In both cases,
    global size is the maximum possible number of devices, but we may use fewer, if specified in the config
    :return:
    """
    max_world_size_from_config = config.workers_per_node * config.node_count
    if is_aml_mpi_run(config):
        global_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    else:
        global_size = torch.cuda.device_count()
    return min(global_size, max_world_size_from_config)


def get_local_size(config: ModelConfigBase) -> int:
    """
    Get the number of devices on current machine (whether running in AML or locally).In both cases,
    local size is the maximum possible number of devices, but we may use fewer, if specified in the config
    """
    if is_aml_mpi_run(config):
        local_size = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    else:
        local_size = torch.cuda.device_count()
    return min(local_size, config.workers_per_node)


def get_az_batch_master_node() -> Optional[str]:
    """
    If AML MPI job, environment variable named should exist
    :return:
    """
    master_node_addr = os.environ['$AZ_BATCH_MASTER_NODE']
    return master_node_addr


def is_aml_mpi_run(config: ModelConfigBase) -> bool:
    """
    Proxy for whether run is an AML MPI job (in which case init_method will be replaced with
    tcp communication address instead of using environment vars)
    Another proxy could be whether the environment variable $AZ_BATCH_MASTER_NODE has been set? (see above)
    :param config:
    :return:
    """
    return (config.distributed_training_init_method.startswith('tcp://')) and (not config.is_offline_run)
