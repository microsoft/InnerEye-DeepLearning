#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os

from typing import Optional

import torch


def get_local_rank() -> int:
    """Returns the local rank of the current process for AML (online) runs."""
    rank = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK')
    return int(rank)


def get_global_rank() -> int:
    """Returns the global rank of the current process for AML (online) runs."""
    rank = os.environ.get("OMPI_COMM_WORLD_RANK")
    return int(rank)


def get_global_size(config) -> int:
    """
    If running in AML, will return the total number of devices across all machines. Otherwise,
    asssumes 1 machine only, and will return all devices on current machine
    :return:
    """
    if is_aml_mpi_run(config):
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    return torch.cuda.device_count()


def get_local_size(config) -> int:
    """Get the number of devices on current machine (whether running in AML or locally)"""
    if is_aml_mpi_run(config):
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    return torch.cuda.device_count()


def is_aml_mpi_run(config):
    """
    Proxy for whether run is an AML MPI job (in which case init_method will be replaced with
    tcp communication address instead of using environment vars)
    :param config:
    :return:
    """
    return (config.init_method.startswith('tcp://')) and (not config.is_offline_run)
