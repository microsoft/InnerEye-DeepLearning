#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
from typing import Optional

import torch


def get_local_rank() -> int:
    """Returns the local rank of the current process for AML (online) runs"""
    return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])


def get_global_rank(is_offline_run: Optional[bool] = True) -> int:
    """Returns the global rank of the current process for AML (online) runs."""
    if is_offline_run:
        # assume 1 machine only
        return torch.cuda.device_count()
    return int(os.environ["OMPI_COMM_WORLD_RANK"])


def get_global_size(is_offline_run: Optional[bool] = True) -> int:
    """
    If running in AML, will return the total number of devices across all machines. Otherwise,
    will return all devices on current machine
    :return:
    """
    if is_offline_run:
        # assume 1 machine only
        return torch.cuda.device_count()
    return int(os.environ['OMPI_COMM_WORLD_SIZE'])


def get_local_size(is_offline_run: Optional[bool] = True) -> int:
    """Get the number of devices on current machine (whether running in AML or locally)"""
    if is_offline_run:
        return torch.cuda.device_count()
    return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
