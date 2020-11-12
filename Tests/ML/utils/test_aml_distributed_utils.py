#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import pytest
from typing import Optional
from unittest import mock

from torch.cuda import device_count

from InnerEye.ML.utils.aml_distributed_utils import get_local_rank, get_global_rank, get_global_size, get_local_size, \
    is_aml_mpi_run
from Tests.ML.configs import DummyModel


@pytest.mark.parametrize("local_rank_env_var", [None, 1, 10])
def test_get_local_rank(local_rank_env_var: Optional[int]) -> None:
    """
    Test that get_local_rank returns the correct environment variable value if it exists
    (e.g. for an AML MPI run) and otherwise raises a TypeError
    :param local_rank_env_var:
    :return:
    """
    if local_rank_env_var is None:
        with pytest.raises(AssertionError):
            get_local_rank()
    else:
        with mock.patch.dict(os.environ, {'OMPI_COMM_WORLD_LOCAL_RANK': str(local_rank_env_var)}):
            rank = get_local_rank()
        assert rank == local_rank_env_var


@pytest.mark.parametrize("global_rank_env_var", [None, 5, 10])
def test_get_global_rank(global_rank_env_var: Optional[int]) -> None:
    """
    Test that get_global_rank returns the correct environment variable value if it exists
    (e.g. for an AML MPI run) and otherwise raises a TypeError
    :param global_rank_env_var:
    :return:
    """
    if global_rank_env_var is None:
        with pytest.raises(AssertionError):
            get_global_rank()
    else:
        with mock.patch.dict(os.environ, {'OMPI_COMM_WORLD_RANK': str(global_rank_env_var)}):
            rank = get_global_rank()
        assert rank == global_rank_env_var


@pytest.mark.parametrize(["num_nodes", "num_workers_per_node"], [(1, 1), (5, 4), (2, 3)])
def test_get_global_size_offline(num_nodes, num_workers_per_node) -> None:
    """
    Assert that, for an offline run, get_global_size returns the number of cuda devices
    on the current machine
    :param num_nodes:
    :param num_workers_per_node:
    :return:
    """
    with mock.patch("Tests.ML.configs.DummyModel") as MockConfig:
        MockConfig.return_value.num_nodes = num_nodes
        MockConfig.return_value.num_workers_per_node = num_workers_per_node
        mock_config = MockConfig()
    available_local_workers = min(device_count(), num_workers_per_node)
    # for offline run, we assume 1 node regardless of config
    expected_global_size = available_local_workers
    global_size = get_global_size(mock_config)
    assert global_size == expected_global_size


@pytest.mark.parametrize("expected_global_size", [1, 5, 10])
def test_get_global_size_aml(expected_global_size: int) -> None:
    """
    Assert that, for an AML run, get_global_size returns the value of the appropriate
    environment variable
    :param expected_global_size:
    :return:
    """
    with mock.patch("os.environ", {'OMPI_COMM_WORLD_SIZE': expected_global_size}):
        with mock.patch("Tests.ML.configs.DummyModel") as MockConfig:
            MockConfig.return_value.is_offline_run = False
            MockConfig.return_value.num_workers_per_node = expected_global_size
            MockConfig.return_value.num_nodes = 1
            config = MockConfig()
            global_size = get_global_size(config)
    assert global_size == expected_global_size


@pytest.mark.parametrize("requested_num_workers_per_node", [1, 2, 5, 10])
def test_get_local_size_offline(requested_num_workers_per_node) -> None:
    """
    Assert that, for an offline run, get_local_size returns the number of cuda devices
    on the current machine
    :param requested_num_workers_per_node:
    :return:
    """
    with mock.patch("Tests.ML.configs.DummyModel") as MockConfig:
        MockConfig.return_value.num_workers_per_node = requested_num_workers_per_node
        mock_config = MockConfig()
    available_local_workers = device_count()
    local_size = get_local_size(mock_config)
    assert local_size == min(requested_num_workers_per_node, available_local_workers)


@pytest.mark.parametrize("requested_num_workers_per_node", [1, 2, 3])
def test_get_local_size_aml(requested_num_workers_per_node: int) -> None:
    """
    Assert that, for an AML run, get_local_size returns the value of the appropriate
    environment variable
    :param requested_num_workers_per_node:
    :return:
    """
    with mock.patch.dict(os.environ, {'OMPI_COMM_WORLD_LOCAL_SIZE': str(requested_num_workers_per_node)}):
        with mock.patch("Tests.ML.configs.DummyModel") as MockConfig:
            MockConfig.return_value.is_offline_run = False
            MockConfig.return_value.num_workers_per_node = requested_num_workers_per_node
            config = MockConfig()
            local_size = get_local_size(config)
    assert local_size == requested_num_workers_per_node


def test_is_aml_mpi_run() -> None:
    """
    Assert that is_aml_mpi_run returns False, unless we have an AML run where the init_method uses TCP
    (by default it would use environment variables, but Azure's MPI job alters this).
    :return:
    """
    # By default, is_offline = True and init_method = "env://", so expect is_aml_mpi_run = False
    with mock.patch("Tests.ML.configs.DummyModel") as MockConfig:
        MockConfig.return_value.is_offline_run = True
        MockConfig.return_value.distributed_training_init_method = "env://"
        config1 = MockConfig()
        assert is_aml_mpi_run(config1) is False
    # if is_offline = True, still expect  is_aml_mpi_run = False, due to init_method
    with mock.patch("Tests.ML.configs.DummyModel") as MockConfig:
        MockConfig.return_value.is_offline_run = False
        MockConfig.return_value.distributed_training_init_method = "env://"
        config2 = MockConfig()
        assert is_aml_mpi_run(config2) is False
    # if init_method starts with "tcp://" but is_offline = True, still expect is_aml_mpi_run = False
    with mock.patch("Tests.ML.configs.DummyModel") as MockConfig:
        MockConfig.return_value.is_offline_run = True
        MockConfig.return_value.distributed_training_init_method = "tcp://"
        config = MockConfig()
        assert is_aml_mpi_run(config) is False
    # if init_method starts with "tcp://" and is_offline = False, expect is_aml_mpi_run = True
    with mock.patch("Tests.ML.configs.DummyModel") as MockConfig:
        MockConfig.return_value.is_offline_run = False
        MockConfig.return_value.distributed_training_init_method = "tcp://"
        config = MockConfig()
        assert is_aml_mpi_run(config) is True
