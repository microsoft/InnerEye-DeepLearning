#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from unittest import mock

from GPUtil import GPU

from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Common.resource_monitor import GpuUtilization, ResourceMonitor


def test_utilization_enumerate() -> None:
    """
    Test if metrics are converted correctly into a loggable format.
    """
    u1 = GpuUtilization(
        id=1,
        load=0.1,
        mem_util=0.2,
        mem_allocated_gb=30,
        mem_reserved_gb=40,
        count=1
    )
    metrics1 = u1.enumerate()
    assert len(metrics1) == 4
    assert metrics1 == [
        # Utilization should be multiplied by 100 to get per-cent
        ('GPU1/MemUtil_Percent', 20.0),
        ('GPU1/Load_Percent', 10.0),
        ('GPU1/MemReserved_GB', 40),
        ('GPU1/MemAllocated_GB', 30),
    ]
    metrics2 = u1.enumerate(prefix="Foo")
    assert len(metrics2) == 4
    assert metrics2[0] == ('GPU1/FooMemUtil_Percent', 20.0)


def test_utilization_add() -> None:
    """
    Test arithmetic operations on a GPUUtilization object.
    """
    u1 = GpuUtilization(
        id=1,
        load=10,
        mem_util=20,
        mem_allocated_gb=30,
        mem_reserved_gb=40,
        count=1
    )
    u2 = GpuUtilization(
        id=2,
        load=100,
        mem_util=200,
        mem_allocated_gb=300,
        mem_reserved_gb=400,
        count=9
    )
    sum = u1 + u2
    assert sum == GpuUtilization(
        id=1,
        load=110,
        mem_util=220,
        mem_allocated_gb=330,
        mem_reserved_gb=440,
        count=10
    )


def test_utilization_average() -> None:
    """
    Test averaging on GpuUtilization objects.
    """
    sum = GpuUtilization(
        id=1,
        load=110,
        mem_util=220,
        mem_allocated_gb=330,
        mem_reserved_gb=440,
        count=10
    )
    # Average is the metric value divided by count
    assert sum.average() == GpuUtilization(
        id=1,
        load=11,
        mem_util=22,
        mem_allocated_gb=33,
        mem_reserved_gb=44,
        count=1
    )


def test_utilization_max() -> None:
    """
    Test if metric-wise maximum is computed correctly.
    """
    u1 = GpuUtilization(
        id=1,
        load=1,
        mem_util=200,
        mem_allocated_gb=3,
        mem_reserved_gb=400,
        count=1
    )
    u2 = GpuUtilization(
        id=2,
        load=100,
        mem_util=2,
        mem_allocated_gb=300,
        mem_reserved_gb=400,
        count=9
    )
    assert u1.max(u2) == GpuUtilization(
        id=1,
        load=100,
        mem_util=200,
        mem_allocated_gb=300,
        mem_reserved_gb=400,
        count=10
    )


def test_resource_monitor(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if metrics are correctly updated in the ResourceMonitor class.
    """
    tensorboard_folder = test_output_dirs.root_dir
    r = ResourceMonitor(interval_seconds=5, tensorboard_folder=tensorboard_folder)

    def create_gpu(id: int, load: float, mem_total: float, mem_used: float) -> GPU:
        return GPU(ID=id, uuid=None, load=load, memoryTotal=mem_total, memoryUsed=mem_used,
                   memoryFree=None, driver=None, gpu_name=None,
                   serial=None, display_mode=None, display_active=None, temp_gpu=None)

    # Fake objects coming from GPUtil: Two entries for GPU1, 1 entry only for GPU2
    gpu1 = create_gpu(1, 0.1, 10, 2)  # memUti=0.2
    gpu2 = create_gpu(2, 0.2, 10, 3)  # memUti=0.3
    gpu3 = create_gpu(1, 0.3, 10, 5)  # memUti=0.5
    # Mock torch calls so that we can run on CPUs. memory allocated: 2GB, reserved: 1GB
    with mock.patch("torch.cuda.memory_allocated", return_value=2 ** 31):
        with mock.patch("torch.cuda.memory_reserved", return_value=2 ** 30):
            # Update with results for both GPUs
            r.update_metrics([gpu1, gpu2])
            # Next update with data for GPU2 missing
            r.update_metrics([gpu3])
    # Element-wise maximum of metrics
    assert r.gpu_max == {
        1: GpuUtilization(id=1, load=0.3, mem_util=0.5, mem_allocated_gb=2.0, mem_reserved_gb=1.0, count=2),
        2: GpuUtilization(id=2, load=0.2, mem_util=0.3, mem_allocated_gb=2.0, mem_reserved_gb=1.0, count=1),
    }
    # Aggregates should contain the sum of metrics that were observed.
    assert r.gpu_aggregates == {
        1: GpuUtilization(id=1, load=0.4, mem_util=0.7, mem_allocated_gb=4.0, mem_reserved_gb=2.0, count=2),
        2: GpuUtilization(id=2, load=0.2, mem_util=0.3, mem_allocated_gb=2.0, mem_reserved_gb=1.0, count=1),
    }
    r.writer.flush()
    r.store_to_file()
    tb_file = list(tensorboard_folder.rglob("*tfevents*"))[0]
    assert os.path.getsize(str(tb_file)) > 100
    assert r.aggregate_metrics_file.is_file
    assert len(r.aggregate_metrics_file.read_text().splitlines()) == 17
    parsed_metrics = r.read_aggregate_metrics()
    assert len(parsed_metrics) == 16


def test_resource_monitor_store_to_file(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if storing metrics to a file works correctly.
    """
    tensorboard_folder = test_output_dirs.root_dir
    r = ResourceMonitor(interval_seconds=5, tensorboard_folder=tensorboard_folder)
    r.gpu_aggregates = {
        1: GpuUtilization(id=1, mem_util=1, load=2, mem_reserved_gb=30.0, mem_allocated_gb=40.0, count=10),
    }
    r.gpu_max = {
        1: GpuUtilization(id=1, mem_util=0.4, load=0.5, mem_reserved_gb=6.0, mem_allocated_gb=7.0, count=10),
    }
    r.store_to_file()
    # Write a second time - we expect that to overwrite and only produce one set of metrics
    r.store_to_file()
    parsed_metrics = r.read_aggregate_metrics()
    assert parsed_metrics == [
        ("GPU1/MemUtil_Percent", 10.0),
        ("GPU1/Load_Percent", 20.0),
        ("GPU1/MemReserved_GB", 3.0),
        ("GPU1/MemAllocated_GB", 4.0),
        ("GPU1/MaxMemUtil_Percent", 40.0),
        ("GPU1/MaxLoad_Percent", 50.0),
        ("GPU1/MaxMemReserved_GB", 6.0),
        ("GPU1/MaxMemAllocated_GB", 7.0),
    ]
