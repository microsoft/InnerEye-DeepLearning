#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List

import pytest
import torch
from torch import Tensor

from InnerEye.Common import common_util
from InnerEye.ML.models.architectures.base_model import BaseModel, CropSizeConstraints
from InnerEye.ML.models.losses.soft_dice import SoftDiceLoss
from InnerEye.ML.models.parallel.data_parallel import DataParallelCriterion
from InnerEye.ML.models.parallel.model_parallel import group_layers_with_balanced_memory, is_model_parallel, \
    move_to_device, partition_layers
from InnerEye.ML.utils.ml_util import is_gpu_available

no_gpu = not is_gpu_available()
no_or_single_gpu = not torch.cuda.is_available() or torch.cuda.device_count() <= 1


class SimpleModel(BaseModel):
    """
    A simple neural network model to test model parallelisation functions. 
    """

    def __init__(self, input_channels: Any, channels: Any, n_classes: int, kernel_size: int):
        # minimum crop size: Network first reduces size by 4, then halves, then multiplies by 2 and adds 1
        # 64 -> 62 -> 30 -> 61 -> 61
        super().__init__(name='SimpleModel',
                         input_channels=input_channels,
                         crop_size_constraints=CropSizeConstraints(minimum_size=6))
        self.channels = channels
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self._model = torch.nn.Sequential(
            torch.nn.Conv3d(input_channels, channels[0], kernel_size=self.kernel_size),
            torch.nn.Conv3d(channels[0], channels[1], kernel_size=self.kernel_size, stride=2),
            torch.nn.ConvTranspose3d(channels[1], channels[0], kernel_size=self.kernel_size, stride=2),
            torch.nn.ConvTranspose3d(channels[0], n_classes, kernel_size=1)
        )

    def forward(self, x: Any):  # type: ignore
        return self._model(x)

    def get_all_child_layers(self) -> List[torch.nn.Module]:
        return list(self._model.children())


@pytest.mark.skipif(common_util.is_windows(), reason="Has issues on windows build")
@pytest.mark.gpu
@pytest.mark.skipif(no_gpu, reason="CUDA capable GPU is not available")
def test_move_to_device() -> None:
    def assert_device_matches(tensors: List[Tensor], target_device: torch.device) -> None:
        for tensor in tensors:
            assert tensor.device == target_device

    target_device = torch.device('cuda:0')
    input_tensor_1 = torch.tensor(3, device=torch.device('cpu'))
    input_tensor_2 = torch.tensor(3, device=torch.device('cuda:0'))
    tensors = [input_tensor_1, input_tensor_2]
    moved = list(move_to_device(tensors, target_device=target_device))
    assert_device_matches(moved, target_device)

    if torch.cuda.device_count() > 1:
        target_device = torch.device('cuda:1')
        moved = list(move_to_device(tensors, target_device=target_device))
        assert_device_matches(moved, target_device)

    # Not supplying a target device should leave the tensor untouched
    moved = list(move_to_device(tensors, target_device=None))
    assert moved[0].device == tensors[0].device
    assert moved[1].device == tensors[1].device


@pytest.mark.gpu
@pytest.mark.skipif(no_or_single_gpu, reason="CUDA capable GPUs are not available")
def test_group_layers_with_balanced_memory() -> None:
    model = SimpleModel(input_channels=1, channels=[2, 3], n_classes=2, kernel_size=1).cuda()
    model.generate_model_summary(crop_size=(8, 8, 8))
    groups = group_layers_with_balanced_memory(model.get_all_child_layers(), num_groups=2, summary=model.summary)

    for group_id, group in enumerate(groups):
        assert len(group) == 2
        if group_id == 0:
            assert isinstance(group[0], torch.nn.Conv3d)
            assert isinstance(group[1], torch.nn.Conv3d)
        elif group_id == 1:
            assert isinstance(group[0], torch.nn.ConvTranspose3d)
            assert isinstance(group[1], torch.nn.ConvTranspose3d)


@pytest.mark.gpu
@pytest.mark.skipif(no_or_single_gpu, reason="CUDA capable GPUs are not available")
def test_partition_layers() -> None:
    model = SimpleModel(input_channels=1, channels=[2, 3], n_classes=2, kernel_size=1).cuda()
    model.generate_model_summary(crop_size=(8, 8, 8))
    summary = model.summary
    devices = [torch.device('cuda:{}'.format(ii)) for ii in range(2)]
    all_layers = model.get_all_child_layers()

    if summary is None:
        raise RuntimeError(
            "Network summary is required to partition UNet3D. Call model.generate_model_summary() first.")
    for layer in all_layers:
        assert not is_model_parallel(layer)

    partition_layers(layers=all_layers, summary=summary, target_devices=devices)

    assert all_layers[0].weight.device == torch.device("cuda:0")
    assert all_layers[1].weight.device == torch.device("cuda:0")
    assert all_layers[2].weight.device == torch.device("cuda:1")
    assert all_layers[3].weight.device == torch.device("cuda:1")

    for layer in all_layers:
        assert is_model_parallel(layer)


@pytest.mark.gpu
@pytest.mark.skipif(no_gpu, reason="CUDA capable GPU is not available")
def test_dataparallel_criterion() -> None:
    num_batches = torch.cuda.device_count()
    array_shape = [num_batches, 2, 8, 4, 8]
    segmentation = torch.rand(array_shape).cuda()
    ground_truth = torch.rand(array_shape).cuda()
    target_loss_values = torch.zeros(num_batches).cuda()

    # Sequential computation with multi-hardware parallelisation
    for ii in range(num_batches):
        loss_fn = SoftDiceLoss()
        loss = loss_fn(output=segmentation[ii:ii + 1], target=ground_truth[ii:ii + 1])
        target_loss_values[ii] = loss

    # Use parallel criterion
    parallel_loss_fn = DataParallelCriterion(SoftDiceLoss(), device_ids=range(torch.cuda.device_count()))
    segmentation_as_parallel = [segmentation[ii:ii + 1].to("cuda:{}".format(ii)) for ii in range(num_batches)]
    computed_loss_values = \
        parallel_loss_fn(segmentation_as_parallel[0], ground_truth) if num_batches == 1 \
            else parallel_loss_fn(segmentation_as_parallel, ground_truth)

    if num_batches == 1:
        assert torch.equal(target_loss_values[0], computed_loss_values)
    else:
        assert torch.equal(target_loss_values, computed_loss_values)
