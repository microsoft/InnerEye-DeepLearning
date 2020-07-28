#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging

import pytest
import torch

from InnerEye.Common import common_util
from InnerEye.ML.models.architectures.unet_3d import UNet3D
from InnerEye.ML.visualizers.model_summary import ModelSummary
from Tests.ML.util import machine_has_gpu, no_gpu_available


@pytest.mark.gpu
def test_unet_summary_generation() -> None:
    """Checks unet summary generation works either in CPU or GPU"""
    model = UNet3D(input_image_channels=1,
                   initial_feature_channels=2,
                   num_classes=2,
                   kernel_size=1,
                   num_downsampling_paths=2)
    if machine_has_gpu:
        model.cuda()
    summary = ModelSummary(model=model).generate_summary(input_sizes=[(1, 4, 4, 4)])
    assert summary is not None


@pytest.mark.skipif(common_util.is_windows(), reason="Has issues on windows build")
@pytest.mark.gpu
def test_unet_model_parallel() -> None:
    """Checks model parallel utilises all the available GPU devices for forward pass"""
    if no_gpu_available:
        logging.warning("CUDA capable GPU is not available - UNet Model Parallel cannot be tested")
        return
    model = UNet3D(input_image_channels=1,
                   initial_feature_channels=2,
                   num_classes=2,
                   kernel_size=1,
                   num_downsampling_paths=2).cuda()
    # Partition the network across all available gpu
    available_devices = [torch.device('cuda:{}'.format(ii)) for ii in range(torch.cuda.device_count())]
    model.generate_model_summary()
    model.partition_model(devices=available_devices)

    # Verify that all the devices are utilised by the layers of the model
    summary = ModelSummary(model=model).generate_summary(input_sizes=[(1, 4, 4, 4)])
    layer_devices = set()
    for layer_summary in summary.values():
        if layer_summary.device:
            layer_devices.add(layer_summary.device)

    assert layer_devices == set(available_devices)
