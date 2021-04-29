#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pl_bolts.models.self_supervised.resnets import ResNet

from InnerEye.ML.SSL.encoders import DenseNet121Encoder, SSLEncoder
from InnerEye.ML.SSL.lightning_containers.ssl_container import EncoderName


def test_get_encoder_dim_within_encoder_class() -> None:
    """
    Tests initialization of various SSLEncoder and computation of corresponding output_feature_dim
    """
    resnet18 = SSLEncoder(EncoderName.resnet18.value)
    assert isinstance(resnet18.cnn_model, ResNet)
    assert resnet18.get_output_feature_dim() == 512
    resnet50 = SSLEncoder(EncoderName.resnet50.value)
    assert isinstance(resnet18.cnn_model, ResNet)
    assert resnet50.get_output_feature_dim() == 2048
    densenet121 = SSLEncoder(EncoderName.densenet121.value)
    assert isinstance(densenet121.cnn_model, DenseNet121Encoder)
    assert densenet121.get_output_feature_dim() == 1024


def test_use7x7conv_flag_in_encoder() -> None:
    """
    Tests the use_7x7_first_conv_in_resnet flag effect on encoder definition
    """
    resnet18 = SSLEncoder(EncoderName.resnet18.value, use_7x7_first_conv_in_resnet=True)
    assert resnet18.cnn_model.conv1.kernel_size == (7, 7)
    resnet18_for_cifar = SSLEncoder(EncoderName.resnet18.value, use_7x7_first_conv_in_resnet=False)
    assert resnet18_for_cifar.cnn_model.conv1.kernel_size == (3, 3)
