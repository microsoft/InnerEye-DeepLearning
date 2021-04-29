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
    assert isinstance(resnet18.cnn_model, DenseNet121Encoder)
    assert densenet121.get_output_feature_dim() == 1024
