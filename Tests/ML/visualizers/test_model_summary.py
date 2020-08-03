#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from abc import ABC
from typing import List, Tuple

import torch

from InnerEye.Common.common_util import logging_to_stdout
from InnerEye.Common.fixed_paths import DEFAULT_MODEL_SUMMARIES_DIR_PATH
from InnerEye.ML.configs.classification.GlaucomaPublic import GlaucomaPublic
from InnerEye.ML.models.architectures.base_model import BaseModel, CropSizeConstraints
from InnerEye.ML.models.architectures.classification.image_encoder_with_mlp import ImageEncoderWithMlp, \
    ImagingFeatureType
from InnerEye.ML.models.layers.basic import BasicLayer
from InnerEye.ML.utils.image_util import HDF5_NUM_SEGMENTATION_CLASSES
from InnerEye.ML.visualizers.model_summary import ModelSummary

logging_to_stdout(logging.INFO)


def test_model_summary() -> None:
    model = MyFavModel()
    input_size = (16, 16, 32)
    model.generate_model_summary(input_size)
    assert model.summary_crop_size == input_size
    assert model.summary is not None

    layer_summary = model.summary["Conv3d-1"]
    assert list(layer_summary.input_shape[0][1:]) == [1, 16, 16, 32]
    assert list(layer_summary.output_shape[1:]) == [2, 14, 16, 32]
    assert layer_summary.n_trainable_params == 6
    assert layer_summary.n_params == 6

    layer_summary = model.summary["ReLU-3"]
    assert layer_summary.n_trainable_params == 0

    layer_summary = model.summary["Conv3d-4"]
    assert list(layer_summary.input_shape[0][1:]) == [2, 14, 16, 32]
    assert list(layer_summary.output_shape[1:]) == [3, 14, 8, 30]
    assert layer_summary.n_trainable_params == 18

    layer_summary = model.summary["DuplicateLayer-7"]
    assert list(layer_summary.output_shape[1][1:]) == [3, 14, 8, 30]
    assert len(layer_summary.output_shape) == 2


def test_model_summary_on_minimum_crop_size() -> None:
    """
    Test that a model summary is generated when no specific crop size is specified.
    """
    model = MyFavModel()
    min_crop_size = (5, 6, 7)
    model.crop_size_constraints = CropSizeConstraints(minimum_size=min_crop_size)
    model.generate_model_summary()
    assert model.summary_crop_size == min_crop_size
    assert model.summary is not None


def test_model_summary_on_classification1() -> None:
    model = GlaucomaPublic().create_model()
    ModelSummary(model).generate_summary(input_sizes=[(1, 6, 64, 60)])


def test_model_summary_on_classification2() -> None:
    image_channels = 2
    model = ImageEncoderWithMlp(imaging_feature_type=ImagingFeatureType.Segmentation,
                                encode_channels_jointly=False,
                                num_encoder_blocks=3,
                                initial_feature_channels=2,
                                num_image_channels=image_channels,
                                mlp_dropout=0.5)
    summarizer = ModelSummary(model)
    summarizer.generate_summary(input_sizes=[(image_channels * HDF5_NUM_SEGMENTATION_CLASSES, 6, 32, 32)])
    assert summarizer.n_params != 0
    assert summarizer.n_trainable_params != 0


def test_log_model_summary_to_file() -> None:
    model = MyFavModel()
    input_size = (16, 16, 32)
    expected_log_file = DEFAULT_MODEL_SUMMARIES_DIR_PATH / "model_log001.txt"
    if expected_log_file.exists():
        expected_log_file.unlink()
    model.generate_model_summary(input_size, log_summaries_to_files=True)
    assert expected_log_file.exists()
    with expected_log_file.open() as inpt:
        assert len(inpt.readlines()) >= 3


class MyFavModel(BaseModel, ABC):
    def __init__(self) -> None:
        super().__init__(input_channels=1, name='MyFavModel')

        class DuplicateLayer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
                return x, x

        self._model = torch.nn.Sequential(
            BasicLayer(channels=(1, 2), kernel_size=(3, 1, 1)),
            BasicLayer(channels=(2, 3), kernel_size=(1, 1, 3), stride=(1, 2, 1), dilation=(2, 1, 1)),
            DuplicateLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self._model(x)

    def get_all_child_layers(self) -> List[torch.nn.Module]:
        return list(self._model.children())
