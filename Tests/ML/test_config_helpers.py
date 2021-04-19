#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import List, Optional, Union

import pytest
import torch
from pandas import DataFrame

from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import ModelArchitectureConfig, SegmentationModelBase, equally_weighted_classes
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.models.architectures.base_model import BaseSegmentationModel
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.utils import ml_util
from InnerEye.ML.utils.model_util import build_net
from Tests.ML.configs.DummyModel import DummyModel


def test_validate_inference_stride_size() -> None:
    SegmentationModelBase.validate_inference_stride_size(inference_stride_size=(5, 3, 1), output_size=(5, 3, 1))
    SegmentationModelBase.validate_inference_stride_size(inference_stride_size=(5, 3, 1), output_size=None)
    with pytest.raises(ValueError):
        SegmentationModelBase.validate_inference_stride_size(inference_stride_size=(5, 3, 1), output_size=(3, 3, 3))
        SegmentationModelBase.validate_inference_stride_size(inference_stride_size=(5, 3, 0), output_size=None)


def test_inference_stride_size_setter() -> None:
    """Tests setter function raises an error when stride size is larger than output patch size"""
    test_output_size = (7, 3, 5)
    test_stride_size = (3, 3, 3)
    test_fail_stride_size = (1, 1, 9)
    model = IdentityModel()
    model_config = SegmentationModelBase(test_crop_size=test_output_size, should_validate=False)

    model_config.inference_stride_size = test_stride_size
    assert model_config.inference_stride_size == test_stride_size

    model_config.set_derived_model_properties(model)
    assert model_config.inference_stride_size == test_stride_size

    model_config.inference_stride_size = None
    model_config.set_derived_model_properties(model)
    assert model_config.inference_stride_size == test_output_size

    with pytest.raises(ValueError):
        model_config.inference_stride_size = test_fail_stride_size


def test_crop_size() -> None:
    """Checks if test crop size is equal to train crop size if it's not specified at init time"""
    model_config = DummyModel()
    assert model_config.test_crop_size == model_config.crop_size


def test_set_model_config_attributes() -> None:
    """Tests setter function for model config attributes"""
    train_output_size = (3, 5, 3)
    test_output_size = (7, 7, 7)
    model = IdentityModel()
    model_config = SegmentationModelBase(crop_size=train_output_size,
                                         test_crop_size=test_output_size,
                                         should_validate=False)

    model_config.set_derived_model_properties(model)
    assert model_config.inference_stride_size == test_output_size


# noinspection PyArgumentList
def test_get_output_size() -> None:
    """Tests config properties related to output tensor size"""
    train_output_size = (5, 5, 5)
    test_output_size = (7, 7, 7)

    model_config = SegmentationModelBase(crop_size=train_output_size,
                                         test_crop_size=test_output_size,
                                         should_validate=False)
    assert model_config.get_output_size(execution_mode=ModelExecutionMode.TRAIN) is None
    assert model_config.get_output_size(execution_mode=ModelExecutionMode.TEST) is None

    model = IdentityModel()
    model_config.set_derived_model_properties(model)
    assert model_config.get_output_size(execution_mode=ModelExecutionMode.TRAIN) == train_output_size
    assert model_config.get_output_size(execution_mode=ModelExecutionMode.TEST) == test_output_size


class IdentityModel(BaseSegmentationModel):
    def __init__(self) -> None:
        super().__init__(input_channels=1, name='IdentityModel')

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        # returns the input as it is
        return x

    def get_all_child_layers(self) -> List[torch.nn.Module]:
        return list()


@pytest.mark.parametrize(["num_fg_classes", "background_weight", "expected"],
                         [
                             (1, 0.2, [0.2, 0.8]),
                             (1, None, [0.5] * 2),
                             (9, None, [0.1] * 10),
                             (3, None, [1 / 4] * 4),
                             (3, 0.4, [0.4, 0.2, 0.2, 0.2]),
                         ])
def test_equally_weighted_classes(num_fg_classes: int, background_weight: Optional[float],
                                  expected: List[float]) -> None:
    classes = [""] * num_fg_classes
    actual = equally_weighted_classes(classes, background_weight)
    assert isinstance(actual, list)
    assert len(actual) == num_fg_classes + 1
    assert sum(actual) == pytest.approx(1.0)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(["num_fg_clases", "background_weight"],
                         [
                             (0, 0.5),
                             (1, 1.0),
                             (1, -0.1)
                         ])
def test_equally_weighted_classes_fails(num_fg_clases: int, background_weight: Optional[float]) -> None:
    classes = [""] * num_fg_clases
    with pytest.raises(ValueError):
        equally_weighted_classes(classes, background_weight)


def test_fields_are_set() -> None:
    """
    Tests that expected fields are set when creating config classes.
    """
    expected = [("hello", None), ("world", None)]
    config = SegmentationModelBase(
        should_validate=False,
        ground_truth_ids=[x[0] for x in expected],
        largest_connected_component_foreground_classes=expected
    )
    assert hasattr(config, CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY)
    assert config.largest_connected_component_foreground_classes == expected


@pytest.mark.cpu_and_gpu
def test_dataset_reader_workers() -> None:
    """
    Test to make sure the number of dataset reader workers are set correctly
    """
    config = ScalarModelBase(
        should_validate=False,
        num_dataset_reader_workers=-1
    )
    if config.is_offline_run:
        assert config.num_dataset_reader_workers == -1
    else:
        assert config.num_dataset_reader_workers == 0


def create_dataset_csv(test_output_dirs: OutputFolderForTests) -> Path:
    """Create dummy dataset csv file for tests,
    deleting any pre-existing file."""
    test_csv = "test_dataset.csv"
    root_dir = test_output_dirs.root_dir
    dataset_csv_path = root_dir / test_csv
    if dataset_csv_path.exists():
        dataset_csv_path.unlink()
    dataset_csv_path.write_text("""subject,channel,filePath""")
    return dataset_csv_path


def validate_dataset_paths(
        model_config: Union[ScalarModelBase, SegmentationModelBase]) -> None:
    """Check that validation of dataset paths is succeeds when csv file exists,
    and fails when it's missing."""
    assert model_config.local_dataset is not None
    ml_util.validate_dataset_paths(model_config.local_dataset, model_config.dataset_csv)

    dataset_csv_path = model_config.local_dataset / model_config.dataset_csv
    dataset_csv_path.unlink()

    ex_message = f"The dataset file {model_config.dataset_csv} is not present"
    with pytest.raises(ValueError) as ex:
        ml_util.validate_dataset_paths(model_config.local_dataset, model_config.dataset_csv)
    assert ex_message in str(ex)


def test_dataset_csv_with_SegmentationModelBase(
        test_output_dirs: OutputFolderForTests) -> None:
    dataset_csv_path = create_dataset_csv(test_output_dirs)
    model_config = SegmentationModelBase(should_validate=False)
    model_config.local_dataset = dataset_csv_path.parent
    model_config.dataset_csv = dataset_csv_path.name
    dataframe = model_config.read_dataset_if_needed()
    assert dataframe is not None
    validate_dataset_paths(model_config)


def test_dataset_csv_with_ScalarModelBase(
        test_output_dirs: OutputFolderForTests) -> None:
    dataset_csv_path = create_dataset_csv(test_output_dirs)
    model_config = ScalarModelBase(should_validate=False)
    model_config.local_dataset = dataset_csv_path.parent
    model_config.dataset_csv = dataset_csv_path.name
    model_config.read_dataset_into_dataframe_and_pre_process()
    assert model_config.dataset_data_frame is not None
    validate_dataset_paths(model_config)


def test_unet3_num_downsampling_paths() -> None:
    for num_downsampling_paths in range(1, 5):
        j = int(2 ** num_downsampling_paths)

        # Test that num_downsampling_paths for built UNet3D
        # is set via model configuration
        crop_size = (j, j, j)
        config = SegmentationModelBase(
            architecture=ModelArchitectureConfig.UNet3D,
            image_channels=["ct"],
            feature_channels=[1],
            crop_size=crop_size,
            num_downsampling_paths=num_downsampling_paths,
            should_validate=False)
        network = build_net(config)
        assert network.num_downsampling_paths == num_downsampling_paths

        # Test that exception is raised if crop size is smaller than is allowed
        # by num_downsampling_paths
        too_small_crop_size = (j // 2, j // 2, j // 2)
        ex_msg = f"Crop size is not valid. The required minimum is {crop_size}"
        config = SegmentationModelBase(
            architecture=ModelArchitectureConfig.UNet3D,
            image_channels=["ct"],
            feature_channels=[1],
            crop_size=too_small_crop_size,
            num_downsampling_paths=num_downsampling_paths,
            should_validate=False)
        with pytest.raises(ValueError) as ex:
            build_net(config)
        assert ex_msg in str(ex)


def test_config_str() -> None:
    """
    Check if dataframe fields are omitted from the string conversion of a config object.
    """
    config = DeepLearningConfig()
    df = DataFrame(columns=["foobar"], data=[1.0, 2.0])
    config.dataset_data_frame = df
    s = str(config)
    assert "foobar" not in s, f"Incorrect output: {s}"
