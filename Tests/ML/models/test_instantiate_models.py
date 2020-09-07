#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import io
import logging
import os
from pathlib import Path
from tempfile import mkstemp
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest
from azure.storage.blob import BlockBlobService
from pandas import DataFrame, Series

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_util import RUN_CONTEXT, is_offline_run_context
from InnerEye.Common.common_util import logging_to_stdout, namespace_to_path
from InnerEye.Common.fixed_paths import DATASETS_ACCOUNT_NAME
from InnerEye.ML.common import DATASET_CSV_FILE_NAME
from InnerEye.ML.config import PaddingMode, SegmentationModelBase
from InnerEye.ML.dataset import full_image_dataset
from InnerEye.ML.model_training import generate_and_print_model_summary
from InnerEye.ML.utils.config_util import ModelConfigLoader
from InnerEye.ML.utils.csv_util import CSV_INSTITUTION_HEADER, CSV_SUBJECT_HEADER
from InnerEye.ML.utils.io_util import read_image_as_array_with_header
from InnerEye.ML.utils.model_util import create_model_with_temperature_scaling
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import get_default_azure_config, get_model_loader

DIM_X = 'DIM_X'
DIM_Y = 'DIM_Y'
DIM_Z = 'DIM_Z'


def fix_dataset_csv(df: DataFrame, dataset_id: str, blob_service: BlockBlobService, container_name: str) -> DataFrame:
    """
    From an existing DataFrame it creates a new DataFrame that has DIM_X, DIM_Y AND DIM_Z as columns
    :param df: The DataFrame with the dataset.csv
    :param dataset_id: The dataset identifier
    :param blob_service: the blob service for datasets
    :return: a new dataset that contains the columns DimX, DimY, DimZ
    """
    dim_x_series: List[int] = []
    dim_y_series: List[int] = []
    dim_z_series: List[int] = []
    for index, row in df.iterrows():
        img_path = dataset_id + '/' + row[full_image_dataset.CSV_PATH_HEADER]
        all_suffixes = "".join(Path(img_path).suffixes)
        fd, temp_path = mkstemp(suffix=all_suffixes)
        blob_service.get_blob_to_path(container_name=container_name,
                                      file_path=temp_path,
                                      blob_name=img_path)
        image, _ = read_image_as_array_with_header(Path(temp_path))
        image_spatial_shape = np.array(image.shape)
        os.close(fd)
        os.remove(temp_path)
        dim_x_series.append(image_spatial_shape[0])
        dim_y_series.append(image_spatial_shape[1])
        dim_z_series.append(image_spatial_shape[2])

    df = df.assign(DIM_X=dim_x_series)
    df = df.assign(DIM_Y=dim_y_series)
    df = df.assign(DIM_Z=dim_z_series)
    return df


def check_is_not_empty(dataframe: pd.DataFrame, error: str) -> None:
    if dataframe.size == 0:
        raise ValueError(error)


def check_dataset_csv(config: SegmentationModelBase, azure_config: AzureConfig) -> None:
    """
    Checks that the configuration is compatible with the dataset.csv file
        - It checks that all the images are bigger than the crop_size
    :param config: The configuration for the model
    :return: None or raises a ValueError if the configuration is incorrect
    """
    dataset_id = config.azure_dataset_id
    if dataset_id is None:
        return
    dataset_id = str(dataset_id)
    block_blob_service = BlockBlobService(account_name=DATASETS_ACCOUNT_NAME,
                                          account_key=azure_config.get_dataset_storage_account_key())
    csv_content = block_blob_service.get_blob_to_text(container_name=azure_config.datasets_container,
                                                      blob_name=dataset_id + "/" + DATASET_CSV_FILE_NAME)
    csv_content_io = io.StringIO()
    csv_content_io.write(csv_content.content)
    csv_content_io.seek(0)
    df: DataFrame = pd.read_csv(csv_content_io)

    def count_by_institution(df: DataFrame, count_suffix: Optional[str] = None) -> DataFrame:
        count_col = CSV_SUBJECT_HEADER if count_suffix is None else CSV_SUBJECT_HEADER + count_suffix
        df2 = df if count_suffix is None else df.rename(columns={CSV_SUBJECT_HEADER: count_col})
        uniques = df2[[CSV_INSTITUTION_HEADER, count_col]].drop_duplicates()
        count = uniques.groupby(by=CSV_INSTITUTION_HEADER).count()
        return count

    if CSV_INSTITUTION_HEADER in df and CSV_SUBJECT_HEADER in df:
        institution_count = count_by_institution(df).sort_values(by=CSV_SUBJECT_HEADER, ascending=False)
        config.dataset_data_frame = df
        splits = config.get_dataset_splits()
        split_train = count_by_institution(splits.train, count_suffix="_train")
        split_test = count_by_institution(splits.test, count_suffix="_test")
        split_val = count_by_institution(splits.val, count_suffix="_val")
        totals = institution_count.join(split_train, on=CSV_INSTITUTION_HEADER)
        totals = totals.join(split_val, on=CSV_INSTITUTION_HEADER)
        totals = totals.join(split_test, on=CSV_INSTITUTION_HEADER)
        totals = totals.fillna(0)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 150):
            print(totals)

    if DIM_X not in df.columns or DIM_Y not in df.columns or DIM_Z not in df.columns:
        df = fix_dataset_csv(df, dataset_id, block_blob_service, azure_config.datasets_container)
        # Override the dataset.csv in blob so we dont have to repeat this fix again
        s = io.StringIO()
        df.to_csv(s, mode='w', index=False)
        new_dataset_csv_text = s.getvalue()
        block_blob_service.create_blob_from_text(container_name=azure_config.datasets_container,
                                                 blob_name=dataset_id + "/" + DATASET_CSV_FILE_NAME,
                                                 text=new_dataset_csv_text)

    def print_min_max(series: Series, message: str) -> None:
        print("{}: min = {}, max = {}".format(message, series.min(), series.max()))

    # Check that the test, validation and training sets do not overlap.
    config.dataset_data_frame = df
    splits = config.get_dataset_splits()

    # check crop size >= spatial size of all Train and Val images as we do not perform padding during training.
    train_val_df = pd.concat([splits.train, splits.val])
    print_min_max(train_val_df.DIM_X, "Train/Val DIM_X")
    print_min_max(train_val_df.DIM_Y, "Train/Val DIM_Y")
    print_min_max(train_val_df.DIM_Z, "Train/Val DIM_Z")

    # Check that the crop size is compatible with the Training and Validation splits
    is_padding_enabled = config.padding_mode != PaddingMode.NoPadding
    crop_size_differences: List[str] = []
    for index, row in train_val_df.iterrows():
        # Reversing to be able to compare with crop_size that is Z Y X
        image_spatial_shape = np.array([row[DIM_Z], row[DIM_Y], row[DIM_X]])
        diff = image_spatial_shape - np.array(config.crop_size)
        if not all(x >= 0 for x in diff):
            if is_padding_enabled:
                msg = f"subjectID {row.subject} with crop size {config.crop_size} in volume with " \
                      f"dimensions {image_spatial_shape} for dataset {dataset_id} will be padded with " \
                      f"padding scheme: {config.padding_mode.value}"
            else:
                msg = f"Invalid subjectID {row.subject} with crop size {config.crop_size} in volume with dimensions " \
                      f"{image_spatial_shape} for dataset {dataset_id}"
            crop_size_differences.append(msg)

    if len(crop_size_differences) > 0:
        diff_str = "\n".join(crop_size_differences)
        if is_padding_enabled:
            logging.warning(diff_str)
        else:
            raise ValueError(diff_str)


def find_models() -> List[str]:
    """
    Lists all Python files in the configs folder. Each of them is assumed to contain one model config.
    :return: list of models
    """
    path = namespace_to_path(ModelConfigLoader.get_default_search_module())
    folders = [path / "segmentation", path / "classification", path / "regression"]
    names = [str(f.stem) for folder in folders for f in folder.glob("*.py") if folder.exists()]
    return [name for name in names if not name.endswith("Base") and not name.startswith("__")]


def test_any_models_found() -> None:
    """
    Test that the basic setup for finding all model configs works: At least one of
    the models in master must be found.
    """
    model_names = find_models()
    assert len(model_names) > 0
    assert "Lung" in model_names
    # Test that all configs in the classification folder are picked up as well
    assert "DummyClassification" in model_names


@pytest.mark.parametrize("model_name", find_models())
@pytest.mark.gpu
def test_load_all_configs(model_name: str) -> None:
    """
    Loads all model configurations that are present in the ML/src/configs folder,
    and carries out basic validations of the configuration.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    config = ModelConfigLoader[SegmentationModelBase]().create_model_config_from_name(model_name)
    assert config.model_name == model_name, "Mismatch between definition .py file and model name"
    azure_config = get_default_azure_config()
    if config.is_segmentation_model:
        # Execute dataset checking only on local runs, in particular we don't want that to happen repeatedly in
        # AzureML runs
        if is_offline_run_context(RUN_CONTEXT):
            check_dataset_csv(config, azure_config)
        # Reduce the feature channels to a minimum, to make tests run fast on CPU.
        minimal_feature_channels = 1
        config.feature_channels = [minimal_feature_channels] * len(config.feature_channels)
        print("Model architecture after restricting to 2 feature channels only:")
        model = create_model_with_temperature_scaling(config)
        generate_and_print_model_summary(config, model)
    else:
        # For classification models, we can't always print a model summary: The model could require arbitrary
        # numbers of input tensors, and we'd only know once we load the training data.
        # Hence, only try to create the model, but don't attempt to print the summary.
        create_model_with_temperature_scaling(config)


def test_cross_validation_config() -> None:
    CrossValidationDummyModel(0, -1)
    CrossValidationDummyModel(10, 1)
    CrossValidationDummyModel(10, -1)

    with pytest.raises(ValueError):
        CrossValidationDummyModel(10, 11)
    with pytest.raises(ValueError):
        CrossValidationDummyModel(10, 10)


class CrossValidationDummyModel(DummyModel):
    def __init__(self, number_of_cross_validation_splits: int, cross_validation_split_index: int):
        self.number_of_cross_validation_splits = number_of_cross_validation_splits
        self.cross_validation_split_index = cross_validation_split_index
        super().__init__()


def test_model_config_loader() -> None:
    logging_to_stdout(log_level=logging.DEBUG)
    default_loader = get_model_loader()
    assert default_loader.create_model_config_from_name("BasicModel2Epochs") is not None
    with pytest.raises(ValueError):
        default_loader.create_model_config_from_name("DummyModel")
    loader_including_tests = get_model_loader(namespace="Tests.ML.configs")
    assert loader_including_tests.create_model_config_from_name("BasicModel2Epochs") is not None
    assert loader_including_tests.create_model_config_from_name("DummyModel") is not None


def test_config_loader_as_in_registration() -> None:
    """
    During model registration, the model config namespace is read out from the present model. Ensure that we
    can create a config loader that has that value as an input.
    """
    loader1 = ModelConfigLoader[SegmentationModelBase]()
    model_name = "BasicModel2Epochs"
    model = loader1.create_model_config_from_name(model_name)
    assert model is not None
    namespace = model.__module__
    loader2 = ModelConfigLoader[SegmentationModelBase](model_configs_namespace=namespace)
    assert len(loader2.module_search_specs) == 2
    model2 = loader2.create_model_config_from_name(model_name)
    assert model2 is not None
