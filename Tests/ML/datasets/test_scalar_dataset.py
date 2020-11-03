#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
import shutil
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
import torch
from pandas.util.testing import assert_frame_equal

from InnerEye.Common import common_util
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.dataset.sample import GeneralSampleMetadata
from InnerEye.ML.dataset.scalar_dataset import DataSourceReader, ScalarDataSource, ScalarDataset, \
    _get_single_channel_row, _string_to_float, extract_label_classification, extract_label_regression, files_by_stem, \
    is_valid_item_index, load_single_data_source
from InnerEye.ML.photometric_normalization import WindowNormalizationForScalarItem, mri_window
from InnerEye.ML.scalar_config import LabelTransformation, ScalarLoss, ScalarModelBase
from InnerEye.ML.utils.dataset_util import CategoricalToOneHotEncoder
from Tests.ML.util import create_dataset_csv_file
from Tests.fixed_paths_for_tests import full_ml_test_data_path


def test_get_single_row() -> None:
    """
    Test if we can extract the unique row that has a given value in the "channel" column.
    """
    csv_string = StringIO("""id,channel,value
S1,single,foo
S2,double,bar
S2,double,baz
""")
    df = pd.read_csv(csv_string, sep=",")
    row1 = _get_single_channel_row(df, "single", "subject")
    assert isinstance(row1, dict)
    # The returned dictionary appears to have its columns sorted by name, not in the order in the datafile!
    assert row1 == {"channel": "single", "id": "S1", "value": "foo"}
    with pytest.raises(ValueError) as ex:
        _get_single_channel_row(df, "notPresent", "NoSubject")
    assert "NoSubject" in str(ex)
    assert "0 rows" in str(ex)
    with pytest.raises(ValueError) as ex:
        _get_single_channel_row(df, "double", "NoSubject")
    assert "NoSubject" in str(ex)
    assert "2 rows" in str(ex)


def test_get_single_row_if_channel_missing() -> None:
    """
    Test if we can convert a single row of data into a dictionary.
    """
    csv_string = StringIO("""id,channel,value
S1,single,foo
""")
    df = pd.read_csv(csv_string, sep=",")
    row1 = _get_single_channel_row(df, None, "subject")
    assert isinstance(row1, dict)
    assert row1 == {"channel": "single", "id": "S1", "value": "foo"}
    csv_string2 = StringIO("""id,channel,value
    S1,double,foo
    S1,double,foo
    """)
    df2 = pd.read_csv(csv_string2, sep=",")
    with pytest.raises(ValueError) as ex:
        _get_single_channel_row(df2, None, "NoSubject")
    assert "NoSubject" in str(ex)
    assert "2 rows" in str(ex)


def test_load_items(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test loading file paths and labels from a datafrome.
    """
    csv_string = """USUBJID,week,path,value,scalar1,scalar2,categorical1,categorical2
S1,image,foo.nii
S1,label,,True,1.1,1.2,A1,A2
S2,image,bar.nii
S2,label,,False,2.1,2.2,B1,A2
"""
    dataset = _create_test_dataset(create_dataset_csv_file(csv_string, test_output_dirs.root_dir),
                                   categorical_columns=["categorical1", "categorical2"])
    items = dataset.items
    metadata0 = items[0].metadata
    assert isinstance(metadata0, GeneralSampleMetadata)
    assert metadata0.id == "S1"
    assert items[0].label.tolist() == [1.0]
    assert items[0].channel_files == ["foo.nii"]
    assert items[0].numerical_non_image_features.shape == (2,)
    assert items[0].numerical_non_image_features.tolist() == pytest.approx(
        [-0.7071067094802856, -0.7071067690849304])
    assert items[0].categorical_non_image_features.tolist() == [1.0, 0.0, 1.0]
    assert items[1].categorical_non_image_features.tolist() == [0.0, 1.0, 1.0]
    metadata1 = items[1].metadata
    assert isinstance(metadata1, GeneralSampleMetadata)
    assert metadata1.id == "S2"
    assert items[1].label.tolist() == [0.0]
    assert items[1].channel_files == ["bar.nii"]
    assert items[1].numerical_non_image_features.shape == (2,)
    assert items[1].numerical_non_image_features.tolist() == pytest.approx([0.7071068286895752, 0.7071067690849304])


def test_load_items_classification_versus_regression(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test loading file paths and labels from a datafrome with diferrent configuration
    """
    csv_string_classification = """USUBJID,week,path,value,scalar1,scalar2,categorical1,categorical2
S1,image,foo.nii
S1,label,,True,1.1,1.2,True,False
S2,image,bar.nii
S2,label,,False,2.1,2.2,False,True
"""
    csv_string_regression = """USUBJID,week,path,value,scalar1,scalar2,categorical1,categorical2
S1,image,foo.nii
S1,label,,1,1.1,1.2,Male,True
S2,image,bar.nii
S2,label,,-1.3,2.1,2.2,Female,True
"""
    with pytest.raises(ValueError):
        _create_test_dataset(create_dataset_csv_file(csv_string_regression, test_output_dirs.root_dir),
                             scalar_loss=ScalarLoss.BinaryCrossEntropyWithLogits)
    with pytest.raises(ValueError):
        _create_test_dataset(create_dataset_csv_file(csv_string_classification, test_output_dirs.root_dir),
                             scalar_loss=ScalarLoss.MeanSquaredError)
    dataset_classification = _create_test_dataset(create_dataset_csv_file(csv_string_classification,
                                                                          test_output_dirs.root_dir),
                                                  scalar_loss=ScalarLoss.BinaryCrossEntropyWithLogits)
    assert len(dataset_classification.items) == 2
    dataset_regression = _create_test_dataset(create_dataset_csv_file(csv_string_regression,
                                                                      test_output_dirs.root_dir),
                                              scalar_loss=ScalarLoss.MeanSquaredError)
    assert len(dataset_regression.items) == 2


def _create_test_dataset(csv_path: Path, scalar_loss: ScalarLoss = ScalarLoss.BinaryCrossEntropyWithLogits,
                         categorical_columns: Optional[List[str]] = None) -> ScalarDataset:
    # Load items indirectly via a ScalarDataset object, to see if the wiring up of all column names works
    args = ScalarModelBase(image_channels=["image"],
                           image_file_column="path",
                           label_channels=["label"],
                           label_value_column="value",
                           non_image_feature_channels=["label"],
                           numerical_columns=["scalar1", "scalar2"],
                           categorical_columns=categorical_columns or list(),
                           subject_column="USUBJID",
                           channel_column="week",
                           local_dataset=csv_path,
                           should_validate=False,
                           loss_type=scalar_loss,
                           num_dataload_workers=0)
    args.read_dataset_into_dataframe_and_pre_process()
    return ScalarDataset(args)


def _get_non_image_dict(list_channels: List[str],
                        numerical_cols: List[str],
                        categorical_cols: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Converts list of channels to dictionary of features to channels.
    Mimics the behavior of convert_non_image_features_channels_to_dict in ScalarConfig.
    """
    res = {}
    columns = numerical_cols.copy()
    if categorical_cols:
        columns += categorical_cols
    for col in columns:
        res[col] = list_channels
    return res


def test_load_items_when_channel_missing() -> None:
    """
    Test loading file paths from a dataframe when a subject misses a channel.
    """
    # S1 has both required image channels, S2 only has image2. Loader should only return subject S1.
    csv_string = StringIO("""subject,channel,path,value
S1,image1,img11.nii
S1,image2,img12.nii,True
S2,image2,image22.nii,False
""")
    df = pd.read_csv(csv_string, sep=",", dtype=str)
    items: List[ScalarDataSource] = DataSourceReader[ScalarDataSource](
        data_frame=df,
        image_channels=["image1", "image2"],
        image_file_column="path",
        label_channels=["image2"],
        label_value_column="value").load_data_sources()
    assert len(items) == 1
    assert items[0].metadata.id == "S1"


def test_load_items_errors() -> None:
    """
    Test error cases when creating a list of classificationItems from a dataframe
    """

    def load(csv_string: StringIO) -> str:
        df = pd.read_csv(csv_string, sep=",", dtype=str)
        numerical_columns = ["scalar2", "scalar1"]
        non_image_channels = _get_non_image_dict(["label", "image2"], ["scalar2", "scalar1"])
        with pytest.raises(Exception) as ex:
            DataSourceReader(data_frame=df,
                             # Provide values in a different order from the file!
                             image_channels=["image2", "image1"],
                             image_file_column="path",
                             label_channels=["label"],
                             label_value_column="value",
                             # Provide values in a different order from the file!
                             non_image_feature_channels=non_image_channels,
                             numerical_columns=numerical_columns).load_data_sources()
        return str(ex)

    csv_string = StringIO("""subject,channel,path,value,scalar1
S1,image1,foo1.nii,,2.1
""")
    assert "columns are missing: scalar2" in load(csv_string)
    csv_string = StringIO("""subject,channel,path,scalar1,scalar2
S1,image1,foo1.nii,2.1,2.2
""")
    assert "columns are missing: value" in load(csv_string)
    csv_string = StringIO("""id,channel,path,value,scalar1,scalar2
S1,image,foo.nii
S1,label,,True,1.1,1.2
""")
    assert "columns are missing: subject" in load(csv_string)


def test_load_single_item_1() -> None:
    """
    Test if we can create a classificationItem from the rows for a single subject,
    including NaN scalar and categorical values.
    """
    csv_string = StringIO("""subject,channel,path,value,scalar1,scalar2,categorical1,categorical2
S1,image1,foo1.nii,,2.1,2.2,True,False
S1,image2,foo2.nii,,3.1,,True,False
S1,label,,True,1.1,1.2,,False
""")
    df = pd.read_csv(csv_string, sep=",", dtype=str)
    numerical_columns = ["scalar2", "scalar1"]
    categorical_columns = ["categorical1", "categorical2"]
    non_image_feature_channels = _get_non_image_dict(["label", "image2"],
                                                     ["scalar2", "scalar1"],
                                                     ["categorical1", "categorical2"])
    item: ScalarDataSource = load_single_data_source(df,
                                                     subject_id="S1",
                                                     # Provide values in a different order from the file!
                                                     image_channels=["image2", "image1"],
                                                     image_file_column="path",
                                                     label_channels=["label"],
                                                     label_value_column="value",
                                                     non_image_feature_channels=non_image_feature_channels,
                                                     # Provide values in a different order from the file!
                                                     numerical_columns=numerical_columns,
                                                     categorical_data_encoder=CategoricalToOneHotEncoder.create_from_dataframe(
                                                         dataframe=df,
                                                         columns=categorical_columns
                                                     ),
                                                     channel_column="channel")
    assert item.channel_files[0] == "foo2.nii"
    assert item.channel_files[1] == "foo1.nii"
    assert item.label == torch.tensor([1.0])
    assert item.label.dtype == torch.float32
    assert item.numerical_non_image_features[0] == 1.2
    assert item.numerical_non_image_features[2] == 1.1
    assert item.numerical_non_image_features[3] == 3.1
    assert math.isnan(item.numerical_non_image_features[1].item())
    assert np.all(np.isnan(item.categorical_non_image_features[0].numpy()))
    assert item.categorical_non_image_features[1:].tolist() == [1.0, 1.0, 1.0]
    assert item.numerical_non_image_features.dtype == torch.float32

    item_no_scalars: ScalarDataSource = load_single_data_source(df,
                                                                subject_id="S1",
                                                                # Provide values in a different order from the file!
                                                                image_channels=["image2", "image1"],
                                                                image_file_column="path",
                                                                label_channels=["label"],
                                                                label_value_column="value",
                                                                non_image_feature_channels={},
                                                                numerical_columns=[],
                                                                channel_column="channel")
    assert item_no_scalars.numerical_non_image_features.shape == (0,)


def test_load_single_item_2() -> None:
    """
    Test error cases when creating a classificationItem from the rows for a single subject.
    """

    def load_item(csv_string: StringIO) -> str:
        df = pd.read_csv(csv_string, sep=",", dtype=str)
        numerical_columns = ["scalar2", "scalar1"]
        non_image_feature_channels = _get_non_image_dict(["label", "image2"],
                                                         ["scalar2", "scalar1"])
        with pytest.raises(Exception) as ex:
            load_single_data_source(df,
                                    subject_id="S1",
                                    # Provide values in a different order from the file!
                                    image_channels=["image2", "image1"],
                                    image_file_column="path",
                                    label_channels=["label"],
                                    label_value_column="value",
                                    # Provide values in a different order from the file!
                                    non_image_feature_channels=non_image_feature_channels,
                                    numerical_columns=numerical_columns,
                                    channel_column="channel")
        return str(ex)

    # Duplicate row for image2
    csv_string = StringIO("""subject,channel,path,value,scalar1,scalar2
S1,image1,foo1.nii,,2.1,2.2
S1,image2,foo2.nii,,3.1,
S1,image2,foo2.nii,,3.1,
S1,label,,True,1.1,1.2
""")
    assert "There should be exactly one row to read from" in load_item(csv_string)


def test_load_single_item_3() -> None:
    """
    Test if we can create a classificationItem from a single row of data (no channels available).
    """
    csv_string = StringIO("""subject,path,value,scalar1,scalar2,label
S1,foo1.nii,,2.1,2.2,True
""")
    df = pd.read_csv(csv_string, sep=",", dtype=str)
    item: ScalarDataSource = load_single_data_source(df,
                                                     subject_id="S1",
                                                     image_channels=[],
                                                     image_file_column="path",
                                                     label_channels=None,
                                                     label_value_column="label",
                                                     non_image_feature_channels={},
                                                     numerical_columns=["scalar2", "scalar1"],
                                                     channel_column="foo")
    assert item.channel_files[0] == "foo1.nii"
    assert item.label == torch.tensor([1.0])
    assert item.label.dtype == torch.float32
    assert item.numerical_non_image_features.tolist() == pytest.approx([2.2, 2.1])
    assert item.numerical_non_image_features.dtype == torch.float32


def test_load_single_item_4() -> None:
    """
    Test if we can create a classificationItem from the rows for a single subject, including NaN scalar values.
    """

    def _test_load_labels(label_channels: List[str],
                          transform_labels: Union[Callable, List[Callable]]) -> ScalarDataSource:
        csv_string = StringIO("""subject,channel,path,value,scalar1,scalar2
    S1,label_w1,,1,1.1,1.2
    S1,label_w2,,3,,
    """)
        df = pd.read_csv(csv_string, sep=",", dtype=str)
        numerical_columns = ["scalar2", "scalar1"]
        non_image_feature_channels = _get_non_image_dict(["label_w1"],
                                                         ["scalar2", "scalar1"])
        return load_single_data_source(df,
                                       subject_id="S1",
                                       channel_column="channel",
                                       label_channels=label_channels,
                                       label_value_column="value",
                                       transform_labels=transform_labels,
                                       # Provide values in a different order from the file!
                                       non_image_feature_channels=non_image_feature_channels,
                                       numerical_columns=numerical_columns,
                                       is_classification_dataset=False)

    item = _test_load_labels(["label_w1"], LabelTransformation.identity)
    assert item.label == torch.tensor([1.0])
    assert item.label.dtype == torch.float32

    # Valid label difference
    item = _test_load_labels(["label_w1", "label_w2"], LabelTransformation.difference)
    assert item.label == torch.tensor([2.0])
    assert item.label.dtype == torch.float32

    # Valid scaling
    item = _test_load_labels(["label_w1"], LabelTransformation.get_scaling_transform(min_value=0, max_value=4))
    assert item.label == torch.tensor([0.25])
    assert item.label.dtype == torch.float32

    # Test pipeline
    pipeline = [LabelTransformation.get_scaling_transform(min_value=0, max_value=4, last_in_pipeline=False),
                LabelTransformation.difference]
    item = _test_load_labels(["label_w1", "label_w2"], pipeline)  # type: ignore
    assert item.label == torch.tensor([0.5])
    assert item.label.dtype == torch.float32

    # Invalid cases
    with pytest.raises(AssertionError):
        _test_load_labels(["label_w1"], LabelTransformation.difference)
    with pytest.raises(AssertionError):
        _test_load_labels(["label_w1", "label_w2"], LabelTransformation.identity)


def test_load_single_item_5() -> None:
    """
    Test loading of different channels for different numerical features.
    """
    csv_string = StringIO("""subject,path,channel,scalar1,scalar2,label
S1,foo1.nii,week1,2.1,2.2,True
S1,foo2.nii,week2,2.3,2.2,True
""")
    df = pd.read_csv(csv_string, sep=",", dtype=str)
    item: ScalarDataSource = load_single_data_source(df,
                                                     subject_id="S1",
                                                     image_channels=["week1"],
                                                     image_file_column="path",
                                                     label_channels=["week1"],
                                                     label_value_column="label",
                                                     non_image_feature_channels={"scalar1": ["week1", "week2"],
                                                                                 "scalar2": ["week1"]},
                                                     numerical_columns=["scalar2", "scalar1"],
                                                     channel_column="channel")
    assert item.channel_files[0] == "foo1.nii"
    assert item.label == torch.tensor([1.0])
    assert item.label.dtype == torch.float32
    assert torch.all(item.numerical_non_image_features == torch.tensor([2.2, 2.1, 2.3]))
    assert item.numerical_non_image_features.dtype == torch.float32


def test_load_single_item_6() -> None:
    """
    Test loading of different channels for different categorical features.
    """
    csv_string = StringIO("""subject,path,channel,cat1,cat2,scalar1,label
S1,foo1.nii,week1,True,True,1.2,True
S1,foo2.nii,week2,False,False,1.2,True
S1,foo2.nii,week3,False,True,1.3,True
""")
    df = pd.read_csv(csv_string, sep=",", dtype=str)
    item: ScalarDataSource = load_single_data_source(df,
                                                     subject_id="S1",
                                                     image_channels=["week1"],
                                                     image_file_column="path",
                                                     label_channels=["week1"],
                                                     label_value_column="label",
                                                     numerical_columns=["scalar1"],
                                                     non_image_feature_channels={"scalar1": ["week3"],
                                                                                 "cat1": ["week1", "week2"],
                                                                                 "cat2": ["week3"]},
                                                     categorical_data_encoder=CategoricalToOneHotEncoder.create_from_dataframe(
                                                         dataframe=df,
                                                         columns=["cat1", "cat2"]
                                                     ),
                                                     channel_column="channel")
    assert torch.all(item.categorical_non_image_features == torch.tensor([0, 1, 1, 0, 0, 1]))


def test_load_single_item_7() -> None:
    """
    Test loading of different channels for different categorical features.
    Case where one column value is invalid.
    """
    # Fit the encoder on the valid labels.
    csv_string_valid = StringIO("""subject,path,channel,cat1,cat2,label
    S1,foo1.nii,week1,True,True,True
    S1,foo2.nii,week2,False,False,True
    S1,foo2.nii,week3,False,,True
    """)
    df = pd.read_csv(csv_string_valid, sep=",", dtype=str)
    encoder = CategoricalToOneHotEncoder.create_from_dataframe(
        dataframe=df,
        columns=["cat1", "cat2"]
    )

    # Try to encode a dataframe with invalid value
    csv_string_invalid = StringIO("""subject,path,channel,cat1,cat2,label
    S1,foo1.nii,week1,True,True,True
    S1,foo2.nii,week2,houhou,False,False
    S1,foo2.nii,week3,False,,True
    """)
    df = pd.read_csv(csv_string_invalid, sep=",", dtype=str)
    item: ScalarDataSource = load_single_data_source(df,
                                                     subject_id="S1",
                                                     image_channels=["week1"],
                                                     image_file_column="path",
                                                     label_channels=["week1"],
                                                     label_value_column="label",
                                                     non_image_feature_channels={"cat1": ["week1", "week2"],
                                                                                 "cat2": ["week3"]},
                                                     categorical_data_encoder=encoder,
                                                     channel_column="channel")
    # cat1 - week1 is valid
    assert torch.all(item.categorical_non_image_features[0:2] == torch.tensor([0, 1]))
    # cat1 - week2 is invalid test regression
    assert torch.all(torch.isnan(item.categorical_non_image_features[2:4]))
    # cat2 - week 3 is invalid
    assert torch.all(torch.isnan(item.categorical_non_image_features[4:6]))


@pytest.mark.parametrize(["text", "expected_classification", "expected_regression"],
                         [
                             ("true", 1, None),
                             ("tRuE", 1, None),
                             ("false", 0, None),
                             ("False", 0, None),
                             ("nO", 0, None),
                             ("Yes", 1, None),
                             ("1.23", None, 1.23),
                             (3.45, None, None),
                             (math.nan, math.nan, math.nan),
                             ("", math.nan, math.nan),
                             (None, math.nan, math.nan),
                             ("abc", None, None),
                             ("1", 1, 1.0),
                             ("-1", None, -1.0)
                         ])
def test_extract_label(text: Union[float, str], expected_classification: Optional[float],
                       expected_regression: Optional[float]) -> None:
    _check_label_extraction_function(extract_label_classification, text, expected_classification)
    _check_label_extraction_function(extract_label_regression, text, expected_regression)


def _check_label_extraction_function(extract_fn: Callable, text: Union[float, str], expected: Optional[float]) -> None:
    if expected is None:
        with pytest.raises(ValueError) as ex:
            extract_fn(text, "foo")
        assert "Subject foo:" in str(ex)
    else:
        actual = extract_fn(text, "foo")
        assert isinstance(actual, type(expected))
        if math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert actual == expected


@pytest.mark.parametrize(["text", "expected"],
                         [
                             (" ", math.nan),
                             ("", math.nan),
                             (None, math.nan),
                             ("1.2", 1.2),
                             ("abc", math.nan),
                             (3.4, 3.4),
                             (math.nan, math.nan),
                         ])
def test_string_to_float(text: str, expected: float) -> None:
    actual = _string_to_float(text, "foo")
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected


def test_files_by_stem(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test enumeration of files recursively.
    """
    root = test_output_dirs.root_dir / "foo"
    folder1 = root / "bar"
    folder1.mkdir(parents=True)
    f1 = root / "1.txt"
    f2 = folder1 / "2.txt.gz"
    for f in [f1, f2]:
        f.touch()
    file_mapping = files_by_stem(root)
    assert file_mapping == {
        "1": f1,
        "2": f2
    }
    f3 = root / "2.h5"
    f3.touch()
    with pytest.raises(ValueError) as ex:
        files_by_stem(root)
    assert "1 files have duplicates" in str(ex)


@pytest.mark.parametrize("center_crop_size", [(2, 2, 2), None])
def test_dataset_traverse_dirs(test_output_dirs: OutputFolderForTests, center_crop_size: Optional[TupleInt3]) -> None:
    """
    Test dataset loading when the dataset file only contains file name stems, not full paths.
    """
    # Copy the existing test dataset to a new folder, two levels deep. Later will initialize the
    # dataset with only the root folder given, to check if the files are still found.
    source_folder = str(full_ml_test_data_path() / "classification_data")
    target_folder = str(Path(test_output_dirs.make_sub_dir("foo")) / "bar")
    shutil.copytree(source_folder, target_folder)
    # The dataset should only contain the file name stem, without extension.
    csv_string = StringIO("""subject,channel,path,value,scalar1
S1,image,4be9beed-5861-fdd2-72c2-8dd89aadc1ef
S1,label,,True,1.0
S2,image,6ceacaf8-abd2-ffec-2ade-d52afd6dd1be
S2,label,,True,2.0
S3,image,61bc9d73-9fbb-bd7d-c06b-eeffbafabcc4
S3,label,,False,3.0
S4,image,61bc9d73-9fbb-bd7d-c06b-eeffbafabcc4
S4,label,,False,3.0
""")
    df = pd.read_csv(csv_string, sep=",", dtype=str)
    args = ScalarModelBase(image_channels=["image"],
                           image_file_column="path",
                           label_channels=["label"],
                           label_value_column="value",
                           non_image_feature_channels={},
                           numerical_columns=[],
                           traverse_dirs_when_loading=True,
                           center_crop_size=center_crop_size,
                           local_dataset=test_output_dirs.root_dir)
    dataset = ScalarDataset(args, data_frame=df)
    assert len(dataset) == 4
    for i in range(4):
        item = dataset[i]
        assert isinstance(item, dict)
        images = item["images"]
        assert images is not None
        assert torch.is_tensor(images)
        expected_image_size = center_crop_size or (4, 5, 7)
        assert images.shape == (1,) + expected_image_size


def test_dataset_normalize_image(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test dataset loading with window normalization image processing.
    """
    source_folder = str(full_ml_test_data_path() / "classification_data")
    target_folder = str(Path(test_output_dirs.make_sub_dir("foo")) / "bar")
    shutil.copytree(source_folder, target_folder)
    csv_string = StringIO("""subject,channel,path,value,scalar1
S1,image,4be9beed-5861-fdd2-72c2-8dd89aadc1ef
S1,label,,True,1.0
S2,image,6ceacaf8-abd2-ffec-2ade-d52afd6dd1be
S2,label,,True,2.0
S3,image,61bc9d73-9fbb-bd7d-c06b-eeffbafabcc4
S3,label,,False,3.0
S4,image,61bc9d73-9fbb-bd7d-c06b-eeffbafabcc4
S4,label,,False,3.0
""")
    df = pd.read_csv(csv_string, sep=",", dtype=str)
    args = ScalarModelBase(image_channels=["image"],
                           image_file_column="path",
                           label_channels=["label"],
                           label_value_column="value",
                           non_image_feature_channels={},
                           numerical_columns=[],
                           traverse_dirs_when_loading=True,
                           local_dataset=test_output_dirs.root_dir)
    raw_dataset = ScalarDataset(args, data_frame=df)
    normalized = ScalarDataset(args, data_frame=df, sample_transforms=WindowNormalizationForScalarItem())
    assert len(raw_dataset) == 4
    for i in range(4):
        raw_item = raw_dataset[i]
        normalized_item = normalized[i]
        normalized_images = normalized_item["images"]
        assert isinstance(raw_item, dict)
        expected_normalized_images = torch.tensor(mri_window(raw_item["images"].numpy(),
                                                             mask=None,
                                                             output_range=(0, 1))[0])
        assert normalized_images is not None
        assert torch.is_tensor(normalized_images)
        assert expected_normalized_images.shape == normalized_images.shape
        expected_image_size = (4, 5, 7)
        assert normalized_images.shape == (1,) + expected_image_size
        assert torch.all(expected_normalized_images == normalized_images)


def test_filter_dataset_by_expected_size() -> None:
    """
    Test that we can filter images that do not follow specific size
    """

    classification_config = ScalarModelBase(image_channels=["image"],
                                            image_file_column="path",
                                            label_channels=["label"],
                                            label_value_column="value",
                                            non_image_feature_channels={},
                                            numerical_columns=[],
                                            traverse_dirs_when_loading=True,
                                            expected_column_values=[("DIM", "512x49x496")],
                                            local_dataset=Path("fakepath"))
    data = {'Subject': ['1', '2', '3', '4'], 'DIM': ["1024x49x496", "512x49x496", "512x49x496", "512x49x496"]}
    df = pd.DataFrame(data)
    print(df.head())
    filtered = classification_config.filter_dataframe(df)
    assert filtered.shape == (3, 2)
    subjects = filtered['Subject'].values
    assert '1' not in subjects
    assert '2' in subjects
    assert '3' in subjects
    assert '4' in subjects


@pytest.mark.parametrize("expected_column_value", [[], None])
def test_filter_dataset_with_empty_list(expected_column_value: List[Tuple[str, str]]) -> None:
    """
    Test that empty filter has no effect
    """

    classification_config = ScalarModelBase(image_channels=["image"],
                                            image_file_column="path",
                                            label_channels=["label"],
                                            label_value_column="value",
                                            non_image_feature_channels={},
                                            numerical_columns=[],
                                            traverse_dirs_when_loading=True,
                                            expected_column_values=[],
                                            local_dataset=Path("fakepath"))
    data = {'Subject': ['1', '2', '3', '4'], 'DIM': ["1024x49x496", "512x49x496", "512x49x496", "512x49x496"]}
    df = pd.DataFrame(data)
    print(df.head())
    filtered = classification_config.filter_dataframe(df)
    assert_frame_equal(df, filtered)


@pytest.mark.parametrize(["channel_files", "numerical_features", "categorical_features", "is_valid"],
                         [
                             ([], torch.tensor([1]), torch.tensor([[0, 0, 1]]), True),
                             ([], torch.tensor([1]), torch.tensor([[np.NaN, np.NaN, np.NaN], [0, 0, 0]]), False),
                             (["foo"], torch.tensor([1]), torch.tensor([[0, 1, 1]]), True),
                             (["foo"], torch.tensor([]), torch.tensor([]), True),
                             ([""], torch.tensor([]), torch.tensor([]), True),
                             ([None], torch.tensor([]), torch.tensor([]), False),
                             ([], torch.tensor([1.0, math.inf]), torch.tensor([[0, 0, 1]]), True),
                             ([], torch.tensor([1.0, math.nan, math.inf]), torch.tensor([[0, 0, 1]]), False),
                         ])
def test_item_is_valid(channel_files: List[Optional[str]],
                       numerical_features: torch.Tensor,
                       categorical_features: torch.Tensor,
                       is_valid: bool) -> None:
    c = ScalarDataSource(channel_files=channel_files,
                         numerical_non_image_features=numerical_features,
                         categorical_non_image_features=categorical_features,
                         label=torch.empty(0),
                         metadata=GeneralSampleMetadata(id="foo"))
    assert c.is_valid() == is_valid


def test_is_index_valid() -> None:
    """
    Test if checks for valid sequence positions work.
    """

    def _create(pos: int) -> ScalarDataSource:
        z = torch.empty(0)
        return ScalarDataSource(metadata=GeneralSampleMetadata(id="", sequence_position=pos),
                                categorical_non_image_features=z,
                                label=z, numerical_non_image_features=z, channel_files=[])

    # If no filtering for maximum index is done, the index itself can be anything
    assert is_valid_item_index(_create(1), max_sequence_position_value=None)
    # Filter for maximum index: index must be present and no larger than the maximum
    assert is_valid_item_index(_create(1), max_sequence_position_value=1)
    assert not is_valid_item_index(_create(1), max_sequence_position_value=0)


def test_categorical_and_numerical_columns_are_mutually_exclusive(test_output_dirs: OutputFolderForTests) -> None:
    csv_string = """USUBJID,week,path,value,scalar1,categorical1
    S1,image,foo.nii
    S1,label,,True,1.1,False
    """
    with pytest.raises(ValueError):
        _create_test_dataset(create_dataset_csv_file(csv_string, test_output_dirs.root_dir),
                             categorical_columns=["scalar1"])


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
def test_imbalanced_sampler() -> None:
    # Simulate a highly imbalanced dataset with only one data point
    # with a negative label.
    csv_string = StringIO("""subject,channel,value,scalar1
    S1,label,True,1.0
    S2,label,True,1.0
    S3,label,True,1.0
    S4,label,True,1.0
    S5,label,True,1.0
    S6,label,False,1.0
    """)
    torch.manual_seed(0)
    df = pd.read_csv(csv_string, sep=",", dtype=str)
    args = ScalarModelBase(label_value_column="value",
                           numerical_columns=["scalar1"],
                           local_dataset=Path("fakepath"))
    dataset = ScalarDataset(args, data_frame=df)
    drawn_subjects = []
    for _ in range(10):
        data_loader = dataset.as_data_loader(use_imbalanced_sampler=True,
                                             shuffle=True, batch_size=6,
                                             num_dataload_workers=0)
        for batch in data_loader:
            drawn_subjects.extend([i.id.strip() for i in batch["metadata"]])
    counts_per_subjects = Counter(drawn_subjects)
    count_negative_subjects = counts_per_subjects["S6"]
    assert count_negative_subjects / float(len(drawn_subjects)) > 0.3


def test_get_class_weights_dataset(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test training and testing of sequence models that predicts at multiple time points,
    when it is started via run_ml.
    """
    dataset_folder = Path(test_output_dirs.make_sub_dir("dataset"))
    dataset_contents = """subject,channel,path,label,numerical1,numerical2,CAT1
   S1,week0,scan1.npy,,1,10,A
   S1,week1,scan2.npy,True,2,20,A
   S2,week0,scan3.npy,,3,30,A
   S2,week1,scan4.npy,False,4,40,A
   S3,week0,scan1.npy,,5,50,A
   S3,week1,scan3.npy,True,6,60,A
   """
    config = ScalarModelBase(
        local_dataset=dataset_folder,
        label_channels=["week1"],
        label_value_column="label",
        non_image_feature_channels=["week0", "week1"],
        numerical_columns=["numerical1", "numerical2"],
        should_validate=False
    )
    config.set_output_to(test_output_dirs.root_dir)
    train_dataset = ScalarDataset(config, pd.read_csv(StringIO(dataset_contents), dtype=str))
    class_counts = train_dataset.get_class_counts()
    assert class_counts == {0.0: 1, 1.0: 2}
