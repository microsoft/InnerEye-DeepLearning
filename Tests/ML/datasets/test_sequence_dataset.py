#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from io import StringIO
from pathlib import Path
from typing import List, Optional, Union
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from InnerEye.Common.common_util import is_windows
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.dataset.full_image_dataset import collate_with_metadata
from InnerEye.ML.dataset.sample import GeneralSampleMetadata
from InnerEye.ML.dataset.scalar_dataset import DataSourceReader, filter_valid_classification_data_sources_items
from InnerEye.ML.dataset.scalar_sample import ScalarDataSource, ScalarItem, SequenceDataSource
from InnerEye.ML.dataset.sequence_dataset import SequenceDataset, add_difference_features, \
    group_samples_into_sequences
from InnerEye.ML.dataset.sequence_sample import ClassificationItemSequence, ListOfSequences
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils.features_util import FeatureStatistics
from InnerEye.ML.utils.io_util import ImageAndSegmentations
from InnerEye.ML.utils.ml_util import set_random_seed
from InnerEye.ML.utils.sequence_utils import sequences_to_padded_tensor
from InnerEye.ML.utils.split_dataset import DatasetSplits
from Tests.ML.models.architectures.sequential.test_rnn_classifier import ToyMultiLabelSequenceModel, \
    _get_multi_label_sequence_dataframe
from Tests.ML.util import assert_tensors_equal, create_dataset_csv_file
from Tests.fixed_paths_for_tests import full_ml_test_data_path


def test_load_items_seq() -> None:
    """
    Test loading file paths and labels from a datafrome if
    """
    csv_string = StringIO("""subject,seq,path,value,scalar1,scalar2,META
S1,0,foo.nii,,0,0,M1
S1,1,,True,1.1,1.2,M2
S2,1,bar.nii,False,2.1,2.2,
""")
    df = pd.read_csv(csv_string, sep=",", dtype=str)
    items: List[SequenceDataSource] = DataSourceReader[SequenceDataSource](
        data_frame=df,
        image_channels=None,
        image_file_column="path",
        label_channels=None,
        label_value_column="value",
        numerical_columns=["scalar1", "scalar2"],
        sequence_column="seq").load_data_sources()

    assert len(items) == 3
    assert isinstance(items[0].metadata, GeneralSampleMetadata)
    assert items[0].metadata.id == "S1"
    assert items[0].metadata.props == {"META": "M1"}
    assert items[0].metadata.sequence_position == 0
    assert len(items[0].label.tolist()) == 1
    assert math.isnan(items[0].label.item())
    assert items[0].channel_files == ["foo.nii"]
    assert_tensors_equal(items[0].numerical_non_image_features, [0.0, 0.0])
    assert isinstance(items[1].metadata, GeneralSampleMetadata)
    assert items[1].metadata.id == "S1"
    assert items[1].metadata.props == {"META": "M2"}
    assert items[1].metadata.sequence_position == 1
    assert_tensors_equal(items[1].label, [1.0])
    assert items[1].channel_files == ['']
    assert_tensors_equal(items[1].numerical_non_image_features, [1.1, 1.2])
    assert isinstance(items[2].metadata, GeneralSampleMetadata)
    assert items[2].metadata.id == "S2"
    assert items[2].metadata.props == {"META": ''}
    assert items[2].metadata.sequence_position == 1
    assert_tensors_equal(items[2].label, [0.0])
    assert items[2].channel_files == ["bar.nii"]
    assert_tensors_equal(items[2].numerical_non_image_features, [2.1, 2.2])


def test_load_items_seq_from_dataset() -> None:
    """
    Test loading a sequence dataset with numerical, categorical features and images.
    """
    dummy_dataset = full_ml_test_data_path() / "sequence_data_for_classification" / "dataset.csv"
    df = pd.read_csv(dummy_dataset, sep=",", dtype=str)
    items: List[SequenceDataSource] = DataSourceReader[SequenceDataSource](
        data_frame=df,
        image_channels=None,
        image_file_column="IMG",
        label_channels=None,
        label_value_column="Label",
        numerical_columns=["NUM1", "NUM2", "NUM3", "NUM4"],
        sequence_column="Position").load_data_sources()
    assert len(items) == 3 * 9  # 3 subjects, 9 visits each, no missing
    assert items[0].metadata.id == "2137.00005"
    assert items[0].metadata.sequence_position == 0
    assert items[0].metadata.props["CAT2"] == "category_A"
    # One of the labels is missing, missing labels should be encoded as NaN
    assert math.isnan(items[0].label[0])
    assert items[0].channel_files == ["img_1"]
    assert str(items[0].numerical_non_image_features.tolist()) == str([362.0, np.nan, np.nan, 71.0])
    assert items[8].metadata.id == "2137.00005"
    assert items[8].metadata.sequence_position == 8
    assert items[8].label.tolist() == [0.0]
    assert items[8].channel_files == ['']
    assert str(items[8].numerical_non_image_features.tolist()) == str([350.0, np.nan, np.nan, 8.0])
    assert items[16].metadata.id == "2627.00001"
    assert items[16].label.tolist() == [0.0]
    assert items[16].channel_files == ["img_2"]
    assert_tensors_equal(items[16].numerical_non_image_features, [217.0, 0.0, 0.01, 153.0])
    assert items[26].metadata.id == "3250.00005"
    assert items[26].metadata.sequence_position == 8
    assert_tensors_equal(items[26].label, [0.0])
    assert items[26].channel_files == ["img_11"]
    assert_tensors_equal(items[26].numerical_non_image_features, [238.0, 0.0, 0.02, 84.0])

    grouped = group_samples_into_sequences(
        filter_valid_classification_data_sources_items(items, file_to_path_mapping=None,
                                                       max_sequence_position_value=None))
    # There are 3 patients total, but one of them has missing measurements for all visits
    assert len(grouped) == 2
    assert grouped[0].id == "2627.00001"
    assert grouped[1].id == "3250.00005"
    # 2627.00001 has full information for weeks 0, 4, and 8
    assert len(grouped[0].items) == 3
    assert grouped[0].items[0].metadata["VISIT"] == "V1"
    assert grouped[0].items[2].metadata["VISIT"] == "VST 3"
    assert len(grouped[1].items) == 9
    assert items[16].metadata.sequence_position == 7


def test_seq_dataset_loader() -> None:
    dummy_dataset = full_ml_test_data_path() / "sequence_data_for_classification" / "dataset.csv"
    df = pd.read_csv(dummy_dataset, sep=",", dtype=str)
    dataset = SequenceDataset(
        args=SequenceModelBase(
            image_file_column="IMG",
            label_value_column="Label",
            numerical_columns=["NUM1", "NUM2", "NUM3", "NUM4"],
            sequence_target_positions=[8],
            sequence_column="Position",
            local_dataset=Path(),
            should_validate=False
        ),
        data_frame=df
    )
    assert len(dataset) == 2
    # Patch the load_images function that well be called once we access a dataset item
    with mock.patch('InnerEye.ML.dataset.scalar_sample.load_images_and_stack',
                    return_value=ImageAndSegmentations[torch.Tensor](images=torch.ones(1),
                                                                     segmentations=torch.empty(0))):
        item0 = ClassificationItemSequence(**dataset[0])
        item1 = ClassificationItemSequence(**dataset[1])
        assert item0.id == "2627.00001"
        len_2627 = 3
        assert len(item0.items) == len_2627
        assert item1.id == "3250.00005"
        len_3250 = 9
        assert len(item1.items) == len_3250

        # Data loaders use a customized collate function, that must work with the sequences too.
        collated = collate_with_metadata([dataset[0], dataset[1]])
        assert collated["id"] == ["2627.00001", "3250.00005"]
        # All subject sequences should be turned into lists of lists.
        assert isinstance(collated["items"], list)
        assert len(collated["items"]) == 2
        assert isinstance(collated["items"][0], list)
        assert isinstance(collated["items"][1], list)
        assert len(collated["items"][0]) == len_2627
        assert len(collated["items"][1]) == len_3250
        back_to_items = ClassificationItemSequence(**collated)
        assert back_to_items.id == ["2627.00001", "3250.00005"]


def test_group_items() -> None:
    """
    Test if grouping and filtering of sequence data sets works.
    """

    def _create(id: str, sequence_position: int, file: Optional[str], metadata: str) -> SequenceDataSource:
        return SequenceDataSource(channel_files=[file],
                                  numerical_non_image_features=torch.tensor([]),
                                  categorical_non_image_features=torch.tensor([]),
                                  label=torch.tensor([]),
                                  metadata=GeneralSampleMetadata(id=id, sequence_position=sequence_position,
                                                                 props={"M": metadata}))

    items = [
        _create("a", 1, "f", "a.1"),
        _create("a", 0, "f", "a.0"),
        _create("a", 4, "f", "a.4"),
        _create("b", 1, None, "b.1"),
        _create("b", 0, None, "b.0"),
        _create("c", 0, "f", "c.0"),
        _create("d", 1, "f", "d.1"),
    ]
    grouped = group_samples_into_sequences(items)
    assert len(grouped) == 3

    def assert_group(group: ClassificationItemSequence, subject: str, props: List[str]) -> None:
        assert isinstance(group, ClassificationItemSequence)
        assert group.id == subject
        assert [i.metadata.props["M"] for i in group.items] == props

    # For subject a, item a.4 should be dropped because the consecutive sequence is only [0, 1]
    assert_group(grouped[0], "a", ["a.0", "a.1"])
    assert_group(grouped[1], "b", ["b.0", "b.1"])
    assert_group(grouped[2], "c", ["c.0"])
    # Group should not contain subject d because its only item is at index 1


def _create_item(id: str, sequence_position: int, metadata: str, label: Optional[float] = None) -> SequenceDataSource:
    return SequenceDataSource(channel_files=["foo"],
                              numerical_non_image_features=torch.tensor([]),
                              categorical_non_image_features=torch.tensor([]),
                              label=(torch.tensor([label]) if label else torch.tensor([])),
                              metadata=GeneralSampleMetadata(id=id, sequence_position=sequence_position,
                                                             props={"M": metadata}))


def _assert_group(group: ClassificationItemSequence, subject: str, props: List[str]) -> None:
    assert group.id == subject
    assert [i.metadata.props["M"] for i in group.items] == props


def test_group_items_with_min_and_max_sequence_position_values() -> None:
    """
    Test if grouping of sequence data works when requiring a full set of items.
    """
    items = [
        _create_item("a", 1, "a.1"),
        _create_item("a", 0, "a.0"),
        _create_item("a", 2, "a.2"),
        _create_item("b", 1, "b.1"),
        _create_item("b", 0, "b.0"),
    ]
    # When not providing a max_sequence_position_value, sequences of any length are OK.
    grouped = group_samples_into_sequences(items, max_sequence_position_value=None)
    assert len(grouped) == 2
    _assert_group(grouped[0], "a", ["a.0", "a.1", "a.2"])
    _assert_group(grouped[1], "b", ["b.0", "b.1"])
    # With a max_sequence_position_value, the set must be complete up to the given index.
    grouped = group_samples_into_sequences(items, min_sequence_position_value=1, max_sequence_position_value=2)
    assert len(grouped) == 2
    _assert_group(grouped[0], "a", ["a.1", "a.2"])
    # When a max position is given, the sequence will be truncated to at most contain the given value.
    grouped = group_samples_into_sequences(items, min_sequence_position_value=0, max_sequence_position_value=1)
    assert len(grouped) == 2
    _assert_group(grouped[0], "a", ["a.0", "a.1"])
    _assert_group(grouped[1], "b", ["b.0", "b.1"])
    grouped = group_samples_into_sequences(items, min_sequence_position_value=1, max_sequence_position_value=1)
    assert len(grouped) == 2
    _assert_group(grouped[0], "a", ["a.1"])
    _assert_group(grouped[1], "b", ["b.1"])
    # Allow sequences upto max_sequence_position_value=2
    grouped = group_samples_into_sequences(items, min_sequence_position_value=1, max_sequence_position_value=2)
    assert len(grouped) == 2
    _assert_group(grouped[0], "a", ["a.1", "a.2"])
    _assert_group(grouped[1], "b", ["b.1"])

    # There are no items that have sequence position == 3, hence the next two calls should not return any items.
    grouped = group_samples_into_sequences(items, min_sequence_position_value=3)
    assert len(grouped) == 0
    # Check that items upto max_sequence_position_value=3 are included
    grouped = group_samples_into_sequences(items, max_sequence_position_value=3)
    assert len(grouped) == 2

    # Sequence positions must be unique
    with pytest.raises(ValueError) as ex:
        group_samples_into_sequences([_create_item("a", 0, "a.0")] * 2)
    assert "contains duplicates" in str(ex)


def test_group_items_with_label_positions() -> None:
    items = [
        _create_item("a", 0, "a.0", 1),
        _create_item("a", 3, "a.3", math.inf),
        _create_item("a", 1, "a.1", 0),
        _create_item("a", 2, "a.2", 1),
    ]

    # Extracting the sequence from 2 to 3
    grouped = group_samples_into_sequences(items, min_sequence_position_value=2, max_sequence_position_value=3)
    assert len(grouped) == 1
    _assert_group(grouped[0], "a", ["a.2", 'a.3'])


def test_filter_valid_items() -> None:
    """
    Test if filtering of sequence data sets works.
    """

    def _create(id: str, sequence_position: int, file: Optional[str], metadata: str) -> SequenceDataSource:
        return SequenceDataSource(channel_files=[file],
                                  numerical_non_image_features=torch.tensor([]),
                                  categorical_non_image_features=torch.tensor([]),
                                  label=torch.tensor([]),
                                  metadata=GeneralSampleMetadata(id=id, sequence_position=sequence_position,
                                                                 props={"M": metadata}))

    items = [
        _create("a", 1, "f1", "a.1"),  # Valid item
        _create("b", 0, None, "b.0"),  # Invalid because no file
        _create("b", 1, "d", "b.1"),  # valid
        _create("c", 0, "f3", "c.0"),  # valid item for subject "c"
    ]

    def assert_items(filtered: List[SequenceDataSource], props: List[str]) -> None:
        assert [i.metadata.props["M"] for i in filtered] == props

    # Standard filtering should remove items with missing file name only, that is b.0
    filtered1 = filter_valid_classification_data_sources_items(items, file_to_path_mapping=None,
                                                               max_sequence_position_value=None)
    assert_items(filtered1, ["a.1", "b.1", "c.0"])

    # Filtering also for max_sequence_position_value
    filtered2 = filter_valid_classification_data_sources_items(items, file_to_path_mapping=None,
                                                               max_sequence_position_value=1)
    assert_items(filtered2, ["a.1", "b.1", "c.0"])
    filtered3 = filter_valid_classification_data_sources_items(items, file_to_path_mapping=None,
                                                               max_sequence_position_value=0)
    assert_items(filtered3, ["c.0"])

    # Filtering also for min_sequence_position_value
    filtered4 = filter_valid_classification_data_sources_items(items, file_to_path_mapping=None,
                                                               min_sequence_position_value=1,
                                                               max_sequence_position_value=None)
    assert_items(filtered4, ["a.1", "b.1"])

    filtered5 = filter_valid_classification_data_sources_items(items, file_to_path_mapping=None,
                                                               min_sequence_position_value=2,
                                                               max_sequence_position_value=None)
    assert_items(filtered5, [])

    # Now also filter by file name mapping: only "d" is in the mapping, hence only b.1 should survive
    file_mapping = {"d": Path("d"), "foo": Path("bar")}
    filtered4 = filter_valid_classification_data_sources_items(items, file_to_path_mapping=file_mapping,
                                                               max_sequence_position_value=1)
    assert_items(filtered4, ["b.1"])


# noinspection PyUnresolvedReferences
@pytest.mark.skipif(is_windows(),
                    reason="This test runs fine on local Windows boxes, but leads to odd timeouts in Azure")
def test_sequence_dataloader() -> None:
    """
    Test if we can create a data loader from the dataset, and recover the items as expected in batched form.
    Including instances where not all elements of the sequence have labels.
    """
    csv_string = StringIO("""subject,seq,path,value,scalar1,scalar2,META
S1,0,foo.nii,,0,0,M1
S1,1,,True,1.1,1.2,M2
S2,0,bar.nii,False,2.1,2.2,M3
S2,1,,False,2.0,2.0,M4
""")
    df = pd.read_csv(csv_string, sep=",", dtype=str)
    config = SequenceModelBase(
        image_file_column=None,
        label_value_column="value",
        numerical_columns=["scalar1"],
        sequence_target_positions=[1],
        sequence_column="seq",
        local_dataset=Path.cwd(),
        should_validate=False
    )
    dataset = SequenceDataset(config, data_frame=df)
    assert len(dataset) == 2
    data_loader = dataset.as_data_loader(shuffle=False, batch_size=2, num_dataload_workers=0)
    # We have 2 subjects, with a batch size of 2 those should be turned into 1 batch
    data_loader_output = list(i for i in data_loader)
    assert len(data_loader_output) == 1
    loaded = list(ClassificationItemSequence(**i) for i in data_loader_output)
    assert loaded[0].id == ["S1", "S2"]
    assert isinstance(loaded[0].items[0][0], ScalarItem)
    assert loaded[0].items[0][0].metadata.id == "S1"
    assert loaded[0].items[0][1].metadata.id == "S1"
    assert loaded[0].items[1][0].metadata.id == "S2"
    assert loaded[0].items[1][1].metadata.id == "S2"

    # The batched sequence data are awkward to work with. Check if we can un-roll them correctly via
    # from_minibatch
    un_batched = ClassificationItemSequence.from_minibatch(data_loader_output[0])
    assert len(un_batched) == 2
    for i in range(2):
        assert un_batched[i].id == dataset.items[i].id
        assert len(un_batched[i].items) == len(dataset.items[i].items)
        for j in range(len(un_batched[i].items)):
            assert un_batched[i].items[j].metadata.id == dataset.items[i].items[j].metadata.id


def test_standardize_features() -> None:
    """
    Test if the non-image feature can be normalized to mean 0, std 1.
    :return:
    """
    set_random_seed(1234)
    expected_mean = torch.tensor([[123, 2, 3], [4, 5, 6]])
    expected_std = torch.tensor([[0, 2, 3], [3, 4, 4]])
    feature_size = (2, 3)
    sequences: List[ClassificationItemSequence] = []
    for s in range(1000):
        items = []
        seq_length = torch.randint(low=3, high=6, size=(1,)).item()
        for i in range(seq_length):  # type: ignore
            # All features are random Gaussian, apart from feature 0 which is constant.
            # Normalization must be able to deal with constant features when dividing by standard deviation.
            features = torch.randn(size=feature_size, dtype=torch.float32) * expected_std + expected_mean
            # Randomly put some infinite values in the vector
            features[s % 2, s % 3] = np.inf if torch.rand(1) > 0.9 else features[s % 2, s % 3]
            features[0, 0] = expected_mean[0, 0]
            item = ScalarItem(metadata=GeneralSampleMetadata(id="foo"),
                              numerical_non_image_features=features,
                              categorical_non_image_features=features,
                              label=torch.tensor([]),
                              images=torch.tensor([]),
                              segmentations=torch.tensor([]))
            items.append(item)
        sequences.append(ClassificationItemSequence(id="foo", items=items))
    mean_std = FeatureStatistics.from_data_sources(sequences)
    assert mean_std.mean.shape == feature_size
    assert mean_std.std.shape == feature_size

    assert_tensors_equal(mean_std.mean, expected_mean, 0.07)
    assert_tensors_equal(mean_std.std, expected_std, 0.07)

    # After normalization, mean should be 0, and std should be 1.
    standardized_seq = mean_std.standardize(sequences)
    mean_std_from_standardized = FeatureStatistics.from_data_sources(standardized_seq)
    # After normalization, the mean should be 0, apart from the constant feature, which should be left untouched,
    # hence its mean is the original feature value.
    expected_mean_from_standardized = torch.zeros(feature_size)
    expected_mean_from_standardized[0, 0] = expected_mean[0, 0]
    expected_std_from_standardized = torch.ones(feature_size)
    expected_std_from_standardized[0, 0] = 0.0
    assert_tensors_equal(mean_std_from_standardized.mean, expected_mean_from_standardized, abs=1e-5)
    assert_tensors_equal(mean_std_from_standardized.std, expected_std_from_standardized, abs=1e-5)


@pytest.mark.parametrize("is_sequence", [True, False])
def test_standardize_features_when_singleton(is_sequence: bool) -> None:
    """
    Test how feature standardize copes with datasets that only have 1 entry.
    """
    numerical_features = torch.ones((1, 3))
    categorical_features = torch.tensor([[0, 1, 1], [1, 0, 0]])
    item: Union[SequenceDataSource, ScalarDataSource]
    sources: Union[ListOfSequences, List[ScalarDataSource]]
    if is_sequence:
        item = SequenceDataSource(metadata=GeneralSampleMetadata(id="foo"),
                                  numerical_non_image_features=numerical_features,
                                  categorical_non_image_features=categorical_features,
                                  label=torch.tensor([]),
                                  channel_files=[])
        sources = [ClassificationItemSequence(id="foo", items=[item])]
        mean_std = FeatureStatistics.from_data_sources(sources)
    else:
        item = ScalarDataSource(metadata=GeneralSampleMetadata(id="foo"),
                                numerical_non_image_features=numerical_features,
                                categorical_non_image_features=categorical_features,
                                label=torch.tensor([]),
                                channel_files=[])

        sources = [item]
        mean_std = FeatureStatistics.from_data_sources(sources)

    assert_tensors_equal(mean_std.mean, numerical_features)
    # Standard deviation can't be computed because there is only one element, hence becomes nan.
    assert torch.all(torch.isnan(mean_std.std))
    # When applying such a standardization to the sequences, they should not be changed (similar to features that
    # are constant)
    standardized_sources = mean_std.standardize(sources)
    if is_sequence:
        assert_tensors_equal(standardized_sources[0].items[0].numerical_non_image_features, numerical_features)
        assert_tensors_equal(standardized_sources[0].items[0].categorical_non_image_features, categorical_features)
    else:
        assert_tensors_equal(standardized_sources[0].numerical_non_image_features, numerical_features)
        assert_tensors_equal(standardized_sources[0].categorical_non_image_features, categorical_features)


def test_add_difference_features() -> None:
    """
    Test if we can add difference features for sequence data sets (differences from position i compared to position 0
    in the sequence)
    """

    def _create(features: List) -> SequenceDataSource:
        return SequenceDataSource(metadata=GeneralSampleMetadata(id="foo"),
                                  channel_files=[],
                                  label=torch.tensor([]),
                                  categorical_non_image_features=torch.tensor([]),
                                  numerical_non_image_features=torch.tensor(features).float())

    item1 = _create([[1, 2, 3], [4, 5, 6]])
    item2 = _create([[11, 22, 33], [44, 55, 66]])
    items = [ClassificationItemSequence[SequenceDataSource](id="bar", items=[item1, item2])]
    updated = add_difference_features(items, [0, 2])
    # The two difference features should be added along dimension 1 of the tensor
    assert updated[0].items[0].numerical_non_image_features.shape == (2, 5)
    # Item 0 should have differences of 0
    assert_tensors_equal(updated[0].items[0].numerical_non_image_features[:, 0:3], item1.numerical_non_image_features)
    assert_tensors_equal(updated[0].items[0].numerical_non_image_features[:, 3:5], [[0, 0], [0, 0]])
    # Item 1 should have non-zero diff, and keep the original non-image features in the first few dim
    assert_tensors_equal(updated[0].items[1].numerical_non_image_features[:, 0:3], item2.numerical_non_image_features)
    assert_tensors_equal(updated[0].items[1].numerical_non_image_features[:, 3:5], [[10, 30], [40, 60]])


def test_seq_to_tensor() -> None:
    """
    Test if we can create a tensor from a variable length sequence.
    """

    def _create(features: List) -> torch.Tensor:
        return ScalarItem(
            segmentations=torch.empty(0),
            metadata=GeneralSampleMetadata(id="foo"),
            images=torch.tensor([]),
            label=torch.tensor([]),
            categorical_non_image_features=torch.tensor(features).float(),
            numerical_non_image_features=torch.tensor(features).float()
        ).get_all_non_imaging_features()

    item1 = _create([1, 2, 3, 4, 5, 6])
    item2 = _create([11, 22, 33])
    items = [item1, item1, item2, item1]
    stacked = sequences_to_padded_tensor(items)
    assert torch.is_tensor(stacked)
    # pad_sequence will pad the tensors to the maximum sequence length
    assert stacked.shape == (len(items), item1.numel())


def test_sequence_dataset_all(test_output_dirs: OutputFolderForTests) -> None:
    """
    Check that the sequence dataset works end-to-end, including applying the right standardization.
    """
    csv_string = """subject,seq,value,scalar1,scalar2,META,BETA
S1,0,False,0,0,M1,B1
S1,1,True,1,10,M2,B2
S2,0,False,2,20,M2,B1
S3,0,True,3,30,M1,B1
S4,0,True,4,40,M2,B1
"""
    csv_path = create_dataset_csv_file(csv_string, test_output_dirs.root_dir)
    config = SequenceModelBase(
        local_dataset=csv_path,
        image_file_column=None,
        label_value_column="value",
        numerical_columns=["scalar1", "scalar2"],
        sequence_target_positions=[0],
        categorical_columns=["META", "BETA"],
        sequence_column="seq",
        num_dataload_workers=0,
        train_batch_size=2,
        should_validate=False,
        shuffle=False
    )
    config.read_dataset_if_needed()
    df = config.dataset_data_frame
    assert df is not None
    df1 = df[df.subject.isin(["S1", "S2"])]
    df2 = df[df.subject == "S3"]
    df3 = df[df.subject == "S4"]
    splits = DatasetSplits(train=df1, val=df2, test=df3)
    with mock.patch.object(SequenceModelBase,
                           'get_model_train_test_dataset_splits',
                           return_value=splits):
        train_val_loaders = config.create_data_loaders()
        # Expected feature mean: Mean of the training data (0, 0), (1, 10), (2, 20) = (1, 10)
        # Expected (biased corrected) std estimate: Std of (0, 0), (1, 10), (2, 20) = (1, 10)
        feature_stats = config.get_torch_dataset_for_inference(ModelExecutionMode.TRAIN).feature_statistics
        assert feature_stats is not None
        assert_tensors_equal(feature_stats.mean, [1, 10])
        assert_tensors_equal(feature_stats.std, [1, 10])

        train_items = list(ClassificationItemSequence.from_minibatch(b)
                           for b in train_val_loaders[ModelExecutionMode.TRAIN])
        assert len(train_items) == 1, "2 items in training set with batch size of 2 should return 1 minibatch"
        assert len(train_items[0]) == 2
        assert train_items[0][0].id == "S1"
        assert_tensors_equal(train_items[0][0].items[0].get_all_non_imaging_features(), [-1., -1., 1., 0., 1., 0.])
        assert_tensors_equal(train_items[0][0].items[1].get_all_non_imaging_features(), [0., 0., 0., 1., 0., 1.])
        assert train_items[0][1].id == "S2"
        assert_tensors_equal(train_items[0][1].items[0].get_all_non_imaging_features(), [1., 1., 0., 1., 1., 0.])
        val_items = list(ClassificationItemSequence.from_minibatch(b)
                         for b in train_val_loaders[ModelExecutionMode.VAL])
        assert len(val_items) == 1
        assert len(val_items[0]) == 1
        assert val_items[0][0].id == "S3"
        # Items in the validation set should be normalized using the mean and std on the training data.
        # Hence, the non-image features (3, 30) should turn into (2, 2)
        assert_tensors_equal(val_items[0][0].items[0].get_all_non_imaging_features(), [2., 2., 1., 0., 1., 0.])

        # Check that the test set is also normalized correctly using the training mean and std.
        test_items = list(ClassificationItemSequence(**b)
                          for b in config.get_torch_dataset_for_inference(ModelExecutionMode.TEST))
        assert test_items[0].id == "S4"
        # Check Non-image features of (4, 40)
        assert_tensors_equal(test_items[0].items[0].get_all_non_imaging_features(), [3., 3., 0., 1., 1., 0.])


def test_get_class_weights_dataset(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test training and testing of sequence models that predicts at multiple time points,
    when it is started via run_ml.
    """
    dataset_contents = _get_multi_label_sequence_dataframe()
    config = ToyMultiLabelSequenceModel(should_validate=False)
    assert config.get_target_indices() == [1, 2, 3]
    expected_prediction_targets = ["Seq_pos 01", "Seq_pos 02", "Seq_pos 03"]
    assert len(config.get_target_indices()) == len(expected_prediction_targets)  # type: ignore
    config.set_output_to(test_output_dirs.root_dir)
    config.dataset_data_frame = dataset_contents
    config.pre_process_dataset_dataframe()
    splits = config.get_dataset_splits()
    train_dataset = config.create_torch_datasets(splits)[ModelExecutionMode.TRAIN]
    class_counts = train_dataset.get_class_counts()
    assert class_counts == {0.0: 9, 1.0: 2}


def test_get_labels_at_target_indices() -> None:
    """
    Test to ensure label selection based on target indices is as expected
    """
    sequence_items = _create_scalar_items(length=3)

    sequence = ClassificationItemSequence(id="A", items=sequence_items)

    # since label at sequence position 3 will not exist, we expect the result tensor to be padded with a nan
    labels = sequence.get_labels_at_target_indices(target_indices=[0, 1, 2, 3])
    assert torch.allclose(labels, torch.tensor([[1.0], [1.0], [1.0], [np.nan]]), equal_nan=True)

    # test we can extract all of the labels in the sequence
    labels = sequence.get_labels_at_target_indices(target_indices=[0, 1, 2])
    assert torch.equal(labels, torch.tensor([[1.0], [1.0], [1.0]]))

    # test we can extract only a subset of the labels in the sequence
    labels = sequence.get_labels_at_target_indices(target_indices=[0, 1])
    assert torch.equal(labels, torch.tensor([[1.0], [1.0]]))

    # test we raise an exception for invalid target indices
    with pytest.raises(Exception):
        sequence.get_labels_at_target_indices(target_indices=[-1])


def test_create_labels_tensor_for_minibatch() -> None:
    """
    Test to make sure labels tensor is created as expected for minibatch
    """

    sequences = [ClassificationItemSequence(id=x, items=_create_scalar_items(length=i + 1))
                 for i, x in enumerate(["A", "B"])]

    labels = ClassificationItemSequence.create_labels_tensor_for_minibatch(sequences, target_indices=[0, 1, 2])
    assert torch.allclose(labels, torch.tensor([
        [[1.0], [np.nan], [np.nan]],
        [[1.0], [1.0], [np.nan]]]
    ), equal_nan=True)

    labels = ClassificationItemSequence.create_labels_tensor_for_minibatch(sequences, target_indices=[0, 1])
    assert torch.allclose(labels, torch.tensor([
        [[1.0], [np.nan]],
        [[1.0], [1.0]]]
    ), equal_nan=True)

    labels = ClassificationItemSequence.create_labels_tensor_for_minibatch(sequences, target_indices=[0])
    assert torch.equal(labels, torch.tensor([
        [[1.0]],
        [[1.0]]]
    ))


def _create_scalar_items(length: int, label_value: float = 1.0) -> List[ScalarItem]:
    return [ScalarItem(metadata=GeneralSampleMetadata(id="foo", sequence_position=x),
                       numerical_non_image_features=torch.tensor([]),
                       categorical_non_image_features=torch.tensor([]),
                       label=torch.tensor([label_value]),
                       images=torch.tensor([]),
                       segmentations=torch.tensor([])) for x in range(length)]
