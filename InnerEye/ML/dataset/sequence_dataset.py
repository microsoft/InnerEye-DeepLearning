#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from InnerEye.ML.dataset.scalar_dataset import ScalarDatasetBase, filter_valid_classification_data_sources_items
from InnerEye.ML.dataset.scalar_sample import ScalarItem, SequenceDataSource
from InnerEye.ML.dataset.sequence_sample import ClassificationItemSequence, ListOfSequences
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils.features_util import FeatureStatistics
from InnerEye.ML.utils.transforms import ComposeTransforms, Transform3D
from InnerEye.ML.scalar_config import ImageDimension


def get_longest_contiguous_sequence(items: List[SequenceDataSource],
                                    min_sequence_position_value: int = 0,
                                    max_sequence_position_value: Optional[int] = None) -> List[SequenceDataSource]:
    """
    From a list of classification items, extract the longest contiguous sequence of items starting
    at position value min_sequence_position_value.

    For example:

    if min_sequence_position_value = 1 and the
    input has sequence positions [0, 1, 3, 4], the result retains the items for positions [1].

    if min_sequence_position_value = 1 and max_sequence_position_value = 2, then if
    input has sequence positions [0, 1, 2, 3], the result retains the items for positions [1, 2].

    if min_sequence_position_value = 1 and max_sequence_position_value = 4, then if
    input has sequence positions [0, 1, 2, 3], the result retains the items for positions [1, 2, 3].

    If the input sequence is [2, 3], the result is an empty list
    (the longest sequence must start at 1 but there is no item with position 1)

    :param items: A list of classification items, sorted by sequence_position.
    :param min_sequence_position_value: The minimum sequence position all sequences start from, 0 is default.
    :param max_sequence_position_value: If provided then this is the maximum sequence position the sequence can
    end with. Longer sequences will be truncated. None is default.
    :return: A list of classification items, with a maximum sequence_position that is the
    len(result) - 1.
    """
    result: List[SequenceDataSource] = []

    # make sure the input sequence is sorted by sequence position first
    items = list(sorted(items, key=lambda x: x.metadata.sequence_position))

    _last_seq_item = next((x for x in items if x.metadata.sequence_position == min_sequence_position_value), None)

    if _last_seq_item is None:
        return result
    else:
        result.append(_last_seq_item)
        for item in items[items.index(_last_seq_item) + 1:]:
            if max_sequence_position_value is not None and \
                    item.metadata.sequence_position > max_sequence_position_value:
                break
            elif item.metadata.sequence_position - _last_seq_item.metadata.sequence_position == 1:
                _last_seq_item = item
                result.append(item)
            else:
                break

        return result


def group_samples_into_sequences(items: Iterable[SequenceDataSource],
                                 min_sequence_position_value: int = 0,
                                 max_sequence_position_value: Optional[int] = None) -> ListOfSequences:
    """
    Turns a flat list of classification items into a list of per-subject classification items. The resulting list
    has one entry per unique sample ID in the input. With a single sample ID, the items
    are sorted by metadata.sequence_position in ascending order.
    Also, all subject data is restricted to the largest contiguous sequence starting at 0
    (e.g., if sequence positions are [0, 1, 4], only [0, 1] are retained,
    if sequence positions are [1, 2, 3] nothing is retained)
    :param items: The items that should be grouped.
    :param max_sequence_position_value: If provided then this is the maximum sequence position the sequence can
    end with. Longer sequences will be truncated. None is default.
    up to and including this value. Entries beyond that sequence_position will be dropped.
    :param min_sequence_position_value: All sequences must have a entries with sequence_position starting
    from and including this value, 0 is default.
    :return:
    """
    if min_sequence_position_value < 0:
        raise ValueError("Argument min_sequence_position_value must be >= 0")

    if max_sequence_position_value:
        if max_sequence_position_value < min_sequence_position_value:
            raise ValueError(f"Argument max_sequence_position_value: {max_sequence_position_value} must "
                             f"be >= min_sequence_position_value: {min_sequence_position_value}")

    grouped: DefaultDict[str, List[SequenceDataSource]] = defaultdict(list)
    for item in items:
        grouped[item.id].append(item)
    result: List[ClassificationItemSequence[SequenceDataSource]] = []
    for sample_id, items in grouped.items():
        unique_positions = set(x.metadata.sequence_position for x in items)
        if len(unique_positions) != len(items):
            raise ValueError(f"The set of sequence positions for subject {sample_id} contains duplicates.")

        group_sorted = get_longest_contiguous_sequence(
            items=items,
            min_sequence_position_value=min_sequence_position_value,
            max_sequence_position_value=max_sequence_position_value
        )

        if len(group_sorted) > 0:
            result.append(ClassificationItemSequence(id=sample_id, items=group_sorted))
        else:
            # No contiguous sequence at all
            logging.warning(f"Skipped sequence for subject {sample_id} as it was not contiguous")

    return result


def add_difference_features(sequences: ListOfSequences, feature_indices: List[int]) -> ListOfSequences:
    """
    For each sequence in the argument, compute feature differences to the first sequence element, and adds them
    as new features at the end of the non-image features. Feature differences are only compute for those columns
    in numerical_non_image_features that are given in the feature_indices argument.
    The first sequence elements gets feature differences that are all zeros. The i.th sequence element will get
    additional features that are the differences of numerical_non_image_features[:,j] and the same element in the
    0.th sequence
    element.
    :param sequences: The input sequences.
    :param feature_indices: The column indices in numerical_non_image_features for which differences should be computed.
    :return: A new list of sequences with the feature differences added as columns in the
    numerical_non_image_features field.
    """

    def add_features(seq: ClassificationItemSequence) -> ClassificationItemSequence:
        items_mapped: List[SequenceDataSource] = []
        feature_baseline = None
        for item_index, item in enumerate(seq.items):
            if item_index == 0:
                feature_baseline = torch.stack([item.numerical_non_image_features[:, i] for i in feature_indices],
                                               dim=0)
            features_for_diff = torch.stack([item.numerical_non_image_features[:, i] for i in feature_indices], dim=0)
            diff = features_for_diff - feature_baseline
            new_features = torch.cat([item.numerical_non_image_features, diff.t()], dim=1)
            items_mapped.append(item.clone_with_overrides(numerical_non_image_features=new_features))
        return ClassificationItemSequence(id=seq.id, items=items_mapped)

    return list(map(add_features, sequences))


"""
Example for the use of SequenceDataset:

A sequence dataset groups rows not only by subject ID (as the normal ClassificationDataset does), but also
by a sequence position. That sequence position is read out from a column specified in the `sequence_column`
field of the model configuration.

Within a given subject, a sequence dataset returns instance of ClassificationItemSequence, each of which contains
a ClassificationItem for each individual sequence position.

Example use case:
subject,POSITION,measure0,measure0,image,Label
1,0,92,362,img1,0
1,1,92,357,img1,1
1,2,92,400,,0
2,0,82,477,img2,0
2,1,82,,img2,1
2,2,82,220,img2,0

To read images and measure1 as a non-imaging feature from this file, you would specify:
    image_channels = []
    image_file_column = "image"
    label_channel = None
    label_value_column = "Label"
    non_image_feature_channels = []
    numerical_columns = ["measure1"]
    sequence_column = "POSITION"

All of the "*_channel" arguments can be left empty. After grouping by subject and sequence position,
only 1 row remains, and it is hence clear to the data loader which row to read from.

After reading the CSV files, the data loader will remove all rows where
* there is no image file path given in the file
* there is a missing value in the non-image features (missing measure1 in the example above)
* If the traverse_dirs_when_loading is given in the model config, the data loader will also remove items where
the image file does not exist.

After this filtering, the data loader will group the items by subject, and sort by position within a subject.
Within a subject, the sequences must start at position 0, and are kept up to the first "gap". Hence, if only positions
0, 1, and 2 are valid, the sequence that is kept contains items [0, 1]

Assuming that the image files all exist, this would return
* result[0] containing "1" with POSITION numbers 0 and 1 (position 2 has no image file)
* result[1] containing "2" with POSITION number 0 only (position 1 has missing measure1, and hence position 2 has to
be dropped as well)
"""


class SequenceDataset(ScalarDatasetBase[SequenceDataSource]):
    """
    A dataset class that groups its raw dataset rows by subject ID and a sequence index. Each item in the dataset
    has all the rows for a given subject, and within each subject, a sorted sequence of rows.
    """
    items: List[ClassificationItemSequence[SequenceDataSource]]  # type: ignore

    def __init__(self,
                 args: SequenceModelBase,
                 data_frame: pd.DataFrame,
                 feature_statistics: Optional[
                     FeatureStatistics[ClassificationItemSequence[SequenceDataSource]]] = None,
                 name: Optional[str] = None,
                 sample_transforms: Optional[Union[ComposeTransforms[ScalarItem], Transform3D[ScalarItem]]] = None,
                 image_dimension: ImageDimension = ImageDimension.Image_3D):
        """
        Creates a new sequence dataset from a dataframe.
        :param args: The model configuration object.
        :param data_frame: The dataframe to read from.
        :param feature_statistics: If given, the normalization factor for the non-image features is taken
        :param sample_transforms: optional transformation to apply to each sample in the loading step.
        from the values provided. If None, the normalization factor is computed from the data in the present dataset.
        :param name: Name of the dataset, used for logging
        """
        super().__init__(args=args,
                         data_frame=data_frame,
                         feature_statistics=feature_statistics,
                         name=name,
                         sample_transforms=sample_transforms,
                         image_dimension=image_dimension)
        if self.args.sequence_column is None:
            raise ValueError("This class requires a value in the `sequence_column`, specifying where the "
                             "sequence index should be read from.")

        data_sources = self.load_all_data_sources()
        grouped = group_samples_into_sequences(
            data_sources,
            min_sequence_position_value=self.args.min_sequence_position_value,
            max_sequence_position_value=self.args.max_sequence_position_value
        )
        if self.args.add_differences_for_features:
            missing_columns = set(self.args.add_differences_for_features) - set(self.args.numerical_columns)
            if len(missing_columns) > 0:
                raise ValueError(f"Unable to add differences for these columns because they have not been specified "
                                 f"in the `non_image_feature_channels` property: {missing_columns}")
            feature_indices = [self.args.numerical_columns.index(f) for f in self.args.add_differences_for_features]
            grouped = add_difference_features(grouped, feature_indices)
        self.status += f"After grouping: {len(grouped)} subjects."
        self.items = grouped
        self.normalize_non_image_features()

    def get_status(self) -> str:
        """
        Creates a human readable string that describes the contents of the dataset.
        """
        return self.status

    def filter_valid_data_sources_items(self, data_sources: List[SequenceDataSource]) -> List[SequenceDataSource]:
        return filter_valid_classification_data_sources_items(
            items=data_sources,
            file_to_path_mapping=self.file_to_full_path,
            min_sequence_position_value=self.args.min_sequence_position_value,
            max_sequence_position_value=self.args.max_sequence_position_value
        )

    def get_labels_for_imbalanced_sampler(self) -> List[float]:
        """
        Returns a list of all the labels at the target_index position. Is used to
        compute the weights in the ImbalancedSampler. If more than on target position
        is specified the ImbalancedSampler cannot be used.
        :return:
        """
        if len(self.args.get_target_indices()) > 1:
            raise NotImplementedError("You cannot use the ImbalancedSampler if you"
                                      "want to predict more than one sequence position."
                                      "Use loss weighting instead.")
        return [seq.get_labels_at_target_indices(self.args.get_target_indices())[-1].item()
                for seq in self.items]

    def get_class_counts(self) -> Dict:
        """
        Return class weights that are proportional to the inverse frequency of label counts (summed
        over all target indices).
        :return: Dictionary of {"label": count}
        """
        all_labels_per_target = torch.stack([seq.get_labels_at_target_indices(self.args.get_target_indices())
                                             for seq in self.items])  # [N, T, 1]
        non_nan_labels = list(filter(lambda x: not np.isnan(x), all_labels_per_target.flatten().tolist()))
        return dict(Counter(non_nan_labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        loaded = list(map(self.load_item, self.items[i].items))
        return vars(ClassificationItemSequence(id=self.items[i].id, items=loaded))
