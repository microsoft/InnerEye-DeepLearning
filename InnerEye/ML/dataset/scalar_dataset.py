#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
import sys
import typing
from abc import abstractmethod
from collections import Counter, defaultdict
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Set, TypeVar, Union

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from more_itertools import flatten

from InnerEye.ML.dataset.full_image_dataset import GeneralDataset
from InnerEye.ML.dataset.sample import GeneralSampleMetadata
from InnerEye.ML.dataset.scalar_sample import ScalarDataSource, ScalarItem, SequenceDataSource
from InnerEye.ML.scalar_config import LabelTransformation, ScalarModelBase
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils.csv_util import CSV_CHANNEL_HEADER, CSV_SUBJECT_HEADER
from InnerEye.ML.utils.dataset_util import CategoricalToOneHotEncoder
from InnerEye.ML.utils.features_util import FeatureStatistics
from InnerEye.ML.utils.transforms import Compose3D, Transform3D

T = TypeVar('T', bound=ScalarDataSource)


def extract_label_classification(label_string: str, sample_id: str, num_classes: int,
                                 is_classification_dataset: bool) -> List[float]:
    """
    Converts a string from a dataset.csv file that contains a model's label to a scalar.

    For classification datasets:
    If num_classes is 1 (binary classification tasks):
        The function maps ["1", "true", "yes"] to [1], ["0", "false", "no"] to [0].
        If the entry in the CSV file was missing (no string given at all) or an empty string, it returns math.nan.
    If num_classes is greater than 1 (multilabel datasets):
        The function maps a pipe-separated set of classes to a tensor with ones at the indices
        of the positive classes and 0 elsewhere (for example if we have a task with 6 label classes,
        map "1|3|4" to [0, 1, 0, 1, 1, 0]).
        If the entry in the CSV file was missing (no string given at all) or an empty string,
        this function returns an all-zero tensor (none of the label classes were positive for this sample).

    For regression datasets:
    The function casts a string label to float. Raises an exception if the conversion is
    not possible.
    If the entry in the CSV file was missing (no string given at all) or an empty string, it returns math.nan.

    :param label_string: The value of the label as read from CSV via a DataFrame.
    :param sample_id: The sample ID where this label was read from. This is only used for creating error messages.
    :param num_classes: Number of classes. This should be equal the size of the model output.
    For binary classification tasks, num_classes should be one. For multilabel classification tasks, num_classes should
    correspond to the number of label classes in the problem.
    :param is_classification_dataset: If the model is a classification model
    :return: A list of floats with the same size as num_classes
    """

    if num_classes < 1:
        raise ValueError(f"Subject {sample_id}: Invalid number of classes: '{num_classes}'")

    if isinstance(label_string, float):
        if math.isnan(label_string):
            if num_classes == 1:
                # Pandas special case: When loading a dataframe with dtype=str, missing values can be encoded as NaN, and get into here.
                return [label_string]
            else:
                return [0] * num_classes
        else:
            raise ValueError(f"Subject {sample_id}: Unexpected float input {label_string} - did you read the "	
                             f"dataframe column as a string?")

    if not label_string:
        if not is_classification_dataset or num_classes == 1:
            return [math.nan]
        else:
            return [0] * num_classes

    if is_classification_dataset:
        if num_classes == 1:
            label_lower = label_string.lower()
            if label_lower in ["true", "yes"]:
                return [1.0]
            elif label_lower in ["false", "no"]:
                return [0.0]
            elif label_string in ["0", "1"]:
                return [float(label_string)]
            else:
                raise ValueError(f"Subject {sample_id}: Label string not recognized: '{label_string}'. "
                             f"Should be one of true/false, yes/no or 0/1.")

        if '|' in label_string or label_string.isdigit():
            classes = [int(a) for a in label_string.split('|')]

            out_of_range = [_class for _class in classes if _class >= num_classes]
            if out_of_range:
                raise ValueError(f"Subject {sample_id}: Indices {out_of_range} are out of range, for number of classes "
                                 f"= {num_classes}")

            one_hot_array = np.zeros(num_classes, dtype=np.float)
            one_hot_array[classes] = 1.0
            return one_hot_array.tolist()
    else:
        try:
            return [float(label_string)]
        except ValueError:
            pass

    raise ValueError(f"Subject {sample_id}: Label string not recognized: '{label_string}'")


def _get_single_channel_row(subject_rows: pd.DataFrame,
                            channel: Optional[str],
                            subject_id: str,
                            channel_column: str = CSV_CHANNEL_HEADER) -> Dict[str, Any]:
    """
    Extracts a single row from a set of rows, where the column `channel_column` has the value given in the
    'channel' argument. Throws a ValueError if there is no or more than 1 such row.
    The result is returned as a dictionary, not a DataFrame!
    If the 'channel' argument is null, the input is expected to be already 1 row, which is returned as a dictionary.
    :param subject_rows: A set of rows all belonging to the same subject.
    :param channel: The value to look for in the `channel_column` column. This can be null. If it is null,
    the input `subject_rows` is expected to have exactly 1 row.
    :param subject_id: A string describing the presently processed subject. This is only used for error reporting.
    :return: A dictionary mapping from column names to values, created from the unique row that was found.
    """
    if channel:
        subject_rows = subject_rows[np.in1d(subject_rows[channel_column].values, [channel])]

    num_rows = len(subject_rows)
    if num_rows != 1:
        with pd.option_context('display.max_columns', None,
                               'display.max_rows', 30,
                               'display.expand_frame_repr', False):
            logging.error(f"Invalid subject data: {subject_rows}")
        raise ValueError(f"Subject {subject_id}: There should be exactly "
                         f"one row to read from, but got {num_rows} rows.")
    return subject_rows.head().to_dict('records')[0]


def _string_to_float(text: Union[str, float], error_message_prefix: str = None) -> float:
    """
    Converts a string coming from a dataset.csv file to a floating point number, taking into account all the
    corner cases that can happen when the dataset file is malformed.
    :param text: The element coming from the dataset.csv file.
    :param error_message_prefix: A prefix string that will go into the error message if the conversion fails.
    :return: A floating point number, possibly np.nan.
    """
    if text is None:
        return math.nan
    if isinstance(text, float):
        # Even when loading a dataframe with dtype=str, missing values can be encoded as NaN, and get into here.
        return text
    text = (text or "").strip()
    if len(text) == 0:
        return math.nan
    try:
        return float(text)
    except:
        logging.warning(f"{error_message_prefix}: Unable to parse value '{text}'")
        return math.nan


def load_single_data_source(subject_rows: pd.DataFrame,
                            subject_id: str,
                            label_value_column: str,
                            channel_column: str,
                            image_channels: Optional[List[str]] = None,
                            image_file_column: Optional[str] = None,
                            label_channels: Optional[List[str]] = None,
                            transform_labels: Union[Callable, List[Callable]] = LabelTransformation.identity,
                            non_image_feature_channels: Optional[Dict] = None,
                            numerical_columns: Optional[List[str]] = None,
                            categorical_data_encoder: Optional[CategoricalToOneHotEncoder] = None,
                            metadata_columns: Optional[Set[str]] = None,
                            is_classification_dataset: bool = True,
                            num_classes: int = 1,
                            sequence_position_numeric: Optional[int] = None) -> T:
    """
    Converts a set of dataset rows for a single subject to a ScalarDataSource instance, which contains the
    labels, the non-image features, and the paths to the image files.
    :param num_classes: Number of classes, this is equivalent to model output tensor size
    :param channel_column: The name of the column that contains the row identifier ("channels")
    :param metadata_columns: A list of columns that well be added to the item metadata as key/value pairs.
    :param subject_rows: All dataset rows that belong to the same subject.
    :param subject_id: The identifier of the subject that is being processed.
    :param image_channels: The names of all channels (stored in the CSV_CHANNEL_HEADER column of the dataframe)
    that are expected to be loaded from disk later because they are large images.
    :param image_file_column: The name of the column that contains the image file names.
    :param label_channels: The name of the channel where the label scalar or vector is read from.
    :param label_value_column: The column that contains the value for the label scalar or vector.
    :param non_image_feature_channels: non_image_feature_channels: A dictonary of the names of all channels where
    additional scalar values should be read from. THe keys should map each feature to its channels.
    :param numerical_columns: The names of all columns where additional scalar values should be read from.
    :param categorical_data_encoder: Encoding scheme for categorical data.
    :param is_classification_dataset: If True, the dataset will be used in a classification model. If False,
    assume that the dataset will be used in a regression model.
    :param transform_labels: a label transformation or a list of label transformation to apply to the labels.
    If a list is provided, the transformations are applied in order from left to right.
    :param sequence_position_numeric: Numeric position of the data source in a data sequence. Assumed to be
    a non-sequential dataset item if None provided (default).
    :return:
    """

    def _get_row_for_channel(channel: Optional[str]) -> Dict[str, str]:
        return _get_single_channel_row(subject_rows, channel, subject_id, channel_column)

    def _get_label_as_tensor(channel: Optional[str]) -> torch.Tensor:
        label_row = _get_row_for_channel(channel)
        label_string = label_row[label_value_column]
        return torch.tensor(
            extract_label_classification(label_string=label_string, sample_id=subject_id, num_classes=num_classes,
                                         is_classification_dataset=is_classification_dataset),
            dtype=torch.float)

    def _apply_label_transforms(labels: Any) -> Any:
        """
        Apply the transformations in order.
        """
        if isinstance(transform_labels, List):
            for transform in transform_labels:
                labels = transform(labels)
            label = labels
        else:
            label = transform_labels(labels)
        return label

    def create_none_list(x: Optional[List]) -> List:
        return [None] if x is None or len(x) == 0 else x

    def get_none_list_from_dict(non_image_channels: Dict[str, List[str]], feature: str) -> Sequence[Optional[str]]:
        """
        Return either the list of channels for a given column or if None was passed as
        numerical channels i.e. there are no channel to be specified return [None].
        :param non_image_channels: Dict mapping features name to their channels
        :param feature: feature name for which to return the channels
        :return: List of channels for the given feature.
        """
        if non_image_channels == {}:
            return [None]
        else:
            return non_image_channels[feature]

    def is_empty(x: Optional[List]) -> bool:
        return x is None or len(x) == 0

    def none_if_missing_in_csv(x: Any) -> Optional[str]:
        # If the CSV contains missing values they turn into NaN here, but mark them as None rather.
        return None if isinstance(x, float) and np.isnan(x) else x

    subject_rows = subject_rows.fillna('')
    labels = []
    if label_channels:
        for channel in label_channels:
            labels.append(_get_label_as_tensor(channel))
    else:
        labels.append(_get_label_as_tensor(None))

    label = _apply_label_transforms(labels)

    channel_for_metadata = label_channels[0] if label_channels else None
    label_row = _get_row_for_channel(channel_for_metadata)
    metadata = GeneralSampleMetadata(id=subject_id, props={key: none_if_missing_in_csv(label_row[key])
                                                           for key in metadata_columns or set()})

    image_files = []
    if image_file_column:
        for image_channel in create_none_list(image_channels):
            # Alternative: restrict rows to given channels first, then read out the relevant columns.
            file_path = _get_row_for_channel(image_channel)[image_file_column]
            image_files.append(none_if_missing_in_csv(file_path))

    numerical_columns = numerical_columns or []
    categorical_columns = categorical_data_encoder.get_supported_dataset_column_names() if categorical_data_encoder \
        else []
    _feature_columns = numerical_columns + categorical_columns

    if not non_image_feature_channels:
        non_image_feature_channels = {}

    numerical = []
    categorical = {}
    if not is_empty(_feature_columns):
        for column in _feature_columns:
            list_channels: Sequence[Optional[str]] = [str(sequence_position_numeric)] \
                if sequence_position_numeric is not None else get_none_list_from_dict(non_image_feature_channels,
                                                                                      column)
            numerical_col, categorical_col = [], []
            for channel in list_channels:  # type: ignore
                row = _get_row_for_channel(channel)
                prefix = f"Channel {channel}, column {column}"
                if column in numerical_columns:
                    numerical_col.append(_string_to_float(row[column], error_message_prefix=prefix))
                else:
                    categorical_col.append(row[column])
            if column in numerical_columns:
                numerical.extend(numerical_col)
            else:
                categorical[column] = categorical_col

    categorical_non_image_features = categorical_data_encoder.encode(categorical) \
        if categorical_data_encoder else torch.tensor(list(categorical.values()))

    datasource: Union[SequenceDataSource, ScalarDataSource]
    if sequence_position_numeric is not None:
        metadata.sequence_position = sequence_position_numeric
        datasource = SequenceDataSource(
            label=label,
            channel_files=image_files,
            numerical_non_image_features=torch.tensor(numerical).float(),
            categorical_non_image_features=categorical_non_image_features.float(),
            metadata=metadata
        )
        return datasource  # type: ignore

    datasource = ScalarDataSource(
        label=label,
        channel_files=image_files,
        numerical_non_image_features=torch.tensor(numerical).float(),
        categorical_non_image_features=categorical_non_image_features.float(),
        metadata=metadata
    )
    return datasource  # type: ignore


class DataSourceReader(Generic[T]):
    """
    Class that allows reading of data sources from a scalar dataset data frame.
    """

    def __init__(self,
                 data_frame: pd.DataFrame,
                 label_value_column: str,
                 image_file_column: Optional[str] = None,
                 image_channels: Optional[List[str]] = None,
                 label_channels: Optional[List[str]] = None,
                 transform_labels: Union[Callable, List[Callable]] = LabelTransformation.identity,
                 non_image_feature_channels: Optional[Dict[str, List[str]]] = None,
                 numerical_columns: Optional[List[str]] = None,
                 sequence_column: Optional[str] = None,
                 subject_column: str = CSV_SUBJECT_HEADER,
                 channel_column: str = CSV_CHANNEL_HEADER,
                 is_classification_dataset: bool = True,
                 num_classes: int = 1,
                 categorical_data_encoder: Optional[CategoricalToOneHotEncoder] = None):
        """
        :param label_value_column: The column that contains the value for the label scalar or vector.
        :param image_file_column: The name of the column that contains the image file names.
        :param image_channels: The names of all channels (stored in the CSV_CHANNEL_HEADER column of the dataframe)
        :param label_channels: The name of the channel where the label scalar or vector is read from.
        :param transform_labels: a label transformation or a list of label transformation to apply to the labels.
        If a list is provided, the transformations are applied in order from left to right.
        :param non_image_feature_channels: non_image_feature_channels: A dictionary of the names of all channels where
        additional scalar values should be read from. The keys should map each feature to its channels.
        :param numerical_columns: The names of all columns where additional scalar values should be read from.
        :param sequence_column: The name of the column that contains the sequence index, that will be stored in
        metadata.sequence_position. If this column name is not provided, the sequence_position will be 0.
        :param subject_column: The name of the column that contains the subject identifier
        :param channel_column: The name of the column that contains the row identifier ("channels")
        that are expected to be loaded from disk later because they are large images.
        :param is_classification_dataset: If the current dataset is classification or not.
        :param categorical_data_encoder: Encoding scheme for categorical data.
        """
        self.categorical_data_encoder = categorical_data_encoder
        self.is_classification_dataset = is_classification_dataset
        self.channel_column = channel_column
        self.subject_column = subject_column
        self.sequence_column = sequence_column
        self.numerical_columns = numerical_columns
        self.non_image_feature_channels = non_image_feature_channels
        self.transform_labels = transform_labels
        self.label_channels = label_channels
        self.image_channels = image_channels
        self.image_file_column = image_file_column
        self.label_value_column = label_value_column
        self.data_frame = data_frame
        self.num_classes = num_classes
        self.expected_non_image_channels: Union[List[None], Set[str]]

        if self.non_image_feature_channels is None:
            self.expected_non_image_channels = []
        else:
            self.expected_non_image_channels = set(sum(self.non_image_feature_channels.values(), []))
        self.expected_channels = {
            *(self.image_channels or []),
            *self.expected_non_image_channels,
            *(self.label_channels or [])
        }
        self.expected_columns = {
            self.subject_column,
            self.label_value_column,
            *(self.numerical_columns or []),
            *(self.categorical_data_encoder.get_supported_dataset_column_names()
              if self.categorical_data_encoder else [])
        }

        if self.numerical_columns and self.categorical_data_encoder:
            _intersection = set(self.numerical_columns).intersection(
                self.categorical_data_encoder.get_supported_dataset_column_names())
            if len(_intersection) > 0:
                raise ValueError(f"Following columns are defined as scalar and categorical: {_intersection}")

        if self.sequence_column:
            self.channel_column = self.sequence_column
            self.expected_columns.add(self.sequence_column)
        if len(self.expected_channels) > 0:
            self.expected_columns.add(self.channel_column)

        if self.image_file_column:
            self.expected_columns.add(self.image_file_column)
        missing_columns = self.expected_columns - set(self.data_frame)
        if len(missing_columns) > 0:
            raise ValueError(f"The following columns are missing: {', '.join(missing_columns)}")

        self.metadata_columns = set(self.data_frame) - self.expected_columns

    @staticmethod
    def load_data_sources_as_per_config(data_frame: pd.DataFrame,
                                        args: ScalarModelBase) -> List[T]:
        """
        Loads dataset items from the given dataframe, where all column and channel configurations are taken from their
        respective model config elements.
        :param data_frame: The dataframe to read dataset items from.
        :param args: The model configuration object.
        :return: A list of all dataset items that could be read from the dataframe.
        """
        # create a one hot encoder if non provided
        if args.categorical_columns and not args.categorical_feature_encoder:
            raise ValueError(f"One hot encoder not found to handle categorical_columns={args.categorical_columns}")

        if args.categorical_feature_encoder is not None:
            assert isinstance(args.categorical_feature_encoder, CategoricalToOneHotEncoder)  # mypy

        sequence_column = None
        if isinstance(args, SequenceModelBase):
            sequence_column = args.sequence_column

        return DataSourceReader[T](
            data_frame=data_frame,
            image_channels=args.image_channels,
            image_file_column=args.image_file_column,
            label_channels=args.label_channels,
            label_value_column=args.label_value_column,
            transform_labels=args.get_label_transform(),
            non_image_feature_channels=args.get_non_image_feature_channels_dict(),
            numerical_columns=args.numerical_columns,
            categorical_data_encoder=args.categorical_feature_encoder,
            sequence_column=sequence_column,
            subject_column=args.subject_column,
            channel_column=args.channel_column,
            num_classes=len(args.class_names),
            is_classification_dataset=args.is_classification_model
        ).load_data_sources(num_dataset_reader_workers=args.num_dataset_reader_workers)

    def load_data_sources(self, num_dataset_reader_workers: int = 0) -> List[T]:
        """
        Extracts information from a dataframe to create a list of ClassificationItem. This will create one entry per
        unique
        value of subject_id in the dataframe. The file is structured around "channels", indicated by specific values in
        the CSV_CHANNEL_HEADER column. The result contains paths to image files, a label vector, and a matrix of
        additional values that are specified by rows and columns given in non_image_feature_channels and
        numerical_columns.
        :param num_dataset_reader_workers: Number of worker processes to use, if 0 then single threaded execution,
        otherwise if -1 then multiprocessing with all available cpus will be used.
        :return: A list of ScalarDataSource or SequenceDataSource instances
        """
        subject_ids = self.data_frame[self.subject_column].unique()
        _backend: Optional[str] = None
        if num_dataset_reader_workers == 0:
            _n_jobs = 1
            _backend = "threading"
        elif num_dataset_reader_workers == -1:
            _n_jobs = cpu_count()
        else:
            _n_jobs = max(1, num_dataset_reader_workers)

        results = Parallel(n_jobs=_n_jobs, backend=_backend)(
            delayed(self.load_datasources_for_subject)(subject_id) for subject_id in subject_ids)

        return list(flatten(filter(None, results)))

    def load_datasources_for_subject(self, subject_id: str) -> Optional[List[T]]:

        rows = self.data_frame[np.in1d(self.data_frame[self.subject_column].values, [subject_id])]

        def _load_single_data_source(_rows: pd.DataFrame,
                                     _sequence_position_numeric: Optional[int] = None) -> T:
            return load_single_data_source(
                subject_rows=_rows,
                subject_id=subject_id,
                image_channels=self.image_channels,
                image_file_column=self.image_file_column,
                label_channels=self.label_channels,
                label_value_column=self.label_value_column,
                transform_labels=self.transform_labels,
                non_image_feature_channels=self.non_image_feature_channels,
                numerical_columns=self.numerical_columns,
                categorical_data_encoder=self.categorical_data_encoder,
                metadata_columns=self.metadata_columns,
                channel_column=self.channel_column,
                is_classification_dataset=self.is_classification_dataset,
                num_classes=self.num_classes,
                sequence_position_numeric=_sequence_position_numeric
            )

        def _load_sequence_data_source(_sequence_position: Any) -> T:
            _sequence_position_numeric = int(_sequence_position)
            if _sequence_position_numeric < 0:
                raise ValueError(
                    f"Sequence positions must be non-negative integers, but got: {_sequence_position}")
            else:
                seq_rows = rows[np.in1d(rows[self.sequence_column].values, [_sequence_position])]
                return _load_single_data_source(seq_rows, _sequence_position_numeric)

        if self.sequence_column:
            seq_positions = rows[self.sequence_column].unique()
            return list(map(_load_sequence_data_source, seq_positions))
        else:
            if len(self.expected_channels) > 0:
                missing_channels = self.expected_channels - set(rows[self.channel_column])
                if len(missing_channels) > 0:
                    logging.warning(f"Subject {subject_id} will be skipped completely because the following "
                                    f"channels are missing: {','.join(missing_channels)}.")
                    return None
            return [_load_single_data_source(rows)]


def files_by_stem(root_path: Path) -> Dict[str, Path]:
    """
    Lists all files under the given root directory recursively, and returns a mapping from file name stem to full path.
    The file name stem is computed more restrictively than what Path.stem returns: file.nii.gz will use "file" as the
    stem, not "file.nii" as Path.stem would.
    Only actual files are returned in the mapping, no directories.
    If there are multiple files that map to the same stem, the function raises a ValueError.
    :param root_path: The root directory from which the file search should start.
    :return: A dictionary mapping from file name stem to the full path to where the file is found.
    """
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError("The root_path must be a directory that exists.")
    result: Dict[str, Path] = dict()
    duplicates: Dict[str, List[Path]] = defaultdict(list)
    for item in root_path.rglob("*"):
        if item.is_file():
            key = item.name
            i = key.find('.')
            if 0 < i < len(key) - 1:
                key = key[:i]
            if key in result:
                duplicates[key].append(item)
            else:
                result[key] = item
    if len(duplicates) > 0:
        for key, files in duplicates.items():
            logging.info(f"{key} maps to {len(files) + 1} locations: ")
            for f in files + [result[key]]:
                logging.info(f"    {key} -> {f}")
        raise ValueError(f"Unable to create a unique file name mapping. {len(duplicates)} files have duplicates")
    return result


def is_valid_item_index(item: ScalarDataSource,
                        max_sequence_position_value: Optional[int],
                        min_sequence_position_value: int = 0) -> bool:
    """
    Returns True if the item metadata in metadata.sequence_position is a valid sequence index.
    :param item: The item to check.
    :param min_sequence_position_value: Check if the item has a metadata.sequence_position that is at least
    the value given here. Default is 0.
    :param max_sequence_position_value: If provided then this is the maximum sequence position the sequence can
    end with. Longer sequences will be truncated. None is default.
    :return: True if the item has a valid index.
    """
    # If no max_sequence_position_value is given, we don't care about
    # the sequence position, it must not even be provided.
    max_sequence_position_value = max_sequence_position_value if max_sequence_position_value is not None else \
        sys.maxsize
    return min_sequence_position_value <= item.metadata.sequence_position <= max_sequence_position_value


def filter_valid_classification_data_sources_items(items: Iterable[T],
                                                   file_to_path_mapping: Optional[Dict[str, Path]],
                                                   max_sequence_position_value: Optional[int] = None,
                                                   min_sequence_position_value: int = 0) -> List[T]:
    """
    Consumes a list of classification data sources, and removes all of those that have missing file names,
    or that have NaN or Inf features. If the file_to_path_mapping is given too, all items that have any missing files
    (files not present on disk) are dropped too. Items that have sequence position larger than the
    max_sequence_position_value are removed.

    :param items: The list of items to filter.
    :param min_sequence_position_value: Restrict the data to items with a metadata.sequence_position that is at least
    the value given here. Default is 0.
    :param max_sequence_position_value: If provided then this is the maximum sequence position the sequence can
    end with. Longer sequences will be truncated. None is default.
    :param file_to_path_mapping: A mapping from a file name stem (without extension) to its full path.
    :return: A list of items, all of which are valid now.
    """

    def all_files_present(item: T) -> bool:
        if file_to_path_mapping:
            return all(f in file_to_path_mapping for f in item.channel_files)
        else:
            return True

    return [item for item in items
            if item.is_valid() and all_files_present(item)
            and is_valid_item_index(item, max_sequence_position_value=max_sequence_position_value,
                                    min_sequence_position_value=min_sequence_position_value)]


"""
Example for use of the ClassificationDataset:

Dataset file 1:

subject,channel,filePath,value,What,Ever
foo,label,,True,what,ever
foo,week0,something1.h5.gz,,
foo,week1,something2.h5.gz

Say you want to put something1.h5.gz and something2.h5.gz into a tensor, and read out the True value in the
label row:
    image_channels = ["week0","week1"]
    image_file_column = "filepath"
    label_channel: "label"
    label_column: "value"
    non_image_feature_channels: []
    numerical_columns: []

Dataset file 2:

subject,channel,filePath,label,scalar1,What,Ever
foo,week0,something1.h5.gz,True,75,
foo,week1,something1.h5.gz,False,78

You now want to get the label from the "week0" row, and read out Scalar1 at week0 and week1 as features:
    image_channels = ["week0","week1"]
    image_file_column = "filepath"
    label_channel: "week0"
    label_column: "label"
    non_image_feature_channels: ["week0", "week1"]
    numerical_columns: ["scalar1"]
"""


class ScalarDatasetBase(GeneralDataset[ScalarModelBase], Generic[T]):
    """
    A base class for datasets for classification tasks. It contains logic for loading images from disk,
    either from a fixed folder or traversing into subfolders.
    """
    one_hot_encoder: Optional[CategoricalToOneHotEncoder] = None
    status: str = ""
    items: List[T]

    def __init__(self, args: ScalarModelBase,
                 data_frame: Optional[pd.DataFrame] = None,
                 feature_statistics: Optional[FeatureStatistics] = None,
                 name: Optional[str] = None,
                 sample_transforms: Optional[Union[Compose3D[ScalarItem], Transform3D[ScalarItem]]] = None):
        """
        High level class for the scalar dataset designed to be inherited for specific behaviour
        :param args: The model configuration object.
        :param data_frame: The dataframe to read from.
        :param feature_statistics: If given, the normalization factor for the non-image features is taken
        :param name: Name of the dataset, used for diagnostics logging
        """
        super().__init__(args, data_frame, name)
        self.transforms = sample_transforms
        self.feature_statistics = feature_statistics
        self.file_to_full_path: Optional[Dict[str, Path]] = None
        if args.traverse_dirs_when_loading:
            if args.local_dataset is None:
                raise ValueError("Unable to load dataset because no `local_dataset` property is set.")
            logging.info(f"Starting to traverse folder {args.local_dataset} to locate image files.")
            self.file_to_full_path = files_by_stem(args.local_dataset)
            logging.info("Finished traversing folder.")

    def load_all_data_sources(self) -> List[T]:
        """
        Uses the dataframe to create data sources to be used by the dataset.
        :return:
        """
        all_data_sources = DataSourceReader.load_data_sources_as_per_config(self.data_frame, self.args)  # type: ignore
        self.status += f"Loading: {self.create_status_string(all_data_sources)}"
        all_data_sources = self.filter_valid_data_sources_items(all_data_sources)
        self.status += f"After filtering: {self.create_status_string(all_data_sources)}"
        return all_data_sources

    def filter_valid_data_sources_items(self, data_sources: List[T]) -> List[T]:
        raise NotImplementedError("filter_valid_data_source_items must be implemented by child classes")

    @abstractmethod
    def get_labels_for_imbalanced_sampler(self) -> List[float]:
        raise NotImplementedError

    def standardize_non_imaging_features(self) -> None:
        """
        Modifies the non image features that this data loader stores, such that they have mean 0, variance 1.
        Mean and variances are either taken from the argument feature_mean_and_variance (use that when
        the data set contains validation or test sequences), or computed from the dataset itself (use for the
        training set).
        If None, they will be computed from the data in the present object.
        """
        if self.items:
            self.feature_statistics = self.feature_statistics or FeatureStatistics[T].from_data_sources(self.items)
            self.items = self.feature_statistics.standardize(self.items)

    def load_item(self, item: ScalarDataSource) -> ScalarItem:
        """
        Loads the images and/or segmentations as given in the ClassificationDataSource item and
        applying the optional transformation specified by the class.
        :param item: The item to load.
        :return: A ClassificationItem instances with the loaded images, and the labels and non-image features copied
        from the argument.
        """
        sample = item.load_images(
            root_path=self.args.local_dataset,
            file_mapping=self.file_to_full_path,
            load_segmentation=self.args.load_segmentation,
            center_crop_size=self.args.center_crop_size,
            image_size=self.args.image_size)

        return Compose3D.apply(self.transforms, sample)

    def create_status_string(self, items: List[T]) -> str:
        """
        Creates a human readable string that contains the number of items, and the distinct number of subjects.
        :param items: Use the items provided to create the string
        :return: A string like "12 items for 5 subjects"
        """
        distinct = len({item.id for item in items})
        return f"{len(items)} items for {distinct} subjects. "


class ScalarDataset(ScalarDatasetBase[ScalarDataSource]):
    """
    A dataset class that can read CSV files with a flexible schema, and extract image file paths and non-image features.
    """

    def __init__(self, args: ScalarModelBase,
                 data_frame: Optional[pd.DataFrame] = None,
                 feature_statistics: Optional[FeatureStatistics[ScalarDataSource]] = None,
                 name: Optional[str] = None,
                 sample_transforms: Optional[Union[Compose3D[ScalarItem], Transform3D[ScalarItem]]] = None):
        """
        Creates a new scalar dataset from a dataframe.
        :param args: The model configuration object.
        :param data_frame: The dataframe to read from.
        :param feature_statistics: If given, the normalization factor for the non-image features is taken
        from the values provided. If None, the normalization factor is computed from the data in the present dataset.
        :param sample_transforms: Sample transforms that should be applied.
        :param name: Name of the dataset, used for diagnostics logging
        """
        super().__init__(args,
                         data_frame=data_frame,
                         feature_statistics=feature_statistics,
                         name=name,
                         sample_transforms=sample_transforms)
        self.items = self.load_all_data_sources()
        self.standardize_non_imaging_features()

    def get_status(self) -> str:
        """
        Creates a human readable string that describes the contents of the dataset.
        """
        return self.status

    def filter_valid_data_sources_items(self, data_sources: List[ScalarDataSource]) -> List[ScalarDataSource]:
        return filter_valid_classification_data_sources_items(
            items=data_sources,
            file_to_path_mapping=self.file_to_full_path,
            max_sequence_position_value=0
        )

    def get_labels_for_imbalanced_sampler(self) -> List[float]:
        """
        Returns a list of all the labels in the dataset. Used to compute
        the sampling weights in Imbalanced Sampler
        """
        if len(self.args.class_names) > 1:
            raise NotImplementedError("ImbalancedSampler is not supported for multilabel tasks.")

        return [item.label.item() for item in self.items]

    def get_class_counts(self) -> Dict[float, int]:
        """
        Return the label counts as a dictionary with the key-value pairs being the class indices and per-class counts.
        In the binary case, the dictionary will have a single element. The key will be 0 as there is only one class and
        one class index. The value stored will be the number of samples that belong to the positive class.
        In the multilabel case, this returns a dictionary with class indices and samples per class as the key-value
        pairs.
        :return: Dictionary of {class_index: count}
        """
        all_labels = [torch.flatten(torch.nonzero(item.label)).tolist() for item in self.items]  # [N, 1]
        flat_list = list(flatten(all_labels))
        freq_iter: typing.Counter = Counter()
        freq_iter.update({x: 0 for x in range(len(self.args.class_names))})
        freq_iter.update(flat_list)
        result = dict(freq_iter)
        return result

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return vars(self.load_item(self.items[i]))
