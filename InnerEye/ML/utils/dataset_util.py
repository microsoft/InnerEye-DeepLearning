#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder

from InnerEye.Common import common_util
from InnerEye.ML.common import OneHotEncoderBase
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.dataset.full_image_dataset import load_dataset_sources
from InnerEye.ML.utils import image_util, io_util, metrics_util, ml_util
from InnerEye.ML.utils.io_util import ImageHeader


class CategoricalToOneHotEncoder(OneHotEncoderBase):
    """
    Encoder to handle conversion to one-hot encoding for categorical data.
    """

    def __init__(self, columns_and_possible_categories: OrderedDict[str, List[str]]):
        """
        :param columns_and_possible_categories: Mapping between dataset column names
        to their possible values. eg: {'Inject': ['True', 'False']}. This is required
        to establish the one-hot encoding each of the possible values.
        """
        super().__init__()
        self._columns_and_possible_categories = columns_and_possible_categories
        self._feature_length = {}
        self._encoder = {}
        for col, value in columns_and_possible_categories.items():
            # Fit only once during initialization with all possible values.
            if np.inf in value:
                value.remove(np.inf)
            self._encoder[col] = OneHotEncoder(handle_unknown='ignore').fit(np.array(value).reshape(-1, 1))
            self._feature_length[col] = len(value)

    def encode(self, x: Dict[str, List[str]]) -> torch.Tensor:
        """
        Encode a dictonary mapping features to a list of values (one per channel). The values are expected to be string
        or NaN (if missing).

        Example for features "A" and "B"
        A| True, False
        B| False, True
        => {"A": ['True', 'False'], "B": ['False', 'True']} => [1, 0, 0, 1, 0, 1, 1, 0]

        In the case of missing values:
        A| True, False
        B| False,
        => {"A": ['True', 'False'], "B": ['False', nan]} => [1, 0, 0, 1, 0, 1, nan, nan]

        :param x: A dictonary mapping features to their categorical values (one value per channel).
        :return: A one-hot encoded Tensor of shape: [total feature length,]
        """
        encoded: np.ndarray = np.empty(0)
        for col in x:
            input_col = np.reshape(x[col], (-1, 1)).astype(str)
            encoded_col = self._encoder[col].transform(input_col).toarray()
            # By default OneHotEncoder will set all values of the encoded vector to be 0 if an illegal column
            # value was provided. Replace this with NaN.
            encoded_col[np.where(~encoded_col.any(axis=1))[0]] = np.NaN
            encoded = np.append(encoded, encoded_col)
        return torch.tensor(encoded)

    def get_supported_dataset_column_names(self) -> List[str]:
        """
        :returns list of categorical columns that are supported by this encoder
        """
        return list(self._columns_and_possible_categories.keys())

    def get_feature_length(self, feature_name: str) -> int:
        """
        The expected length of the one-hot encoded feature vector for a given feature.
        For example, a feature that takes 3 values, will be encoded as a one-hot vector
        of length 3.

        :param feature_name: the name of the column for which to compute the feature
        length.
        :returns the feature length i.e. number of possible values for this feature.
        """
        return self._feature_length[feature_name]

    @staticmethod
    def create_from_dataframe(dataframe: pd.DataFrame, columns: List[str]) -> CategoricalToOneHotEncoder:
        """
        Create an encoder that handles the conversion of the provided columns from a dataframe.

        :param dataframe: Dataframe to create the encoder from.
        :param columns: Supported columns for this encoder.
        :return: CategoricalToOneHotEncoder that handles the conversion of the `columns`.
        """

        def _get_unique_column_values(df: pd.Series) -> List[str]:
            """
            :param df: the column to analyze
            :return: all unique values present in df
            """
            # select all non nan or empty strings to identify the unique column values
            return df[df != ''].dropna().unique().tolist()

        return CategoricalToOneHotEncoder(
            columns_and_possible_categories=OrderedDict({x: _get_unique_column_values(dataframe[x]) for x in columns})
        )


@dataclass(frozen=True)
class DatasetExample:
    """
    Dataset sample with predictions after being passed through a model.
    """
    epoch: int  # the epoch this example belongs to.
    patient_id: int  # the patient id this example belongs to.
    header: ImageHeader  # the image header
    image: np.ndarray  # the example image data in Z x X x Y.
    prediction: np.ndarray  # the predictions for this image as multi-label mask shape: Z x X x Y.
    labels: np.ndarray  # the labels for this image as in shape: C x Z x X x Y.

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)

        ml_util.check_size_matches(arg1=self.image, arg2=self.prediction,
                                   dim1=3, dim2=3,
                                   matching_dimensions=[])

        ml_util.check_size_matches(arg1=self.image, arg2=self.labels,
                                   dim1=3, dim2=4,
                                   matching_dimensions=[-1, -2, -3])


def store_and_upload_example(dataset_example: DatasetExample,
                             args: Optional[SegmentationModelBase],
                             images_folder: Optional[Path] = None) -> None:
    """
    Stores an example input and output of the network to Nifti files.

    :param dataset_example: The dataset example, with image, label and prediction, that should be written.
    :param args: configuration information to be used for normalization. TODO: This should not be optional why is this
    assigning to example_images_folder
    :param images_folder: The folder to which the result Nifti files should be written. If args is not None,
    the args.example_images_folder is used instead.
    """

    folder = Path("") if images_folder is None else images_folder
    if args is not None:
        folder = args.example_images_folder
    if folder != "" and not os.path.exists(folder):
        os.mkdir(folder)

    def create_file_name(suffix: str) -> str:
        fn = "p" + str(dataset_example.patient_id) + "_e_" + str(dataset_example.epoch) + "_" + suffix + ".nii.gz"
        fn = os.path.join(folder, fn)
        return fn

    io_util.store_image_as_short_nifti(image=dataset_example.image,
                                       header=dataset_example.header,
                                       file_name=create_file_name(suffix="image"),
                                       args=args)

    # merge multiple binary masks (one per class) into a single multi-label map image
    labels = image_util.merge_masks(dataset_example.labels)
    io_util.store_as_ubyte_nifti(image=labels,
                                 header=dataset_example.header,
                                 file_name=create_file_name(suffix="label"))
    io_util.store_as_ubyte_nifti(image=dataset_example.prediction,
                                 header=dataset_example.header,
                                 file_name=create_file_name(suffix="prediction"))


def add_label_stats_to_dataframe(input_dataframe: pd.DataFrame,
                                 dataset_root_directory: Path,
                                 target_label_names: List[str]) -> pd.DataFrame:
    """
    Loops through all available subject IDs, generates ground-truth label statistics and updates input dataframe
    with the computed stats by adding new columns. In particular, it checks the overlapping regions between
    different structures and volume of labels.

    :param input_dataframe: Input Pandas dataframe object containing subjectIds and label names
    :param dataset_root_directory: Path to dataset root directory
    :param target_label_names: A list of label names that are used in label stat computations
    """
    dataset_sources = load_dataset_sources(input_dataframe,
                                           local_dataset_root_folder=dataset_root_directory,
                                           image_channels=["ct"],
                                           ground_truth_channels=target_label_names,
                                           mask_channel=None)

    # Iterate over subjects and check overlapping labels
    for subject_id in [*dataset_sources.keys()]:
        labels = io_util.load_labels_from_dataset_source(dataset_sources[subject_id])
        overlap_stats = metrics_util.get_label_overlap_stats(labels=labels[1:, ...],
                                                             label_names=target_label_names)

        header = io_util.load_nifti_image(dataset_sources[subject_id].ground_truth_channels[0]).header
        volume_stats = metrics_util.get_label_volume(labels=labels[1:, ...],
                                                     label_names=target_label_names,
                                                     label_spacing=header.spacing)

        # Log the extracted label statistics
        for col_name, col_stats in zip(("LabelOverlap", "LabelVolume (mL)"), (overlap_stats, volume_stats)):
            input_dataframe.loc[input_dataframe.subject == subject_id, col_name] = \
                input_dataframe.loc[input_dataframe.subject == subject_id, "channel"].map(col_stats)

    return input_dataframe
