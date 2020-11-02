#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
import torch

from InnerEye.Common import common_util
from InnerEye.Common.type_annotations import PathOrString, T, TupleFloat3
from InnerEye.ML.utils import ml_util
from InnerEye.ML.utils.csv_util import CSV_INSTITUTION_HEADER, CSV_SERIES_HEADER, CSV_SUBJECT_HEADER, \
    CSV_TAGS_HEADER
from InnerEye.ML.utils.image_util import ImageHeader

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
SAMPLE_METADATA_FIELD = "metadata"


@dataclass
class PatientMetadata:
    """
    Patient metadata
    """
    patient_id: str
    image_header: Optional[ImageHeader] = None
    institution: Optional[str] = None
    series: Optional[str] = None
    tags_str: Optional[str] = None

    @staticmethod
    def from_dataframe(dataframe: pd.DataFrame, patient_id: str) -> PatientMetadata:
        """
        Extracts the patient metadata columns from a dataframe that represents a full dataset.
        For each of the columns "seriesId", "instituionId" and "tags", the distinct values for the given patient are
        computed. If there is exactly 1 distinct value, that is returned as the respective patient metadata. If there is
        more than 1 distinct value, the metadata column is set to None.
        :param dataframe: The dataset to read from.
        :param patient_id: The ID of the patient for which the metadata should be extracted.
        :return: An instance of PatientMetadata for the given patient_id
        """
        rows = dataframe.loc[dataframe[CSV_SUBJECT_HEADER] == patient_id]
        if len(rows) == 0:
            raise ValueError(f"There is no row for patient ID {patient_id} (expected in column {CSV_SUBJECT_HEADER}")
        actual_columns = set(rows)

        def get_single_value(column: str) -> Optional[str]:
            if column in actual_columns:
                values = rows[column].unique()
                if len(values) == 1:
                    return str(values[0])
            return None

        # Tags string is enclosed in brackets, separated by semicolon. Just strip off the brackets, but don't split.
        tags = get_single_value(CSV_TAGS_HEADER)
        if tags is not None:
            tags = tags.lstrip("[").rstrip("]")
        return PatientMetadata(
            patient_id=patient_id,
            institution=get_single_value(CSV_INSTITUTION_HEADER),
            series=get_single_value(CSV_SERIES_HEADER),
            tags_str=tags
        )


@dataclass
class GeneralSampleMetadata:
    """
    A very generic class to store information about a sample inside of a dataset.
    Each sample has a string identifier, and a dictionary for attributes.
    """
    id: str
    props: Dict[str, Any] = field(default_factory=dict)
    sequence_position: int = field(default=0)

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)

    def __getitem__(self, item: str) -> Any:
        """
        Gets the metadata entry for the given key.
        """
        return self.props[item]


@dataclass(frozen=True)
class SampleBase:
    """
    All flavours of dataset samples should inherit from this class.
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls: Type[T], sample: Dict[str, Any]) -> T:
        """
        Create an instance of the sample class, based on the provided sample dictionary
        :param sample: dictionary of arguments
        :return:
        """
        return cls(**sample)  # type: ignore

    def clone_with_overrides(self: T, **overrides: Any) -> T:
        """
        Create a clone of the current sample, with the provided overrides to replace the
        existing properties if they exist.
        :param overrides:
        :return:
        """
        return type(self)(**{**vars(self), **(overrides if overrides else {})})  # type: ignore

    def get_dict(self) -> Dict[str, Any]:
        """
        Get the current sample as a dictionary of property names and their values.
        :return:
        """
        return vars(self)


@dataclass(frozen=True)
class SegmentationSampleBase(SampleBase):
    """
    A base class for all samples for segmentation models.
    """
    metadata: PatientMetadata


@dataclass(frozen=True)
class PatientDatasetSource(SegmentationSampleBase):
    """
    Dataset source locations for channels associated with a given patient in a particular dataset.
    """
    image_channels: List[PathOrString]
    ground_truth_channels: List[PathOrString]
    mask_channel: Optional[PathOrString]

    def __post_init__(self) -> None:
        # make sure all properties are populated
        common_util.check_properties_are_not_none(self, ignore=["mask_channel"])

        if not self.image_channels:
            raise ValueError("image_channels cannot be empty")
        if not self.ground_truth_channels:
            raise ValueError("ground_truth_channels cannot be empty")


@dataclass(frozen=True)
class Sample(SegmentationSampleBase):
    """
    Instance of a dataset sample that contains full 3D images, and is compatible with PyTorch data loader.
    """
    # (Batches if from data loader) x Channels x Z x Y x X
    image: Union[np.ndarray, torch.Tensor]
    # (Batches if from data loader) x Z x Y x X
    mask: Union[np.ndarray, torch.Tensor]
    # (Batches if from data loader) x Classes x Z X Y x X
    labels: Union[np.ndarray, torch.Tensor]

    def __post_init__(self) -> None:
        # make sure all properties are populated
        common_util.check_properties_are_not_none(self)

        ml_util.check_size_matches(arg1=self.image, arg2=self.mask,
                                   matching_dimensions=self._get_matching_dimensions())

        ml_util.check_size_matches(arg1=self.image, arg2=self.labels,
                                   matching_dimensions=self._get_matching_dimensions())

    @property
    def patient_id(self) -> int:
        assert isinstance(self.metadata, PatientMetadata)  # GeneralSampleMetadata has no patient_id
        return int(self.metadata.patient_id)

    @property
    def image_spacing(self) -> TupleFloat3:
        # Hdf5PatientMetadata and GeneralSampleMetadata have no spacing
        assert isinstance(self.metadata, PatientMetadata)
        if self.metadata.image_header is None:
            raise ValueError("metadata.image_spacing cannot be None")
        return self.metadata.image_header.spacing

    def _get_matching_dimensions(self) -> List[int]:
        # adjust the dimensions as there will be a batch dimension if this is loaded by a data loader
        matching_dimensions = [-1, -2, -3]
        return [0] + matching_dimensions if self._is_batched() else matching_dimensions

    def _is_batched(self) -> bool:
        """ Signifies the sample has an added batch dimension"""
        return len(self.image.shape) == 5


@dataclass(frozen=True)
class CroppedSample(Sample):
    """
    Instance of a dataset sample (compatible with PyTorch data loader)
    used for training that contains (possibly) cropped images
    as well as the center crops for the mask and the labels.
    """
    # (Batches if from data loader) x Z x Y x X
    mask_center_crop: Union[torch.Tensor, np.ndarray]
    # (Batches if from data loader) x Classes x Z X Y x X
    labels_center_crop: Union[torch.Tensor, np.ndarray]
    # The indices of the crop center point in the original image. Size: Batches x 3
    center_indices: Union[torch.Tensor, np.ndarray]

    def __post_init__(self) -> None:
        # make sure all properties are populated
        common_util.check_properties_are_not_none(self)

        # ensure the center crops for the labels and mask are compatible with each other
        ml_util.check_size_matches(arg1=self.mask_center_crop,
                                   arg2=self.labels_center_crop,
                                   matching_dimensions=self._get_matching_dimensions())
