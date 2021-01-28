#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

import h5py
import numpy as np

from InnerEye.ML.utils.image_util import ImageDataType

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


class HDF5Field(Enum):
    PATIENT_ID = "id"
    DATE = "acquisition_date"
    VOLUME = "volume"
    SEGMENTATION = "segmentation"


class HDF5ImageDataType(Enum):
    """
    Data type of medical image data (e.g. masks and labels)
    Segmentation label maps (LABEL) are one-hot encoded.
    """
    IMAGE = ImageDataType.IMAGE.value
    SEGMENTATION = ImageDataType.SEGMENTATION.value
    MASK = ImageDataType.MASK.value
    THICKNESS = np.float32
    VESSELS = np.float32
    QUANTITY = np.float64


T = TypeVar('T', bound='HDF5Object')


class HDF5Object:
    """
    An HDF5 file. Each of volume (images), segmentation (labels), acquisition date and patient ID must be provided
    """

    def __init__(self,
                 patient_id: str,
                 volume: np.ndarray,
                 acquisition_date: Union[str, datetime],
                 segmentation: Optional[np.ndarray]) -> None:
        """

        :param patient_id: The id of the patient
        :param volume: the image for this patient
        :param acquisition_date: (str or datetime)
        :param segmentation: the segmentation maps for the volume
        """
        self.patient_id = patient_id
        self.volume = volume
        self.segmentation = segmentation
        parsed_date: Optional[datetime]
        if isinstance(acquisition_date, datetime):
            parsed_date = acquisition_date
        else:
            parsed_date = HDF5Object.parse_acquisition_date(acquisition_date)
            if not parsed_date:
                raise ValueError(
                    f"Stored acquisition date is not ISO601 format {DATE_FORMAT} - found {acquisition_date}")
        self.acquisition_date = parsed_date

    @staticmethod
    def parse_acquisition_date(date: str) -> Optional[datetime]:
        """
        Converts a string representing a date to a datetime object
        :param date: string representing a date
        :return: converted date, None if the string is invalid for
        date conversion.
        """
        try:
            return datetime.strptime(date, DATE_FORMAT)
        except:
            return None

    @staticmethod
    def _hdf5_data_path(data_field: HDF5Field) -> str:
        root_path = "/"
        return root_path + data_field.value

    @staticmethod
    def _load_image(hdf5_data: h5py.File, data_field: HDF5Field) -> np.ndarray:
        """
        Load the volume from the HDF5 file.
        :param hdf5_data: path to the hdf5 file
        :param data_field: field of the hdf5 file containing the data
        :return: image as numpy array
        """
        img = hdf5_data[HDF5Object._hdf5_data_path(data_field)][()]  # N x C x H x W
        # ensure a 4D image is loaded
        if img.ndim != 4:
            raise ValueError(f"The loaded image should be 4D (image.shape: {img.shape})")
        n_channels = img.shape[1]
        if n_channels != 1:
            raise ValueError(f"Expected number of channels to be 1 but instead found {n_channels}")
        # squeeze channels dim (N == 1) - return N x H x W
        return np.squeeze(img, axis=1)

    @classmethod
    def from_file(cls: Type[T], hdf5_path: Path, load_segmentation: bool) -> T:
        """
        Load HDF5 object from file

        :param hdf5_path: Path to an HDF5 file
        :param load_segmentation: If True it loads segmentation (if present on the same file as the image).
        :return: HDF5 object
        """
        hdf5_data = h5py.File(str(hdf5_path), 'r')
        expected_keys = set([k.value for k in HDF5Field])
        act_keys = list(hdf5_data.keys())
        if not expected_keys.issubset(act_keys):
            raise ValueError(f"HDF5 group should at least have the datasets: {expected_keys} but found {act_keys}")

        patient_id = hdf5_data[cls._hdf5_data_path(HDF5Field.PATIENT_ID)][()]
        volume = cls._load_image(hdf5_data, HDF5Field.VOLUME)
        segmentation = cls._load_image(hdf5_data, HDF5Field.SEGMENTATION) if load_segmentation else None
        acquisition_date = hdf5_data[cls._hdf5_data_path(HDF5Field.DATE)][()]
        return cls(patient_id=patient_id,
                   volume=volume,
                   segmentation=segmentation,
                   acquisition_date=acquisition_date)
