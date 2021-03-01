#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from InnerEye.Common.common_util import check_properties_are_not_none
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.dataset.sample import GeneralSampleMetadata, SampleBase
from InnerEye.ML.utils.io_util import load_images_and_stack
from InnerEye.ML.utils.ml_util import is_tensor_nan, is_tensor_nan_or_inf


@dataclass(frozen=True)
class ScalarItemBase(SampleBase):
    """
    This class contains all information that are input to an image classification model, apart from the image itself.
    Labels and numerical_non_image_features can be matrices of arbitrary size.
    """
    metadata: GeneralSampleMetadata
    # A [n, m] tensor that contains the label(s) for this image sample.
    label: torch.Tensor
    # A [q,] size tensor that contains non-image features.
    numerical_non_image_features: torch.Tensor
    # A [r,] size tensor that contains one-hot encoded categorical non-image features.
    categorical_non_image_features: torch.Tensor

    def __post_init__(self) -> None:
        check_properties_are_not_none(self)

    @property
    def id(self) -> str:
        """
        Gets the identifier of the present object from metadata.
        :return:
        """
        return self.metadata.id  # type: ignore

    @property
    def props(self) -> Dict[str, Any]:
        """
        Gets the general metadata dictionary for the present object.
        :return:
        """
        return self.metadata.props  # type: ignore

    def is_valid(self) -> bool:
        """
        Return True if numerical_non_image_features, categorical_non_image_features and label are valid
        ie: none of the elements in the tensors are either Not a Number or Infinity.
        """
        return self.features_valid() and self.labels_valid()

    def features_valid(self) -> bool:
        """
        Return True if numerical_non_image_features and categorical_non_image_features are valid
        ie: none of the elements in the tensors are Not a Number.
        """
        return not (is_tensor_nan(self.numerical_non_image_features)
                    or is_tensor_nan(self.categorical_non_image_features))

    def labels_valid(self) -> bool:
        """
        Checks to make sure label tensor is valid ie: none of the elements in the tensors
        are either Not a Number or Infinity.
        """
        return not is_tensor_nan_or_inf(self.label)


@dataclass(frozen=True)
class ScalarItem(ScalarItemBase):
    """
    This class contains all information that are input to an image classification model, including the images itself.
    Labels and numerical_non_image_features can be matrices of arbitrary size.
    """
    images: torch.Tensor  # (channels, Z, Y, X)
    segmentations: Optional[torch.Tensor]  # (channels, Z, Y, X)

    def get_all_non_imaging_features(self) -> torch.Tensor:
        """
        Returns a concatenation of the numerical_non_image_features and categorical_non_image_features
        """
        _dim = 0 if self.numerical_non_image_features.ndimension() == 1 else 1
        return torch.cat([self.numerical_non_image_features, self.categorical_non_image_features], dim=_dim)

    def to_device(self, device: Any) -> ScalarItem:
        """
        Creates a copy of the present object where all tensors live on the given CUDA device.
        The metadata field is left unchanged.
        :param device: The CUDA or GPU device to move to.
        :return: A new `ScalarItem` with all tensors on the chosen device.
        """
        return ScalarItem(
            metadata=self.metadata,
            label=self.label.to(device),
            categorical_non_image_features=self.categorical_non_image_features.to(device),
            numerical_non_image_features=self.numerical_non_image_features.to(device),
            images=self.images.to(device),
            segmentations=None if self.segmentations is None else self.segmentations.to(device)
        )


@dataclass(frozen=True)
class ScalarDataSource(ScalarItemBase):
    channel_files: List[Optional[str]]

    def load_images(self,
                    root_path: Optional[Path],
                    file_mapping: Optional[Dict[str, Path]],
                    load_segmentation: bool,
                    center_crop_size: Optional[TupleInt3],
                    image_size: Optional[TupleInt3]
                    ) -> ScalarItem:
        """
        Loads all the images that are specified in the channel_files field, and stacks them into a tensor
        along the first dimension. The channel_files field must either contain the image file path, relative to the
        root_path argument, or it must contain a file name stem only (without extension). In this case, the actual
        mapping from file name stem to full path is expected in the file_mapping argument.
        Either of 'root_path' or 'file_mapping' must be provided.
        :param root_path: The root path where all channel files for images are expected. This is ignored if
        file_mapping is given.
        :param file_mapping: A mapping from a file name stem (without extension) to its full path.
        :param load_segmentation: If True it loads segmentation if present on the same file as the image.
        :param center_crop_size: If supplied, all loaded images will be cropped to the size given here. The crop will
        be taken from the center of the image.
        :param image_size: If given, all loaded images will be reshaped to the size given here, prior to the
        center crop.
        :return: An instance of ClassificationItem, with the same label and numerical_non_image_features fields,
        and all images loaded.
        """
        full_channel_files = self.get_all_image_filepaths(root_path=root_path,
                                                          file_mapping=file_mapping)

        imaging_data = load_images_and_stack(files=full_channel_files,
                                             load_segmentation=load_segmentation,
                                             center_crop_size=center_crop_size,
                                             image_size=image_size)
        return ScalarItem(
            label=self.label,
            numerical_non_image_features=self.numerical_non_image_features,
            categorical_non_image_features=self.categorical_non_image_features,
            # HDF5 files can contain float16 images. Convert to float32. AMP may later convert back to float16.
            images=imaging_data.images.float(),
            segmentations=imaging_data.segmentations,
            metadata=self.metadata
        )

    def is_valid(self) -> bool:
        """
        Checks if all file paths and non-image features are present in the object. All image channel files must
        be not None, and none of the non imaging features may be NaN or infinity.
        :return: True if channel files is a list with not-None entries, and all non imaging features are finite
        floating point numbers.
        """
        return self.files_valid() and super().is_valid()

    def files_valid(self) -> bool:
        return not any(f is None for f in self.channel_files)

    def get_all_image_filepaths(self,
                                root_path: Optional[Path],
                                file_mapping: Optional[Dict[str, Path]]) -> List[Path]:
        """
        Get a list of image paths for the object. Either root_path or file_mapping must be specified.
        :param root_path: The root path where all channel files for images are expected. This is ignored if
        file_mapping is given.
        :param file_mapping: A mapping from a file name stem (without extension) to its full path.
        """
        full_channel_files: List[Path] = []
        for f in self.channel_files:
            full_channel_files.append(self.get_full_image_filepath(f, root_path, file_mapping))

        return full_channel_files

    @staticmethod
    def get_full_image_filepath(file: str,
                                root_path: Optional[Path],
                                file_mapping: Optional[Dict[str, Path]]) -> Path:
        """
        Get the full path of an image file given the path relative to the dataset folder and one of
        root_path or file_mapping.
        :param file: Image filepath relative to the dataset folder
        :param root_path: The root path where all channel files for images are expected. This is ignored if
        file_mapping is given.
        :param file_mapping: A mapping from a file name stem (without extension) to its full path.
        """
        if file is None:
            raise ValueError("When loading images, channel_files should no longer contain None entries.")
        elif file_mapping:
            if file in file_mapping:
                return file_mapping[str(file)]
            else:
                raise ValueError(f"File mapping does not contain an entry for {file}")
        elif root_path:
            return root_path / file
        else:
            raise ValueError("One of the arguments 'file_mapping' or 'root_path' must be given.")


@dataclass(frozen=True)
class SequenceDataSource(ScalarDataSource):
    def labels_valid(self) -> bool:
        # for sequence data sources we do not require all sources to
        # have a label associated with them
        return True
