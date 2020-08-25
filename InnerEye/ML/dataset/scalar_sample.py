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
from InnerEye.Common.type_annotations import TupleInt2Or3
from InnerEye.ML.dataset.sample import GeneralSampleMetadata, SampleBase
from InnerEye.ML.utils.io_util import load_images_and_stack
from InnerEye.ML.utils.ml_util import is_tensor_nan_or_inf
from InnerEye.ML.scalar_config import ImageDimension


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
        ie: none of the elements in the tensors are either Not a Number or Infinity.
        """
        return not (is_tensor_nan_or_inf(self.numerical_non_image_features)
                    or is_tensor_nan_or_inf(self.categorical_non_image_features))

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


@dataclass(frozen=True)
class ScalarDataSource(ScalarItemBase):
    channel_files: List[Optional[str]]

    def load_images(self,
                    root_path: Optional[Path],
                    file_mapping: Optional[Dict[str, Path]],
                    load_segmentation: bool,
                    center_crop_size: Optional[TupleInt2Or3],
                    image_size: Optional[TupleInt2Or3],
                    image_dimension: ImageDimension
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
        :param image_dimension: Indicates if the input image is 2D or 3D
        :return: An instance of ClassificationItem, with the same label and numerical_non_image_features fields,
        and all images loaded.
        """
        full_channel_files: List[Path] = []
        for f in self.channel_files:
            if f is None:
                raise ValueError(f"When loading images, channel_files should no longer contain None entries.")
            elif file_mapping:
                if f in file_mapping:
                    full_channel_files.append(file_mapping[str(f)])
                else:
                    raise ValueError(f"File mapping does not contain an entry for {f}")
            elif root_path:
                full_channel_files.append(root_path / f)
            else:
                raise ValueError("One of the arguments 'file_mapping' or 'root_path' must be given.")

        imaging_data = load_images_and_stack(files=full_channel_files,
                                             load_segmentation=load_segmentation,
                                             image_dimension=image_dimension,
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


@dataclass(frozen=True)
class SequenceDataSource(ScalarDataSource):
    def labels_valid(self) -> bool:
        # for sequence data sources we do not require all sources to
        # have a label associated with them
        return True
