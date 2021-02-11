#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List
import numpy as np
import torch
from torch.nn.parameter import Parameter

from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.config import equally_weighted_classes, SegmentationModelBase
from InnerEye.ML.models.architectures.base_model import BaseSegmentationModel
from InnerEye.ML.utils import image_util


# Test fill holes.
FillHoles: List[bool] = [
    True, True, True, True,
    False, False, True, True,
    True, True, False, True,
    True, True, True, False,
    True, False, True, True,
    False, True
]


def convert_hex_to_rgb_colour(colour: str) -> TupleInt3:
    """
    Utility to convert hex strings to RGB triples.

    :param colour: Colour formatted as a hex string.
    :return: RGB colour as a TupleInt3.
    """
    red = int(colour[0:2], 16)
    green = int(colour[2:4], 16)
    blue = int(colour[4:6], 16)
    return (red, green, blue)


# Test structure colors.
StructureColors: List[TupleInt3] = [convert_hex_to_rgb_colour(colour) for colour in [
    "FF0001", "FF0002", "FF0003", "FF0004",
    "FF0101", "FF0102", "FF0103", "FF0103",
    "FF0201", "FF02FF", "FF0203", "FF0204",
    "FF0301", "FF0302", "01FF03", "FF0304",
    "FF0401", "00FFFF", "FF0403", "FF0404",
    "FF0501", "FF0502"
]]

# Test structure names.
StructureNames: List[str] = [
    "External", "parotid_l", "parotid_r", "smg_l",
    "smg_r", "spinal_cord", "brainstem", "globe_l",
    "Globe_r", "mandible", "spc_muscle", "mpc_muscle",
    "Cochlea_l", "cochlea_r", "lens_l", "lens_r",
    "optic_chiasm", "optic_nerve_l", "optic_nerve_r", "pituitary_gland",
    "lacrimal_gland_l", "lacrimal_gland_r"
]


PassThroughCount = 5
PassThroughStructureNames = StructureNames[0:PassThroughCount]
PassThroughStructureColors = StructureColors[0:PassThroughCount]
PassThroughFillHoles = FillHoles[0:PassThroughCount]


class PassThroughModel(SegmentationModelBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            should_validate=False,
            random_seed=42,
            # architecture="Basic",
            local_dataset=full_ml_test_data_path(),
            crop_size=(157, 512, 512),
            # This speeds up loading dramatically. Multi-process data loading is tested via BasicModel2Epochs
            num_dataload_workers=0,
            # Disable monitoring so that we can use VS Code remote debugging
            monitoring_interval_seconds=0,
            image_channels=list(map(str, range(2))),
            ground_truth_ids=PassThroughStructureNames,
            ground_truth_ids_display_names=PassThroughStructureNames,
            colours=PassThroughStructureColors,
            fill_holes=PassThroughFillHoles,
            # mask_id="mask",
            dataset_expected_spacing_xyz=(1.269531011581421, 1.269531011581421, 2.5),
            inference_batch_size=1,
            class_weights=equally_weighted_classes(PassThroughStructureNames),
            feature_channels=[1]
        )
        self.add_and_validate(kwargs)

    def create_model(self) -> torch.nn.Module:
        return PyTorchPassthroughModel(self.number_of_image_channels, self.number_of_classes)


class PyTorchPassthroughModel(BaseSegmentationModel):
    """
    Defines a model that returns a center crop of its input tensor. The center crop is defined by
    shrinking the image dimensions by a given amount, on either size of each axis.
    For example, if shrink_by is (0,1,5), the center crop is the input size in the first dimension unchanged,
    reduced by 2 in the second dimension, and reduced by 10 in the third.
    """

    def __init__(self, input_channels: int, number_of_classes: int):
        super().__init__(input_channels=input_channels, name='PassthroughModel')
        # Create a fake parameter so that we can instantiate an optimizer easily
        self.foo = Parameter(requires_grad=True)
        self.number_of_classes = number_of_classes

    def forward(self, patches: torch.Tensor) -> torch.Tensor:  # type: ignore
        # simulate models where only the center of the patch is returned
        image_shape = patches.shape[2:]

        output_size = (image_shape[0], image_shape[1], image_shape[2])
        predictions = torch.zeros((patches.shape[0], self.number_of_classes) + output_size)
        for i, patch in enumerate(patches):
            for j, channel in enumerate(patch):
                predictions[i, j] = image_util.get_center_crop(image=channel, crop_shape=output_size)

        return predictions

    def get_all_child_layers(self) -> List[torch.nn.Module]:
        return list()


def make_distance_range(length: int) -> np.array:
    """
    Create a numpy array of ints of shape (length,) where each item is the distance from the centre.

    If length is odd, then let hl=(length-1)/2, then the result is:
    [hl, hl-1,..., 1, 0, 1, ... hl-1, hl]
    If length is even, then let hl=(length/2)-1, then the result is:
    [hl, hl-1,..., 1, 0, 0, 1, ... hl-1, hl]
    More concretely:
    For length=7, then the result is [3, 2, 1, 0, 1, 2, 3]
    For length=8, then the result is [3, 2, 1, 0, 0, 1, 2, 3]

    :param length: Size of array to return.
    :return: Array of distances from the centre
    """
    return abs(np.arange(1 - length, length + 1, 2)) // 2


def make_stroke_rectangle(dim0: int, dim1: int, half_side: int) -> np.array:
    """
    Create a stroke rectangle within a rectangle.

    Create a numpy array of shape (dim0, dim1), that is 0 except for an unfilled
    rectangle of 1s centred about the centre of the array.

    :param dim0: Target array dim0.
    :param dim1: Target array dim1.
    :param half_side: Rough rectangle half side length.
    :return: np.array mostly 0s apart from the path of a rectangle.
    """
    x1 = make_distance_range(dim0)
    x2 = make_distance_range(dim1)
    X1, X2 = np.meshgrid(x1, x2, sparse=False, indexing='ij')
    return ((X1 == half_side - 1) & (X2 < half_side)
            | (X1 < half_side) & (X2 == half_side - 1)) * 1


def make_fill_rectangle(dim0: int, dim1: int, half_side: int, invert: bool) -> np.array:
    """
    Create a filled rectangle within a rectangle.

    Create a numpy array of shape (dim0, dim1) that is background except for a filled
    foreground rectangle centred about the centre of the array.
    If dim0 is odd then the length in axis 0 will be 2*half_side - 1, otherwise it will be
        length 2*half_side.
    Similarly for dim1.

    :param dim0: Target array dim0.
    :param dim1: Target array dim1.
    :param half_side: Rough rectangle half side length.
    :param invert: If False then background is 0, foreground 1. If True then v.v.
    :return: np.array mostly background apart from the foreground rectangle.
    """
    x1 = make_distance_range(dim0)
    x2 = make_distance_range(dim1)
    X1, X2 = np.meshgrid(x1, x2, sparse=False, indexing='ij')
    grid = ((X1 < half_side) & (X2 < half_side)) * 1

    return grid if not invert else 1 - grid


def make_nesting_rectangles(dim0: int, dim1: int, num_features: int) -> np.array:
    """
    Create a np.array of shape (num_features, dim0, dim1) of nesting rectangles.

    The first slice is intended to be a background, the remaining slices are
    consecutively smaller rectanges, none overlapping.

    :param dim0: Target array dim0.
    :param dim1: Target array dim1.
    :param num_features: Number of rectangles.
    :return: np.array of background then a set of rectangles.
    """
    nesting = np.empty((num_features, dim0, dim1), dtype=np.int64)
    nesting[0::] = make_fill_rectangle(dim0, dim1, num_features - 1, True)

    for feature in range(1, num_features):
        nesting[feature::] = make_stroke_rectangle(dim0, dim1, num_features - feature)

    return nesting
