#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
from typing import Any, List
import numpy as np
import torch
from torch.nn.parameter import Parameter

from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.ML.config import equally_weighted_classes, SegmentationModelBase
from InnerEye.ML.models.architectures.base_model import BaseSegmentationModel
from InnerEye.ML.utils import image_util
from InnerEye.ML.utils.model_metadata_util import generate_random_colours_list


RANDOM_COLOUR_GENERATOR = random.Random(0)


class PassThroughModel(SegmentationModelBase):
    """
    Dummy model that returns a fixed segmentation.
    """
    def __init__(self, **kwargs: Any) -> None:
        random_seed = 42
        number_of_image_channels = 1
        number_of_classes = 5
        class_names = [f"structure_{i}" for i in range(number_of_classes)]
        
        super().__init__(
            should_validate=False,
            random_seed=random_seed,
            local_dataset=full_ml_test_data_path(),
            crop_size=(40, 55, 65),
            # This speeds up loading dramatically. Multi-process data loading is tested via BasicModel2Epochs
            num_dataload_workers=0,
            # Disable monitoring so that we can use VS Code remote debugging
            monitoring_interval_seconds=0,
            image_channels=[f"image_{i}" for i in range(number_of_image_channels)],
            ground_truth_ids=class_names,
            ground_truth_ids_display_names=class_names,
            colours=generate_random_colours_list(RANDOM_COLOUR_GENERATOR, number_of_classes),
            fill_holes=[True] * number_of_classes,
            # mask_id="mask",
            dataset_expected_spacing_xyz=(1.269531011581421, 1.269531011581421, 2.5),
            inference_batch_size=1,
            class_weights=equally_weighted_classes(class_names),
            feature_channels=[1]
        )
        self.add_and_validate(kwargs)

    def create_model(self) -> torch.nn.Module:
        return PyTorchPassthroughModel(self.number_of_image_channels, self.number_of_classes)


class PyTorchPassthroughModel(BaseSegmentationModel):
    """
    Defines a model that returns a nested set of extruded rectangles.
    """

    def __init__(self, input_channels: int, number_of_classes: int):
        super().__init__(input_channels=input_channels, name='PassthroughModel')
        # Create a fake parameter so that we can instantiate an optimizer easily
        self.foo = Parameter(requires_grad=True)
        self.number_of_classes = number_of_classes

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # simulate models where only the center of the patch is returned
        image_shape = patches.shape[2:]

        output_size = (self.number_of_classes, image_shape[0], image_shape[1], image_shape[2])
        predictions = torch.zeros(patches.shape[0], *output_size)
        for i, patch in enumerate(patches):
            nest = make_nesting_rectangles(self.number_of_classes, image_shape[1], image_shape[2])
            project_nest = nest.reshape(self.number_of_classes, 1, image_shape[1], image_shape[2])
            extrusion = np.broadcast_to(project_nest, output_size)
            predictions[i] = torch.from_numpy(extrusion)

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


def make_nesting_rectangles(num_features: int, dim0: int, dim1: int) -> np.array:
    """
    Create a np.array of shape (num_features, dim0, dim1) of nesting rectangles.

    The first slice is intended to be a background, the remaining slices are
    consecutively smaller rectanges, none overlapping.

    :param num_features: Number of rectangles.
    :param dim0: Target array dim0.
    :param dim1: Target array dim1.
    :return: np.array of background then a set of rectangles.
    """
    nesting = np.empty((num_features, dim0, dim1), dtype=np.int64)
    nesting[0::] = make_fill_rectangle(dim0, dim1, num_features - 1, True)

    for feature in range(1, num_features):
        nesting[feature::] = make_stroke_rectangle(dim0, dim1, num_features - feature)

    return nesting
