#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
from typing import Any, List
import numpy as np
import pandas as pd
import torch
from torch.nn.parameter import Parameter

from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.common import DATASET_CSV_FILE_NAME
from InnerEye.ML.config import equally_weighted_classes, get_center_size, SegmentationModelBase
from InnerEye.ML.models.architectures.base_model import BaseSegmentationModel
from InnerEye.ML.utils.csv_util import CSV_SUBJECT_HEADER, CSV_PATH_HEADER, CSV_CHANNEL_HEADER
from InnerEye.ML.utils.io_util import reverse_tuple_float3, store_as_nifti, ImageHeader
from InnerEye.ML.utils.model_metadata_util import generate_random_colours_list
from InnerEye.ML.utils.split_dataset import DatasetSplits


RANDOM_COLOUR_GENERATOR = random.Random(0)
RECTANGLE_STROKE_THICKNESS = 3


class PassThroughModel(SegmentationModelBase):
    """
    Dummy model that returns a fixed segmentation, explained in make_nesting_rectangles.
    """
    def __init__(self, **kwargs: Any) -> None:
        random_seed = 42
        local_dataset = full_ml_test_data_path("passthrough_data")
        local_dataset.mkdir(exist_ok=True)
        crop_size = (64, 192, 160)
        dataset_expected_spacing_xyz = (1.269531011581421, 1.269531011581421, 2.5)
        # Need at least 3 subjects, 1 each for train, validate, test.
        number_of_subjects = 3
        self.subjects = list(map(str, range(1, number_of_subjects + 1)))
        number_of_image_channels = 1
        image_channels = [f"channel_{i}" for i in range(1, number_of_image_channels + 1)]
        number_of_classes = 5
        class_names = [f"structure_{i}" for i in range(1, number_of_classes + 1)]

        super().__init__(
            should_validate=False,
            random_seed=random_seed,
            local_dataset=local_dataset,
            crop_size=crop_size,
            # This speeds up loading dramatically. Multi-process data loading is tested via BasicModel2Epochs
            num_dataload_workers=0,
            # Disable monitoring so that we can use VS Code remote debugging
            monitoring_interval_seconds=0,
            image_channels=image_channels,
            ground_truth_ids=class_names,
            ground_truth_ids_display_names=class_names,
            colours=generate_random_colours_list(RANDOM_COLOUR_GENERATOR, number_of_classes),
            fill_holes=[False] * number_of_classes,
            dataset_expected_spacing_xyz=dataset_expected_spacing_xyz,
            inference_batch_size=1,
            class_weights=equally_weighted_classes(class_names),
            feature_channels=[1],
            start_epoch=0,
            num_epochs=1
        )
        self.add_and_validate(kwargs)
        self.create_passthrough_dataset()

    def create_model(self) -> torch.nn.Module:
        return PyTorchPassthroughModel(self)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_subject_ids(
            df=dataset_df,
            train_ids=[self.subjects[0]],
            test_ids=[self.subjects[1]],
            val_ids=[self.subjects[2]]
        )

    def create_passthrough_dataset(self) -> None:
        """
        Create all the files expected for training.

        Training expects a data folder containing:
        1) A CSV file, DATASET_CSV_FILE_NAME, describing the data set.
        2) A set of image files, one for each combination of image channel/subject.
        3) A set of binary mask image files, for ground truths. One for each class_name.
        """
        dataset_csv_file_path = self.local_dataset / DATASET_CSV_FILE_NAME
        # Use the same image file for each subject/image channel.
        image_file_name = "dummy_image.nii.gz"
        # Need a different file for each class.
        ground_truth_file_names = {class_name: f"dummy_ground_truth_{class_name}.nii.gz"
                                   for class_name in self.ground_truth_ids}

        with dataset_csv_file_path.open('w') as f:
            f.write(f"{CSV_SUBJECT_HEADER},{CSV_PATH_HEADER},{CSV_CHANNEL_HEADER}\n")
            for subject in self.subjects:
                for image_channel in self.image_channels:
                    f.write(f"{subject},{image_file_name},{image_channel}\n")
                for class_name in self.ground_truth_ids:
                    f.write(f"{subject},{ground_truth_file_names[class_name]},{class_name}\n")

        # Create a shared, random, nifti image file.
        image = np.random.random_sample(self.crop_size)
        spacingzyx = reverse_tuple_float3(self.dataset_expected_spacing_xyz)
        image_file_path = self.local_dataset / image_file_name
        header = ImageHeader(origin=(1, 1, 1), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), spacing=spacingzyx)
        store_as_nifti(image, header, image_file_path, np.float32)

        # For ground truths, use the same data as in make_nesting_rectangles.
        ground_truths = make_nesting_rectangles(self.number_of_classes, self.crop_size[1], self.crop_size[2],
                                                RECTANGLE_STROKE_THICKNESS)
        for i, class_name in enumerate(self.ground_truth_ids):
            ground_truth_file_path = self.local_dataset / ground_truth_file_names[class_name]
            # Skip the background slice.
            ground_truth = ground_truths[i + 1].reshape(1, self.crop_size[1], self.crop_size[2])
            # Extrude to image_size[0]
            project_ground_truth = np.broadcast_to(ground_truth, self.crop_size)
            store_as_nifti(project_ground_truth, header, ground_truth_file_path, np.ubyte)


class PyTorchPassthroughModel(BaseSegmentationModel):
    """
    Defines a model that returns a fixed segmentation, explained in make_nesting_rectangles.
    """

    def __init__(self, config: SegmentationModelBase):
        """
        Creates a new instance of the class.

        :param config: Model config.
        """
        super().__init__(input_channels=config.number_of_image_channels, name='PassthroughModel')
        # Create a fake parameter so that we can instantiate an optimizer easily
        self.foo = Parameter(requires_grad=True)
        self.config = config
        # Cache the fixed segmentation.
        self.cached_patch_size = config.crop_size
        self.cached_patch = self.make_nest(config.crop_size)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Ignore the actual patches and return a fixed segmentation, explained in make_nesting_rectangles.

        The output tensor is shrunk by the amount basic_size_shrinkage in config.py.

        :param patches: Set of patches, of shape (#patches, #image_channels, Z, Y, X). Only the shape
        is used.
        :return: Fixed tensor of shape (#patches, number_of_classes, Z', Y', Z') where Z' = Z - basic_size_shrinkage,
        etc.
        """
        output_size: TupleInt3 = patches.shape[2:]
        if self.cached_patch_size == output_size:
            patch = self.cached_patch
        else:
            patch = self.make_nest(output_size)
        if patches.shape[0] == 1:
            np_predictions = patch
        else:
            np_predictions = np.broadcast_to(patch, (patches.shape[0],) + patch.shape[1:])
        return torch.tensor(np_predictions, requires_grad=True)

    def get_all_child_layers(self) -> List[torch.nn.Module]:
        return list()

    def make_nest(self, output_size: TupleInt3) -> np.ndarray:
        """
        Given a patch shaped (Z, Y, X) return a fixed segmentation shaped to (1, number_of_classes, Z', Y', X').

        :param output_size: Target output size before reduction by basic_size_shrinkage.
        :return: 5d tensor.
        """
        output_size = get_center_size(self.config.architecture, output_size)
        nest = make_nesting_rectangles(self.config.number_of_classes, output_size[1], output_size[2],
                                       RECTANGLE_STROKE_THICKNESS)
        project_nest = nest.reshape(1, self.config.number_of_classes, 1, output_size[1], output_size[2])
        return np.broadcast_to(project_nest, (1, self.config.number_of_classes) + output_size)


def make_distance_range(length: int) -> np.ndarray:
    """
    Create a numpy array of np.float32 of shape (length,) where each item is the distance from the centre.

    If length is odd, then let hl=(length-1)/2, then the result is:
    [hl, hl-1,..., 1, 0, 1, ... hl-1, hl]
    If length is even, then let hl=(length/2)-1, then the result is:
    [hl, hl-1,..., 1, 0, 0, 1, ... hl-1, hl]
    More concretely:
    For length=7, then the result is [3., 2., 1., 0., 1., 2., 3.]
    For length=8, then the result is [3., 2., 1., 0., 0., 1., 2., 3.]

    :param length: Size of array to return.
    :return: Array of distances from the centre.
    """
    return abs(np.arange(1 - length, length + 1, 2, dtype=np.float32)) // 2


def make_stroke_rectangle(dim0: int, dim1: int, half_side: int, thickness: int) -> np.ndarray:
    """
    Create a stroked rectangle within a rectangle.

    Create a numpy array of np.float32 of shape (dim0, dim1), that is 0. except for an unfilled
    rectangle of 1.'s centred about the centre of the array.

    For example, with dim0=5, dim1=8, half_side=2, thickness=1, this will produce this array:
    array([[0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 1., 1., 1., 1., 0., 0.],
           [0., 0., 1., 0., 0., 1., 0., 0.],
           [0., 0., 1., 1., 1., 1., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)

    :param dim0: Target array dim0.
    :param dim1: Target array dim1.
    :param half_side: Inner rectangle approximate half side length.
    :param thickness: Stroke thickness.
    :return: np.ndarray mostly 0s apart from the stroked path of a rectangle.
    """
    x1 = make_distance_range(dim0)
    x2 = make_distance_range(dim1)
    X1, X2 = np.meshgrid(x1, x2, sparse=False, indexing='ij')
    bool_array = (((half_side - thickness <= X1) & (X1 < half_side) & (X2 < half_side))
                  | ((X1 < half_side) & (half_side - thickness <= X2) & (X2 < half_side))) * 1
    return bool_array.astype(np.float32)


def make_fill_rectangle(dim0: int, dim1: int, half_side: int) -> np.ndarray:
    """
    Create a filled rectangle within a rectangle.

    Create a numpy array of np.float32 of shape (dim0, dim1) that is 0. except for a filled
    foreground rectangle of 1.'s centred about the centre of the array.
    If dim0 is odd then the length in axis 0 will be 2*half_side - 1, otherwise it will be
        length 2*half_side.
    Similarly for dim1.

    For example, with dim0=5, dim1=8, half_side=2, this will produce this array:
    array([[0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 1., 1., 1., 1., 0., 0.],
           [0., 0., 1., 1., 1., 1., 0., 0.],
           [0., 0., 1., 1., 1., 1., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)

    :param dim0: Target array dim0.
    :param dim1: Target array dim1.
    :param half_side: Inner rectangle approximate half side length.
    :return: np.ndarray mostly 0s apart from the filled path of a rectangle.
    """
    x1 = make_distance_range(dim0)
    x2 = make_distance_range(dim1)
    X1, X2 = np.meshgrid(x1, x2, sparse=False, indexing='ij')
    bool_array = ((X1 < half_side) & (X2 < half_side))
    return bool_array.astype(np.float32)


def make_nesting_rectangles(dim0: int, dim1: int, dim2: int, thickness: int) -> np.ndarray:
    """
    Create an np.ndarray of np.float32 of shape (dim0, dim1, dim2) of nesting rectangles.

    The first slice is intended to be a background so is an inverted filled rectangle.
    The remaining slices are consecutively larger stroked rectangles, none overlapping.

    For example, with dim0=3, dim1=5, dim2=8, half_side=2, thickness=1 this will produce an
    array of shape (3, 5, 8) where:
    slice 0 =
    array([[1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 0., 0., 0., 0., 1., 1.],
           [1., 1., 0., 0., 0., 0., 1., 1.],
           [1., 1., 0., 0., 0., 0., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.]], dtype=float32)
    slice 1 =
    array([[0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)
    slice 2 =
    array([[0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 1., 1., 1., 1., 0., 0.],
           [0., 0., 1., 0., 0., 1., 0., 0.],
           [0., 0., 1., 1., 1., 1., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)

    :param dim0: Target array dim0.
    :param dim1: Target array dim1.
    :param dim2: Target array dim2.
    :param thickness: Stroke thickness.
    :return: np.ndarray of background then a set of rectangles.
    """
    nesting = np.empty((dim0, dim1, dim2), dtype=np.float32)
    nesting[0] = 1. - make_fill_rectangle(dim1, dim2, (dim0 - 1) * thickness)
    for feature in range(1, dim0):
        nesting[feature] = make_stroke_rectangle(dim1, dim2, feature * thickness, thickness)
    return nesting
