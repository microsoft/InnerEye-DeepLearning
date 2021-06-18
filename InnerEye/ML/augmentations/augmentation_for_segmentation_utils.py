#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
from typing import List, Tuple, Union

import numpy as np
import torch
from monai.transforms import RandAffined, Compose, RandGaussianNoised, RandRotated

from InnerEye.Common.common_util import any_pairwise_larger
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.dataset.sample import Sample
from InnerEye.ML.utils.transforms import Transform3D


def random_select_patch_center(sample: Sample, class_weights: List[float] = None) -> np.ndarray:
    """
    Samples a point to use as the coordinates of the patch center. First samples one
    class among the available classes then samples a center point among the pixels of the sampled
    class.

    :param sample: A set of Image channels, ground truth labels and mask to randomly crop.
    :param class_weights: A weighting vector with values [0, 1] to influence the class the center crop
                          voxel belongs to (must sum to 1), uniform distribution assumed if none provided.
    :return numpy int array (3x1) containing patch center spatial coordinates
    """
    num_classes = sample.labels.shape[0]

    if class_weights is not None:
        if len(class_weights) != num_classes:
            raise Exception("A weight must be provided for each class, found weights:{}, expected:{}"
                            .format(len(class_weights), num_classes))
        SegmentationModelBase.validate_class_weights(class_weights)

    # If class weights are not initialised, selection is made with equal probability for all classes
    available_classes = list(range(num_classes))
    original_class_weights = class_weights
    while len(available_classes) > 0:
        selected_label_class = random.choices(population=available_classes, weights=class_weights, k=1)[0]
        # Check pixels where mask and label maps are both foreground
        indices = np.argwhere(np.logical_and(sample.labels[selected_label_class] == 1.0, sample.mask == 1))
        if not np.any(indices):
            available_classes.remove(selected_label_class)
            if class_weights is not None:
                assert original_class_weights is not None  # for mypy
                class_weights = [original_class_weights[i] for i in available_classes]
                if sum(class_weights) <= 0.0:
                    raise ValueError("Cannot sample a class: no class present in the sample has a positive weight")
        else:
            break

    # Raise an exception if non of the foreground classes are overlapping with the mask
    if len(available_classes) == 0:
        raise Exception("No non-mask voxels found, please check your mask and labels map")

    # noinspection PyUnboundLocalVariable
    choice = random.randint(0, len(indices) - 1)

    return indices[choice].astype(int)  # Numpy usually stores as floats


def slicers_for_random_crop(sample: Sample,
                            crop_size: TupleInt3,
                            class_weights: List[float] = None) -> Tuple[List[slice], np.ndarray]:
    """
    Computes array slicers that produce random crops of the given crop_size.
    The selection of the center is dependant on background probability.
    By default it does not center on background.

    :param sample: A set of Image channels, ground truth labels and mask to randomly crop.
    :param crop_size: The size of the crop expressed as a list of 3 ints, one per spatial dimension.
    :param class_weights: A weighting vector with values [0, 1] to influence the class the center crop
                          voxel belongs to (must sum to 1), uniform distribution assumed if none provided.
    :return: Tuple element 1: The slicers that convert the input image to the chosen crop. Tuple element 2: The
    indices of the center point of the crop.
    :raises ValueError: If there are shape mismatches among the arguments or if the crop size is larger than the image.
    """
    shape = sample.image.shape[1:]

    if any_pairwise_larger(crop_size, shape):
        raise ValueError("The crop_size across each dimension should be greater than zero and less than or equal "
                         "to the current value (crop_size: {}, spatial shape: {})"
                         .format(crop_size, shape))

    # Sample a center pixel location for patch extraction.
    center = random_select_patch_center(sample, class_weights)

    # Verify and fix overflow for each dimension
    left = []
    for i in range(3):
        margin_left = int(crop_size[i] / 2)
        margin_right = crop_size[i] - margin_left
        left_index = center[i] - margin_left
        right_index = center[i] + margin_right
        if right_index > shape[i]:
            left_index = left_index - (right_index - shape[i])
        if left_index < 0:
            left_index = 0
        left.append(left_index)

    return [slice(left[x], left[x] + crop_size[x]) for x in range(0, 3)], center


def random_crop(sample: Sample,
                crop_size: TupleInt3,
                class_weights: List[float] = None) -> Tuple[Sample, np.ndarray]:
    """
    Randomly crops images, mask, and labels arrays according to the crop_size argument.
    The selection of the center is dependant on background probability.
    By default it does not center on background.

    :param sample: A set of Image channels, ground truth labels and mask to randomly crop.
    :param crop_size: The size of the crop expressed as a list of 3 ints, one per spatial dimension.
    :param class_weights: A weighting vector with values [0, 1] to influence the class the center crop
                          voxel belongs to (must sum to 1), uniform distribution assumed if none provided.
    :return: Tuple item 1: The cropped images, labels, and mask. Tuple item 2: The center that was chosen for the crop,
    before shifting to be inside of the image. Tuple item 3: The slicers that convert the input image to the chosen
    crop.
    :raises ValueError: If there are shape mismatches among the arguments or if the crop size is larger than the image.
    """
    slicers, center = slicers_for_random_crop(sample, crop_size, class_weights)
    sample = Sample(
        image=sample.image[:, slicers[0], slicers[1], slicers[2]],
        labels=sample.labels[:, slicers[0], slicers[1], slicers[2]],
        mask=sample.mask[slicers[0], slicers[1], slicers[2]],
        metadata=sample.metadata
    )
    return sample, center


class BasicAugmentations(Transform3D[Sample]):
    """
    Transform3D for basic augmentations on a SegmentationSample
    """
    IMAGE = "image"
    LABELS = "labels"
    augment = Compose([
        RandRotated(
            keys=[IMAGE, LABELS],
            mode=("bilinear", "nearest"),
            range_x=30 * np.pi / 180,
            padding_mode="zeros",
            prob=0.5
        ),
        RandGaussianNoised(
            keys=[IMAGE],
            prob=0.5
        )])

    def __call__(self, sample: Sample) -> Sample:
        image, labels = self.transform(
            image=sample.image,
            labels=sample.labels)
        return sample.clone_with_overrides(
            image=image,
            labels=labels,
        )

    @staticmethod
    def transform(image: Union[np.ndarray, torch.Tensor],
                  labels: Union[np.ndarray, torch.Tensor]) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Applies 1 transform with 0.5 probability
        :param image: channels x image dimensions
        :param labels: channels x image dimensions
        :return: A tuple of image channels transformed and labels transformed
        """

        subject = {
            BasicAugmentations.IMAGE: image,
            BasicAugmentations.LABELS: labels
        }

        augmented_dict = BasicAugmentations.augment(subject)
        return augmented_dict[BasicAugmentations.IMAGE], augmented_dict[BasicAugmentations.LABELS]
