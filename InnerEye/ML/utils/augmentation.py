#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from torchvision.transforms import functional as TF

from InnerEye.Common.common_util import any_pairwise_larger
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.dataset.sample import Sample
from InnerEye.ML.dataset.scalar_sample import ScalarItem
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


class ImageTransformationBase(Transform3D):
    """
    This class is the base class to classes built to define data augmentation transformations
    for 3D image inputs.
    """

    # noinspection PyMissingConstructor
    def __init__(self,
                 is_transformation_for_segmentation_maps: bool = False,
                 use_joint_channel_transformation: bool = False):
        """
        :param is_transformation_for_segmentation_maps: if True, only use geometrical transformation suitable
        for segmentation maps. If False, additionally use color/contrast related transformation suitable for
        images.
        :param use_joint_channel_transformation: if True apply the exact same transformation for all channels of
        a given image. If False, apply a different transformation for each channel.
        """
        self.for_segmentation_input_maps = is_transformation_for_segmentation_maps
        self.use_joint_channel_transformation = use_joint_channel_transformation

    def draw_next_transform(self) -> List[Callable]:
        """
        Samples all parameters defining the transformation pipeline.
        Returns a list of operations to apply to each 2D-slice in a given
        3D volume.
        (defined by the sampled parameters).

        :return: list of transformations to apply to each B-scan.
        """
        raise NotImplementedError("The child class should implement the sampling of transforms")

    @staticmethod
    def apply_transform_on_3d_image(image: torch.Tensor, transforms: List[Callable]) -> torch.Tensor:
        """
        Apply a list of transforms sequentially to a 3D image. Each transformation is assumed to be a 2D-transform to
        be applied to each slice of the given 3D input separately.

        :param image: a 3d tensor dimension [Z, X, Y]. Transform are applied one after another
        separately for each [X, Y] slice along the Z-dimension (assumed to be the first dimension).
        :param transforms: a list of transformations to apply to each slice sequentially.
        :returns image: the transformed 3D-image
         """
        for z in range(image.shape[0]):
            pil = TF.to_pil_image(image[z])
            for transform_fn in transforms:
                pil = transform_fn(pil)
            image[z] = TF.to_tensor(pil).squeeze()
        return image

    @staticmethod
    def _toss_fair_coin() -> bool:
        """
        Simulates the toss of a fair coin.
        :returns the outcome of the toss.
        """
        return random.random() > 0.5

    @staticmethod
    def randomly_negate_level(value: Any) -> Any:
        """
        Negate the value of the input with probability 0.5
        """
        return -value if ImageTransformationBase._toss_fair_coin() else value

    @staticmethod
    def identity() -> Callable:
        """
        Identity transform.
        """
        return lambda img: img

    @staticmethod
    def rotate(angle: float) -> Callable:
        """
        Returns a function that rotates a 2D image by
        a certain angle.
        """
        return lambda img: TF.rotate(img, angle)

    @staticmethod
    def translateX(shift: float) -> Callable:
        """
        Returns a function that shifts a 2D image horizontally by
        a given shift.
        :param shift: a floating point between (-1, 1), the shift is defining as the
        proportion of the image width by which to shift.
        """
        return lambda img: TF.affine(img, 0, (shift * img.size[0], 0), 1, 0)

    @staticmethod
    def translateY(shift: float) -> Callable:
        """
        Returns a function that shifts a 2D image vertically by
        a given shift.
        :param shift: a floating point between (-1, 1), the shift is defining as the
        proportion of the image height by which to shift.
        """
        return lambda img: TF.affine(img, 0, (0, shift * img.size[1]), 1, 0)

    @staticmethod
    def horizontal_flip() -> Callable:
        """
        Returns a function that is flipping a 2D-image horizontally.
        """
        return lambda img: TF.hflip(img)

    @staticmethod
    def adjust_contrast(constrast_factor: float) -> Callable:
        """
        Returns a function that modifies the contrast of a
        2D image by a certain factor.
        :param constrast_factor: Integer > 0. 0 means black image,
        1 means no transformation, 2 means multipyling the contrast
        by two.
        """
        return lambda img: TF.adjust_contrast(img, constrast_factor)

    @staticmethod
    def adjust_brightness(brightness_factor: float) -> Callable:
        """
        Returns a function that modifies the brightness of a
        2D image by a certain factor.
        :param brightness_factor: Integer > 0. 0 means black image,
        1 means no transformation, 2 means multipyling the brightness
        by two.
        """
        return lambda img: TF.adjust_brightness(img, brightness_factor)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Main function to apply the transformation to one 3D-image.
        Assumes the same transformations have to be applied on
        each 2D-slice along the Z-axis.
        Assumes the Z axis is the first dimension.

        :param image: batch of images of size [C, Z, Y, X]
        """
        assert len(image.shape) == 4
        res = image.clone()
        if self.for_segmentation_input_maps:
            res = res.int()
        else:
            res = res.float()
            if res.max() > 1:
                raise ValueError("Image tensor should be in "
                                 "range 0-1 for conversion to PIL")

        # Sample parameters defining the transformation
        transforms = self.draw_next_transform()
        for c in range(image.shape[0]):
            res[c] = self.apply_transform_on_3d_image(res[c], transforms)
            if not self.use_joint_channel_transformation:
                # Resample transformations for the next channel
                transforms = self.draw_next_transform()
        return res.to(dtype=image.dtype)


class RandomSliceTransformation(ImageTransformationBase):
    """
    Class to apply a random set of 2D affine transformations to all
    slices of a 3D volume separately along the z-dimension.
    """

    def __init__(self,
                 probability_transformation: float = 0.8,
                 max_angle: int = 10,
                 max_x_shift: float = 0.05,
                 max_y_shift: float = 0.1,
                 max_contrast: float = 2,
                 min_constrast: float = 0,
                 max_brightness: float = 2,
                 min_brightness: float = 0,
                 **kwargs: Any) -> None:
        """

        :param probability_transformation: probability of applying the transformation pipeline.
        :param max_angle: maximum allowed angle for rotation. For each transformation
        the angle is drawn uniformly between -max_angle and max_angle.
        :param min_constrast: Minimum contrast factor to apply. 1 means no difference.
        2 means doubling the contrast. 0 means a black image. Parameter is sampled
        between min_contrast and max_contrast.
        :param max_contrast: maximum contrast factor
        :param max_brightness: Maximum brightness factor to apply. 1 means no difference.
        2 means doubling the brightness. 0 means a black image. Parameter is sampled
        between min_brightness and max_brightness.
        :param max_x_shift: maximum vertical shift in proportion of the image width
        :param max_y_shift: maximum horizontal shift in proportion of the image height.
        """
        super().__init__(**kwargs)
        self.probability_transformation = probability_transformation
        self.max_angle = max_angle
        self.max_x_shift = max_x_shift
        self.max_y_shift = max_y_shift
        self.max_constrast = max_contrast
        self.min_constrast = min_constrast
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def draw_next_transform(self) -> List[Callable]:
        """
        Samples all parameters defining the transformation pipeline.
        Returns a list of operations to apply to each slice in the
        3D volume.

        :return: list of transformations to apply to each slice.
        """
        # Sample parameters for each transformation
        angle = random.randint(-self.max_angle, self.max_angle)
        x_shift = random.uniform(-self.max_x_shift, self.max_x_shift)
        y_shift = random.uniform(-self.max_y_shift, self.max_y_shift)
        contrast = random.uniform(self.min_constrast, self.max_constrast)
        brightness = random.uniform(self.min_brightness, self.max_brightness)
        horizontal_flip = ImageTransformationBase._toss_fair_coin()
        # Returns the corresponding operations
        if random.random() < self.probability_transformation:
            ops = [self.rotate(angle),
                   self.translateX(x_shift),
                   self.translateY(y_shift)]
            if horizontal_flip:
                ops.append(self.horizontal_flip())
            if self.for_segmentation_input_maps:
                return ops
            ops.extend([self.adjust_contrast(contrast),
                        self.adjust_brightness(brightness)])
        else:
            ops = []
        return ops


class RandAugmentSlice(ImageTransformationBase):
    """
    Implements the RandAugment procedure on a restricted set of
    transformations. https://arxiv.org/abs/1909.13719

    Possible transformations for segmentations maps are: rotation, horizontal
    and vertical shift, horizontal flip, identity. Additional transformations
    for images are brightness adjustment and contrast adjustment.
    """

    def __init__(self,
                 magnitude: int = 3,
                 n_transforms: int = 2,
                 **kwargs: Any) -> None:
        """
        :param magnitude: magnitude to apply to the transformations as defined in the RandAugment paper.
        1 means a weak transform, 10 is the strongest transform.
        :param n_transforms: number of transformation to sample for each image.
        """
        super().__init__(**kwargs)
        self.magnitude = magnitude
        self.n_transforms = n_transforms
        self._max_magnitude = 10.0
        self._max_x_shift = 0.1
        self._max_y_shift = 0.2
        self._max_angle = 30
        self._max_contrast = 1
        self._max_brightness = 1

    def get_all_transforms(self) -> Dict[str, Callable]:
        """
        Defines the possible transformations for one fixed magnitude level
        to sample from.
        """
        # Convert magnitude to argument for each transform
        level = self.magnitude / self._max_magnitude
        angle = self.randomly_negate_level(level) * self._max_angle
        x_shift = self.randomly_negate_level(level) * self._max_x_shift
        y_shift = self.randomly_negate_level(level) * self._max_y_shift
        # Contrast / brightness factor of 1 means no change. 0 means black, 2 means times 2.
        contrast = self.randomly_negate_level(level) * self._max_contrast + 1
        brightness = self.randomly_negate_level(level) * self._max_brightness + 1
        transforms_dict = {
            "identity": self.identity(),
            "rotate": self.rotate(angle),
            "translateX": self.translateX(x_shift),
            "translateY": self.translateY(y_shift),
            "hFlip": self.horizontal_flip()
        }
        if self.for_segmentation_input_maps:
            return transforms_dict

        transforms_dict.update(
            {"constrast": self.adjust_contrast(contrast),
             "brightness": self.adjust_brightness(brightness),
             })

        return transforms_dict

    def draw_next_transform(self) -> List[Callable]:
        """
        Samples all parameters defining the transformation pipeline.
        Returns a list of operations to apply to each slice of a 3D volume
        (defined by the sampled parameters).

        :return: list of transformations to apply to each slice.
        """
        available_transforms = self.get_all_transforms()
        transform_names = np.random.choice(list(available_transforms), self.n_transforms)
        ops = [available_transforms[name] for name in transform_names]
        return ops


class ScalarItemAugmentation(Transform3D[ScalarItem]):
    """
    Wrapper around an augmentation pipeline for applying an image transformation
    to a ScalarItem input and return the transformed sample. Applies the
    transformation either to the images or the segmentation maps depending on the
    defined transformation to apply. Several objects of this class can be applied
    in a row inside a Compose3D object.
    """

    # noinspection PyMissingConstructor
    def __init__(self, transform: ImageTransformationBase):
        """

        :param transform: the transformation to apply to the image.
        """
        self.transform = transform

    def __call__(self, item: ScalarItem) -> ScalarItem:
        if self.transform.for_segmentation_input_maps:
            if item.segmentations is None:
                raise ValueError("A segmentation data augmentation transform has been"
                                 "specified but no segmentations has been loaded.")
            return item.clone_with_overrides(segmentations=self.transform(item.segmentations))
        else:
            return item.clone_with_overrides(images=self.transform(item.images))


class SampleImageAugmentation(Transform3D[Sample]):
    """
    Wrapper around augmentation pipeline for applying an image transformation
    to a Sample input (for segmentation models).
    """

    # noinspection PyMissingConstructor
    def __init__(self, transform: ImageTransformationBase) -> None:
        self.transform = transform

    def __call__(self, item: Sample) -> Sample:
        return item.clone_with_overrides(image=self.transform(item.image))
