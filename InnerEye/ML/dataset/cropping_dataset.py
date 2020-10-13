#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from monai.transforms import Compose, Randomizable, Transform

from InnerEye.Common.common_util import any_pairwise_larger
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.config import PaddingMode, SegmentationModelBase
from InnerEye.ML.dataset.full_image_dataset import FullImageDataset
from InnerEye.ML.dataset.sample import CroppedSample, Sample
from InnerEye.ML.utils import augmentation, image_util
from InnerEye.ML.utils.augmentation import random_select_patch_center
from InnerEye.ML.utils.image_util import pad_images
from InnerEye.ML.utils.io_util import ImageDataType
from InnerEye.ML.utils.transforms import Compose3D


class PadSample(Transform):
    def __init__(self, output_size: TupleInt3, padding_mode: PaddingMode):
        super().__init__()
        self.output_size = output_size
        self.padding_mode = padding_mode

    def __call__(self, data: Sample) -> Sample:
        return self.create_possibly_padded_sample_for_cropping(data, self.output_size, self.padding_mode)

    @staticmethod
    def create_possibly_padded_sample_for_cropping(sample: Sample,
                                                   output_size: TupleInt3,
                                                   padding_mode: PaddingMode) -> Sample:
        """
        Pad the original sample such the the provided images has the same
        (or slightly larger in case of uneven difference) shape to the output_size, using the provided padding mode.
        :param sample: Sample to pad.
        :param output_size: Output size to match.
        :param padding_mode: The padding scheme to apply.
        :return: padded sample
        """
        image_spatial_shape = sample.image.shape[-3:]

        if any_pairwise_larger(output_size, image_spatial_shape):
            if padding_mode == PaddingMode.NoPadding:
                raise ValueError(
                    "The crop_size across each dimension should be greater than zero and less than or equal "
                    f"to the current value (crop_size: {output_size}, spatial shape: {image_spatial_shape}) "
                    "or padding_mode must be set to enable padding")
            else:
                sample = sample.clone_with_overrides(
                    image=pad_images(sample.image, output_size, padding_mode),
                    mask=pad_images(sample.mask, output_size, padding_mode),
                    labels=pad_images(sample.labels, output_size, padding_mode)
                )

                logging.debug(f"Padded sample for patient: {sample.patient_id}, from spatial dimensions: "
                              f"{image_spatial_shape}, to: {sample.image.shape[-3:]}")

        return sample


class RandomCropSample(Randomizable, Transform):
    def __init__(self, random_seed: int, crop_size: TupleInt3,
                 center_size: TupleInt3,
                 class_weights: Optional[List[float]] = None):
        super().__init__()
        self.crop_size = crop_size
        self.center_size = center_size
        self.class_weights = class_weights
        self.set_random_state(seed=random_seed)
        self._random_patch_center = None

    def randomize(self, data: Sample) -> None:
        # Sample a center pixel location for patch extraction.
        self._random_patch_center = random_select_patch_center(data, self.class_weights)

    def __call__(self, data: Sample) -> CroppedSample:
        self.randomize(data)
        return self.create_random_cropped_sample(data, self.crop_size, self.center_size, self.class_weights)

    @staticmethod
    def create_random_cropped_sample(sample: Sample,
                                     crop_size: TupleInt3,
                                     center_size: TupleInt3,
                                     class_weights: Optional[List[float]] = None,
                                     center: Optional[np.ndarray] = None) -> CroppedSample:
        """
        Creates an instance of a cropped sample extracted from full 3D images.
        :param sample: the full size 3D sample to use for extracting a cropped sample.
        :param crop_size: the size of the crop to extract.
        :param center_size: the size of the center of the crop (this should be the same as the spatial dimensions
                            of the posteriors that the model produces)
        :param class_weights: the distribution to use for the crop center class.
        :return: CroppedSample
        """
        # crop the original raw sample
        sample, center_point = augmentation.random_crop(
            sample=sample,
            crop_size=crop_size,
            class_weights=class_weights,
            center=center
        )

        # crop the mask and label centers if required
        if center_size == crop_size:
            mask_center_crop = sample.mask
            labels_center_crop = sample.labels
        else:
            mask_center_crop = image_util.get_center_crop(image=sample.mask, crop_shape=center_size)
            labels_center_crop = np.zeros(shape=[len(sample.labels)] + list(center_size),  # type: ignore
                                          dtype=ImageDataType.SEGMENTATION.value)
            for c in range(len(sample.labels)):  # type: ignore
                labels_center_crop[c] = image_util.get_center_crop(
                    image=sample.labels[c],
                    crop_shape=center_size
                )

        return CroppedSample(
            image=sample.image,
            mask=sample.mask,
            labels=sample.labels,
            mask_center_crop=mask_center_crop,
            labels_center_crop=labels_center_crop,
            center_indices=center_point,
            metadata=sample.metadata
        )


class CroppingDataset(FullImageDataset):
    """
    Dataset class that creates random cropped samples from full 3D images from a given pd.DataFrame. The following
    are the operations performed to generate a sample from this dataset. The crops extracted are of size
    crop_size which is defined in the model config, and the crop center class population is distributed as per the
    class_weights vector in the model config (which by default weights all classes equally)
    """

    def __init__(self, args: SegmentationModelBase, data_frame: pd.DataFrame,
                 cropped_sample_transforms: Optional[List[Callable]] = None,
                 full_image_sample_transforms: Optional[List[Callable]] = None):
        self.cropped_sample_transforms = cropped_sample_transforms or list()
        super().__init__(args, data_frame, full_image_sample_transforms)

    def get_transforms(self) -> List[Callable]:
        base_transforms = super().get_transforms()
        return base_transforms + self.cropped_sample_transforms
