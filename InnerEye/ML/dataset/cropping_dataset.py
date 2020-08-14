#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from InnerEye.Common.common_util import any_pairwise_larger
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.config import PaddingMode, SegmentationModelBase
from InnerEye.ML.dataset.full_image_dataset import FullImageDataset
from InnerEye.ML.dataset.sample import CroppedSample, Sample
from InnerEye.ML.utils import augmentation, image_util
from InnerEye.ML.utils.image_util import pad_images
from InnerEye.ML.utils.io_util import ImageDataType
from InnerEye.ML.utils.transforms import ComposeTransforms


class CroppingDataset(FullImageDataset):
    """
    Dataset class that creates random cropped samples from full 3D images from a given pd.DataFrame. The following
    are the operations performed to generate a sample from this dataset. The crops extracted are of size
    crop_size which is defined in the model config, and the crop center class population is distributed as per the
    class_weights vector in the model config (which by default weights all classes equally)
    """

    def __init__(self, args: SegmentationModelBase, data_frame: pd.DataFrame,
                 cropped_sample_transforms: Optional[ComposeTransforms[CroppedSample]] = None,
                 full_image_sample_transforms: Optional[ComposeTransforms[Sample]] = None):
        super().__init__(args, data_frame, full_image_sample_transforms)
        self.cropped_sample_transforms = cropped_sample_transforms

    def __getitem__(self, i: int) -> Dict[str, Any]:
        sample = CroppingDataset.create_possibly_padded_sample_for_cropping(
            sample=super().get_samples_at_index(index=i)[0],
            crop_size=self.args.crop_size,
            padding_mode=self.args.padding_mode
        )

        sample = self.create_random_cropped_sample(
            sample=sample,
            crop_size=self.args.crop_size,
            center_size=self.args.center_size,
            class_weights=self.args.class_weights
        )

        return ComposeTransforms.apply(self.cropped_sample_transforms, sample).get_dict()

    @staticmethod
    def create_possibly_padded_sample_for_cropping(sample: Sample,
                                                   crop_size: TupleInt3,
                                                   padding_mode: PaddingMode) -> Sample:
        """
        Pad the original sample such the the provided images has the same
        (or slightly larger in case of uneven difference) shape to the output_size, using the provided padding mode.
        :param sample: Sample to pad.
        :param crop_size: Crop size to match.
        :param padding_mode: The padding scheme to apply.
        :return: padded sample
        """
        image_spatial_shape = sample.image.shape[-3:]

        if any_pairwise_larger(crop_size, image_spatial_shape):
            if padding_mode == PaddingMode.NoPadding:
                raise ValueError(
                    "The crop_size across each dimension should be greater than zero and less than or equal "
                    f"to the current value (crop_size: {crop_size}, spatial shape: {image_spatial_shape}) "
                    "or padding_mode must be set to enable padding")
            else:
                sample = sample.clone_with_overrides(
                    image=pad_images(sample.image, crop_size, padding_mode),
                    mask=pad_images(sample.mask, crop_size, padding_mode),
                    labels=pad_images(sample.labels, crop_size, padding_mode)
                )

                logging.debug(f"Padded sample for patient: {sample.patient_id}, from spatial dimensions: "
                              f"{image_spatial_shape}, to: {sample.image.shape[-3:]}")

        return sample

    @staticmethod
    def create_random_cropped_sample(sample: Sample,
                                     crop_size: TupleInt3,
                                     center_size: TupleInt3,
                                     class_weights: Optional[List[float]] = None) -> CroppedSample:
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
            class_weights=class_weights
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
