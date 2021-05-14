#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Callable, Union

from InnerEye.ML.augmentations.transform_pipeline import ImageTransformationPipeline
from InnerEye.ML.dataset.sample import Sample
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.utils.transforms import Transform3D


class ScalarItemAugmentation(Transform3D[ScalarItem]):
    """
    Wrapper around an augmentation pipeline for applying an image transformation
    to a ScalarItem input and return the transformed sample. Applies the
    transformation either to the images or the segmentation maps depending on the
    defined transformation to apply. Several objects of this class can be applied
    in a row inside a Compose3D object.
    """

    # noinspection PyMissingConstructor
    def __init__(self, transform: Union[Callable, ImageTransformationPipeline]):
        """

        :param transform: the transformation to apply to the image.
        """
        self.transform_pipeline = transform

    def __call__(self, item: ScalarItem) -> ScalarItem:
        if hasattr(self.transform_pipeline, "for_segmentation_input_maps") and self.transform_pipeline.for_segmentation_input_maps:
            if item.segmentations is None:
                raise ValueError("A segmentation data augmentation transform_pipeline has been"
                                 "specified but no segmentations has been loaded.")
            return item.clone_with_overrides(segmentations=self.transform_pipeline(item.segmentations))
        else:
            return item.clone_with_overrides(images=self.transform_pipeline(item.images))


class SampleImageAugmentation(Transform3D[Sample]):
    """
    Wrapper around augmentation pipeline for applying an image transformation
    to a Sample input (for segmentation models).
    """

    # noinspection PyMissingConstructor
    def __init__(self, transform: ImageTransformationPipeline) -> None:
        self.transform_pipeline = transform

    def __call__(self, item: Sample) -> Sample:
        return item.clone_with_overrides(image=self.transform_pipeline(item.image))


