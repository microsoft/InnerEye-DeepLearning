#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Callable, List, Union

import PIL
import torch

from torchvision.transforms import CenterCrop, ColorJitter, Compose, RandomAffine, RandomErasing, \
    RandomHorizontalFlip, RandomResizedCrop, Resize
from torchvision.transforms.functional import to_tensor
from yacs.config import CfgNode

from InnerEye.ML.augmentations.image_transforms import AddGaussianNoise, ElasticTransform, ExpandChannels, RandomGamma
from InnerEye.ML.dataset.scalar_sample import ScalarItem


class ImageTransformationPipeline:
    """
    This class is the base class to classes built to define data augmentation transformations
    for 3D or 2D image inputs.
    In the case of 3D images, the transformations are applied slice by slices along the Z dimension (same transformation
    applied for each slice).
    The transformations are applied channel by channel, the user can specify whether to apply the same transformation
    to each channel (no random shuffling) or whether each channel should use a different transformation (random
    parameters of transforms shuffled for each channel).
    """

    # noinspection PyMissingConstructor
    def __init__(self,
                 transforms: List[Callable],
                 use_different_transformation_per_channel: bool = False,
                 apply_pipeline_to_segmentation_maps: bool = False):
        """
        :param transforms: List of transformations to apply to images. Supports out of the boxes torchvision transforms
        as they accept data of arbitrary dimension. If the data is [C, Z, H, W] they will apply the same transformation
        to all leading dimensions (i.e. same transformation for all slices by default). You can also define your own
        transform class but be aware that you function should expect input of shape [C, Z, H, W] and apply the same
        transformation to each C, Z slice.
        :param use_different_transformation_per_channel: if True, apply a different version of the augmentation pipeline
        for each channel. If False, applies the same transformation to each channel, separately.
        :param: apply_transform_to_segmentation_maps. If True, the pipeline will be applied to the segmentations field
        of the scalar item, else it will be applied to the images.
        """
        self.transforms = transforms
        self.use_different_transformation_per_channel = use_different_transformation_per_channel
        self.pipeline = Compose(transforms)
        self.apply_pipeline_to_segmentation_maps = apply_pipeline_to_segmentation_maps

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Main function to apply the transformation pipeline to either slice by slice on one 3D-image or
        on the 2D image.

        Note for 3D images: Assumes the same transformations have to be applied on each 2D-slice along the Z-axis.
        Assumes the Z axis is the first dimension.

        :param image: batch of tensor images of size [C, Z, Y, X] or batch of 2D images as PIL Image
        """

        def _convert_to_tensor_if_necessary(data: Union[PIL.Image.Image, torch.Tensor]) -> torch.Tensor:
            return to_tensor(data) if isinstance(data, PIL.Image.Image) else data

        image = _convert_to_tensor_if_necessary(image)

        # If we have a 2D image [C, H, W] expand to [Z, C, H, W]. Build-in torchvision transforms allow such 4D inputs.
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        else:
            # Some transforms assume the order of dimension is [..., C, H, W] so permute first and last dimension to
            # obtain [Z, C, H, W]
            image = torch.transpose(image, 1, 0)

        if not self.use_different_transformation_per_channel:
            image = _convert_to_tensor_if_necessary(self.pipeline(image))
        else:
            channels = []
            for channel in range(image.shape[1]):
                channels.append(_convert_to_tensor_if_necessary(self.pipeline(image[:, channel, :, :].unsqueeze(1))))
            image = torch.cat(channels, dim=1)
        # Back to [C, Z, H, W]
        image = torch.transpose(image, 1, 0)
        return image.to(dtype=image.dtype)

    def get_scalar_item_transformation(self, item: ScalarItem) -> ScalarItem:
        """
        This function returns the transformation around a ScalarItem, it will apply the pipeline to either the images
        or the segmentations and return a new ScalarItem with the transformed image.
        :param item: item to transform
        :return: item with the new transformed image.
        """
        if self.apply_pipeline_to_segmentation_maps:
            if item.segmentations is None:
                raise ValueError("A segmentation data augmentation transform_pipeline has been"
                                 "specified but no segmentations has been loaded.")
            return item.clone_with_overrides(segmentations=self(item.segmentations))
        else:
            return item.clone_with_overrides(images=self(item.images))

def create_cxr_transform_pipeline_from_config(config: CfgNode,
                                              apply_augmentations: bool) -> ImageTransformationPipeline:
    """
    Defines the image transformations pipeline used in Chest-Xray datasets. Can be used for other types of
    images data, type of augmentations to use and strength are expected to be defined in the config.
    :param config: config yaml file fixing strength and type of augmentation to apply
    :param apply_augmentations: if True return transformation pipeline with augmentations. Else,
    disable augmentations i.e. only resize and center crop the image.
    """
    transforms: List[Any] = [ExpandChannels()]
    if apply_augmentations:
        if config.augmentation.use_random_affine:
            transforms.append(RandomAffine(
                degrees=config.augmentation.random_affine.max_angle,
                translate=(config.augmentation.random_affine.max_horizontal_shift,
                           config.augmentation.random_affine.max_vertical_shift),
                shear=config.augmentation.random_affine.max_shear
            ))
        if config.augmentation.use_random_crop:
            transforms.append(RandomResizedCrop(
                scale=config.augmentation.random_crop.scale,
                size=config.preprocess.resize
            ))
        else:
            transforms.append(Resize(size=config.preprocess.resize))
        if config.augmentation.use_random_horizontal_flip:
            transforms.append(RandomHorizontalFlip(p=config.augmentation.random_horizontal_flip.prob))
        if config.augmentation.use_gamma_transform:
            transforms.append(RandomGamma(scale=config.augmentation.gamma.scale))
        if config.augmentation.use_random_color:
            transforms.append(ColorJitter(
                brightness=config.augmentation.random_color.brightness,
                contrast=config.augmentation.random_color.contrast,
                saturation=config.augmentation.random_color.saturation
            ))
        if config.augmentation.use_elastic_transform:
            transforms.append(ElasticTransform(
                alpha=config.augmentation.elastic_transform.alpha,
                sigma=config.augmentation.elastic_transform.sigma,
                p_apply=config.augmentation.elastic_transform.p_apply
            ))
        transforms.append(CenterCrop(config.preprocess.center_crop_size))
        if config.augmentation.use_random_erasing:
            transforms.append(RandomErasing(
                scale=config.augmentation.random_erasing.scale,
                ratio=config.augmentation.random_erasing.ratio
            ))
        if config.augmentation.add_gaussian_noise:
            transforms.append(AddGaussianNoise(
                p_apply=config.augmentation.gaussian_noise.p_apply,
                std=config.augmentation.gaussian_noise.std
            ))
    else:
        transforms += [Resize(size=config.preprocess.resize),
                       CenterCrop(config.preprocess.center_crop_size)]
    pipeline = ImageTransformationPipeline(transforms=transforms)
    return pipeline
