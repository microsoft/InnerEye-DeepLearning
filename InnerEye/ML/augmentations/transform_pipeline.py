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

ImageData = Union[PIL.Image.Image, torch.Tensor]


class ImageTransformationPipeline:
    """
    This class is the base class to classes built to define data augmentation transformations
    for 3D or 2D image inputs (tensor or PIL.Image).
    In the case of 3D images, the transformations are applied slice by slices along the Z dimension (same transformation
    applied for each slice).
    The transformations are applied channel by channel, the user can specify whether to apply the same transformation
    to each channel (no random shuffling) or whether each channel should use a different transformation (random
    parameters of transforms shuffled for each channel).
    """

    # noinspection PyMissingConstructor
    def __init__(self,
                 transforms: Union[Callable, List[Callable]],
                 use_different_transformation_per_channel: bool = False):
        """
        :param transforms: List of transformations to apply to images. Supports out of the boxes torchvision transforms
            as they accept data of arbitrary dimension. You can also define your own transform class but be aware that you
        function should expect input of shape [C, Z, H, W] and apply the same transformation to each Z slice.

        :param use_different_transformation_per_channel: if True, apply a different version of the augmentation pipeline
            for each channel. If False, applies the same transformation to each channel, separately.
        """
        self.use_different_transformation_per_channel = use_different_transformation_per_channel
        self.pipeline = Compose(transforms) if isinstance(transforms, List) else transforms

    def transform_image(self, image: ImageData) -> torch.Tensor:
        """
        Main function to apply the transformation pipeline to either slice by slice on one 3D-image or
        on the 2D image.

        Note for 3D images: Assumes the same transformations have to be applied on each 2D-slice along the Z-axis.
        Assumes the Z axis is the first dimension.

        :param image: batch of tensor images of size [C, Z, Y, X] or batch of 2D images as PIL Image
        """

        def _convert_to_tensor_if_necessary(data: ImageData) -> torch.Tensor:
            return to_tensor(data) if not isinstance(data, torch.Tensor) else data

        image = _convert_to_tensor_if_necessary(image)
        original_input_is_2d = len(image.shape) == 3
        # If we have a 2D image [C, H, W] expand to [Z, C, H, W]. Build-in torchvision transforms allow such 4D inputs.
        if original_input_is_2d:
            image = image.unsqueeze(0)
        else:
            # Some transforms assume the order of dimension is [..., C, H, W] so permute first and last dimension to
            # obtain [Z, C, H, W]
            if len(image.shape) != 4:
                raise ValueError(f"ScalarDataset should load images as 4D tensor [C, Z, H, W]. The input tensor here"
                                 f"was of shape {image.shape}. This is unexpected.")
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
        if original_input_is_2d:
            image = image.squeeze(1)
        return image.to(dtype=image.dtype)

    def __call__(self, data: ImageData) -> torch.Tensor:
        return self.transform_image(data)


def create_transforms_from_config(config: CfgNode,
                                  apply_augmentations: bool,
                                  expand_channels: bool = True) -> ImageTransformationPipeline:
    """
    Defines the image transformations pipeline from a config file. It has been designed for Chest X-Ray
    images but it can be used for other types of images data, type of augmentations to use and strength are
    expected to be defined in the config. The channel expansion is needed for gray images.

    :param config: config yaml file fixing strength and type of augmentation to apply
    :param apply_augmentations: if True return transformation pipeline with augmentations. Else,
        disable augmentations i.e. only resize and center crop the image.

    :param expand_channels: if True the expand channel transformation from InnerEye.ML.augmentations.image_transforms
        will be added to the transformation passed through the config. This is needed for single channel images as CXR.
    """
    transforms: List[Any] = []
    if expand_channels:
        transforms.append(ExpandChannels())
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
    pipeline = ImageTransformationPipeline(transforms)
    return pipeline
