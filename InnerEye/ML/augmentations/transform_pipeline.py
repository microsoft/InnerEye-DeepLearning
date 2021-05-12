#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List

import PIL
import torch
import torchvision
from torchvision.transforms import functional as F
from yacs.config import CfgNode

from InnerEye.ML.augmentations.image_transforms import AddGaussianNoise, CenterCrop, ElasticTransform, \
    ExpandChannels, ImageTransformBase, RandomAffine, \
    RandomColorJitter, RandomErasing, RandomGamma, \
    RandomHorizontalFlip, \
    RandomResizeCrop, Resize, ToTensor
from InnerEye.ML.utils.transforms import Transform3D


class ImageTransformationPipeline(Transform3D):
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
                 transforms: List[ImageTransformBase],
                 is_transformation_for_segmentation_maps: bool = False,
                 use_joint_channel_transformation: bool = True,
                 use_different_transformation_per_channel: bool = False):
        """
        :param transforms: List of transformations to apply to images.
        :param is_transformation_for_segmentation_maps: if True, only use geometrical transformation suitable
        for segmentation maps. If False, additionally use color/contrast related transformation suitable for
        images.
        :param use_joint_channel_transformation: if True apply one transformation on each slice but using all channels
        as input e.g. for RGB images. If False, apply transformation channel by channel.
        :param use_different_transformation_per_channel: if True, apply a different version of the augmentation pipeline
        for each channel. If False, applies the same transformation to each channel, separately. Incompatible with
        use_joint_channel_transformation set to True.

        """
        self.transforms = transforms
        self.for_segmentation_input_maps = is_transformation_for_segmentation_maps
        self.use_joint_channel_transformation = use_joint_channel_transformation
        self.use_different_transformation_per_channel = use_different_transformation_per_channel
        if self.use_joint_channel_transformation and self.use_different_transformation_per_channel:
            raise ValueError("You cannot specify both use_joint_channel_transformation = True and "
                             "use_different_transformation_per_channel = True")

    def draw_next_transform(self, input_size: List[int]) -> None:
        """
        Samples all parameters defining the transformation pipeline.
        Returns a list of operations to apply to each 2D-slice in a given
        3D volume.
        """
        for transform in self.transforms:
            input_size = transform.draw_transform(input_size)

    def apply_transform_on_3d_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply a list of transforms sequentially to a 3D image. Each transformation is assumed to be a
        2D-transform_pipeline to
        be applied to each slice of the given 3D input separately.

        :param image: a 4D tensor dimension [C, Z, X, Y]. Transform are applied one after another
        separately for each [X, Y] slice along the Z-dimension (assumed to be the first dimension).
        :param transforms: a list of transformations to apply to each slice sequentially.
        :returns image: the transformed 3D-image
         """
        slices = []
        for z in range(image.shape[1]):
            data = image[z]
            data = F.to_pil_image(data)
            slices.append(self.apply_transform_on_2d_image(data))
        return torch.cat(slices, dim=1)

    def apply_transform_on_2d_image(self, image: PIL.Image) -> torch.Tensor:
        """
        Apply a list of transform_pipeline sequentially to a 2D PIL Image. Assumes the first transform_pipeline takes
        a PIL Image as input
        and the last transform_pipeline can either return a Tensor or a PIL Image.
        :param image:
        :return:
        """
        for transform_fn in self.transforms:
            image = transform_fn(image)
        return F.to_tensor(image) if isinstance(image, PIL.Image.Image) else image

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Main function to apply the transformation pipeline to either slice by slice on one 3D-image or
        on the 2D image.

        Note for 3D images: Assumes the same transformations have to be applied on each 2D-slice along the Z-axis.
        Assumes the Z axis is the first dimension.

        :param image: batch of tensor images of size [C, Z, Y, X] or batch of 2D images as PIL Image
        """
        if isinstance(image, PIL.Image.Image):
            image = torchvision.transforms.functional.to_tensor(image).unsqueeze(1)
        assert len(image.shape) == 4
        c, z, x, y = image.shape
        res = image.clone()
        if self.for_segmentation_input_maps:
            res = res.int()
        else:
            res = res.float()
            if res.max() > 1:
                raise ValueError("Image tensor should be in "
                                 "range 0-1 for conversion to PIL")
            if not self.use_joint_channel_transformation:
                for channel in range(image.shape[0]):
                    if self.use_different_transformation_per_channel or channel == 0:
                        self.draw_next_transform(input_size=[1, x, y])
                res[channel] = self.apply_transform_on_3d_image(res[channel].unsqueeze(0)).squeeze(0)
            else:
                self.draw_next_transform(input_size=[1, x, y])
                res = self.apply_transform_on_3d_image(res)
        return res.to(dtype=image.dtype)


def create_transform_pipeline_from_config(config: CfgNode,
                                          apply_augmentations: bool) -> ImageTransformationPipeline:
    """
    Defines the image transformations pipeline used in Chest-Xray datasets.
    Type of augmentation and strength are defined in the config.
    :param config: config yaml file fixing strength and type of augmentation to apply
    :param apply_augmentations: if True return transformation pipeline with augmentations. Else,
    disable augmentations i.e.
    only resize and center crop the image.
    """
    transforms: List[Any] = []
    if apply_augmentations:
        if config.augmentation.use_random_affine:
            transforms.append(RandomAffine(config))
        if config.augmentation.use_random_crop:
            transforms.append(RandomResizeCrop(config))
        else:
            transforms.append(Resize(config))
        if config.augmentation.use_random_horizontal_flip:
            transforms.append(RandomHorizontalFlip(config))
        if config.augmentation.use_gamma_transform:
            transforms.append(RandomGamma(config))
        if config.augmentation.use_random_color:
            transforms.append(RandomColorJitter(config))
        if config.augmentation.use_elastic_transform:
            transforms.append(ElasticTransform(config))
        transforms += [CenterCrop(config), ToTensor()]
        if config.augmentation.use_random_erasing:
            transforms.append(RandomErasing(config))
        if config.augmentation.add_gaussian_noise:
            transforms.append(AddGaussianNoise(config))
    else:
        transforms += [Resize(config), CenterCrop(config), ToTensor()]
    transforms.append(ExpandChannels())
    pipeline = ImageTransformationPipeline(transforms=transforms)
    return pipeline
