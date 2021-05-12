#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
from collections import Callable
from typing import Any, List, Tuple

import PIL
import numpy as np
import torch
import torchvision
from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform
from scipy.ndimage import gaussian_filter, map_coordinates
from torchvision.transforms import ToTensor
from yacs.config import CfgNode


class BaseTransform:
    def transform(self, x: Any) -> Any:
        raise NotImplementedError("Transform needs to be overridden in the child classes")

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class CenterCrop(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.center_crop_size = config.preprocess.center_crop_size

    def transform(self, x: Any) -> Any:
        return torchvision.transforms.CenterCrop(self.center_crop_size)(x)


class RandomResizeCrop(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.resize_size = config.preprocess.resize
        self.crop_scale = config.augmentation.random_crop.scale

    def transform(self, x: Any) -> Any:
        return torchvision.transforms.RandomResizedCrop(
            size=self.resize_size,
            scale=self.crop_scale)(x)


class RandomHorizontalFlip(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.p_apply = config.augmentation.random_horizontal_flip.prob

    def transform(self, x: Any) -> Any:
        return torchvision.transforms.RandomHorizontalFlip(self.p_apply)(x)


class RandomAffine(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.max_angle = config.augmentation.random_affine.max_angle
        self.max_horizontal_shift = config.augmentation.random_affine.max_horizontal_shift
        self.max_vertical_shift = config.augmentation.random_affine.max_vertical_shift
        self.max_shear = config.augmentation.random_affine.max_shear

    def transform(self, x: Any) -> Any:
        return torchvision.transforms.RandomAffine(degrees=self.max_angle,
                                                   translate=(self.max_horizontal_shift, self.max_vertical_shift),
                                                   shear=self.max_shear)(x)


class Resize(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.resize_size = config.preprocess.resize

    def transform(self, x: Any) -> Any:
        return torchvision.transforms.Resize(self.resize_size)(x)


class RandomColorJitter(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.max_brightness = config.augmentation.random_color.brightness
        self.max_contrast = config.augmentation.random_color.contrast
        self.max_saturation = config.augmentation.random_color.saturation

    def transform(self, x: Any) -> Any:
        return torchvision.transforms.ColorJitter(brightness=self.max_brightness,
                                                  contrast=self.max_contrast,
                                                  saturation=self.max_saturation)(x)


class RandomErasing(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.scale = config.augmentation.random_erasing.scale
        self.ratio = config.augmentation.random_erasing.ratio

    def transform(self, x: Any) -> Any:
        return torchvision.transforms.RandomErasing(p=0.5,
                                                    scale=self.scale,
                                                    ratio=self.ratio)(x)


class RandomGamma(BaseTransform):

    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.scale = config.augmentation.gamma.scale

    def transform(self, image: PIL.Image.Image) -> PIL.Image.Image:
        gamma = random.uniform(*self.scale)
        return torchvision.transforms.functional.adjust_gamma(image, gamma=gamma)


class ExpandChannels(BaseTransform):
    """
    Transforms an image with 1 channel to an image with 3 channels by copying pixel intensities of the image along
    the 0th dimension.
    """

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return torch.repeat_interleave(data, 3, dim=0)


class AddGaussianNoise(BaseTransform):

    def __init__(self, config: CfgNode) -> None:
        """
        Transformation to add Gaussian noise N(0, std) to an image. Where std is set with the
        config.augmentation.gaussian_noise.std argument. The transformation will be applied with probability
        config.augmentation.gaussian_noise.p_apply
        """
        super().__init__()
        self.p_apply = config.augmentation.gaussian_noise.p_apply
        self.std = config.augmentation.gaussian_noise.std

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        assert data.max() <= 1 and data.min() >= 0
        if np.random.random(1) > self.p_apply:
            return data
        noise = torch.randn(size=data.shape) * self.std
        data = torch.clamp(data + noise, 0, 1)
        return data


class ElasticTransform(BaseTransform):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.

       https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.160.8494&rep=rep1&type=pdf

        :param sigma: elasticity coefficient
        :param alpha: intensity of the deformation
        :param p_apply: probability of applying the transformation
    """

    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.alpha = config.augmentation.elastic_transform.alpha
        self.sigma = config.augmentation.elastic_transform.sigma
        self.p_apply = config.augmentation.elastic_transform.p_apply

    def transform(self, image: PIL.Image) -> PIL.Image:
        if np.random.random(1) > self.p_apply:
            return image
        image = np.asarray(image).squeeze()
        assert len(image.shape) == 2
        shape = image.shape

        dx = gaussian_filter((np.random.random(shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((np.random.random(shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        return PIL.Image.fromarray(map_coordinates(image, indices, order=1).reshape(shape))


class DualViewTransformWrapper:
    """
    Returns two versions of one image, given a random transformation function.
    """

    def __init__(self, transform: Callable):
        self.transform = transform

    def __call__(self, sample: PIL.Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


def get_cxr_ssl_transforms(config: CfgNode,
                           return_two_views_per_sample: bool,
                           use_training_augmentations_for_validation: bool = False) -> Tuple[Any, Any]:
    """
    Returns training and validation transforms for CXR.
    Transformations are constructed in the following way:
    1. Construct the pipeline of augmentations in create_chest_xray_transform (e.g. resize, flip, affine) as defined
    by the config.
    2. If we just want to construct the transformation pipeline for a classification model or for the linear evaluator
    of the SSL module, return this pipeline.
    2. If we are constructing transforms for the SSL training, we have to return two versions of each image, hence
    apply DualViewTransformWrapper a wrapper around the obtained transformation pipeline so that we return two augmented
    version of each sample.

    :param config: configuration defining which augmentations to apply as well as their intensities.
    :param return_two_views_per_sample: if True the resulting transforms will return two versions of each sample they
    are called on. If False, simply return one transformed version of the sample.
    :param use_training_augmentations_for_validation: If True, use augmentation at validation time too.
    This is required for SSL validation loss to be meaningful. If False, only apply basic processing step
    (no augmentations)
    """
    train_transforms = create_chest_xray_transform(config, apply_augmentations=True)
    val_transforms = create_chest_xray_transform(config, apply_augmentations=use_training_augmentations_for_validation)
    if return_two_views_per_sample:
        train_transforms = DualViewTransformWrapper(train_transforms)
        val_transforms = DualViewTransformWrapper(val_transforms)
    return train_transforms, val_transforms


def create_chest_xray_transform(config: CfgNode,
                                apply_augmentations: bool) -> Callable:
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
    return torchvision.transforms.Compose(transforms)


class InnerEyeCIFARTrainTransform(SimCLRTrainDataTransform):
    """
    Overload lightning-bolts SimCLRTrainDataTransform, to avoid return unused eval transform. Used for training and
    val of SSL models.
    """

    def __call__(self, sample: Any) -> Tuple[Any, Any]:
        transform = self.train_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class InnerEyeCIFARLinearHeadTransform(SimCLRTrainDataTransform):
    """
    Overload lightning-bolts SimCLRTrainDataTransform, to only return linear head eval transform.
    """

    def __call__(self, sample: Any) -> Any:
        return self.online_transform(sample)
