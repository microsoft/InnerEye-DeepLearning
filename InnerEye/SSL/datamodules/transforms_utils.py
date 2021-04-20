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
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
from scipy.ndimage import gaussian_filter, map_coordinates
from torchvision.transforms import ToTensor

from InnerEye.SSL.config_node import ConfigNode


class BaseTransform:
    def __init__(self, config: ConfigNode):
        self.transform = lambda x: x

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class CenterCrop(BaseTransform):
    def __init__(self, config: ConfigNode):
        self.transform = torchvision.transforms.CenterCrop(config.preprocess.center_crop_size)


class RandomResizeCrop(BaseTransform):
    def __init__(self, config: ConfigNode):
        self.transform = torchvision.transforms.RandomResizedCrop(
            size=config.preprocess.resize,
            scale=config.augmentation.random_crop.scale)


class RandomHorizontalFlip(BaseTransform):
    def __init__(self, config: ConfigNode):
        self.transform = torchvision.transforms.RandomHorizontalFlip(
            config.augmentation.random_horizontal_flip.prob)


class RandomAffine(BaseTransform):
    def __init__(self, config: ConfigNode):
        self.transform = torchvision.transforms.RandomAffine(degrees=config.augmentation.random_affine.max_angle,
                                                             translate=(
                                                                 config.augmentation.random_affine.max_horizontal_shift,
                                                                 config.augmentation.random_affine.max_vertical_shift),
                                                             shear=config.augmentation.random_affine.max_shear)


class Resize(BaseTransform):
    def __init__(self, config: ConfigNode):
        self.transform = torchvision.transforms.Resize(config.preprocess.resize)


class RandomColorJitter(BaseTransform):
    def __init__(self, config: ConfigNode) -> None:
        self.transform = torchvision.transforms.ColorJitter(brightness=config.augmentation.random_color.brightness,
                                                            contrast=config.augmentation.random_color.contrast,
                                                            saturation=config.augmentation.random_color.saturation)


class RandomErasing(BaseTransform):
    def __init__(self, config: ConfigNode) -> None:
        self.transform = torchvision.transforms.RandomErasing(p=0.5,
                                                              scale=config.augmentation.random_erasing.scale,
                                                              ratio=config.augmentation.random_erasing.ratio)


class RandomGamma(BaseTransform):
    def __init__(self, config: ConfigNode) -> None:
        self.min, self.max = config.augmentation.gamma.scale

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        gamma = random.uniform(self.min, self.max)
        return torchvision.transforms.functional.adjust_gamma(image, gamma=gamma)


class ExpandChannels:
    """
    Transforms a image with one channel to a an image with
    3 channels by copying pixel intensities of the image along
    the 0 dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return torch.repeat_interleave(data, 3, dim=0)


class AddGaussianNoise:
    def __init__(self, config: ConfigNode) -> None:
        """
        Transformation to add Gaussian noise N(0, std) to
        an image. Where std is set with the config.augmentation.gaussian_noise.std
        argument. The transformation will be applied with probability
        config.augmentation.gaussian_noise.p_apply
        """
        self.std = config.augmentation.gaussian_noise.std
        self.p_apply = config.augmentation.gaussian_noise.p_apply

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if np.random.random(1) > self.p_apply:
            return data
        noise = torch.randn(size=data.shape) * self.std
        data = torch.clamp(data + noise, 0, 1)
        return data


class ElasticTransform:
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

    def __init__(self, config: ConfigNode) -> None:
        self.alpha = config.augmentation.elastic_transform.alpha
        self.sigma = config.augmentation.elastic_transform.sigma
        self.p_apply = config.augmentation.elastic_transform.p_apply

    def __call__(self, image: PIL.Image) -> PIL.Image:
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

    def __init__(self, transforms: Callable):
        self.transforms = transforms

    def __call__(self, sample: PIL.Image.Image) -> Tuple[Any, Any]:
        transform = self.transforms
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


def get_cxr_ssl_transforms(config: ConfigNode, linear_head_module: bool) -> Tuple[Callable, Callable]:
    """
    Applies wrapper around transforms to return two augmented versions of the
    same image
    """
    train_transforms = create_chest_xray_transform(config, is_train=True)
    val_transforms = create_chest_xray_transform(config, is_train=False)
    if linear_head_module:
        return train_transforms, val_transforms
    train_transforms = DualViewTransformWrapper(train_transforms)
    val_transforms = DualViewTransformWrapper(val_transforms)
    return train_transforms, val_transforms


def create_chest_xray_transform(config: ConfigNode,
                                is_train: bool) -> Callable:
    """
    Defines the image transformations pipeline used in Chest-Xray datasets.
    Type of augmentation and strength are defined in the config.
    :param config: config yaml file fixing strength and type of augmentation to apply
    :param is_train: if True return transformation pipeline with augmentations. Else, disable augmentations i.e.
    only resize and center crop the image.
    """
    transforms: List[Any] = []
    if is_train:
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
    Overload lightning-bolts SimCLRTrainDataTransform, to avoid return unused eval transform.
    """

    def __call__(self, sample: Any) -> Tuple[Any, Any]:
        transform = self.train_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class InnerEyeCIFARLinearHeadTransform(SimCLRTrainDataTransform):
    """
    Overload lightning-bolts SimCLRTrainDataTransform, to avoid return unused eval transform.
    """

    def __call__(self, sample: Any) -> Any:
        return self.online_transform(sample)


class InnerEyeCIFARValTransform(SimCLREvalDataTransform):
    def __call__(self, sample: Any) -> Tuple[Any, Any]:
        transform = self.train_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj
