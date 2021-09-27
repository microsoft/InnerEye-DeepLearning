#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import random
from typing import Any, Callable, Tuple

import PIL
import PIL.Image
import numpy as np
import torch
import torchvision
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.filters import rank
from skimage.morphology import disk

from InnerEyeDataQuality.configs.config_node import ConfigNode


class BaseTransform:
    def __init__(self, config: ConfigNode):
        self.transform = lambda x: x

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class Standardize:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image: PIL.Image.Image) -> np.ndarray:
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - self.mean) / self.std
        return image


class CenterCrop(BaseTransform):
    def __init__(self, config: ConfigNode):
        self.transform = torchvision.transforms.CenterCrop(config.preprocess.center_crop_size)


class RandomCrop(BaseTransform):
    def __init__(self, config: ConfigNode):
        self.transform = torchvision.transforms.RandomCrop(
            config.dataset.image_size,
            padding=config.augmentation.random_crop.padding,
            fill=config.augmentation.random_crop.fill,
            padding_mode=config.augmentation.random_crop.padding_mode)


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
        self.transform = torchvision.transforms.RandomAffine(degrees=config.augmentation.random_affine.max_angle,  # 15
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


class HistogramNormalization:
    def __init__(self, config: ConfigNode) -> None:
        self.disk_size = config.preprocess.histogram_normalization.disk_size

    def __call__(self, image: PIL.Image.Image) -> np.ndarray:
        # Apply local histogram equalization
        image = np.array(image)
        return PIL.Image.fromarray(rank.equalize(image, selem=disk(self.disk_size)))


class ExpandChannels:
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return torch.repeat_interleave(data, 3, dim=0)


class ToNumpy:
    def __call__(self, image: PIL.Image.Image) -> np.ndarray:
        return np.array(image)


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
    def __init__(self, transforms: Callable):
        self.transforms = transforms

    def __call__(self, sample: PIL.Image.Image) -> Tuple[Any, Any]:
        transform = self.transforms
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj
