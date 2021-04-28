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
from yacs.config import CfgNode


class BaseTransform:
    def __init__(self, **kwargs: Any) -> None:
        self.transform: Callable = lambda x: NotImplementedError(
            "Transform needs to be overridden in the child classes")

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class CenterCrop(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.transform = torchvision.transforms.CenterCrop(config.preprocess.center_crop_size)


class RandomResizeCrop(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.transform = torchvision.transforms.RandomResizedCrop(
            size=config.preprocess.resize,
            scale=config.augmentation.random_crop.scale)


class RandomHorizontalFlip(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.transform = torchvision.transforms.RandomHorizontalFlip(
            config.augmentation.random_horizontal_flip.prob)


class RandomAffine(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.transform = torchvision.transforms.RandomAffine(degrees=config.augmentation.random_affine.max_angle,
                                                             translate=(
                                                                 config.augmentation.random_affine.max_horizontal_shift,
                                                                 config.augmentation.random_affine.max_vertical_shift),
                                                             shear=config.augmentation.random_affine.max_shear)


class Resize(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.transform = torchvision.transforms.Resize(config.preprocess.resize)


class RandomColorJitter(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.transform = torchvision.transforms.ColorJitter(brightness=config.augmentation.random_color.brightness,
                                                            contrast=config.augmentation.random_color.contrast,
                                                            saturation=config.augmentation.random_color.saturation)


class RandomErasing(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.transform = torchvision.transforms.RandomErasing(p=0.5,
                                                              scale=config.augmentation.random_erasing.scale,
                                                              ratio=config.augmentation.random_erasing.ratio)


class RandomGamma(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()

        def gamma_transform(image: PIL.Image.Image) -> PIL.Image.Image:
            gamma = random.uniform(*config.augmentation.gamma.scale)
            return torchvision.transforms.functional.adjust_gamma(image, gamma=gamma)

        self.transform = gamma_transform


class ExpandChannels(BaseTransform):
    """
    Transforms an image with 1 channel to an image with 3 channels by copying pixel intensities of the image along
    the 0th dimension.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.transform = lambda x: torch.repeat_interleave(x, 3, dim=0)


class AddGaussianNoise(BaseTransform):
    def __init__(self, config: CfgNode) -> None:
        """
        Transformation to add Gaussian noise N(0, std) to an image. Where std is set with the
        config.augmentation.gaussian_noise.std argument. The transformation will be applied with probability
        config.augmentation.gaussian_noise.p_apply
        """
        super().__init__()

        def add_gaussian_noise(data: torch.Tensor) -> torch.Tensor:
            if np.random.random(1) > config.augmentation.gaussian_noise.p_apply:
                return data
            noise = torch.randn(size=data.shape) * config.augmentation.gaussian_noise.std
            data = torch.clamp(data + noise, 0, 1)
            return data

        self.transform = add_gaussian_noise


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
        alpha = config.augmentation.elastic_transform.alpha
        sigma = config.augmentation.elastic_transform.sigma
        p_apply = config.augmentation.elastic_transform.p_apply

        def elastic_transform(image: PIL.Image) -> PIL.Image:
            if np.random.random(1) > p_apply:
                return image
            image = np.asarray(image).squeeze()
            assert len(image.shape) == 2
            shape = image.shape

            dx = gaussian_filter((np.random.random(shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((np.random.random(shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
            return PIL.Image.fromarray(map_coordinates(image, indices, order=1).reshape(shape))

        self.transform = elastic_transform


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


def get_cxr_ssl_transforms(config: CfgNode, return_two_views_per_sample: bool) -> Tuple[Any, Any]:
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
    are called on. If False, simply returned one transformed version of the sample.
    """
    train_transforms = create_chest_xray_transform(config, is_train=True)
    val_transforms = create_chest_xray_transform(config, is_train=False)
    if return_two_views_per_sample:
        return train_transforms, val_transforms
    train_transforms = DualViewTransformWrapper(train_transforms)
    val_transforms = DualViewTransformWrapper(val_transforms)
    return train_transforms, val_transforms


def create_chest_xray_transform(config: CfgNode,
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
