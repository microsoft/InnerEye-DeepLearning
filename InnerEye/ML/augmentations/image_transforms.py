#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
from abc import abstractmethod
from typing import Any, List, Tuple, Union

import PIL
import numpy as np
import torch
import torchvision
from scipy.ndimage import gaussian_filter, map_coordinates
from torchvision.transforms import functional as F

class RandomGamma():

    def __init__(self, scale: Tuple[float, float]) -> None:
        self.scale = scale

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        gamma = random.uniform(*self.scale)
        if len(image.shape)!= 4:
            raise ValueError(f"Expected input of shape [C, Z, H, W], but only got {len(image.shape)} dimensions")
        for c, z in zip(range(image.shape[0]), range(image.shape[1])):
            image[c, z] = torchvision.transforms.functional.adjust_gamma(image[c, z], gamma=gamma)
        return image


class ExpandChannels:
    """
    Transforms an image with 1 channel to an image with 3 channels by copying pixel intensities of the image along
    the 1st dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param: data [Z, C, H, W]
        """
        shape = data.shape
        if len(shape) != 4 or shape[1] != 1:
             raise ValueError(f"Expected input of shape [Z, 1, H, W], found {shape}")
        return torch.repeat_interleave(data, 3, dim=1)


class AddGaussianNoise:

    def __init__(self, p_apply: float, std: float) -> None:
        """
        Transformation to add Gaussian noise N(0, std) to an image. Where std is set with the
        config.augmentation.gaussian_noise.std argument. The transformation will be applied with probability
        config.augmentation.gaussian_noise.p_apply
        """
        super().__init__()
        self.p_apply = p_apply
        self.std = std

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if np.random.random(1) > self.p_apply:
            return data
        noise = torch.randn(size=data.shape[-2:]) * self.std
        data = torch.clamp(data + noise, data.min(), data.max())
        return data


class ElasticTransform:
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.160.8494&rep=rep1&type=pdf
    """

    def __init__(self,
                 sigma: float,
                 alpha: float,
                 p_apply: float
                 ) -> None:
        """
        :param sigma: elasticity coefficient
        :param alpha: intensity of the deformation
        :param p_apply: probability of applying the transformation
        """
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.p_apply = p_apply

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if np.random.random(1) > self.p_apply:
            return data
        result_type = data.dtype
        data = data.cpu().numpy()
        shape = data.shape

        dx = gaussian_filter((np.random.random(shape[-2:]) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((np.random.random(shape[-2:]) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        all_dimensions_axes = [np.arange(dim) for dim in shape]
        grid = np.meshgrid(*all_dimensions_axes, indexing='ij')
        grid[-2] = grid[-2] + dx
        grid[-1] = grid[-1] + dy
        indices = [np.reshape(grid[i], (-1, 1)) for i in range(len(grid))]

        return torch.tensor(map_coordinates(data, indices, order=1).reshape(shape), dtype=result_type)
