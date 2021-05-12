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


class ImageTransformBase:
    def __init__(self, *args, **kwargs):
        pass

    def draw_transform(self, input_size: List[int]) -> List[int]:
        """
        This function should implement how to shuffle the transform_pipeline parameters

        Example: for random rotation the max angle is fixed during init of the class.
        Each call the draw_next_transform should sample a rotation angle within the [-max_angle, max_angle] interval.

        Note: We are using an explicit call to shuffle the transform_pipeline parameters to ensure that we can use
        the same
        transform_pipeline for all 2D slices in a given 3D volume.
        """
        return input_size

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """
        Should implement the transformation itself given the currently set transformation parameters.
        """
        pass


class CenterCrop(ImageTransformBase):
    def __init__(self, center_crop_size: Union[Tuple[int, int], int]) -> None:
        super().__init__()
        self.center_crop_size = center_crop_size

    def draw_transform(self, input_size: List[int]) -> List[int]:
        if isinstance(self.center_crop_size, int):
            return [input_size[0], self.center_crop_size, self.center_crop_size]
        return [input_size[0], self.center_crop_size[0], self.center_crop_size[1]]

    def __call__(self, image: PIL.Image.Image) -> PIL.Image:
        return torchvision.transforms.CenterCrop(self.center_crop_size)(image)


class RandomResizeCrop(ImageTransformBase):
    def __init__(self,
                 random_crop_scale: Tuple[float, float],
                 resize_size: Union[int, Tuple[int, int]]) -> None:
        super().__init__()
        self.size = resize_size
        self._transform_generator = torchvision.transforms.RandomResizedCrop(
            size=self.size,
            scale=random_crop_scale)

    def draw_transform(self, input_size: List[int]) -> List[int]:
        self.params = self._transform_generator.get_params(torch.zeros(input_size),
                                                           self._transform_generator.scale,
                                                           self._transform_generator.ratio)
        size = F._get_image_size(F.resized_crop(
            PIL.Image.fromarray(np.ones(input_size[1:])), *self.params, self.size))
        return [input_size[0], size[1], size[0]]  # Returns [C, new_width, new_height]

    def __call__(self, image: PIL.Image.Image) -> Any:
        return F.resized_crop(image, *self.params, self.size)


class RandomAffine(ImageTransformBase):
    def __init__(self,
                 max_angle: int = 0,
                 max_horizontal_shift: float = 0.0,
                 max_vertical_shift: float = 0.0,
                 max_shear: int = 0.0) -> None:
        super().__init__()
        self.max_angle = max_angle
        self.max_horizontal_shift = max_horizontal_shift
        self.max_vertical_shift = max_vertical_shift
        self.max_shear = max_shear
        self._transform_generator = torchvision.transforms.RandomAffine(degrees=self.max_angle,
                                                                        translate=(self.max_horizontal_shift,
                                                                                   self.max_vertical_shift),
                                                                        shear=self.max_shear)

    def draw_transform(self, input_size: List[int]) -> List[int]:
        self._current_params = self._transform_generator.get_params(self._transform_generator.degrees,
                                                                    self._transform_generator.translate,
                                                                    self._transform_generator.scale,
                                                                    self._transform_generator.shear,
                                                                    input_size)
        return input_size

    def __call__(self, img: PIL.Image.Image) -> Any:
        return F.affine(img, *self._current_params, fill=0)


class RandomHorizontalFlip(ImageTransformBase):
    def __init__(self, p_apply: float) -> None:
        super().__init__()
        self.p_apply = p_apply

    def draw_transform(self, input_size: List[int]) -> List[int]:
        self.apply_flip = torch.rand(1).data < self.p_apply
        return input_size

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        if self.apply_flip:
            return F.hflip(img)
        return img


class Resize(ImageTransformBase):
    def __init__(self, resize_size: Union[Tuple[int, int], int]) -> None:
        super().__init__()
        self.resize_size = resize_size

    def draw_transform(self, input_size: List[int]) -> List[int]:
        if isinstance(self.resize_size, Tuple):
            return [input_size[0], self.resize_size[0], self.resize_size[1]]
        return [input_size[0], self.resize_size, self.resize_size]

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return torchvision.transforms.Resize(self.resize_size)(img)


class RandomColorJitter(ImageTransformBase):
    def __init__(self,
                 max_brightness: float = 0.0,
                 max_contrast: float = 0.0,
                 max_saturation: float = 0.0
                 ) -> None:
        super().__init__()
        self._transform_generator = torchvision.transforms.ColorJitter(
            brightness=max_brightness,
            contrast=max_contrast,
            saturation=max_saturation)

    def draw_transform(self, input_size: List[int]) -> List[int]:
        self.params = self._transform_generator.get_params(self._transform_generator.brightness,
                                                           self._transform_generator.contrast,
                                                           self._transform_generator.saturation,
                                                           None)
        return input_size

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.params
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
        return img


class RandomErasing(ImageTransformBase):
    def __init__(self,
                 scale: Tuple[float, float],
                 ratio: Tuple[float, float]
                 ) -> None:
        super().__init__()
        self.scale = scale
        self.ratio = ratio
        self.p_apply = 0.5
        self._transform_generator = torchvision.transforms.RandomErasing(scale=self.scale, ratio=self.ratio)

    def draw_transform(self, input_size: List[int]) -> List[int]:
        self.params = self._transform_generator.get_params(torch.zeros(input_size),
                                                           scale=self.scale,
                                                           ratio=self.ratio,
                                                           value=[0, ])
        self.apply = torch.rand(1).data < self.p_apply
        return input_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.apply:
            return F.erase(img, *self.params)
        return img


class RandomGamma(ImageTransformBase):

    def __init__(self, scale: Tuple[float, float]) -> None:
        super().__init__()
        self.scale = scale

    def draw_transform(self, input_size: List[int]) -> List[int]:
        self.gamma = random.uniform(*self.scale)
        return input_size

    def __call__(self, image: PIL.Image) -> PIL.Image:
        return F.adjust_gamma(image, gamma=self.gamma)


class ElasticTransform(ImageTransformBase):
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

    def __init__(self,
                 sigma: float,
                 alpha: float,
                 p_apply: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.p_apply = p_apply

    def draw_transform(self, input_size: List[int]) -> List[int]:
        self.dx_pertubation = np.random.random(input_size) * 2 - 1
        self.dy_pertubation = np.random.random(input_size) * 2 - 1
        self.apply = np.random.random(1) < self.p_apply
        return input_size

    def __call__(self, image: PIL.Image) -> PIL.Image:
        if self.apply:
            image = np.asarray(image).squeeze()
            assert len(image.shape) == 2
            shape = image.shape
            dx = gaussian_filter(self.dx_pertubation, self.sigma, mode="constant", cval=0) * self.alpha
            dy = gaussian_filter(self.dy_pertubation, self.sigma, mode="constant", cval=0) * self.alpha
            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
            return PIL.Image.fromarray(map_coordinates(image, indices, order=1).reshape(shape))
        return image


class ToTensor(ImageTransformBase):
    def __call__(self, image: PIL.Image.Image) -> torch.Tensor:
        tensor_data = torchvision.transforms.ToTensor()(image)
        if len(tensor_data) == 2:
            tensor_data = tensor_data.unsqueeze(0)
        return tensor_data


class ExpandChannels(ImageTransformBase):
    """
    Transforms an image with 1 channel to an image with 3 channels by copying pixel intensities of the image along
    the 0th dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return torch.repeat_interleave(data, 3, dim=0)


class AddGaussianNoise(ImageTransformBase):

    def __init__(self,
                 p_apply: float,
                 std: float,
                 ) -> None:
        """
        Transformation to add Gaussian noise N(0, std) to an image. Where std is set with the
        config.augmentation.gaussian_noise.std argument. The transformation will be applied with probability
        config.augmentation.gaussian_noise.p_apply
        """
        super().__init__()
        self.p_apply = p_apply
        self.std = std

    def draw_transform(self, input_size: List[int]) -> List[int]:
        self.apply = torch.rand(1).data < self.p_apply
        self.noise = torch.randn(size=input_size) * self.std
        return input_size

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        assert data.max() <= 1 and data.min() >= 0
        if self.apply:
            data = torch.clamp(data + self.noise, 0, 1)
        return data
