#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import abc
from functools import reduce
from typing import Any, Generic, List, Optional, Tuple, Union

import numpy as np
import param
import torch

from InnerEye.Common.common_util import is_gpu_tensor
from InnerEye.Common.type_annotations import T, TupleFloat2


class Transform3D(param.Parameterized, Generic[T]):
    """
    Class that allows defining a transform function with the possibility of operating on the GPU.
    """
    use_gpu: bool = param.Boolean(False, doc="The use_gpu flag will be "
                                             "set based upon the available GPU devices.")

    def get_gpu_tensor_if_possible(self, data: T) -> Any:
        """"
        Get a cuda tensor if this transform was CUDA enabled and a GPU is available, otherwise
        return the input.
        """
        import torch
        if isinstance(data, torch.Tensor):
            if self.use_gpu and not is_gpu_tensor(data):
                return data.cuda()
            else:
                return data
        else:
            return data

    @abc.abstractmethod
    def __call__(self, sample: T) -> T:
        raise Exception("__call__ function must be implemented by subclasses")


class Compose3D(Generic[T]):
    """
    Class that allows chaining multiple transform functions together, and applying them to a sample
    """

    def __init__(self, transforms: List[Transform3D[T]]):
        self._transforms = transforms

    def __call__(self, sample: T) -> T:
        # pythonic implementation of the foldl function
        # foldl (-) 0 [1,2,3] => (((0 - 1) - 2) - 3) => -6
        return reduce(lambda x, f: f(x), self._transforms, sample)

    @staticmethod
    def apply(compose: Optional[Compose3D[T]], sample: T) -> T:
        """
        Apply a composition of transfer functions to the provided sample

        :param compose: A composition of transfer functions
        :param sample: The sample to apply the composition on
        :return:
        """
        if compose:
            return compose(sample)
        else:
            return sample


class CTRange(Transform3D[Union[torch.Tensor, np.ndarray]]):
    output_range: TupleFloat2 = param.NumericTuple(default=(0.0, 255.0), length=2,
                                                   doc="Desired output range of intensities")
    window: float = param.Number(None, doc="Width of window")
    level: float = param.Number(None, doc="Mid-point of window")

    def __call__(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        return LinearTransform.transform(
            data=self.get_gpu_tensor_if_possible(data),
            input_range=get_range_for_window_level(self.level, self.window),
            output_range=self.output_range,
            use_gpu=self.use_gpu
        )

    @staticmethod
    def transform(data: Union[torch.Tensor, np.ndarray],
                  output_range: TupleFloat2,
                  window: float, level: float,
                  use_gpu: bool = False) -> Union[torch.Tensor, np.ndarray]:
        # find upper and lower values of input range to linearly map to output range. Values outside range are
        # floored and capped at min or max of range.
        transform = CTRange(output_range=output_range, window=window, level=level, use_gpu=use_gpu)
        return transform(data)


class LinearTransform(Transform3D[Union[torch.Tensor, np.ndarray]]):
    input_range: TupleFloat2 = param.NumericTuple(None, length=2, doc="Expected input range of intensities")
    output_range: TupleFloat2 = param.NumericTuple(None, length=2, doc="Desired output range of intensities")

    def __call__(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        data = self.get_gpu_tensor_if_possible(data)
        gradient = (self.output_range[1] - self.output_range[0]) / (self.input_range[1] - self.input_range[0])
        c = self.output_range[1] - gradient * self.input_range[1]

        _apply_transform = lambda: data * gradient + c

        if torch.is_tensor(data):
            gradient = self.get_gpu_tensor_if_possible(torch.tensor(gradient))
            c = self.get_gpu_tensor_if_possible(torch.tensor(c))
            return _apply_transform().clamp(min=self.output_range[0], max=self.output_range[1])
        else:
            return np.clip(_apply_transform(), a_min=self.output_range[0], a_max=self.output_range[1])

    @staticmethod
    def transform(data: Union[torch.Tensor, np.ndarray],
                  input_range: TupleFloat2, output_range: TupleFloat2,
                  use_gpu: bool = False) -> Union[torch.Tensor, np.ndarray]:
        transform = LinearTransform(use_gpu=use_gpu, input_range=input_range, output_range=output_range)
        return transform(data)


def get_range_for_window_level(level: float, window: float) -> Tuple[float, float]:
    upper = level + window / 2
    lower = level - window / 2
    return lower, upper
