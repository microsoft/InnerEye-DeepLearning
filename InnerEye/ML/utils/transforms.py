#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import abc
from typing import Tuple, Union

import numpy as np
import param
import torch
from monai.transforms import Transform

from InnerEye.Common.generic_parsing import CudaAwareConfig
from InnerEye.Common.type_annotations import T, TupleFloat2


class Transform3DBaseMeta(type(CudaAwareConfig), type(Transform)):  # type: ignore
    """
    Metaclass to make the hierarchy explicit for Transform3D
    """
    pass


class Transform3D(CudaAwareConfig[T], Transform, metaclass=Transform3DBaseMeta):
    """
    Class that allows defining a transform function with the possibility of operating on the GPU.
    """

    @abc.abstractmethod
    def __call__(self, sample: T) -> T:
        raise Exception("__call__ function must be implemented by subclasses")


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
