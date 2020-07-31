#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import abc
from abc import ABC
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from InnerEye.Common.common_util import any_pairwise_larger, initialize_instance_variables
from InnerEye.Common.type_annotations import IntOrTuple3, TupleInt2, TupleInt3
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.ML.visualizers.model_summary import ModelSummary, forward_preserve_state


class CropSizeConstraints:
    def __init__(self, multiple_of: Optional[IntOrTuple3] = None,
                 minimum_size: Optional[IntOrTuple3] = None,
                 num_dimensions: int = 3):
        """
        :param multiple_of: Stores minimum size and other conditions that a training crop size must satisfy.
        :param minimum_size: Training crops must have a size that is a multiple of this value, along each dimension.
        For example, if set to (1, 16, 16), the crop size has to be a multiple of 16 along X and Y, and a
        multiple of 1 (i.e., any number) along the Z dimension.
        :param num_dimensions: Training crops must have a size that is at least this value.
        """
        self.multiple_of = multiple_of
        self.minimum_size = minimum_size
        self.num_dimensions = num_dimensions

        def make_tuple3(o: Optional[IntOrTuple3]) -> Optional[TupleInt3]:
            # "type ignore" directives below are because mypy is not clever enough
            if o is None:
                return None
            if isinstance(o, int):
                # noinspection PyTypeChecker
                return (o,) * self.num_dimensions  # type: ignore
            if len(o) != self.num_dimensions:  # type: ignore
                raise ValueError("Object must have length {}, but got: {}"
                                 .format(self.num_dimensions, o))
            return o  # type: ignore

        self.multiple_of = make_tuple3(self.multiple_of)
        self.minimum_size = make_tuple3(self.minimum_size)
        if self.minimum_size is None:
            self.minimum_size = self.multiple_of
        else:
            if self.multiple_of is not None and any_pairwise_larger(self.multiple_of, self.minimum_size):
                raise ValueError(f"Invalid arguments: The minimum size must be at least as large as the multiple_of. "
                                 f"minimum_size: {self.minimum_size}, multiple_of: {self.multiple_of}")

    def validate(self, crop_size: TupleInt3, message_prefix: Optional[str] = None) -> None:
        """
        Checks if the given crop size is a valid crop size for the present model.
        If it is not valid, throw a ValueError.
        :param crop_size: The crop size that should be checked.
        :param message_prefix: A string prefix for the error message if the crop size is found to be invalid.
        :return:
        """
        message_prefix = message_prefix + ": " if message_prefix else ""
        if len(crop_size) != self.num_dimensions:
            raise ValueError(f"{message_prefix}Crop size must have length {self.num_dimensions}, but got: {crop_size}")
        if self.minimum_size is not None:
            assert not isinstance(self.minimum_size, int)
            if any_pairwise_larger(self.minimum_size, crop_size):  # type: ignore
                raise ValueError(f"{message_prefix}Crop size is not valid. The required minimum is {self.minimum_size},"
                                 f" but got: {crop_size}")
        if self.multiple_of is not None:
            assert not isinstance(self.multiple_of, int)
            if any(crop % mult != 0 for (crop, mult) in zip(crop_size, self.multiple_of)):
                raise ValueError(f"{message_prefix}Crop size is not valid. Crop size is should be a multiple of "
                                 f"{self.multiple_of}, but got: {crop_size}")

    def restrict_crop_size_to_image(self,
                                    image_shape: TupleInt3,
                                    crop_size: TupleInt3,
                                    stride_size: TupleInt3) -> Tuple[TupleInt3, TupleInt3]:
        """
        Computes an adjusted crop and stride size for cases where the image is smaller than the chosen crop size
        (at test time). The new crop size will be the largest multiple of self.multiple_of that fits into the
        image_shape.
        The stride size will attempt to maintain the stride-to-crop ratio before adjustment.
        :param image_shape: The shape of the image to process.
        :param crop_size: The present test crop size.
        :param stride_size: The present inference stride size.
        :return: A tuple of (crop_size, stride_size)
        """
        shape = np.array(image_shape)
        crop = np.array(crop_size)
        stride = np.array(stride_size)
        multiple_of = np.array(self.multiple_of)
        minimum = np.array(self.minimum_size)
        if np.any(shape < minimum):
            raise ValueError("The input image must have at least a size of {}, but got: {}"
                             .format(self.minimum_size, image_shape))
        if np.all(shape >= crop):
            return crop_size, stride_size
        stride_to_crop = stride / crop
        crop_new = np.ceil(np.minimum(crop, shape) / multiple_of) * multiple_of
        stride_new = np.maximum(np.floor(stride_to_crop * crop_new), 1)

        def to_tuple(a: np.ndarray) -> TupleInt3:
            return int(a[0]), int(a[1]), int(a[2])

        return to_tuple(crop_new), to_tuple(stride_new)


class BaseModel(DeviceAwareModule, ABC):
    """
    Base neural network segmentation model.
    """

    @initialize_instance_variables
    def __init__(self,
                 name: str,
                 input_channels: int,
                 crop_size_constraints: Optional[CropSizeConstraints] = None
                 ):
        """
        Creates a new instance of the base model class.
        :param name: A human readable name of the model.
        :param input_channels: The number of image input channels.
        :param crop_size_constraints: The size constraints for the training crop size. If not provided,
        a minimum crop size of 1 is assumed.
        """
        super().__init__()
        self.num_dimensions = 3
        self.name = name
        self.input_channels = input_channels
        self.summarizer: Optional[ModelSummary] = None
        self.summary: Optional[OrderedDict] = None
        self.summary_crop_size: Optional[TupleInt3] = None
        if crop_size_constraints is None:
            # Allow any size. With this initialization, both multiple_of and minimum_size will be populated.
            crop_size_constraints = CropSizeConstraints(multiple_of=1)
        self.crop_size_constraints = crop_size_constraints

    def get_output_shape(self, input_shape: Union[TupleInt2, TupleInt3]) -> Tuple[int, ...]:
        """
        Computes model's output tensor shape for given input tensor shape.
        The argument is expected to be either a 2-tuple or a 3-tuple. A batch dimension (1)
        and the number of channels are added as the first dimensions. The result tuple has batch and channel dimension
        stripped off.
        :param input_shape: A tuple (2D or 3D) representing incoming tensor shape.
        """
        # Create a sample tensor for inference
        batch_size = 1
        if len(input_shape) not in [2, 3]:
            raise ValueError("Input shape has to be in 2D or 3D, found {}".format(len(input_shape)))
        input_tensors = \
            [torch.zeros(batch_size, self.input_channels, *input_shape, dtype=torch.float)]

        # Perform a forward pass then restore the state of the module
        output_shape = forward_preserve_state(module=self, inputs=input_tensors).size()
        return tuple(output_shape[2:])

    def partition_model(self, devices: List[torch.device]) -> None:
        """An abstract method to partition a neural network model and map them across multiple devices"""
        raise NotImplementedError(f"Model partitioning is not implemented for '{self.name}'")

    def validate_crop_size(self, crop_size: TupleInt3, message_prefix: Optional[str] = None) -> None:
        """
        Checks if the given crop size is a valid crop size for the present model.
        If it is not valid, throw a ValueError.
        :param crop_size: The crop size that should be checked.
        :param message_prefix: A string prefix for the error message if the crop size is found to be invalid.
        """
        if self.crop_size_constraints is not None:
            self.crop_size_constraints.validate(crop_size, message_prefix)

    def generate_model_summary(self, crop_size: Optional[TupleInt3] = None,
                               log_models_to_files: bool = False) -> None:
        """
        Stores a model summary, containing information about layers, memory consumption and runtime
        in the model.summary field.
        When called again with the same crop_size, the summary is not created again.
        :param crop_size: The crop size for which the summary should be created. If not provided,
        the minimum allowed crop size is used.
        :param log_models_to_files: whether to write the summary to a file
        """
        if crop_size is None:
            crop_size = self.crop_size_constraints.minimum_size  # type: ignore
            assert crop_size is not None
        input_size = [crop_size]
        if self.summary is None or self.summary_crop_size != input_size:
            self.summarizer = ModelSummary(self)
            self.summary = self.summarizer.generate_summary(
                input_sizes=[(self.input_channels, *crop_size)],
                log_models_to_files=log_models_to_files)
            self.summary_crop_size = crop_size

    @abc.abstractmethod
    def forward(self, input: Any) -> Any:  # type: ignore
        raise NotImplementedError("forward must be implemented by subclasses")

    @abc.abstractmethod
    def get_all_child_layers(self) -> List[torch.nn.Module]:
        raise NotImplementedError("get_all_child_layers must be implemented by subclasses")
