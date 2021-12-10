#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from contextlib import contextmanager
from typing import Generator, Iterable, Sized, Tuple, Union

import torch
from torch.nn import init

from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.config import PaddingMode

IntOrTuple3 = Union[int, TupleInt3, Iterable]


def initialise_layer_weights(module: torch.nn.Module) -> None:
    """
    Torch kernel initialisations for conv and batch_norm are based on leaky_relu activation.
    Apply Kaiming initialisation and adapt the kernel weights for relu activation.

    :param module: Torch nn module
    """
    if isinstance(module, torch.nn.Conv3d):
        init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    if isinstance(module, torch.nn.BatchNorm3d):
        init.normal_(module.weight, mean=1.0, std=0.01)
        init.constant_(module.bias, val=0.0)


def get_padding_from_kernel_size(padding: PaddingMode,
                                 kernel_size: Union[Iterable[int], int],
                                 dilation: Union[Iterable[int], int] = 1,
                                 num_dimensions: int = 3) -> Tuple[int, ...]:
    """
    Returns padding value required for convolution layers based on input kernel size and dilation.

    :param padding: Padding type (Enum) {`zero`, `no_padding`}. Option `zero` is intended to preserve the tensor shape.
    In `no_padding` option, padding is not applied and the function returns only zeros.
    :param kernel_size: Spatial support of the convolution kernel. It is used to determine the padding size. This can be
    a scalar, tuple or array.
    :param dilation: Dilation of convolution kernel. It is used to determine the padding size. This can be a scalar,
    tuple or array.
    :param num_dimensions: The number of dimensions that the returned padding tuple should have, if both
    kernel_size and dilation are scalars.
    :return padding value required for convolution layers based on input kernel size and dilation.
    """
    if isinstance(kernel_size, Sized):
        num_dimensions = len(kernel_size)
    elif isinstance(dilation, Sized):
        num_dimensions = len(dilation)
    if not isinstance(kernel_size, Iterable):
        kernel_size = [kernel_size] * num_dimensions
    if not isinstance(dilation, Iterable):
        dilation = [dilation] * num_dimensions

    if padding == PaddingMode.NoPadding:
        return tuple([0] * num_dimensions)
    else:
        out = [0 if k == 1 else (k - 1) // 2 + d - 1 for k, d in zip(kernel_size, dilation)]
        return tuple(out)


def get_upsampling_kernel_size(downsampling_factor: IntOrTuple3, num_dimensions: int) -> TupleInt3:
    """
    Returns the kernel size that should be used in the transpose convolution in the decoding blocks of the UNet.
    Use a value that is a multiple of the downsampling factor to avoid checkerboard artefacts, see
    https://distill.pub/2016/deconv-checkerboard/

    :param downsampling_factor: downsampling factor use for each dimension of the kernel. Can be
    either a list of len(num_dimension) with one factor per dimension or an int in which case the
    same factor will be applied for all dimension.
    :param num_dimensions: number of dimensions of the kernel
    :return: upsampling_kernel_size
    """

    def upsample_size(down: int) -> int:
        return 2 * down if down > 1 else 1

    if isinstance(downsampling_factor, int):
        downsampling_factor = (downsampling_factor,) * num_dimensions
    if any(down < 1 for down in downsampling_factor):
        raise ValueError(f"The downsampling_factor must be >= 1 in each component, but got: {downsampling_factor}")
    if len(downsampling_factor) != num_dimensions:  # type: ignore
        raise ValueError(f"The downsampling_factor must be an integer or an Iterable of length 3, but got: "
                         f"{downsampling_factor}")
    # Writing this as a tuple so that the type checker knows we really return a 3-tuple.
    upsampling_kernel_size = (upsample_size(downsampling_factor[0]),  # type: ignore
                              upsample_size(downsampling_factor[1]),  # type: ignore
                              upsample_size(downsampling_factor[2]))  # type: ignore
    return upsampling_kernel_size


@contextmanager
def set_model_to_eval_mode(model: torch.nn.Module) -> Generator:
    """
    Puts the given torch model into eval mode. At the end of the context, resets the state of the training flag to
    what is was before the call.
    :param model: The model to modify.
    """
    old_mode = model.training
    model.eval()
    yield
    model.train(old_mode)
