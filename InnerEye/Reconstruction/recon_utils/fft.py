#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Tuple, Union

import torch

FftDimensions = Union[int, Tuple[int, ...]]


def fft(data: torch.Tensor, dim: FftDimensions = None) -> torch.Tensor:
    """ 
    Computes the Fourier transform along specified dimensions.

    :param data: image data as complex torch.Tensor
    :param dim: dimensions to transform
    :returns: transformed data
    """

    dim = validate_fft_dimensions(len(data.shape), dim)

    data = ifftshift(data, dim)

    # Permute, ifft, unpermute
    permute_dim, unpermute_dim = get_fft_permute_dims(len(data.shape), dim)
    data = data.permute(permute_dim)
    data = torch.view_as_complex(torch.fft(torch.view_as_real(data), len(dim)))
    data = data.permute(unpermute_dim)

    data = fftshift(data, dim)

    # Scaling
    data *= 1 / torch.sqrt(torch.prod(torch.Tensor([data.shape[d] for d in dim])))

    return data


def ifft(data: torch.Tensor, dim: FftDimensions = None) -> torch.Tensor:
    """ 
    Computes the inverse Fourier transform along specified dimensions.

    :param data: k-space data as complex torch.Tensor
    :param dim: dimensions to transform
    :returns: transformed data
    """

    dim = validate_fft_dimensions(len(data.shape), dim)

    data = ifftshift(data, dim)

    # Permute, ifft, unpermute
    permute_dim, unpermute_dim = get_fft_permute_dims(len(data.shape), dim)
    data = data.permute(permute_dim)
    data = torch.view_as_complex(torch.ifft(torch.view_as_real(data), len(dim)))
    data = data.permute(unpermute_dim)

    data = fftshift(data, dim)

    # Scaling
    data *= torch.sqrt(torch.prod(torch.Tensor([data.shape[d] for d in dim])))

    return data


def fftshift(data: torch.Tensor, dim: Tuple[int, ...]) -> torch.Tensor:
    """
    Shift the zero-frequency component to the center.
    See: https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html

    PyTorch does not have an fftshift function so we implement it using roll

    :param data: torch.Tensor
    :param dim: tuple of dimensions to shift
    :returns torch.Tensor shifted along dimensions
    """

    shifts = [data.shape[x] // 2 for x in dim]
    return torch.roll(data, shifts, dim)


def ifftshift(data: torch.Tensor, dim: Tuple[int, ...]) -> torch.Tensor:
    """
    Shift the zero-frequency component to the center.
    See: https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftshift.html

    PyTorch does not have an ifftshift function so we implement it using roll.
    Note, it is identical to fftshift except for odd-length dimensions

    :param data: torch.Tensor
    :param dim: tuple of dimensions to shift
    :returns torch.Tensor shifted along dimensions
    """

    shifts = [(data.shape[x] + 1) // 2 for x in dim]
    return torch.roll(data, shifts, dim)


def validate_fft_dimensions(datandim: int, dim: FftDimensions = None) -> Tuple[int, ...]:
    """
    Checks that requested FFT dims are valid and adjusts accordingly.

    :param datandim: number of dimensions in data tensor
    :param dim: FFT dimensions
    :returns validated and adjusted array.
    """

    if not dim:
        dim = tuple(range(datandim))
        # Since torch can only do up to 3 dimensions, we will pick the last 3
        if len(dim) > 3:
            dim = dim[-3:]

    if isinstance(dim, int):
        dim = (dim,)

    # Torch ffts only support 1, 2, or 3 dimensions
    assert len(dim) <= 3 and len(dim) >= 1
    return dim


def get_fft_permute_dims(ndim: int, transform_dim: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Helper function for determiming permute dimensions for FFT/IFFT.

    Returns permutation orders needed for moving the transform dimensions `transform_dim` to the highest dimensions
    and back.

    :param ndim: total number of dimensions
    :param transform_dim: dimensions to transform
    :return permute_dim,unpermute_dim: tuples needed to permute and unpermute

    """
    dd = [d % ndim for d in transform_dim]
    permute_dims = []
    for d in range(ndim):
        if d not in dd:
            permute_dims.append(d)
    for d in dd:
        permute_dims.append(d % ndim)

    permute_dims_tuple = tuple(permute_dims)

    unpermute_dims = [0 for _ in range(ndim)]
    for i, d in enumerate(permute_dims):
        unpermute_dims[d] = i
    unpermute_dims_tuple = tuple(unpermute_dims)

    return permute_dims_tuple, unpermute_dims_tuple
