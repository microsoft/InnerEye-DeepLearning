#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import random
from typing import Tuple, Union

import torch


def generate_cartesian_sampling_pattern(dimensions: Tuple[int, int],
                                        acceleration_factor: Union[float, int] = 1.0,
                                        use_random_sampling: bool = True,
                                        centerlines: int = 8,
                                        seed: int = None) -> torch.Tensor:
    """
    Creates a sampling pattern for undersampling of Cartesian MRI data.

    The sampling pattern generator treats the first dimension as the ky (or phase encoding) dimensions
    and selects which lines to sample. The undersampling can either be random (a random set of phase encodings) or
    equispaced. For equispaced (regular) undersampling, the acceleration factor must be an integer and the first
    dimension must be a multiple of the acceleration factor.

    :param dimensions: tuple (size_ky, size_kx)
    :param acceleration_factor: Acceleration factor, 1.0 is fully sampled, 2.0 is half the data, etc.
    :param use_random_sampling: boolean
    :param centerlines: fully sampled lines in the center
    :param seed: random seed for random lines
    :returns PyTorch tensor with sampling pattern with size ``dimensions``
    """

    if use_random_sampling is False:
        assert isinstance(acceleration_factor, int)
        assert dimensions[0] % acceleration_factor == 0

    if use_random_sampling is False:
        lines = torch.Tensor([x % int(acceleration_factor) == 0 for x in range(dimensions[0])])
    else:
        if seed:
            random.seed(seed)
        number_of_lines = int(dimensions[0] / acceleration_factor)
        lines = torch.zeros(dimensions[0])
        idx = list(range(dimensions[0]))
        random.shuffle(idx)
        lines[idx[:number_of_lines]] = 1

    if centerlines > 0:
        leftmargin = dimensions[0] // 2 - centerlines // 2
        rightmargin = leftmargin + centerlines
        lines[leftmargin:rightmargin] = 1

    sp = torch.zeros(dimensions)
    sp[lines > 0, :] = 1

    return sp
