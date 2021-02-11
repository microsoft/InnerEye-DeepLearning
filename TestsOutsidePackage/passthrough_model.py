#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np


def make_distance_range(length: int) -> np.array:
    """
    Create a numpy array of ints of shape (length,) where each item is the distance from the centre.

    If length is odd, then let hl=(length-1)/2, then the result is:
    [hl, hl-1,..., 1, 0, 1, ... hl-1, hl]
    If length is even, then let hl=(length/2)-1, then the result is:
    [hl, hl-1,..., 1, 0, 0, 1, ... hl-1, hl]
    More concretely:
    For length=7, then the result is [3, 2, 1, 0, 1, 2, 3]
    For length=8, then the result is [3, 2, 1, 0, 0, 1, 2, 3]

    :param length: Size of array to return.
    :return: Array of distances from the centre
    """
    return abs(np.arange(1 - length, length + 1, 2)) // 2


def make_stroke_rectangle(dim0: int, dim1: int, half_side: int) -> np.array:
    """
    Create a stroke rectangle within a rectangle.

    Create a numpy array of shape (dim0, dim1), that is 0 except for an unfilled
    rectangle of 1s centred about the centre of the array.

    :param dim0: Target array dim0.
    :param dim1: Target array dim1.
    :param half_side: Rough rectangle half side length.
    :return: np.array mostly 0s apart from the path of a rectangle.
    """
    x1 = make_distance_range(dim0)
    x2 = make_distance_range(dim1)
    X1, X2 = np.meshgrid(x1, x2, sparse=False, indexing='ij')
    return ((X1 == half_side - 1) & (X2 < half_side)
            | (X1 < half_side) & (X2 == half_side - 1)) * 1


def make_fill_rectangle(dim0: int, dim1: int, half_side: int, invert: bool) -> np.array:
    """
    Create a filled rectangle within a rectangle.

    Create a numpy array of shape (dim0, dim1) that is background except for a filled
    foreground rectangle centred about the centre of the array.
    If dim0 is odd then the length in axis 0 will be 2*half_side - 1, otherwise it will be
        length 2*half_side.
    Similarly for dim1.

    :param dim0: Target array dim0.
    :param dim1: Target array dim1.
    :param half_side: Rough rectangle half side length.
    :param invert: If False then background is 0, foreground 1. If True then v.v.
    :return: np.array mostly background apart from the foreground rectangle.
    """
    x1 = make_distance_range(dim0)
    x2 = make_distance_range(dim1)
    X1, X2 = np.meshgrid(x1, x2, sparse=False, indexing='ij')
    grid = ((X1 < half_side) & (X2 < half_side)) * 1

    return grid if not invert else 1 - grid


def make_nesting_rectangles(dim0: int, dim1: int, num_features: int) -> np.array:
    """
    Create a np.array of shape (num_features, dim0, dim1) of nesting rectangles.

    The first slice is intended to be a background, the remaining slices are
    consecutively smaller rectanges, none overlapping.

    :param dim0: Target array dim0.
    :param dim1: Target array dim1.
    :param num_features: Number of rectangles.
    :return: np.array of background then a set of rectangles.
    """
    nesting = np.empty((num_features, dim0, dim1), dtype=np.int64)
    nesting[0::] = make_fill_rectangle(dim0, dim1, num_features - 1, True)

    for feature in range(1, num_features):
        nesting[feature::] = make_stroke_rectangle(dim0, dim1, num_features - feature)

    return nesting
