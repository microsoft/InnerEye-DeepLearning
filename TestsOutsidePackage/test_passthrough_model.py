#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import List, Tuple
import pytest
import numpy as np
from passthrough_model import make_distance_range, make_stroke_rectangle, \
    make_fill_rectangle, make_nesting_rectangles


# Test data for test_make_distance_range
make_distance_range_test_data: List[List[int]] = [
    [],
    [0],
    [0, 0],
    [1, 0, 1],
    [1, 0, 0, 1],
    [2, 1, 0, 1, 2],
    [2, 1, 0, 0, 1, 2],
    [3, 2, 1, 0, 1, 2, 3],
    [3, 2, 1, 0, 0, 1, 2, 3],
    [4, 3, 2, 1, 0, 1, 2, 3, 4],
]


@pytest.mark.parametrize("expected", make_distance_range_test_data)
def test_make_distance_range(expected: List[int]) -> None:
    """
    Test make_distance_range.

    :param expected: Expected range.
    """
    length = len(expected)
    actual = make_distance_range(length)
    assert actual.shape == (length,)
    expected_array = np.asarray(expected, dtype=np.int64)
    assert np.array_equal(actual, expected_array)


# Shorthands for some binary lists.
e0 = [0]
e1 = [1]
e00 = [0, 0]
e11 = [1, 1]
e000 = [0, 0, 0]
e010 = [0, 1, 0]
e101 = [1, 0, 1]
e111 = [1, 1, 1]
e0000 = [0, 0, 0, 0]
e0110 = [0, 1, 1, 0]
e1001 = [1, 0, 0, 1]
e1111 = [1, 1, 1, 1]
e00000 = [0, 0, 0, 0, 0]
e00100 = [0, 0, 1, 0, 0]
e01010 = [0, 1, 0, 1, 0]
e01110 = [0, 1, 1, 1, 0]
e10001 = [1, 0, 0, 0, 1]
e11111 = [1, 1, 1, 1, 1]

# Test data for test_make_stroke_rectangle
# Data format is: (half_side, expected)
# Where expected data is in row order.
make_stroke_rectangle_test_data: List[Tuple[int, List[List[int]]]] = [
    (0, [e0]),
    (1, [e1]),
    (2, [e0]),
    (0, [e00]),
    (1, [e11]),
    (2, [e00]),
    (0, [e00, e00]),
    (1, [e11, e11]),
    (2, [e00, e00]),
    (0, [e000]),
    (1, [e010]),
    (2, [e101]),
    (3, [e000]),
    (0, [e000, e000]),
    (1, [e010, e010]),
    (2, [e101, e101]),
    (3, [e000, e000]),
    (0, [e000, e000, e000]),
    (1, [e000, e010, e000]),
    (2, [e111, e101, e111]),
    (3, [e000, e000, e000]),
    (0, [e0000]),
    (1, [e0110]),
    (2, [e1001]),
    (3, [e0000]),
    (0, [e0000, e0000]),
    (1, [e0110, e0110]),
    (2, [e1001, e1001]),
    (3, [e0000, e0000]),
    (0, [e0000, e0000, e0000]),
    (1, [e0000, e0110, e0000]),
    (2, [e1111, e1001, e1111]),
    (3, [e0000, e0000, e0000]),
    (0, [e0000, e0000, e0000, e0000]),
    (1, [e0000, e0110, e0110, e0000]),
    (2, [e1111, e1001, e1001, e1111]),
    (3, [e0000, e0000, e0000, e0000]),
    (0, [e00000]),
    (1, [e00100]),
    (2, [e01010]),
    (3, [e10001]),
    (0, [e00000, e00000]),
    (1, [e00100, e00100]),
    (2, [e01010, e01010]),
    (3, [e10001, e10001]),
    (4, [e00000, e00000]),
    (0, [e00000, e00000, e00000]),
    (1, [e00000, e00100, e00000]),
    (2, [e01110, e01010, e01110]),
    (3, [e10001, e10001, e10001]),
    (4, [e00000, e00000, e00000]),
    (0, [e00000, e00000, e00000, e00000]),
    (1, [e00000, e00100, e00100, e00000]),
    (2, [e01110, e01010, e01010, e01110]),
    (3, [e10001, e10001, e10001, e10001]),
    (4, [e00000, e00000, e00000, e00000]),
    (0, [e00000, e00000, e00000, e00000, e00000]),
    (1, [e00000, e00000, e00100, e00000, e00000]),
    (2, [e00000, e01110, e01010, e01110, e00000]),
    (3, [e11111, e10001, e10001, e10001, e11111]),
    (4, [e00000, e00000, e00000, e00000, e00000]),
]


@pytest.mark.parametrize("half_side,expected", make_stroke_rectangle_test_data)
def test_make_stroke_rectangle(half_side: int, expected: List[List[int]]) -> None:
    """
    Test make_stroke_rectangle.

    :param half_side: Rectangle half side length.
    :param expected: Expected output.
    """
    dim0 = len(expected)  # number of rows
    dim1 = len(expected[0])  # number of columns
    actual = make_stroke_rectangle(dim0, dim1, half_side)
    assert actual.shape == (dim0, dim1)
    expected_array = np.asarray(expected, dtype=np.int64)
    assert np.array_equal(actual, expected_array)

    if dim1 != dim0:
        actual_transpose = make_stroke_rectangle(dim1, dim0, half_side)
        assert actual_transpose.shape == (dim1, dim0)
        expected_array_transpose = np.transpose(expected_array)
        assert np.array_equal(actual_transpose, expected_array_transpose)


# Test data for test_make_fill_rectangle_small
# Data format is: (half_side, expected)
# Where expected data is in row order.
make_fill_rectangle_small_test_data: List[Tuple[int, List[List[int]]]] = [
    (0, [e0]),
    (1, [e1]),
    (2, [e1]),
    (0, [e00]),
    (1, [e11]),
    (2, [e11]),
    (0, [e00, e00]),
    (1, [e11, e11]),
    (2, [e11, e11]),
    (0, [e000]),
    (1, [e010]),
    (2, [e111]),
    (3, [e111]),
    (0, [e000, e000]),
    (1, [e010, e010]),
    (2, [e111, e111]),
    (3, [e111, e111]),
    (0, [e000, e000, e000]),
    (1, [e000, e010, e000]),
    (2, [e111, e111, e111]),
    (3, [e111, e111, e111]),
    (0, [e0000]),
    (1, [e0110]),
    (2, [e1111]),
    (3, [e1111]),
    (0, [e0000, e0000]),
    (1, [e0110, e0110]),
    (2, [e1111, e1111]),
    (3, [e1111, e1111]),
    (0, [e0000, e0000, e0000]),
    (1, [e0000, e0110, e0000]),
    (2, [e1111, e1111, e1111]),
    (3, [e1111, e1111, e1111]),
    (0, [e0000, e0000, e0000, e0000]),
    (1, [e0000, e0110, e0110, e0000]),
    (2, [e1111, e1111, e1111, e1111]),
    (3, [e1111, e1111, e1111, e1111]),
    (0, [e00000]),
    (1, [e00100]),
    (2, [e01110]),
    (3, [e11111]),
    (4, [e11111]),
    (0, [e00000, e00000]),
    (1, [e00100, e00100]),
    (2, [e01110, e01110]),
    (3, [e11111, e11111]),
    (4, [e11111, e11111]),
    (0, [e00000, e00000, e00000]),
    (1, [e00000, e00100, e00000]),
    (2, [e01110, e01110, e01110]),
    (3, [e11111, e11111, e11111]),
    (4, [e11111, e11111, e11111]),
    (0, [e00000, e00000, e00000, e00000]),
    (1, [e00000, e00100, e00100, e00000]),
    (2, [e01110, e01110, e01110, e01110]),
    (3, [e11111, e11111, e11111, e11111]),
    (4, [e11111, e11111, e11111, e11111]),
    (0, [e00000, e00000, e00000, e00000, e00000]),
    (1, [e00000, e00000, e00100, e00000, e00000]),
    (2, [e00000, e01110, e01110, e01110, e00000]),
    (3, [e11111, e11111, e11111, e11111, e11111]),
    (4, [e11111, e11111, e11111, e11111, e11111]),
]


@pytest.mark.parametrize("half_side,expected", make_fill_rectangle_small_test_data)
def test_make_fill_rectangle_small(half_side: int, expected: List[List[int]]) -> None:
    """
    Test make_fill_rectangle for smaller sizes.

    :param half_side: Rectangle half side length.
    :param expected: Expected output.
    """
    dim0 = len(expected)  # number of rows
    dim1 = len(expected[0])  # number of columns
    actual = make_fill_rectangle(dim0, dim1, half_side, False)
    assert actual.shape == (dim0, dim1)
    expected_array = np.asarray(expected, dtype=np.int64)
    assert np.array_equal(actual, expected_array)

    actual_inverted = make_fill_rectangle(dim0, dim1, half_side, True)
    assert actual_inverted.shape == (dim0, dim1)
    expected_array_inverted = [1] - expected_array
    assert np.array_equal(actual_inverted, expected_array_inverted)

    total = actual + actual_inverted
    assert np.array_equal(total, np.ones((dim0, dim1)))

    if dim1 != dim0:
        actual_transpose = make_fill_rectangle(dim1, dim0, half_side, False)
        assert actual_transpose.shape == (dim1, dim0)
        expected_array_transpose = np.transpose(expected_array)
        assert np.array_equal(actual_transpose, expected_array_transpose)


# Test data for test_make_fill_rectangle_large
make_fill_rectangle_large_test_data: List[Tuple[int, int, int]] = [
    (20, 30, 0),
    (20, 30, 5),
    (21, 30, 5),
    (20, 31, 5),
    (20, 30, 10),
    (20, 30, 11),
    (20, 30, 17),
    (30, 20, 10),
    (30, 20, 11),
    (30, 20, 17),
]

@pytest.mark.parametrize("dim0,dim1,half_side", make_fill_rectangle_large_test_data)
def test_make_fill_rectangle_large(dim0: int, dim1: int, half_side: int) -> None:
    """
    Test make_fill_rectangle for larger sizes.

    :param dim0: Target array dim0.
    :param dim1: Target array dim1.
    :param half_side: Rough rectangle half side length.
    """
    filled = make_fill_rectangle(dim0, dim1, half_side, True)
    assert filled.shape == (dim0, dim1)

    x_centre_length = min(half_side * 2 if dim1 % 2 == 0 else half_side * 2 - 1, dim1)
    x_buffer_length = max(int((dim1 - x_centre_length) / 2), 0)

    y_centre_length = min(half_side * 2 if dim0 % 2 == 0 else half_side * 2 - 1, dim0)
    y_buffer_length = max(int((dim0 - y_centre_length) / 2), 0)

    expected_edge_row = np.ones(dim1, dtype=np.int64)
    expected_centre_row = np.ones(dim1, dtype=np.int64)
    for x in range(x_buffer_length, x_buffer_length + x_centre_length):
        expected_centre_row[x] = 0

    expected_edge_column = np.ones(dim0, dtype=np.int64)
    expected_centre_column = np.ones(dim0, dtype=np.int64)
    for y in range(y_buffer_length, y_buffer_length + y_centre_length):
        expected_centre_column[y] = 0

    for x in range(x_buffer_length):
        column = filled[:, x]
        assert np.array_equal(column, expected_edge_column)

    for x in range(x_buffer_length, x_buffer_length + x_centre_length):
        column = filled[:, x]
        assert np.array_equal(column, expected_centre_column)

    for x in range(x_buffer_length + x_centre_length, dim1):
        column = filled[:, x]
        assert np.array_equal(column, expected_edge_column)

    for y in range(y_buffer_length):
        row = filled[y]
        assert np.array_equal(row, expected_edge_row)

    for y in range(y_buffer_length, y_buffer_length + y_centre_length):
        row = filled[y]
        assert np.array_equal(row, expected_centre_row)

    for y in range(y_buffer_length + y_centre_length, dim0):
        row = filled[y]
        assert np.array_equal(row, expected_edge_row)


make_nesting_rectangles_test_data: List[Tuple[int, int, int]] = [
    (20, 30, 1),
    (20, 30, 2),
    (20, 30, 3),
    (20, 30, 10),
    (20, 30, 20),
    (20, 30, 30),
]


@pytest.mark.parametrize("dim0,dim1,num_features", make_nesting_rectangles_test_data)
def test_make_nesting_rectangles(dim0: int, dim1: int, num_features: int) -> None:
    """
    Test make_nesting_rectangles.

    :param dim0: Target array dim0.
    :param dim1: Target array dim1.
    :param num_features: Number of rectangles.
    """
    nesting = make_nesting_rectangles(dim0, dim1, num_features)
    assert nesting.shape == (num_features, dim0, dim1)
    actual_background_slice = nesting[0]
    expected_background_slice = make_fill_rectangle(dim0, dim1, num_features - 1, True)
    assert np.array_equal(actual_background_slice, expected_background_slice)

    for feature in range(1, num_features):
        actual_feature_slice = nesting[feature]
        expected_feature_slice = make_stroke_rectangle(dim0, dim1, num_features - feature)
        assert np.array_equal(actual_feature_slice, expected_feature_slice)

    total = nesting.sum(axis=0)
    assert total.shape == (dim0, dim1)
