#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import pytest
import numpy as np
from passthrough_model import convert_hex_to_rgb_colour, make_distance_range, make_stroke_rectangle, \
    make_fill_rectangle, make_nesting_rectangles
from score import convert_rgb_colour_to_hex

rgb_colour_testdata = [
    (0x00, 0x00, 0x00, "000000"),
    (0x0f, 0x00, 0x00, "0F0000"),
    (0xff, 0x00, 0x00, "FF0000"),
    (0x00, 0x0f, 0x00, "000F00"),
    (0x00, 0xff, 0x00, "00FF00"),
    (0x00, 0x00, 0x0f, "00000F"),
    (0x00, 0x00, 0xff, "0000FF"),
    (0x04, 0x08, 0x0a, "04080A"),
    (0xf4, 0xf8, 0xfa, "F4F8FA")
]


@pytest.mark.parametrize("red,green,blue,colour", rgb_colour_testdata)
def test_convert_hex_to_rgb_colour(red: int, green: int, blue: int, colour: str) -> None:
    """
    Test that test colours, which are strings, can be formatted as
    TupleInt3's.

    :param red: Expected red component.
    :param green: Expected green component.
    :param blue: Expected blue component.
    :param colour: Hex string.
    """
    assert convert_hex_to_rgb_colour(colour) == (red, green, blue)


@pytest.mark.parametrize("red,green,blue,colour", rgb_colour_testdata)
def test_convert_rgb_colour_to_hex(red: int, green: int, blue: int, colour: str) -> None:
    """
    Test that config colours, which are TupleInt3's, can be formatted as
    strings.

    :param red: Red component.
    :param green: Green component.
    :param blue: Blue component.
    :param colour: Expected hex string.
    """
    assert convert_rgb_colour_to_hex((red, green, blue)) == colour


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


@dataclass(frozen=True)
class RectangleInRectangleLineData:
    """
    Contains information about lines through a rectangle containing another
    rectangle.
    """
    dim: int  # Dimension
    centre_start: int  # Coordinate where the inner rectangle starts.
    centre_end: int  # Coordinate where the inner rectangle end.
    border_slice: np.array  # Slice through the rectangle missing the inner.
    centre_slice: np.array  # Slice through the rectangle intercepting filled inner.
    stroke_slice: np.array  # Slice through the rectangle intercepting stroked inner.

    @staticmethod
    def create(dim: int, half_side: int) -> RectangleInRectangleLineData:
        """
        Given a dimension and half side length, create RectangleInRectangleLineData.

        :param dim: Outer rectangle dimension.
        :param half_side: Inner rectangle half side length.
        :return: RectangleInRectangleLineData.
        """
        centre_length = half_side * 2 if dim % 2 == 0 else half_side * 2 - 1
        border_length = int((dim - centre_length) / 2)
        centre_start, centre_end = border_length, border_length + centre_length
        border_slice = np.zeros(dim, dtype=np.int64)
        centre_slice = np.zeros(dim, dtype=np.int64)
        centre_slice[max(centre_start, 0):min(centre_end, dim)] = 1
        stroke_slice = np.zeros(dim, dtype=np.int64)
        if 0 <= centre_start < dim:
            stroke_slice[centre_start] = 1
        if 0 <= centre_end - 1 < dim:
            stroke_slice[centre_end - 1] = 1
        return RectangleInRectangleLineData(dim, centre_start, centre_end, border_slice, centre_slice,
                                            stroke_slice)


@dataclass(frozen=True)
class RectangleInRectangleData:
    """
    Contains information about a rectangle containing another
    rectangle.
    """
    dim0_data: RectangleInRectangleLineData
    dim1_data: RectangleInRectangleLineData

    @staticmethod
    def create(dim0: int, dim1: int, half_side: int) -> RectangleInRectangleData:
        """
        Given dimensions and half side length, create RectangleInRectangleData.

        :param dim0: Outer rectangle dimension 0.
        :param dim1: Outer rectangle dimension 1.
        :param half_side: Inner rectangle half side length.
        :return: RectangleInRectangleData.
        """
        dim0_data = RectangleInRectangleLineData.create(dim0, half_side)
        dim1_data = RectangleInRectangleLineData.create(dim1, half_side)
        return RectangleInRectangleData(dim0_data, dim1_data)

    def make_by_rows(self, fill: bool, invert: bool) -> np.array:
        """
        Create filled or stroked rectangle in rectangle, by rows.

        :param fill: True for filled inner rectangle, false for stroked inner rectangle.
        :param invert: True to invert when fill is true. Ignored for stroked.
        :return: 2d np.array representing a rectangle in a rectangle.
        """
        rows = [self.slice_line(row, fill, invert)
                for row in range(self.dim0_data.dim)]
        return np.asarray(rows, dtype=np.int64)

    def slice_line(self, i: int, fill: bool, invert: bool) -> np.array:
        """
        Calculate what a line sliced through the outer rectangle would look like.

        :param i: Line coordindate.
        :param fill: True for filled inner rectangle, false for stroked inner rectangle.
        :param invert: True to invert.
        :return: Expected line slice.
        """
        if not fill and (self.dim0_data.centre_start < i < self.dim0_data.centre_end - 1):
            plain_slice = self.dim1_data.stroke_slice
        elif self.dim0_data.centre_start <= i < self.dim0_data.centre_end:
            plain_slice = self.dim1_data.centre_slice
        else:
            plain_slice = self.dim1_data.border_slice
        return plain_slice if not invert else 1 - plain_slice


def test_make_fill_rectangle() -> None:
    """
    Test make_fill_rectangle.
    """
    for dim0 in range(1, 30):
        for dim1 in range(1, 30):
            for half_side in range(max(dim0, dim1) + 1):
                for invert in [False, True]:
                    filled = make_fill_rectangle(dim0, dim1, half_side, invert)
                    assert filled.shape == (dim0, dim1)

                    rectangle_data = RectangleInRectangleData.create(dim0, dim1, half_side)

                    filled_by_rows = rectangle_data.make_by_rows(True, invert)
                    assert np.array_equal(filled, filled_by_rows)

                filled_0s = make_fill_rectangle(dim0, dim1, half_side, False)
                filled_1s = make_fill_rectangle(dim0, dim1, half_side, True)
                total = filled_0s + filled_1s
                assert np.array_equal(total, np.ones((dim0, dim1)))


def test_make_stroke_rectangle() -> None:
    """
    Test make_stroke_rectangle.
    """
    for dim0 in range(1, 30):
        for dim1 in range(1, 30):
            for half_side in range(max(dim0, dim1) + 1):
                stroked = make_stroke_rectangle(dim0, dim1, half_side)
                assert stroked.shape == (dim0, dim1)

                rectangle_data = RectangleInRectangleData.create(dim0, dim1, half_side)

                filled2 = rectangle_data.make_by_rows(False, False)
                assert np.array_equal(stroked, filled2)


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
    total = nesting.sum(axis=0)
    assert total.shape == (dim0, dim1)
