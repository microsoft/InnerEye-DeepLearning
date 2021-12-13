#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np
import pytest

from InnerEye.ML.Histopathology.preprocessing.tiling import assemble_tiles_2d, get_1d_padding, \
    pad_for_tiling_2d, tile_array_2d


@pytest.mark.parametrize("length,tile_size",
                         [(8, 4), (9, 4), (8, 3), (4, 4), (3, 4)])
def test_1d_padding(length: int, tile_size: int) -> None:
    pad_pre, pad_post = get_1d_padding(length, tile_size)

    assert pad_pre >= 0 and pad_post >= 0
    assert pad_pre < tile_size and pad_post < tile_size
    assert abs(pad_post - pad_pre) <= 1, "Asymmetric padding"

    padded_length = pad_pre + length + pad_post
    assert padded_length % tile_size == 0

    n_tiles = padded_length // tile_size
    expected_n_tiles = int(np.ceil(length / tile_size))
    assert n_tiles == expected_n_tiles


@pytest.mark.parametrize("width,height", [(8, 6)])
@pytest.mark.parametrize("tile_size", [3, 4, 5])
@pytest.mark.parametrize("channels_first", [True, False])
def test_2d_padding(width: int, height: int, tile_size: int, channels_first: bool) -> None:
    channels = 2
    pad_value = 0
    array = np.random.rand(channels, height, width)

    input_array = array if channels_first else array.transpose(1, 2, 0)
    padded_array, (offset_w, offset_h) = pad_for_tiling_2d(input_array, tile_size, channels_first,
                                                           constant_values=pad_value)
    if not channels_first:
        padded_array = padded_array.transpose(2, 0, 1)

    padded_channels, padded_height, padded_width = padded_array.shape
    assert padded_channels == channels and padded_height >= height and padded_width >= width
    assert padded_height % tile_size == 0 and padded_width % tile_size == 0
    assert 0 <= offset_h < tile_size and 0 <= offset_w < tile_size

    crop = padded_array[:, offset_h:offset_h + height, offset_w:offset_w + width]
    assert np.array_equal(crop, array)

    # np.array_equiv() broadcasts the shapes
    assert np.array_equiv(padded_array[:, :offset_h, :], pad_value)
    assert np.array_equiv(padded_array[:, :, :offset_w], pad_value)
    assert np.array_equiv(padded_array[:, offset_h + height:, :], pad_value)
    assert np.array_equiv(padded_array[:, :, offset_w + width:], pad_value)


def _get_2d_meshgrid(width: int, height: int, channels_first: bool = True) -> np.ndarray:
    array = np.stack(np.meshgrid(np.arange(width), np.arange(height)),
                     axis=0 if channels_first else -1)
    assert array.shape == ((2, height, width) if channels_first else (height, width, 2))
    return array


@pytest.mark.parametrize("width,height", [(8, 6)])
@pytest.mark.parametrize("tile_size", [3, 4, 5])
@pytest.mark.parametrize("channels_first", [True, False])
def test_tile_array_2d_both(width: int, height: int, tile_size: int, channels_first: bool) -> None:
    channels = 2
    array = _get_2d_meshgrid(width, height, channels_first)

    padded_array, (offset_w, offset_h) = pad_for_tiling_2d(array, tile_size, channels_first,
                                                           constant_values=0)

    tiles, coords = tile_array_2d(array, tile_size, channels_first)
    assert tiles.shape[0] == coords.shape[0]

    expected_n_tiles_w = int(np.ceil(width / tile_size))
    expected_n_tiles_h = int(np.ceil(height / tile_size))
    expected_n_tiles = expected_n_tiles_w * expected_n_tiles_h

    if channels_first:
        assert tiles.shape == (expected_n_tiles, channels, tile_size, tile_size)
    else:
        assert tiles.shape == (expected_n_tiles, tile_size, tile_size, channels)
    assert coords.shape == (expected_n_tiles, 2)

    for idx in range(tiles.shape[0]):
        row = coords[idx, 1] + offset_h
        col = coords[idx, 0] + offset_w
        if channels_first:
            expected_tile = padded_array[:, row:row + tile_size, col:col + tile_size]
        else:
            expected_tile = padded_array[row:row + tile_size, col:col + tile_size, :]
        assert np.array_equal(tiles[idx], expected_tile)

        expected_x = tile_size * (idx % expected_n_tiles_w) - offset_w
        expected_y = tile_size * (idx // expected_n_tiles_w) - offset_h
        assert tuple(coords[idx]) == (expected_x, expected_y)


@pytest.mark.parametrize("width,height", [(8, 6)])
@pytest.mark.parametrize("tile_size", [3, 4, 5])
@pytest.mark.parametrize("channels_first", [True, False])
def test_assemble_tiles_2d(width: int, height: int, tile_size: int, channels_first: bool) -> None:
    array = _get_2d_meshgrid(width, height, channels_first)
    fill_value = 0
    padded_array, padding_offset = pad_for_tiling_2d(array, tile_size, channels_first,
                                                     constant_values=fill_value)

    tiles, coords = tile_array_2d(array, tile_size, channels_first)

    assembled_array, assembly_offset = assemble_tiles_2d(tiles, coords, fill_value=fill_value,
                                                         channels_first=channels_first)
    assert np.array_equal(assembled_array, padded_array)
    assert np.array_equal(assembly_offset, padding_offset)

    for idx in range(tiles.shape[0]):
        row = coords[idx, 1] + assembly_offset[1]
        col = coords[idx, 0] + assembly_offset[0]
        if channels_first:
            crop = assembled_array[:, row:row + tile_size, col:col + tile_size]
        else:
            crop = assembled_array[row:row + tile_size, col:col + tile_size, :]
        assert np.array_equal(crop, tiles[idx])
