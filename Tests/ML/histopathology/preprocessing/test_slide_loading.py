from typing import Optional

import numpy as np
import pytest
from monai.data.image_reader import WSIReader

from InnerEye.Common.common_util import is_windows
from InnerEye.Common.fixed_paths_for_tests import tests_root_directory
from InnerEye.ML.Histopathology.preprocessing.tiling import tile_array_2d
from InnerEye.ML.Histopathology.preprocessing.loading import (LoadROId, get_luminance, load_slide_at_level,
                                                              segment_foreground)
from InnerEye.ML.Histopathology.utils.naming import SlideKey
from Tests.ML.histopathology.datasets.test_slides_dataset import MockSlidesDataset

TEST_IMAGE_PATH = str(tests_root_directory("ML/histopathology/test_data/panda_wsi_example.tiff"))


@pytest.mark.skipif(is_windows(), "cucim package is not available on Windows")
def test_load_slide() -> None:
    level = 2
    reader = WSIReader('cuCIM')
    slide_obj: 'CuImage' = reader.read(TEST_IMAGE_PATH)
    dims = slide_obj.resolutions['level_dimensions'][level][::-1]

    slide = load_slide_at_level(reader, slide_obj, level)
    assert isinstance(slide, np.ndarray)
    expected_shape = (3, *dims)
    assert slide.shape == expected_shape
    frac_empty = (slide == 0).mean()
    assert frac_empty == 0.0

    larger_dims = (2 * dims[0], 2 * dims[1])
    larger_slide, _ = reader.get_data(slide_obj, size=larger_dims, level=level)
    assert isinstance(larger_slide, np.ndarray)
    assert larger_slide.shape == (3, *larger_dims)
    # Overlapping parts match exactly
    assert np.array_equal(larger_slide[:, :dims[0], :dims[1]], slide)
    # Non-overlapping parts are all empty
    empty_fill_value = 0  # fill value seems to depend on the image
    assert np.array_equiv(larger_slide[:, dims[0]:, :], empty_fill_value)
    assert np.array_equiv(larger_slide[:, :, dims[1]:], empty_fill_value)


@pytest.mark.skipif(is_windows(), "cucim package is not available on Windows")
def test_get_luminance() -> None:
    level = 2  # here we only need to test at a single resolution
    reader = WSIReader('cuCIM')
    slide_obj: 'CuImage' = reader.read(TEST_IMAGE_PATH)

    slide = load_slide_at_level(reader, slide_obj, level)
    slide_luminance = get_luminance(slide)
    assert isinstance(slide_luminance, np.ndarray)
    assert slide_luminance.shape == slide.shape[1:]
    assert (slide_luminance <= 255).all() and (slide_luminance >= 0).all()

    tiles, _ = tile_array_2d(slide, tile_size=224, constant_values=255)
    tiles_luminance = get_luminance(tiles)
    assert isinstance(tiles_luminance, np.ndarray)
    assert tiles_luminance.shape == (tiles.shape[0], *tiles.shape[2:])
    assert (tiles_luminance <= 255).all() and (tiles_luminance >= 0).all()

    slide_luminance_tiles, _ = tile_array_2d(np.expand_dims(slide_luminance, axis=0),
                                             tile_size=224, constant_values=255)
    assert np.array_equal(slide_luminance_tiles.squeeze(1), tiles_luminance)


@pytest.mark.skipif(is_windows(), "cucim package is not available on Windows")
def test_segment_foreground() -> None:
    level = 2  # here we only need to test at a single resolution
    reader = WSIReader('cuCIM')
    slide_obj: 'CuImage' = reader.read(TEST_IMAGE_PATH)
    slide = load_slide_at_level(reader, slide_obj, level)

    auto_mask, auto_threshold = segment_foreground(slide, threshold=None)
    assert isinstance(auto_mask, np.ndarray)
    assert auto_mask.dtype == bool
    assert auto_mask.shape == slide.shape[1:]
    assert 0 < auto_mask.sum() < auto_mask.size  # auto-seg should not produce trivial mask
    luminance = get_luminance(slide)
    assert luminance.min() < auto_threshold < luminance.max()

    mask, returned_threshold = segment_foreground(slide, threshold=auto_threshold)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == slide.shape[1:]
    assert np.array_equal(mask, auto_mask)
    assert returned_threshold == auto_threshold

    tiles, _ = tile_array_2d(slide, tile_size=224, constant_values=255)
    tiles_mask, _ = segment_foreground(tiles, threshold=auto_threshold)
    assert isinstance(tiles_mask, np.ndarray)
    assert tiles_mask.dtype == bool
    assert tiles_mask.shape == (tiles.shape[0], *tiles.shape[2:])

    slide_mask_tiles, _ = tile_array_2d(np.expand_dims(mask, axis=0),
                                        tile_size=224, constant_values=False)
    assert np.array_equal(slide_mask_tiles.squeeze(1), tiles_mask)


@pytest.mark.parametrize('level', [1, 2])
@pytest.mark.parametrize('foreground_threshold', [None, 215])
@pytest.mark.skipif(is_windows(), "cucim package is not available on Windows")
def test_get_bounding_box(level: int, foreground_threshold: Optional[float]) -> None:
    margin = 0
    reader = WSIReader('cuCIM')
    loader = LoadROId(reader, image_key=SlideKey.IMAGE, level=level, margin=margin,
                      foreground_threshold=foreground_threshold)
    slide_obj: 'CuImage' = reader.read(TEST_IMAGE_PATH)
    level0_bbox, _ = loader._get_bounding_box(slide_obj)

    highest_level = slide_obj.resolutions['level_count'] - 1
    # level = highest_level
    slide = load_slide_at_level(reader, slide_obj, level=level)
    scale = slide_obj.resolutions['level_downsamples'][level]
    bbox = level0_bbox / scale
    assert bbox.x >= 0 and bbox.y >= 0
    assert bbox.x + bbox.w <= slide.shape[1]
    assert bbox.y + bbox.h <= slide.shape[2]

    # Now with nonzero margin
    margin = 42
    loader_margin = LoadROId(reader, image_key=SlideKey.IMAGE, level=level, margin=margin,
                             foreground_threshold=foreground_threshold)
    level0_bbox_margin, _ = loader_margin._get_bounding_box(slide_obj)
    # Here we test the box differences at the highest resolution, because margin is
    # specified in low-res pixels. Otherwise could fail due to rounding error.
    level0_scale: float = slide_obj.resolutions['level_downsamples'][highest_level]
    level0_margin = int(level0_scale * margin)
    assert level0_bbox_margin.x == level0_bbox.x - level0_margin
    assert level0_bbox_margin.y == level0_bbox.y - level0_margin
    assert level0_bbox_margin.w == level0_bbox.w + 2 * level0_margin
    assert level0_bbox_margin.h == level0_bbox.h + 2 * level0_margin


@pytest.mark.parametrize('level', [1, 2])
@pytest.mark.parametrize('margin', [0, 42])
@pytest.mark.parametrize('foreground_threshold', [None, 215])
@pytest.mark.skipif(is_windows(), "cucim package is not available on Windows")
def test_load_roi(level: int, margin: int, foreground_threshold: Optional[float]) -> None:
    dataset = MockSlidesDataset()
    sample = dataset[0]
    reader = WSIReader('cuCIM')
    loader = LoadROId(reader, image_key=SlideKey.IMAGE, level=level, margin=margin,
                      foreground_threshold=foreground_threshold)
    loaded_sample = loader(sample)
    assert isinstance(loaded_sample, dict)
    # Check that none of the input keys were removed
    assert all(key in loaded_sample for key in sample)

    # Check that the expected new keys were inserted
    additional_keys = [SlideKey.ORIGIN, SlideKey.SCALE, SlideKey.FOREGROUND_THRESHOLD]
    assert all(key in loaded_sample for key in additional_keys)

    assert isinstance(loaded_sample[SlideKey.IMAGE], np.ndarray)
    image_shape = loaded_sample[SlideKey.IMAGE].shape
    assert len(image_shape)
    assert image_shape[0] == 3

    origin = loaded_sample[SlideKey.ORIGIN]
    assert isinstance(origin, tuple)
    assert len(origin) == 2
    assert all(isinstance(coord, int) for coord in origin)

    assert isinstance(loaded_sample[SlideKey.SCALE], (int, float))
    assert loaded_sample[SlideKey.SCALE] >= 1.0

    assert isinstance(loaded_sample[SlideKey.FOREGROUND_THRESHOLD], (int, float))
