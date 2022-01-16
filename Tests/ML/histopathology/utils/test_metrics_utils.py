#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
import numpy as np
from typing import List

import matplotlib
from torch.functional import Tensor
import pytest

from InnerEye.ML.Histopathology.utils.metrics_utils import plot_scores_hist, select_k_tiles, plot_slide, plot_heatmap_slide
from InnerEye.ML.Histopathology.utils.naming import ResultsKey


def assert_equal_lists(pred: List, expected: List) -> None:
    assert len(pred) == len(expected)
    for i, slide in enumerate(pred):
        for j, value in enumerate(slide):
            if type(value) in [int, float]:
                assert math.isclose(value, expected[i][j], rel_tol=1e-06)
            elif isinstance(value, List):
                for k, idx in enumerate(value):
                    if type(idx) in [int, float]:
                        assert math.isclose(idx, expected[i][j][k], rel_tol=1e-06)
                    elif type(idx) == Tensor:
                        assert math.isclose(idx.item(), expected[i][j][k].item(), rel_tol=1e-06)
            else:
                raise TypeError("Unexpected list composition")


test_dict = {ResultsKey.SLIDE_ID: [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
             ResultsKey.IMAGE_PATH: [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
             ResultsKey.PROB: [Tensor([0.5]), Tensor([0.7]), Tensor([0.4]), Tensor([1.0])],
             ResultsKey.TRUE_LABEL: [0, 1, 1, 1],
             ResultsKey.BAG_ATTN:
                  [Tensor([[0.1, 0.0, 0.2, 0.15]]),
                  Tensor([[0.10, 0.18, 0.15, 0.13]]),
                  Tensor([[0.25, 0.23, 0.20, 0.21]]),
                  Tensor([[0.33, 0.31, 0.37, 0.35]])]
             }

def test_select_k_tiles() -> None:
    top_tn = select_k_tiles(test_dict, n_slides=1, label=0, n_tiles=2, select=('lowest_pred', 'highest_att'))
    assert_equal_lists(top_tn, [(1, 0.5, [3, 4], [Tensor([0.2]), Tensor([0.15])])])

    nslides = 2
    ntiles = 2
    top_fn = select_k_tiles(test_dict, n_slides=nslides, label=1, n_tiles=ntiles, select=('lowest_pred', 'highest_att'))
    bottom_fn = select_k_tiles(test_dict, n_slides=nslides, label=1, n_tiles=ntiles, select=('lowest_pred', 'lowest_att'))
    assert_equal_lists(top_fn, [(3, 0.4, [1, 2], [Tensor([0.25]), Tensor([0.23])]), (2, 0.7, [2, 3], [Tensor([0.18]), Tensor([0.15])])])
    assert_equal_lists(bottom_fn, [(3, 0.4, [3, 4], [Tensor([0.20]), Tensor([0.21])]), (2, 0.7, [1, 4], [Tensor([0.10]), Tensor([0.13])])])

    top_tp = select_k_tiles(test_dict, n_slides=nslides, label=1, n_tiles=ntiles, select=('highest_pred', 'highest_att'))
    bottom_tp = select_k_tiles(test_dict, n_slides=nslides, label=1, n_tiles=ntiles, select=('highest_pred', 'lowest_att'))
    assert_equal_lists(top_tp, [(4, 1.0, [3, 4], [Tensor([0.37]), Tensor([0.35])]), (2, 0.7, [2, 3], [Tensor([0.18]), Tensor([0.15])])])
    assert_equal_lists(bottom_tp, [(4, 1.0, [2, 1], [Tensor([0.31]), Tensor([0.33])]), (2, 0.7, [1, 4], [Tensor([0.10]), Tensor([0.13])])])


def test_plot_scores_hist() -> None:
    fig = plot_scores_hist(test_dict)
    assert isinstance(fig, matplotlib.figure.Figure)

@pytest.mark.parametrize("scale", [0.1, 1.2, 2.4, 3.6])
def test_plot_slide(scale: int) -> None:
    slide_image = np.random.rand(3, 1000, 2000)
    fig = plot_slide(slide_image=slide_image, scale=scale)
    assert isinstance(fig, matplotlib.figure.Figure)

@pytest.mark.parametrize("level", [0, 1, 2])
def test_plot_heatmap_slide(level: int) -> None:
    slide_image = np.random.rand(3, 1000, 2000)
    location_bbox = [100, 100]
    slide = 1  
    fig = plot_heatmap_slide(slide, slide_image, test_dict, location_bbox)
    assert isinstance(fig, matplotlib.figure.Figure)

    tile_coords = np.array([[100, 100], [200, 100], [200, 200]])
    level_dict = {"0": 1, "1": 4, "2": 16}
    factor = level_dict[str(level)]
    x_tr, y_tr = location_bbox
    tile_xs, tile_ys = tile_coords.T
    tile_xs = tile_xs - x_tr 
    tile_ys = tile_ys - y_tr 
    tile_xs = tile_xs//factor
    tile_ys = tile_ys//factor
    assert min(tile_xs) >= 0 
    assert max(tile_xs) <= slide_image.shape[1]//factor
    assert min(tile_ys) >= 0 
    assert max(tile_ys) <= slide_image.shape[2]//factor
