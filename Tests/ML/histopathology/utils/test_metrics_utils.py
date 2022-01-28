#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path

import math
import numpy as np
from typing import List

import matplotlib
from torch.functional import Tensor
import pytest

from InnerEye.ML.Histopathology.utils.metrics_utils import plot_scores_hist, select_k_tiles, plot_slide, plot_heatmap_overlay, plot_normalized_confusion_matrix
from InnerEye.ML.Histopathology.utils.naming import ResultsKey
from InnerEye.ML.Histopathology.utils.heatmap_utils import location_selected_tiles
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.plotting import resize_and_save
from InnerEye.ML.utils.ml_util import set_random_seed
from Tests.ML.util import assert_binary_files_match


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
                  Tensor([[0.33, 0.31, 0.37, 0.35]])],
             ResultsKey.TILE_X:
                  [Tensor([200, 200, 424, 424]), 
                  Tensor([200, 200, 424, 424]),
                  Tensor([200, 200, 424, 424]), 
                  Tensor([200, 200, 424, 424])],
             ResultsKey.TILE_Y: 
                  [Tensor([200, 424, 200, 424]),
                  Tensor([200, 200, 424, 424]),
                  Tensor([200, 200, 424, 424]), 
                  Tensor([200, 200, 424, 424])]
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


def test_plot_scores_hist(test_output_dirs: OutputFolderForTests) -> None:
    fig = plot_scores_hist(test_dict)
    assert isinstance(fig, matplotlib.figure.Figure)
    file = Path(test_output_dirs.root_dir) / "plot_score_hist.png"
    resize_and_save(5, 5, file)
    assert file.exists()
    expected = full_ml_test_data_path("histo_heatmaps") / "score_hist.png"
    # To update the stored results, uncomment this line:
    # expected.write_bytes(file.read_bytes())
    assert_binary_files_match(file, expected)


@pytest.mark.parametrize("scale", [0.1, 1.2, 2.4, 3.6])
def test_plot_slide(test_output_dirs: OutputFolderForTests, scale: int) -> None:
    set_random_seed(0)
    slide_image = np.random.rand(3, 1000, 2000)
    fig = plot_slide(slide_image=slide_image, scale=scale)
    assert isinstance(fig, matplotlib.figure.Figure)
    file = Path(test_output_dirs.root_dir) / "plot_slide.png"
    resize_and_save(5, 5, file)
    assert file.exists()
    expected = full_ml_test_data_path("histo_heatmaps") / f"slide_{scale}.png"
    # To update the stored results, uncomment this line:
    # expected.write_bytes(file.read_bytes())
    assert_binary_files_match(file, expected)


def test_plot_heatmap_overlay(test_output_dirs: OutputFolderForTests) -> None:
    set_random_seed(0)
    slide_image = np.random.rand(3, 1000, 2000)
    location_bbox = [100, 100]
    slide = 1 
    tile_size = 224
    level = 0
    fig = plot_heatmap_overlay(slide=slide,                                             # type: ignore
                               slide_image=slide_image,
                               results=test_dict,                                       # type: ignore
                               location_bbox=location_bbox,
                               tile_size=tile_size,
                               level=level)
    assert isinstance(fig, matplotlib.figure.Figure)
    file = Path(test_output_dirs.root_dir) / "plot_heatmap_overlay.png"
    resize_and_save(5, 5, file)
    assert file.exists()
    expected = full_ml_test_data_path("histo_heatmaps") / "heatmap_overlay.png"
    # To update the stored results, uncomment this line:
    # expected.write_bytes(file.read_bytes())
    assert_binary_files_match(file, expected)


@pytest.mark.parametrize("n_classes", [1, 3])
def test_plot_normalized_confusion_matrix(test_output_dirs: OutputFolderForTests, n_classes: int) -> None:
    set_random_seed(0)
    if n_classes > 1:
        cm = np.random.randint(1, 1000, size=(n_classes, n_classes))
        class_names = [str(i) for i in range(n_classes)]
    else:
        cm = np.random.randint(1, 1000, size=(n_classes+1, n_classes+1))
        class_names = [str(i) for i in range(n_classes+1)]
    cm_n = cm/cm.sum(axis=1, keepdims=True)
    assert (cm_n <= 1).all()

    fig = plot_normalized_confusion_matrix(cm=cm_n, class_names=class_names)
    assert isinstance(fig, matplotlib.figure.Figure)
    file = Path(test_output_dirs.root_dir) / f"plot_confusion_matrix_{n_classes}.png"
    resize_and_save(5, 5, file)
    assert file.exists()
    expected = full_ml_test_data_path("histo_heatmaps") / f"confusion_matrix_{n_classes}.png"
    # To update the stored results, uncomment this line:
    expected.write_bytes(file.read_bytes())
    assert_binary_files_match(file, expected)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_location_selected_tiles(level: int) -> None:
    set_random_seed(0)
    slide = 1 
    location_bbox = [100, 100]
    slide_image = np.random.rand(3, 1000, 2000)

    coords = []
    slide_ids = [item[0] for item in test_dict[ResultsKey.SLIDE_ID]]                                            # type: ignore
    slide_idx = slide_ids.index(slide)
    for tile_idx in range(len(test_dict[ResultsKey.IMAGE_PATH][slide_idx])):                                    # type: ignore
        tile_coords = np.transpose(np.array([test_dict[ResultsKey.TILE_X][slide_idx][tile_idx].cpu().numpy(),   # type: ignore
                                    test_dict[ResultsKey.TILE_Y][slide_idx][tile_idx].cpu().numpy()]))          # type: ignore
        coords.append(tile_coords)

    coords = np.array(coords)
    tile_coords_transformed = location_selected_tiles(tile_coords=coords, 
                                                          location_bbox=location_bbox,
                                                          level=level)
    tile_xs, tile_ys = tile_coords_transformed.T
    level_dict = {0: 1, 1: 4, 2: 16}
    factor = level_dict[level]
    assert min(tile_xs) >= 0 
    assert max(tile_xs) <= slide_image.shape[2]//factor
    assert min(tile_ys) >= 0 
    assert max(tile_ys) <= slide_image.shape[1]//factor
