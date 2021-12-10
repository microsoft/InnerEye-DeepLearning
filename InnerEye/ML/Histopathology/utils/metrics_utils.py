#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Tuple, List, Any, Dict
import torch
import matplotlib.pyplot as plt
from math import ceil

from InnerEye.ML.Histopathology.models.transforms import load_pil_image
from InnerEye.ML.Histopathology.utils.naming import ResultsKey


def select_k_tiles(results: Dict, n_tiles: int = 5, n_slides: int = 5, label: int = 1,
                   select: Tuple = ('lowest_pred', 'highest_att'),
                   slide_col: str = ResultsKey.SLIDE_ID, gt_col: str = ResultsKey.TRUE_LABEL,
                   attn_col: str = ResultsKey.BAG_ATTN, prob_col: str = ResultsKey.PROB,
                   return_col: str = ResultsKey.IMAGE_PATH) -> List[Tuple[Any, Any, List[Any], List[Any]]]:
    """
    :param results: List that contains slide_level dicts
    :param n_tiles: number of tiles to be selected for each slide
    :param n_slides: number of slides to be selected
    :param label: which label to use to select slides
    :param select: criteria to be used to sort the slides (select[0]) and the tiles (select[1])
    :param slide_col: column name that contains slide identifiers
    :param gt_col: column name that contains labels
    :param attn_col: column name that contains scores used to sort tiles
    :param prob_col: column name that contains scores used to sort slides
    :param return_col: column name of the values we want to return for each tile
    :return: tuple containing the slides id, the slide score, the tile ids, the tiles scores
    """
    tmp_s = [(results[prob_col][i], i) for i, gt in enumerate(results[gt_col]) if gt == label]  # type ignore
    if select[0] == 'lowest_pred':
        tmp_s.sort(reverse=False)
    elif select[0] == 'highest_pred':
        tmp_s.sort(reverse=True)
    else:
        ValueError('select value not recognised')
    _, sorted_idx = zip(*tmp_s)
    k_idx = []
    if select[1] == 'highest_att':
        descending = True
    elif select[1] == 'lowest_att':
        descending = False
    for _, slide_idx in enumerate(sorted_idx[:n_slides]):
        tmp = results[attn_col][slide_idx]
        _, t_indices = torch.sort(tmp, descending=descending)
        k_tiles = []
        scores = []
        for t_idx in t_indices[0][:n_tiles]:
            k_tiles.append(results[return_col][slide_idx][t_idx])
            scores.append(results[attn_col][slide_idx][0][t_idx])
        # slide_ids are duplicated
        k_idx.append((results[slide_col][slide_idx][0],
                      results[prob_col][slide_idx].item(),
                      k_tiles, scores))
    return k_idx


def plot_scores_hist(results: Dict, prob_col: str = ResultsKey.PROB,
                     gt_col: str = ResultsKey.TRUE_LABEL) -> plt.figure:
    """
    :param results: List that contains slide_level dicts
    :param prob_col: column name that contains the scores
    :param gt_col: column name that contains the true label
    :return: matplotlib figure of the scores histogram by class
    """
    pos_scores = [results[prob_col][i][0].cpu().item() for i, gt in enumerate(results[gt_col]) if gt == 1]
    neg_scores = [results[prob_col][i][0].cpu().item() for i, gt in enumerate(results[gt_col]) if gt == 0]
    fig, ax = plt.subplots()
    ax.hist([pos_scores, neg_scores], label=['1', '0'], alpha=0.5)
    ax.set_xlabel("Predicted Score")
    ax.legend()
    return fig


def plot_slide_noxy(slide: str, score: float, paths: List, attn: List, case: str, ncols: int = 5,
                    size: Tuple = (10, 10)) -> plt.figure:
    """
    :param slide: slide identifier
    :param score: predicted score for the slide
    :param paths: list of paths to tiles belonging to the slide
    :param attn: list of scores belonging to the tiles in paths. paths and attn are expected to have the same shape
    :param case: string used to define the title of the plot e.g. TP
    :param ncols: number of cols the produced figure should have
    :param size: size of the plot
    :return: matplotlib figure of each tile in paths with attn score
    """
    nrows = int(ceil(len(paths) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)
    fig.suptitle(f"{case}: {slide} P=%.2f" % score)
    for i in range(len(paths)):
        img = load_pil_image(paths[i])
        axs.ravel()[i].imshow(img, clim=(0, 255), cmap='gray')
        axs.ravel()[i].set_title("%.6f" % attn[i].cpu().item())
    for i in range(len(axs.ravel())):
        axs.ravel()[i].set_axis_off()
    return fig
