#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import param

from InnerEye.Common.common_util import SCATTERPLOTS_SUBDIR_NAME
from InnerEye.Common.generic_parsing import GenericConfig


class MetricsScatterplotConfig(GenericConfig):
    """
    Parameters for metrics_scatterplot
    """
    x_path: Path = param.ClassSelector(class_=Path, default=Path(), doc="Path to metrics.csv file to plot along x axis")
    y_path: Path = param.ClassSelector(class_=Path, default=Path(), doc="Path to metrics.csv file to plot along y axis")
    x_name: Optional[str] = param.String(default=None, doc="Name to print below x axis")
    y_name: Optional[str] = param.String(default=None, doc="Name to print beside y axis")
    min_dice: float = param.Number(default=0.0, doc="Minimum Dice score to show in plot")
    max_dice: Optional[float] = param.Number(default=None, allow_None=True, doc="Maximum Dice score to show in plot")
    prefixes: List[str] = param.List(default=[], class_=str, doc="List of prefixes of structure names to include")


def create_scatterplots(data: Dict[str, Dict[str, Dict[str, float]]], against: Optional[List[str]],
                        max_dice: Optional[float] = None) -> Dict[str, plt.Figure]:
    """
    :param data: dictionary such that data[run][structure][seriesId] = dice_score
    :param against: run names to plot against (as y axis); if None or empty, do all against all
    :param max_dice: maximum Dice score to expect; if None, either 1.0 or 100.0 will be inferred from the data
    in code called from here.
    """
    runs = sorted(data.keys())
    result = {}
    if not against:
        against = None
    for i, run1 in enumerate(runs):
        for run2 in runs[i+1:]:
            if against is not None and run2 not in against:
                if run1 not in against:
                    continue
                x_run, y_run = run2, run1
            else:
                x_run, y_run = run1, run2
            x_dct = data[x_run]
            y_dct = data[y_run]
            x_name = x_run.replace(":", "_")
            y_name = y_run.replace(":", "_")
            plot_name = f"{x_name}_vs_{y_name}"
            config = MetricsScatterplotConfig(x_name=x_run, y_name=y_run, max_dice=max_dice)
            result[plot_name] = metrics_scatterplot_from_dicts(config, x_dct, y_dct)
    return result


def satisfies_prefixes(prefixes: List[str], structure: str) -> bool:
    """
    Returns whether "structure" starts with any of the elements of "prefixes", or "prefixes" is empty.
    """
    if not prefixes:
        return True
    for prefix in prefixes:
        if structure.startswith(prefix):
            return True
    return False


def metrics_scatterplot(config: MetricsScatterplotConfig) -> plt.Figure:
    """
    Creates and returns a figure showing how two models compare on a test set. For each point (x, y), x is the
    Dice score of some structure for some patient in the first model (at config.x_path), and y is the Dice score for
    the same structure and patient in the second (at config.y_path). Different structures are distinguished by colour
    and shape. Each model is represented by a metrics.csv file as produced during inference.
    """
    dct1 = to_dict(pd.read_csv(config.x_path))
    dct2 = to_dict(pd.read_csv(config.y_path))
    return metrics_scatterplot_from_dicts(config, dct1, dct2)


def inferred_max_dice(dicts: List[Dict[Any, Dict[Any, float]]]) -> float:
    """
    Given one or more two-level dictionaries of Dice scores, return a guess at what the maximum value for the
    axes should be. If no score is much more than 1.0, we return 1.0, otherwise we guess the numbers are percentages
    and return 100.0.
    """
    max_dice = 1.0
    for dct in dicts:
        for sub_dct in dct.values():
            for value in sub_dct.values():
                max_dice = max(max_dice, value)
    if max_dice > 1.001:
        max_dice = 100.0
    return max_dice


def metrics_scatterplot_from_dicts(config: MetricsScatterplotConfig,
                                   x_dct: Dict[str, Dict[str, float]],
                                   y_dct: Dict[str, Dict[str, float]]) -> plt.Figure:
    """
    :param config: configuration
    :param x_dct: dictionary of structures to patients to Dice scores, for x axis
    :param y_dct: dictionary of structures to patients to Dice scores, for y axis
    :return: figure to display
    """
    fig = plt.figure()
    fig.set_size_inches(17.0, 12.0)
    ax = fig.add_subplot(1, 1, 1)
    # By default, the full Dice score range is shown, but the user can restrict this to focus attention on some
    # part of the plot, e.g. the high end which can get very crowded.
    min_dice = config.min_dice
    max_dice = config.max_dice
    if max_dice is None:
        max_dice = inferred_max_dice([x_dct, y_dct])
    prefixes = config.prefixes
    x_name = config.x_name or str(config.x_path)
    y_name = config.y_name or str(config.y_path)
    dice_span = max_dice - min_dice
    # We set the axes to have a little space (factor 0.02) around the [0, 1] region, so that points for Dice scores of 0
    # are not clipped. We extend the X axis more than (factor 0.2) that to the left, so the legend can appear in the
    # region x < 0 where there will be no points to obscure.
    ax.set_xlim(min_dice - 0.20 * dice_span, max_dice + 0.02 * dice_span)
    ax.set_ylim(min_dice - 0.02 * dice_span, max_dice + 0.02 * dice_span)
    # We'll cycle through these colours and, at the end of the colour sequence, progress to another shape.
    colors = "bgrcmy"   # blue, green, red, cyan, magenta, yellow
    shapes = "ov^sPDX"  # circle, down triangle, up triangle, square, plus, diamond, cross
    index = 0
    for structure in sorted(x_dct):
        if not satisfies_prefixes(prefixes, structure):
            # we don't want to plot points for this structure name
            continue
        d1 = x_dct[structure]
        if structure not in y_dct:
            # structure missing second model, so nothing to plot
            continue
        d2 = y_dct[structure]
        x_list = []  # list of x values
        y_list = []  # list of corresponding y values
        x_better = 0  # number of points (for this structure) for which x > y
        y_better = 0  # number of points for which y > x
        for patient in d1:
            if patient not in d2:
                continue
            x = d1[patient]
            y = d2[patient]
            if x > y:
                x_better += 1
            elif y > x:
                y_better += 1
            if (x < min_dice or x > max_dice) and (y < min_dice or y > max_dice):
                # If both x and y values are outside the range for the plot, ignore the point
                continue
            # Plot (x, y), except when one of x and y (can't be both) is outside the plotting square, show it as only
            # just outside.
            x_list.append(min(max_dice + dice_span * 0.01, max(min_dice - dice_span * 0.01, x)))
            y_list.append(min(max_dice + dice_span * 0.01, max(min_dice - dice_span * 0.01, y)))
        if not x_list:
            continue
        # Choose colour from current index value
        color = colors[index % len(colors)]
        # Show "_l" (left) structures as left-pointing triangles.
        if structure.endswith('_l'):
            shape = '<'
        elif structure.endswith('_r'):
            # Show "_r" (right) structures as right-pointing triangles. We assume the corresponding _l was present
            # and occurred just before this one in the (sorted) structure order, so the colour will be the same
            # because we didn't increment index for the "_l" structure.
            shape = '>'
            index += 1
        else:
            shape = shapes[int(index / len(colors)) % len(shapes)]
            index += 1
        # Plot all points for this structure, and display the structure name and x_better:y_better scores in the
        # legend.
        ax.scatter(x_list, y_list, c=color, marker=shape, label=f"{structure}, {x_better}:{y_better}")
    # Draw a diagonal line representing x=y.
    ax.add_line(mlines.Line2D([min_dice, max_dice], [min_dice, max_dice], color='black'))
    # Draw a box around the region [min_dice, max_dice].
    ax.add_line(mlines.Line2D([min_dice, min_dice], [min_dice, max_dice], color='black'))
    ax.add_line(mlines.Line2D([max_dice, max_dice], [min_dice, max_dice], color='black'))
    ax.add_line(mlines.Line2D([min_dice, max_dice], [min_dice, min_dice], color='black'))
    ax.add_line(mlines.Line2D([min_dice, max_dice], [max_dice, max_dice], color='black'))
    plt.title(f"Dice scores for {x_name} (x axis) and {y_name} (y axis)")
    plt.legend(loc="upper left")
    return fig


def to_dict(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    :param data: data frame with Patient, Structure and Dice columns
    :return: dictionary from Structure to dictionary from Patient to Dice score.
    """
    dct: Dict[str, Dict[str, float]] = {}
    for index, row in data.iterrows():
        patient = row['Patient']
        structure = row['Structure']
        if structure not in dct:
            dct[structure] = {}
        dct[structure][patient] = row['Dice']
    return dct


def write_to_scatterplot_directory(root_folder: Path, plots: Dict[str, plt.Figure]) -> None:
    """
    Writes a file root_folder/scatterplots/basename.png for every plot in plots with key "basename".
    :param root_folder: path to a folder
    :param plots: dictionary from plot basenames to plots (plt.Figure objects)
    """
    scatterplot_dir = root_folder / SCATTERPLOTS_SUBDIR_NAME
    if not plots:
        logging.info(f"There are no plots to write to {scatterplot_dir}, so not creating it.")
        return
    scatterplot_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"There are {len(plots)} plots to write to {scatterplot_dir}")
    for basename, fig in plots.items():
        fig.savefig(scatterplot_dir / f"{basename}.png")


def main() -> None:
    """
    Main function
    """
    metrics_scatterplot(MetricsScatterplotConfig.parse_args())
    plt.show()


if __name__ == '__main__':
    main()
