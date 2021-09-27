#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import re

import numpy as np
import pandas as pd
import seaborn as sns

from itertools import cycle
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from typing import Any, Dict, List, Optional, Tuple, Union

from default_paths import MAIN_SIMULATION_DIR
from InnerEyeDataQuality.selection.simulation_statistics import SimulationStatsDistribution
from InnerEyeDataQuality.utils.generic import create_folder, save_obj


def _plot_mean_and_confidence_intervals(ax: plt.Axes, y: np.ndarray, color: str, label: str, linestyle: str = '-',
                                        clip: Tuple[float, float] = (-np.inf, np.inf),
                                        use_derivative: bool = False,
                                        include_auc: bool = True,
                                        linewidth: float = 3.0) -> None:
    if use_derivative:
        y = (y - y[:, 0][:, np.newaxis]) / (np.arange(y.shape[1]) + 1)

    x = np.arange(0, y.shape[1])
    mean = np.mean(y, axis=0)
    total_auc = auc([0, np.max(x)], [100, 100])
    auc_mean = auc(x, mean) / total_auc
    label += f" - AUC: {auc_mean:.4f}" if include_auc else ""
    std_error = np.std(y, axis=0) / np.sqrt(y.shape[0]) * 1.96
    ax.plot(x, mean, color=color, label=label, linestyle=linestyle, linewidth=linewidth)
    ax.fill_between(x,
                    np.clip((mean - std_error), a_min=clip[0], a_max=clip[1]),
                    np.clip((mean + std_error), a_min=clip[0], a_max=clip[1]),
                    color=color, alpha=.2)


def plot_stats_for_all_selectors(stats: Dict[str, SimulationStatsDistribution],
                                 y_attr_names: List[str],
                                 y_attr_labels: List[str],
                                 title: str = '',
                                 x_label: str = '',
                                 y_label: str = '',
                                 legend_loc: Union[int, str] = 2,
                                 fontsize: int = 12,
                                 figsize: Tuple[int, int] = (14, 10),
                                 plot_n_not_ambiguous_noise_cases: bool = False,
                                 **plot_kwargs: Any) -> Tuple[plt.Figure, plt.Axes]:
    """
    Given the dictionary with SimulationStatsDistribution plot a curve for each attribute in the y_attr_names list for
    each selector. Total number of curves will be num_selectors * num_y_attr_names.
    :param stats: A dictionary where each entry corresponds to the SimulationStatsDistribution of each selector.
    :param y_attr_names: The names of the attributes to plot on the y axis.
    :param y_attr_labels: The labels for the legend of the attributes to plot on the y axis.
    :param title: The title of the figure.
    :param x_label: The title of the x-axis.
    :param y_label: The title of the y-axis.
    :param legend_loc: The location of the legend.
    :param plot_n_not_ambiguous_noise_cases: If True, indicate the number of noise easy cases left at the end of the
    simulation
    for each selector. If all cases are selected indicate the corresponding iteration.
    :return: The figure and axis with all the curves plotted.
    """
    colors = ['blue', 'red', 'orange', 'brown', 'black', 'purple', 'green', 'gray', 'olive', 'cyan', 'yellow', 'pink']
    linestyles = ['-', '--']
    fig, ax = plt.subplots(figsize=figsize)

    # Sort alphabetically
    ordered_keys = sorted(stats.keys())
    # Put always oracle and random first (this crashes if oracle / random not there)
    ordered_keys.remove("Oracle")
    ordered_keys.remove("Random")
    ordered_keys.insert(0, "Oracle")
    ordered_keys.insert(1, "Random")

    for _stat_key, color in zip(ordered_keys, cycle(colors)):
        _stat = stats[_stat_key]
        if '(Posterior)' in _stat.name:
            _stat.name = _stat.name.split('(Posterior)')[0]
        for y_attr_name, y_attr_label, linestyle in zip(y_attr_names, y_attr_labels, cycle(linestyles)):
            color = assign_preset_color(name=_stat.name, color=color)
            _plot_mean_and_confidence_intervals(ax=ax, y=_stat.__getattribute__(y_attr_name), color=color,
                                                label=y_attr_label + _stat.name, linestyle=linestyle, **plot_kwargs)
        if plot_n_not_ambiguous_noise_cases:
            average_value_attribute = np.mean(_stat.__getattribute__(y_attr_name), axis=0)
            std_value_attribute = np.std(_stat.__getattribute__(y_attr_name), axis=0)
            logging.debug(f"Method {_stat.name} - {y_attr_name} std: {std_value_attribute[-1]}")
            n_average_mislabelled_not_ambiguous = np.mean(_stat.num_remaining_mislabelled_not_ambiguous, axis=0)
            ax.text(average_value_attribute.size + 1, average_value_attribute[-1],
                    f"{average_value_attribute[-1]:.2f}", fontsize=fontsize - 4)
            no_mislabelled_not_ambiguous_remaining = np.where(n_average_mislabelled_not_ambiguous == 0)[0]
            if no_mislabelled_not_ambiguous_remaining.size > 0:
                idx = np.min(no_mislabelled_not_ambiguous_remaining)
                ax.scatter(idx + 1, average_value_attribute[idx], color=color, marker="s")
    ax.grid()
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.legend(loc=legend_loc, fontsize=fontsize - 2)
    ax.xaxis.set_tick_params(labelsize=fontsize - 2)
    ax.yaxis.set_tick_params(labelsize=fontsize - 2)

    return fig, ax


def assign_preset_color(name: str, color: str) -> str:
    if re.search('oracle', name, re.IGNORECASE):
        return 'blue'
    elif re.search('random', name, re.IGNORECASE):
        return 'red'
    elif re.search('bald', name, re.IGNORECASE):
        if re.search('active', name, re.IGNORECASE):
            return 'green'
        return 'purple'
    elif re.search('clean', name, re.IGNORECASE):
        return 'pink'
    elif re.search('self-supervision', name, re.IGNORECASE) \
            or re.search('SSL', name, re.IGNORECASE) or re.search('pretrained', name, re.IGNORECASE):
        return 'orange'
    elif re.search('imagenet', name, re.IGNORECASE):
        return 'green'
    elif re.search('self-supervision', name, re.IGNORECASE):
        if re.search('graph', name, re.IGNORECASE):
            return 'green'
        elif re.search('With entropy', name, re.IGNORECASE):
            return 'olive'
        elif re.search('active', name, re.IGNORECASE):
            return 'cyan'
        else:
            return 'orange'
    elif re.search('coteaching', name, re.IGNORECASE):
        return 'brown'
    elif re.search('less', name, re.IGNORECASE):
        return 'grey'
    elif re.search('vanilla', name, re.IGNORECASE):
        if re.search('active', name, re.IGNORECASE):
            return 'grey'
        return 'black'
    else:
        return color


def plot_minimal_sampler(stats: Dict[str, SimulationStatsDistribution],
                         n_samples: int, ax: plt.Axes, fontsize: int = 12, legend_loc: int = 2) -> None:
    """
    Starting with one initial label, a noisy sample needs to be relabeled at least twice (best case scenario
    the sampled class during relabeling is equal to the correct label
    """

    n_mislabelled_ambiguous = list(stats.values())[0].num_initial_mislabelled_ambiguous
    n_mislabelled_not_ambiguous = list(stats.values())[0].num_initial_mislabelled_not_ambiguous

    accuracy_beginning = 100 * float(n_samples - n_mislabelled_ambiguous - n_mislabelled_not_ambiguous) / n_samples
    ax.plot([0, 2 * (n_mislabelled_not_ambiguous + n_mislabelled_ambiguous)], [accuracy_beginning, 100], linestyle="--",
            label="Minimal sampler")

    # accuracy_easy_cases = 100 * float(n_samples - n_mislabelled_ambiguous) / n_samples
    # max_accuracy = max([np.max(_stat.accuracy) for _stat in stats.values()])
    # plt.scatter(2 * n_mislabelled_not_ambiguous, accuracy_easy_cases,
    #            marker='s', label="No non-ambiguous noise cases left")
    # ax.set_ylim(accuracy_beginning - 0.25, max_accuracy + 0.25)

    ax.legend(loc=legend_loc, fontsize=fontsize - 2)


def plot_stats(stats: Dict[str, SimulationStatsDistribution],
               dataset_name: str,
               n_samples: int,
               filename_suffix: str,
               save_path: Optional[Path] = None,
               sample_indices: Optional[List[int]] = None) -> None:
    """
    :param stats:
    :param dataset_name:
    :param n_samples:
    :param filename_suffix:
    :param save_path:
    :param sample_indices: Image indices used in the dataset to visualise the selected cases
    """

    if save_path:
        input_args = locals()
        save_path = save_path / dataset_name
        save_path.mkdir(exist_ok=True)
        save_obj(input_args, save_path / "inputs_to_plot_stats.pkl")

    noise_rate = 100 * float(
        list(stats.values())[0].num_initial_mislabelled_ambiguous + list(stats.values())[
            0].num_initial_mislabelled_not_ambiguous) / n_samples
    # Label accuracy vs num relabels
    fontsize, legend_loc = 20, 2
    fig, ax = plot_stats_for_all_selectors(stats, ['accuracy'], [''],
                                           title=f'Dataset Curation - {dataset_name} (N={n_samples}, '
                                                 f'{noise_rate:.1f}% noise)',
                                           x_label='Number of collected relabels on the dataset',
                                           y_label='Percentage of correct labels',
                                           legend_loc=legend_loc,
                                           fontsize=fontsize,
                                           figsize=(11, 10),
                                           plot_n_not_ambiguous_noise_cases=True)
    if dataset_name != "NoisyChestXray":
        plot_minimal_sampler(stats, n_samples, ax, fontsize=fontsize, legend_loc=legend_loc)
    if save_path:
        fig.savefig(save_path / f"simulation_label_accuracy_{filename_suffix}.pdf", bbox_inches='tight')
        fig.savefig(save_path / f"simulation_label_accuracy_{filename_suffix}.png", bbox_inches='tight')

    # Label accuracy vs num relabels - To Origin
    fig, ax = plot_stats_for_all_selectors(stats, ['accuracy'], [''],
                                           title=f'Dataset Curation - {dataset_name} (N={n_samples})',
                                           x_label='Number of collected relabels on the dataset',
                                           y_label='Percentage of correct labels',
                                           legend_loc=1,
                                           plot_n_not_ambiguous_noise_cases=False,
                                           use_derivative=True)
    if save_path:
        fig.savefig(save_path / f"simulation_label_accuracy_to_origin_{filename_suffix}.png", bbox_inches='tight')

    # Average total variation vs num relabels
    fig, ax = plot_stats_for_all_selectors(stats, ['avg_total_variation'], [''],
                                           title=f'Dataset Curation - {dataset_name} (N={n_samples})',
                                           x_label='Number of collected relabels on the dataset',
                                           y_label='Total Variation (Full vs Sampled Distributions)',
                                           fontsize=fontsize,
                                           include_auc=False,
                                           figsize=(10, 11),
                                           legend_loc=1)
    if save_path:
        fig.savefig(save_path / f"simulation_avg_total_variation_{filename_suffix}.png", bbox_inches='tight')
        fig.savefig(save_path / f"simulation_avg_total_variation_{filename_suffix}.pdf", bbox_inches='tight')

    # Remaining number of noisy cases vs num relabels
    fig, ax = plot_stats_for_all_selectors(stats,
                                           ['num_remaining_mislabelled_not_ambiguous'], [''],
                                           x_label='Number of collected relabels on the dataset',
                                           y_label='# of remaining clear noisy samples',
                                           fontsize=fontsize + 1,
                                           figsize=(10, 10),
                                           include_auc=False,
                                           linewidth=5.0,
                                           legend_loc="lower left")
    if save_path:
        fig.savefig(save_path / f"simulation_remaining_clear_mislabelled_{filename_suffix}.png", bbox_inches='tight')
        fig.savefig(save_path / f"simulation_remaining_clear_mislabelled_{filename_suffix}.pdf", bbox_inches='tight')

    # Remaining number ambiguous cases vs num relabels
    fig, ax = plot_stats_for_all_selectors(stats,
                                           ['num_remaining_mislabelled_ambiguous'], [''],
                                           x_label='Number of collected relabels on the dataset',
                                           y_label='# of remaining difficult noisy samples',
                                           fontsize=fontsize + 1,
                                           include_auc=False,
                                           figsize=(10, 10),
                                           linewidth=5.0,
                                           legend_loc="lower left")
    if save_path:
        fig.savefig(save_path / f"simulation_remaining_ambiguous_mislabelled_{filename_suffix}.png",
                    bbox_inches='tight')
        fig.savefig(save_path / f"simulation_remaining_ambiguous_mislabelled_{filename_suffix}.pdf",
                    bbox_inches='tight')


def plot_roc_curve(scores: np.ndarray, labels: np.ndarray,
                   type_of_cases: str = "mislabelled",
                   ax: Optional[plt.Axes] = None,
                   color: str = "b",
                   legend: str = "AUC",
                   linestyle: str = "-") -> None:
    fpr, tpr, threshold = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color=color, label=f"{legend}: {roc_auc:.2f}", linestyle=linestyle)
    ax.set_ylabel("True positive rate")
    ax.set_xlabel("False positive rate")
    ax.set_title(f"ROC curve - {type_of_cases} detection")
    ax.legend(loc="lower right")
    ax.grid(b=True)


def plot_pr_curve(scores: np.ndarray, labels: np.ndarray,
                  type_of_cases: str = "mislabelled",
                  ax: Optional[plt.Axes] = None,
                  color: str = "b",
                  legend: str = "AUC",
                  linestyle: str = "-") -> None:
    precision, recall, _ = precision_recall_curve(y_true=labels, probas_pred=scores)
    pr_auc = auc(recall, precision)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(recall, precision, color=color, label=f"{legend}: {pr_auc:.2f}", linestyle=linestyle)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve - {type_of_cases} detection")
    ax.legend(loc="lower right")
    ax.grid(b=True)


def plot_binary_confusion(scores: np.ndarray, labels: np.ndarray, type_of_cases: str = "mislabelled",
                          ax: Optional[plt.Axes] = None) -> None:
    if ax is None:
        fig, ax = plt.subplots()
    fpr, tpr, threshold = roc_curve(labels, scores)
    optimal_threshold = threshold[np.argmax(tpr - fpr)]
    prediction = scores > optimal_threshold
    tn, fp, fn, tp = confusion_matrix(labels, prediction).ravel()
    rates = tn / (tn + tp), fp / (tn + fp), fn / (tp + fn), tp / (tp + fn)
    cf_matrix = confusion_matrix(labels, prediction)
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = [f"{value:.0f}" for value in cf_matrix.flatten()]
    group_percentages = [f"{value:.3f}" for value in rates]
    annotations = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    annotations = np.asarray(annotations).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=annotations, fmt="", cmap=sns.light_palette("navy", reverse=True), ax=ax,
                cbar=False)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - {type_of_cases} cases\nThreshold = {optimal_threshold:.2f}")


def plot_stats_scores(selector_name: str, scores_mislabelled: np.ndarray, labels_mislabelled: np.ndarray,
                      scores_ambiguous: Optional[np.ndarray] = None, labels_ambiguous: Optional[np.ndarray] = None,
                      save_path: Optional[Path] = None) -> None:
    if scores_ambiguous is None:
        fig, ax = plt.subplots(3, 1, figsize=(5, 10))
    else:
        fig, ax = plt.subplots(3, 2, figsize=(10, 15))

    fig.suptitle(selector_name)

    ax = ax.ravel(order="F")
    plot_roc_curve(scores_mislabelled, labels_mislabelled, type_of_cases="mislabelled", ax=ax[0])
    plot_pr_curve(scores_mislabelled, labels_mislabelled, type_of_cases="mislabelled", ax=ax[1])
    plot_binary_confusion(scores_mislabelled, labels_mislabelled, type_of_cases="mislabelled", ax=ax[2])

    if scores_ambiguous is not None and labels_ambiguous is not None and np.sum(labels_ambiguous) != 0:
        plot_roc_curve(scores_ambiguous, labels_ambiguous, type_of_cases="ambiguous", ax=ax[3])
        plot_pr_curve(scores_ambiguous, labels_ambiguous, type_of_cases="ambiguous", ax=ax[4])
        plot_binary_confusion(scores_ambiguous, labels_ambiguous, "ambiguous", ax[5])
        # ambiguous detection given that is mislabelled
        scores_ambiguous_given_mislabelled = scores_ambiguous[labels_mislabelled == 1]
        labels_ambiguous_given_mislabelled = labels_ambiguous[labels_mislabelled == 1]
        if roc_auc_score(labels_ambiguous_given_mislabelled, scores_ambiguous_given_mislabelled) < .5:
            scores_ambiguous_given_mislabelled *= -1

        plot_roc_curve(scores_ambiguous_given_mislabelled,
                       labels_ambiguous_given_mislabelled,
                       type_of_cases="ambiguous",
                       ax=ax[3],
                       color="red", legend="AUC given mislabelled=True", linestyle="--")
        plot_pr_curve(scores_ambiguous_given_mislabelled,
                      labels_ambiguous_given_mislabelled,
                      type_of_cases="ambiguous",
                      ax=ax[4],
                      color="red", legend="AUC given mislabelled=True", linestyle="--")

    if save_path:
        plt.savefig(save_path / f"{selector_name}_stats_scoring.png", bbox_inches="tight")
    plt.show()


def plot_relabeling_score(true_majority: np.ndarray,
                          starting_scores: np.ndarray,
                          current_majority: np.ndarray,
                          selector_name: str) -> None:
    """
    Plots for ranking of samples score-wise
    """
    create_folder(MAIN_SIMULATION_DIR / "scores_histogram")
    create_folder(MAIN_SIMULATION_DIR / "noise_detection_heatmaps")

    is_noisy = true_majority != current_majority
    total_noise_rate = np.mean(is_noisy)

    # Compute how many noisy sampled where present in the
    # highest n_noisy samples.
    target_perc = int((1 - total_noise_rate) * 100)
    q = np.percentile(starting_scores, q=target_perc)
    noisy_cases_detected = is_noisy[starting_scores > q]
    percentage_noisy_detected = 100 * float(noisy_cases_detected.sum()) / is_noisy.sum()

    # Plot histogram of scores differentiated by noisy or not.
    df = pd.DataFrame({"scores": starting_scores,
                       "is_noisy": is_noisy})
    plt.close()
    sns.histplot(data=df, x="scores", hue="is_noisy", multiple="dodge", bins=10)
    plt.title(f"Histogram of relabeling scores {selector_name}\n"
              f"{percentage_noisy_detected:.1f}% noise cases > {target_perc}th percentile of scores")
    plt.savefig(MAIN_SIMULATION_DIR / "scores_histogram" / selector_name)
    plt.close()

    # Plot heatmap showing where the noisy cases are located in the score ranking.
    idx = np.argsort(starting_scores)
    sorted_is_noisy = is_noisy[idx]
    plt.title(f"{selector_name}\n"
              f"Location of noisy cases by increasing scores (from left to right)\n"
              f"{percentage_noisy_detected:.1f}% noise cases > {target_perc}th percentile of scores")
    sns.heatmap(sorted_is_noisy.reshape(1, -1), yticklabels=False, vmax=1.3, cbar=False)
    plt.savefig(MAIN_SIMULATION_DIR / "noise_detection_heatmaps" / selector_name)
    plt.close()
