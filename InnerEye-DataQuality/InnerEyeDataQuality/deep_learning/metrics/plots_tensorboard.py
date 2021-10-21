#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Optional, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('Agg')  # No display


def get_scatter_plot(x_data: np.ndarray, y_data: np.ndarray, scale: np.ndarray = None,
                     title: str = '', x_label: str = '', y_label: str = '',
                     y_lim: Optional[List[float]] = None) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, alpha=0.3, s=scale)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid()
    if y_lim:
        ax.set_ylim(y_lim)

    return fig


def get_histogram_plot(data: Union[List[np.ndarray], np.ndarray], num_bins: int, title: str = '',
                       x_label: str = '', x_lim: Tuple[float, float] = None) -> plt.Figure:
    """
    Creates a histogram plot for a given set of numpy arrays specified in `data` object.
    Return the generated figure object.
    """

    fig, ax = plt.subplots()
    cm = plt.cm.get_cmap('Pastel1')
    if isinstance(data, list):
        for _d_id, _d in enumerate(data):
            ax.hist(_d, density=True, bins=num_bins, color=cm(_d_id), alpha=0.6)
    else:
        ax.hist(data, density=True, bins=num_bins)
    ax.set_ylabel('Sample density')
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.grid()

    if x_lim:
        ax.set_xlim(x_lim)

    return fig


def plot_excluded_cases_coteaching(case_drop_mask: np.ndarray,
                                   entropy_sorted_indices: np.ndarray,
                                   num_epochs: int,
                                   num_samples: int,
                                   title: Optional[str] = None) -> plt.Figure:
    """
    Plots the excluded cases in co-teaching training - training epochs vs sample_ids
    Samples are sorted based on their true ambiguity score.
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(case_drop_mask[entropy_sorted_indices, :].astype(np.uint8) * 255, cmap="gray",
              extent=[0, num_epochs, 0, num_samples], vmin=0, vmax=10, aspect='auto')
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel("Training sample ids (ordered by true entropy)")
    if title:
        ax.set_title(title)
    return fig


def plot_disagreement_per_sample(prediction_disagreement: np.ndarray, true_label_entropy: np.ndarray) -> plt.Figure:
    """
    Plots predicted class disagreement between two models - training epochs vs sample_ids
    Samples are sorted based on their true ambiguity score.
    """
    entropy_sorted_indices = np.argsort(true_label_entropy)
    fig, ax = plt.subplots()
    num_epochs = prediction_disagreement.shape[1]
    num_samples = prediction_disagreement.shape[0]
    ax.imshow(prediction_disagreement[entropy_sorted_indices, :].astype(np.uint8) * 2, cmap="gray",
              extent=[0, num_epochs, 0, num_samples], vmin=0, vmax=1, aspect='auto')
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel("Training sample ids (ordered by true entropy)")
    return fig
