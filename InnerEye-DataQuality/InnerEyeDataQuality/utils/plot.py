#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from InnerEyeDataQuality.evaluation.metrics import compute_label_entropy
from InnerEyeDataQuality.utils.generic import create_folder
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true: List[int],
                          y_pred: List[int], label_names: List[str],
                          cm: Optional[np.ndarray] = None,
                          save_path: Optional[Path] = None,
                          filename: Optional[str] = None) -> None:
    """
    Generates a figure illustrating nxn confusion matrix
    If param `cm` is specified, y_true and y_pred are not utilised in the plotting function.
    :param y_true: target class ids given in a list form
    :param y_pred: class predictions made by a model or user.
    """

    # Create a confusion matrix for all the classes
    if cm is None:
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize="true")
    cm = np.around(cm, decimals=2)

    fig, ax = plt.subplots(figsize=(7, 8))
    sns.heatmap(cm, annot=True, ax=ax, fmt='g', vmin=0.0, vmax=0.50,
                cmap='BuPu', cbar=False, annot_kws={"fontsize": 13})

    # labels, title and ticks
    ax.set_xlabel('Sampled labels', fontsize=15, labelpad=10)
    ax.set_ylabel('True labels', fontsize=15, labelpad=10)
    ax.set_title('Class Confusion Matrix', fontsize=15)
    ax.set_xticklabels(label_names, fontsize=13)
    ax.set_yticklabels(label_names, fontsize=13)

    if save_path:
        filename = filename + ".png" if filename is not None else "confusion_matrix.png"
        create_folder(save_path)
        fig.savefig(save_path / filename, bbox_inches='tight', dpi=300)


def plot_label_entropy_histogram(labels_entropy: np.ndarray, num_bins: int) -> None:
    num_samples = labels_entropy.shape[0]
    f, ax = plt.subplots()
    ax.hist(labels_entropy, density=False, bins=num_bins)
    ax.set_ylabel('Number of Sample Counts')
    ax.set_xlabel('Entropy of Label Distributions')
    ax.set_title(f'(N={num_samples}) - Purity of Label Distributions')
    ax.grid()


def plot_label_entropy_cumsum(label_entropy: np.ndarray, num_bins: int) -> None:
    num_samples = label_entropy.shape[0]
    # evaluate the histogram and cumulative
    values, base = np.histogram(label_entropy, bins=20)
    cumulative = np.cumsum(values)
    # plot the cumulative function
    f, ax = plt.subplots()
    ax.plot(base[:-1], cumulative, c='blue')
    ax.set_ylabel('Number of Sample Counts')
    ax.set_xlabel('Entropy of Label Distributions')
    ax.set_title(f'(N={num_samples}) - Cumulative Histogram of Label Entropy')
    ax.grid()


def plot_model_embeddings(embeddings: np.ndarray,
                          label_distribution: np.ndarray,
                          label_names: List[str],
                          save_path: Path) -> None:
    # Build a graph and visualise the label entropy on top of it.
    tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, metric='euclidean',
                init='random', random_state=1234, method='barnes_hut', n_jobs=-1)
    model_tsne = tsne.fit_transform(embeddings)

    df = pd.DataFrame({"tsne_dim1": model_tsne[:, 0],
                       "tsne_dim2": model_tsne[:, 1],
                       "labels": np.vectorize(lambda x: label_names[x])(np.argmax(label_distribution, axis=1)),
                       "marker_size": np.power(compute_label_entropy(label_distribution), 2),
                       "label_entropy_bool": compute_label_entropy(label_distribution) > 0.35})

    f, ax = plt.subplots()
    g = sns.scatterplot(x="tsne_dim1", y="tsne_dim2", hue="labels", style="label_entropy_bool",
                        size="marker_size", palette=sns.color_palette("hls", 10),
                        data=df, legend="brief", alpha=0.5, ax=ax)
    g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
    f.savefig(save_path / "tsne_plot_1.png", bbox_inches='tight', dpi=450)

    f, ax = plt.subplots()
    g = sns.scatterplot(x="tsne_dim1", y="tsne_dim2", hue="label_entropy_bool", style="label_entropy_bool",
                        size="marker_size", palette=sns.color_palette("hls", 2),
                        data=df, legend="brief", alpha=0.5, ax=ax)
    g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
    f.savefig(save_path / "tsne_plot_2.png", bbox_inches='tight', dpi=450)


def plot_latent_space_and_noise_cases(ax: plt.Axes,
                                      embeddings: np.ndarray,
                                      labels: np.ndarray,
                                      indicator_noisy_labels: np.ndarray,
                                      selected_cases: np.ndarray,
                                      metric: str = "cosine") -> None:
    """
    Plots the embeddings in 2D. Color is determined by the labels, markers by indicator_noisy_labels
    and size by selected_cases.
    :param ax: matplotlib axis to plot onto.
    :param embeddings: Embeddings shape [n_samples, embedding_size]
    :param labels: array of shape [n_samples,]
    :param indicator_noisy_labels: categorical array to indicate whether a sample is clean, mislabelled clear or
    ambiguous. The value determine the marker used for each point. Shape [n_samples,]
    :param selected_cases: binary array to indicate whether a sample has been selected or not. If yes, the size of
    the corresponding marker is big, if False it is small. Shape [n_samples]
    :param metric: metric to use for T-SNE computation.
    """
    tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, metric=metric,
                init='random', random_state=1234, method='barnes_hut', n_jobs=-1)
    model_tsne = tsne.fit_transform(embeddings)

    df = pd.DataFrame({"tsne_dim1": model_tsne[:, 0],
                       "tsne_dim2": model_tsne[:, 1],
                       "labels": labels,
                       "type_of_labels": indicator_noisy_labels,
                       "selected": selected_cases})
    sns.scatterplot(x="tsne_dim1", y="tsne_dim2", hue="labels",
                    style="type_of_labels", size="selected", sizes={True: 120, False: 40},
                    data=df, legend="brief", alpha=0.5, ax=ax)
