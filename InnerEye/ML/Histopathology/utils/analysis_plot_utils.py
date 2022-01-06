#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np
from typing import List, Any

import umap
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def get_tsne_projection(features: List[Any], n_components: int = 2, n_jobs: int = -1, **kwargs: Any) -> List[Any]:
    """
    Get the t-sne projection of high dimensional data in a lower dimensional space
    :param features: list of features in higher dimensional space (n x f for n samples and f features per sample)
    :param **kwargs: keyword arguments to be passed to TSNE()
    :return: list of features in lower dimensional space (n x c for n samples and c components)
    """
    tsne_2d = TSNE(n_components=n_components, n_jobs=n_jobs, **kwargs)
    tsne_proj = tsne_2d.fit_transform(features)
    return tsne_proj


def get_umap_projection(features: List[Any], n_components: int = 2, n_jobs: int = -1, **kwargs: Any) -> List[Any]:
    """
    Get the umap projection of high dimensional data in a lower dimensional space
    :param features: list of features in higher dimensional space (n x f for n samples and f features per sample)
    :param **kwargs: keyword arguments to be passed to UMAP()
    :return: list of features in lower dimensional space (n x c for n samples and c components)
    """
    umap_2d = umap.UMAP(n_components=n_components, n_jobs=n_jobs, **kwargs)
    umap_proj = umap_2d.fit_transform(features)
    return umap_proj


def normalize_array_minmax(arr: List[float]) -> List[float]:
    """
    Normalize an array in range 0 to 1
    :param arr: array to be normalized
    :return: normalized array
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def normalize_array_mean(arr: List[float]) -> List[float]:
    """
    Normalize an array with zero mean and unit variance
    :param arr: array to be normalized
    :return: normalized array
    """
    return (arr - np.mean(arr)) / np.std(arr)


def plot_projected_features_2d(data: Any, labels: List[int], classes: List[str], title: str = "") -> None:
    """
    Plot a scatter plot of projected features in two dimensions
    :param data: features projected in 2d space (nx2)
    :param labels: corresponding labels of the data (nx1)
    :param classes: list of classes in the dataset
    :param title: plot title string
    """
    plt.figure()
    scatter = plt.scatter(data[:, 0], data[:, 1], 20, labels)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.title(title)


def plot_box_whisker(data_list: List[Any], column_names: List[str], show_outliers: bool, title: str = "") -> None:
    """
    Plot a box whisker plot of column data
    :param columns: data to be plotted in columns
    :param column_names: names of the columns
    :param show_outliers: whether outliers need to be shown
    :param title: plot title string
    """
    plt.figure()
    _, ax = plt.subplots()
    ax.boxplot(data_list, showfliers=show_outliers)
    positions = range(1, len(column_names)+1)
    means = []
    for i in range(len(data_list)):
        means.append(np.mean(data_list[i]))
    ax.plot(positions, means, 'rs')
    plt.xticks(positions, column_names)
    plt.title(title)


def plot_histogram(data: List[Any], title: str = "") -> None:
    """
    Plot a histogram given some data
    :param data: data to be plotted
    :param title: plot title string
    """
    plt.figure()
    plt.hist(data, bins=50)
    plt.gca().set(title=title, xlabel='Values', ylabel='Frequency')
