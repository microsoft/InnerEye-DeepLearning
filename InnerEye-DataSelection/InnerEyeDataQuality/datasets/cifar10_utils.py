#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import io
import logging
import os
import pickle
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests

logging.getLogger('matplotlib').setLevel(logging.WARNING)


def split_dataset(labels: np.ndarray,
                  reference_split: float,
                  shuffle: bool = True,
                  seed: int = 1234) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param labels: Complete label array that is split into two subsets, reference and noisy test set.
    :param reference_split: Ratio of total samples that will be put in the reference set.
    :param shuffle: If set to true, rows of the label matrix are shuffled prior to split.
    :param seed: Random seed used in sample shuffle
    """
    # Create two testing sets from CIFAR10H Test Samples (10k - 10 classes)
    num_samples = labels.shape[0]
    num_samples_set1 = int(num_samples * reference_split)
    perm = np.random.RandomState(seed=seed).permutation(num_samples) if shuffle else np.array(range(num_samples))
    d_set1 = labels[perm[:num_samples_set1], :]
    d_set2 = labels[perm[num_samples_set1:], :]

    return d_set1, d_set2, perm


def get_cifar10h_labels(reference_split: float = 0.5,
                        shuffle: bool = True,
                        immutable: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param reference_split: The sample ratio between gold standard test set and full test set
    :param shuffle: Shuffle samples prior to split.
    :param immutable: If set to True, the returned arrays are only read only.
    """
    cifar10h_counts = download_cifar10h_data()  # Num_samples x n_classes
    d_split1_standard, d_split2_complete, permutation = split_dataset(cifar10h_counts, reference_split, shuffle)
    d_split1_permutation = permutation[:d_split1_standard.shape[0]]

    if immutable:
        d_split1_standard.setflags(write=False)
        d_split2_complete.setflags(write=False)

    return d_split1_standard, d_split1_permutation


def download_cifar10h_data() -> np.ndarray:
    """
    Pulls cifar10h label data stream and returns it in numpy array.
    """

    url = 'https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-counts.npy'
    response = requests.get(url)
    response.raise_for_status()
    if response.status_code == requests.codes.ok:
        cifar10h_data = np.load(io.BytesIO(response.content))
    else:
        raise ValueError('CIFAR10H content was not found.')

    return cifar10h_data


def download_cifar10_data() -> Path:
    """
    Download CIFAR10 dataset and returns path to the test set
    """
    import wget
    local_path = Path.cwd() / 'InnerEyeDataQuality' / 'downloaded_data'
    local_path_to_test_batch = local_path / 'cifar-10-batches-py/test_batch'

    if not local_path_to_test_batch.exists():
        local_path.mkdir(parents=True, exist_ok=True)
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        path_to_tar = local_path / 'cifar10.tar.gz'
        wget.download(url, str(path_to_tar))
        tf = tarfile.open(str(path_to_tar))
        tf.extractall(local_path)
        os.remove(path_to_tar)

    return local_path_to_test_batch


def get_cifar10_label_names(file: Optional[Path] = None) -> List[str]:
    """
    TBD
    """
    if file:
        dict = load_cifar10_file(file)
        label_names = [_s.decode("utf-8") for _s in dict[b"label_names"]]
    else:
        label_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return label_names


def load_cifar10_file(file: Path) -> Dict:
    """
    TBD
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict


def plot_cifar10_images(sample_ids: List[int], save_directory: Path) -> None:
    """
    Displays a set of CIFAR-10 Images based on the input sample ids. In the title of the figure,
    label distribution is displayed as well to understand the sample difficulty.
    """

    path_cifar10_test_batch = download_cifar10_data()
    test_batch = load_cifar10_file(path_cifar10_test_batch)
    plot_images(test_batch[b'data'], sample_ids, save_directory=save_directory)


def plot_images(images: np.ndarray,
                selected_sample_ids: List[int],
                cifar10h_labels: Optional[np.ndarray] = None,
                label_names: Optional[List[str]] = None,
                save_directory: Optional[Path] = None) -> None:
    """
    Displays a set of CIFAR-10 Images based on the input sample ids. In the title of the figure,
    label distribution is displayed as well to understand the sample difficulty.
    """

    f, ax = plt.subplots(figsize=(2, 2))
    for sample_id in selected_sample_ids:
        img = np.reshape(images[sample_id, :], (3, 32, 32)).transpose(1, 2, 0)
        ax.imshow(img)

        if (cifar10h_labels is not None) and (label_names is not None):
            num_k_classes = 3
            label_distribution = cifar10h_labels[sample_id, :]
            k_min_val = np.sort(label_distribution)[-num_k_classes]
            available_classes = np.where(label_distribution >= k_min_val)[0]
            class_counts = label_distribution[available_classes]
            class_names = [label_names[_c] for _c in available_classes]
            ax_title = ''.join([a + '_' + str(b) + ' ' for a, b in zip(class_names, class_counts)])
            ax.set_title(ax_title)
        else:
            ax_title = f'CIFAR10H - Sample ID {sample_id}'
            ax.set_title(ax_title)

        if save_directory:
            save_directory.mkdir(parents=True, exist_ok=True)
            f.savefig(save_directory / f"{sample_id}.png", bbox_inches='tight')
            ax.clear()
            plt.close(f)
        else:
            plt.show()
            f, ax = plt.subplots(figsize=(2, 2))
