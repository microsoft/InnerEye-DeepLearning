#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import io
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import PIL.Image
import numpy as np
import requests
import torchvision

from InnerEyeDataQuality.datasets.cifar10_utils import get_cifar10_label_names
from InnerEyeDataQuality.datasets.label_distribution import LabelDistribution
from InnerEyeDataQuality.evaluation.metrics import compute_label_entropy
from InnerEyeDataQuality.selection.simulation_statistics import SimulationStats, get_ambiguous_sample_ids
from InnerEyeDataQuality.utils.generic import convert_labels_to_one_hot
import wget

TOTAL_CIFAR10H_DATASET_SIZE = 10000

class CIFAR10H(torchvision.datasets.CIFAR10):
    """
    Dataset class for the CIFAR10H dataset. The CIFAR10H dataset is the CIFAR10 test set but all the samples have
    been labelled my multiple annotators
    """

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 num_samples: Optional[int] = None,
                 preset_indices: Optional[np.ndarray] = None,
                 seed: int = 1234,
                 noise_temperature: float = 1.0,
                 noise_offset: float = 0.0) -> None:
        """
        :param root: The directory in which the CIFAR10 images will be stored
        :param transform: Transform to apply to the images
        :param num_samples: The number of samples to use out of a maximum of TOTAL_CIFAR10H_DATASET_SIZE
        :param seed: The random seed that defines which samples are train/test and which labels are sampled
        :param preset_indices: Image indices that will be used to create a subset of CIFAR10H dataset.
                               If not specified and num_samples < 10000 then random sub-selection is performed.
        :param shuffle: Whether to shuffle the data before splitting into training and test sets.
        :param noise_temperature: A temperature a value that is used to temperature scale the label distribution.
        :param noise_offset: Offset parameter to control the noise rate in sampling initial labels.
        """
        super().__init__(root, train=False, transform=transform, target_transform=None, download=True)
        num_samples = TOTAL_CIFAR10H_DATASET_SIZE if num_samples is None else num_samples

        self.seed = seed
        cifar10h_labels = self.download_cifar10h_labels(self.root)
        self.num_classes = cifar10h_labels.shape[1]
        assert cifar10h_labels.shape[0] == TOTAL_CIFAR10H_DATASET_SIZE
        assert self.num_classes == 10
        assert 0 < num_samples <= TOTAL_CIFAR10H_DATASET_SIZE
        self.num_samples = num_samples

        # Create a set indices that 
        self.indices = self.get_dataset_indices(num_samples, cifar10h_labels, keep_hard_samples=True, seed=seed) \
                                                if preset_indices is None else preset_indices    
        self.verify_data_indices()
        self.label_counts = cifar10h_labels[self.indices]
        self.label_counts.flags.writeable = False
        self.true_label_entropy = compute_label_entropy(label_counts=self.label_counts)

        self.label_distribution = LabelDistribution(seed, self.label_counts, noise_temperature, noise_offset)
        self.targets = self.label_distribution.sample_initial_labels_for_all()

        # Check the class distribution
        _, class_counts = np.unique(self.targets, return_counts=True)
        class_distribution = np.array([_c/self.num_samples for _c in class_counts])
        logging.info(f"Preparing dataset: CIFAR10H (N={self.num_samples})")
        logging.info(f"Class distribution (%) (true labels): {class_distribution * 100.0}")
        self.clean_targets = np.argmax(self.label_counts, axis=1)
        # Identify true ambiguous and clear label noise cases
        self._identify_sample_types()

    def _identify_sample_types(self) -> None:
        """
        Stores and logs clear label noise and ambiguous case types.
        """
        label_stats = SimulationStats(name="cifar10h", true_label_counts=self.label_counts,
                                      initial_labels=convert_labels_to_one_hot(self.targets, self.num_classes))
        self.ambiguous_mislabelled_cases = label_stats.mislabelled_ambiguous_sample_ids[0]
        self.clear_mislabeled_cases = label_stats.mislabelled_not_ambiguous_sample_ids[0]
        self.ambiguity_metric_args = {"ambiguous_mislabelled_ids": self.ambiguous_mislabelled_cases,
                                      "clear_mislabelled_ids": self.clear_mislabeled_cases,
                                      "true_label_entropy": self.true_label_entropy}

        # Log dataset details
        logging.info(f"Ambiguous mislabeled cases: {100 * len(self.ambiguous_mislabelled_cases) / self.num_samples}%")
        logging.info(f"Clear mislabeled cases: {100 * len(self.clear_mislabeled_cases) / self.num_samples}%\n")

    @classmethod
    def download_cifar10h_labels(self, root: str = ".") -> np.ndarray:
        """
        Pulls cifar10h label data stream and returns it in numpy array.
        """
        try:
            cifar10h_labels = np.load(Path(root) / "cifar10h-counts.npy")
        except FileNotFoundError:
            url = 'https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-counts.npy'
            response = requests.get(url)
            response.raise_for_status()
            if response.status_code == requests.codes.ok:
                cifar10h_labels = np.load(io.BytesIO(response.content))
            else:
                raise ValueError('Failed to download CIFAR10H labels!')

        return cifar10h_labels

    def __getitem__(self, index: int) -> Tuple[PIL.Image.Image, int]:
        """
        :param index: The index of the sample to be fetched
        :return: The image and label tensors
        """
        img = PIL.Image.fromarray(self.data[self.indices[index]])
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(target)

    def __len__(self) -> int:
        """

        :return: The size of the dataset
        """
        return len(self.indices)

    def get_label_names(self) -> List[str]:
        return get_cifar10_label_names()

    def verify_data_indices(self) -> None:
        assert isinstance(self.indices, np.ndarray)
        assert self.indices.size == self.num_samples
        assert np.all(self.indices < TOTAL_CIFAR10H_DATASET_SIZE)
        assert np.all(0 <= self.indices)
        _, c = np.unique(self.indices, return_counts=True)
        assert np.all(c == 1)

    def get_dataset_indices(self,
                            num_samples: int,
                            true_label_counts: np.ndarray,
                            keep_hard_samples: bool,
                            seed: int = 1234) -> np.ndarray:
        """
        Function to choose a subset of the CIFAR10H dataset. Returns selected subset of sample
        indices in a shuffled order.

        :param num_samples: Number of samples in the selected subset.
                            If the full dataset size is specified then indices are just shuffled and returned.
        :param true_label_counts: True label counts of CIFAR10H images (num_samples x num_classes)
        :param keep_hard_samples: If set to True, all hard examples are kept in the selected subset of points.
        :param seed: Random seed used in shuffling data indices.
        """
        random_state = np.random.RandomState(seed=seed)

        assert num_samples <= TOTAL_CIFAR10H_DATASET_SIZE
        if (not keep_hard_samples) or (num_samples == TOTAL_CIFAR10H_DATASET_SIZE):
            indices = random_state.permutation(true_label_counts.shape[0])
            return indices[:num_samples]

        # Identify difficult samples and keep them in the dataset
        hard_sample_indices = get_ambiguous_sample_ids(true_label_counts)
        if hard_sample_indices.shape[0] > num_samples:
            logging.info(f"Total number of hard samples: {hard_sample_indices.shape[0]} and requested: {num_samples}")
            hard_sample_indices = hard_sample_indices[:num_samples]
        num_hard_samples = hard_sample_indices.shape[0]

        # Sample the remaining indices randomly and aggregate
        remaining_indices = np.setdiff1d(range(TOTAL_CIFAR10H_DATASET_SIZE), hard_sample_indices)
        easy_sample_indices = random_state.choice(remaining_indices, num_samples - num_hard_samples, replace=False)
        indices = np.concatenate([hard_sample_indices, easy_sample_indices], axis=0)
        random_state.shuffle(indices)

        # Assert that there are no repeated indices
        _, _counts = np.unique(indices, return_counts=True)
        assert not np.any(_counts > 1)

        return indices

