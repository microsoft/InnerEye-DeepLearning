#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from typing import Callable, Optional, Tuple

import numpy as np
import torchvision
import PIL.Image

from default_paths import FIGURE_DIR
from InnerEyeDataQuality.datasets.label_distribution import LabelDistribution
from InnerEyeDataQuality.datasets.label_noise_model import get_cifar10h_confusion_matrix, \
    get_cifar10_asym_noise_model, get_cifar10_sym_noise_model
from InnerEyeDataQuality.utils.plot import plot_confusion_matrix
from InnerEyeDataQuality.datasets.cifar10_utils import get_cifar10_label_names


class CIFAR10AsymNoise(torchvision.datasets.CIFAR10):
    """
    Dataset class for the CIFAR10 dataset where target labels are sampled from a confusion matrix.
    """

    def __init__(self,
                 root: str,
                 train: bool,
                 transform: Optional[Callable] = None,
                 download: bool = True,
                 use_fixed_labels: bool = True,
                 seed: int = 1234,
                 ) -> None:
        """
        :param root: The directory in which the CIFAR10 images will be stored.
        :param train: If True, creates dataset from training set, otherwise creates from test set.
        :param transform: Transform to apply to the images.
        :param download: Whether to download the dataset if it is not already in the local disk.
        :param use_fixed_labels: If true labels are sampled only once and are kept fixed. If false labels are sampled at
                                 each get_item() function call from label distribution.
        :param seed: The random seed that defines which samples are train/test and which labels are sampled.
        """
        super().__init__(root, train=train, transform=transform, target_transform=None, download=download)

        self.seed = seed
        self.targets = np.array(self.targets, dtype=np.int64)  # type: ignore
        self.num_classes = np.unique(self.targets, return_counts=False).size
        self.num_samples = len(self.data)
        self.label_counts = np.eye(self.num_classes, dtype=np.int64)[self.targets]
        self.np_random_state = np.random.RandomState(seed)
        self.use_fixed_labels = use_fixed_labels
        self.clean_targets = np.argmax(self.label_counts, axis=1)
        logging.info(f"Preparing dataset: CIFAR10-Asym-Noise (N={self.num_samples})")

        # Create label distribution for simulation of label adjudication
        self.label_distribution = LabelDistribution(seed, self.label_counts, temperature=1.0)

        # Add asymmetric noise on the labels
        self.noise_model = self.create_noise_transition_model(self.targets, self.num_classes, "cifar10_sym")

        # Sample fixed labels from the distribution
        if use_fixed_labels:
            self.targets = self.sample_labels_from_model()
            # Identify label noise cases
            noise_rate = np.mean(self.clean_targets != self.targets) * 100.0

            # Check the class distribution after sampling
            class_distribution = self.get_class_frequencies(targets=self.targets, num_classes=self.num_classes)

            # Log dataset details
            logging.info(f"Class distribution (%) (true labels): {class_distribution * 100.0}")
            logging.info(f"Label noise rate: {noise_rate}")

    @staticmethod
    def create_noise_transition_model(labels: np.ndarray, num_classes: int, noise_model: str) -> np.ndarray:
        logging.info(f"Using {noise_model} label noise model")
        if noise_model == "cifar10h":
            transition_matrix = get_cifar10h_confusion_matrix(temperature=2.0)
        elif noise_model == "cifar10_asym":
            transition_matrix = get_cifar10_asym_noise_model(eta=0.4)
        elif noise_model == "cifar10_sym":
            transition_matrix = get_cifar10_sym_noise_model(eta=0.4)
        else:
            raise ValueError("Unknown noise transition model")
        assert(np.all(np.sum(transition_matrix, axis=1) - 1.00 < 1e-6))  # Checks = it sums up to one.

        # Visualise the noise model
        plot_confusion_matrix(list(), list(), get_cifar10_label_names(), cm=transition_matrix, save_path=FIGURE_DIR)

        # Compute the expected noise rate
        assert labels.ndim == 1
        exp_noise_rate = 1.0 - np.sum(
            np.diag(transition_matrix) * CIFAR10AsymNoise.get_class_frequencies(labels, num_classes))
        logging.info(f"Expected noise rate (transition model): {exp_noise_rate}")

        return transition_matrix

    def sample_labels_from_model(self, sample_index: Optional[int] = None) -> np.ndarray:
        # Sample based on the transition matrix and original labels (labels, transition mat, seed)
        if sample_index is not None:
            cur_label = self.targets[sample_index]
            label = self.np_random_state.choice(self.num_classes, 1, p=self.noise_model[cur_label, :])[0]
            return label

        noisy_targets = np.zeros_like(self.targets)
        for ii in range(self.num_samples):
            cur_label = self.targets[ii]
            noisy_targets[ii] = self.np_random_state.choice(self.num_classes, 1, p=self.noise_model[cur_label, :])[0]      

        return noisy_targets

    @staticmethod
    def get_class_frequencies(targets: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Returns normalised frequency of each semantic class
        """
        assert targets.ndim == 1
        class_ids, class_counts = np.unique(targets, return_counts=True)
        class_distribution = np.zeros(num_classes, dtype=float)
        for ii in range(num_classes):
            if np.any(class_ids == ii):
                class_distribution[ii] = class_counts[class_ids == ii]/targets.size

        return class_distribution

    def __getitem__(self, index: int) -> Tuple[PIL.Image.Image, int]:
        """
        :param index: The index of the sample to be fetched
        :return: The image and label tensors
        """
        img = PIL.Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img)

        if self.use_fixed_labels:
            target = self.targets[index]
        else:
            target = self.sample_labels_from_model(sample_index=index)

        return img, int(target)
