#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np

class LabelDistribution(object):
    """
    LabelDistribution class handles sampling from a label distribution with reproducible behavior given a seed
    """

    def __init__(self,
                 seed: int,
                 label_counts: np.ndarray,
                 temperature: float = 1.0,
                 offset: float = 0.0) -> None:
        """
        :param seed: The random seed used to ensure reproducible behaviour
        :param label_counts: An array of shape (num_samples, num_classes) where each entry represents the number of
        labels available for each sample and class
        :param temperature: A temperature a value that will be used to temperature scale the distribution, default is
        1.0  which is equivalent to no scaling; temperature must be greater than 0.0, values between 0 and 1 will result
        in a sharper distribution and values greater than 1 in a more uniform distribution over classes.
        :param offset: Offset parameter to control the noise rate in sampling initial labels.
        All classes are assigned a uniform fixed offset amount.
        """
        assert label_counts.dtype == np.int64
        assert label_counts.ndim == 2
        self.num_classes = label_counts.shape[1]
        self.num_samples = label_counts.shape[0]
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.label_counts = label_counts
        self.temperature = temperature

        # make the distribution
        self.distribution = label_counts / np.sum(label_counts, axis=1, keepdims=True)
        assert np.isfinite(self.distribution).all()
        assert self.temperature > 0

        # scale distribution based on temperature and offset
        _d = np.power(self.distribution, 1. / temperature)
        _d = _d / np.sum(_d, axis=1, keepdims=True)
        self.distribution_temp_scaled = self.add_noise_to_distribution(offset, _d, 'asym') if offset > 0.0 else _d

        # check if there are multiple labels per data point
        self.is_multi_label_per_sample = np.all(np.sum(label_counts, axis=1) > 1.0)

    def sample_initial_labels_for_all(self) -> np.ndarray:
        """
        Sample one label for each sample in the dataset according to its label distribution
        :return: None
        """
        if not self.is_multi_label_per_sample:
            RuntimeWarning("Sampling labels from one-hot encoded distribution - Multi labels are not available")

        sampling_fn = lambda p: self.random_state.choice(self.num_classes, 1, p=p)

        return np.squeeze(np.apply_along_axis(sampling_fn, arr=self.distribution_temp_scaled, axis=1))

    def sample(self, sample_idx: int) -> int:
        """
        Sample one label for a given sample index
        :param sample_idx: The sample index for which the label will be sampled
        :return: None
        """
        return self.random_state.choice(self.num_classes, 1, p=self.distribution[sample_idx])[0]

    def add_noise_to_distribution(self, offset: float, distribution: np.ndarray, noise_model_type: str) -> np.ndarray:
        from InnerEyeDataQuality.datasets.label_noise_model import get_cifar10_asym_noise_model
        from InnerEyeDataQuality.datasets.label_noise_model import get_cifar10_sym_noise_model

        # Create noise model
        if noise_model_type == 'sym':
            noise_model = get_cifar10_sym_noise_model(eta=1.0)
        elif noise_model_type == 'asym':
            noise_model = get_cifar10_asym_noise_model(eta=1.0)
        else:
            raise ValueError("Unknown noise model type")
        np.fill_diagonal(noise_model, 0.0)
        noise_model *= offset

        # Add this noise on top of every sample in the dataset
        for _ii in range(self.num_samples):
            true_label = np.argmax(self.label_counts[_ii])
            distribution[_ii] += noise_model[true_label]

        # Normalise the distribution
        return distribution / np.sum(distribution, axis=1, keepdims=True)
