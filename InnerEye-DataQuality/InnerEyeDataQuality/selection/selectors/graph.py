#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import timeit
from pathlib import Path
from typing import Generator, List, Optional, Any

import scipy
import numpy as np
import logging
from numpy.random import RandomState
from scipy.sparse import find

from InnerEyeDataQuality.datasets.cifar10_utils import plot_cifar10_images
from InnerEyeDataQuality.algorithms.graph import GraphParameters, build_connectivity_graph, label_diffusion
from InnerEyeDataQuality.selection.selectors.label_based import LabelDistributionBasedSampler, LabelBasedDecisionRule


class GraphBasedSelector(LabelDistributionBasedSampler):
    """
    TBD
    """

    def __init__(self,
                 num_samples: int,
                 num_classes: int,
                 embeddings: np.ndarray,
                 sample_indices: Optional[np.ndarray],
                 graph_params: GraphParameters,
                 name: str = 'Graph Based Sample Selection',
                 allow_repeat_samples: bool = False,
                 bald_score: Optional[np.ndarray] = None,
                 class_priors: Optional[np.ndarray] = None,
                 diffusion_normalizing_factor: float = 0.01,
                 **kwargs: Any) -> None:
        """
        :param initial_labels: The initial set of labels assigned to each sample in the dataset.
                               This can be seen as histogram of labels (n_samples, n_classes)
        :param embeddings: Sample embeddings collected from a NN model/s (n_samples, embedding_dim)
        :param sample_indices: Permutation of the dataset indices - Intended to be used for plotting purposes.
        :param graph_params: GraphParameters to construct to the graph selector. If None, use the
        default parameters (neighbors = num_samples / 200, diffusion_alpha = 0.95, cg_solver_max_iter=10,
        diffusion_batch_size=num_samples / 10, distance_kernel="cosine").
        :param bald_score
        :param class_priors: an array of shape [n_classes,] with the prior probability associated to each class.
        :param diffusion_normalizing_factor: factor to normalize the diffused labels
        """

        super().__init__(num_samples=num_samples,
                         num_classes=num_classes,
                         decision_rule=LabelBasedDecisionRule.CROSS_ENTROPY,
                         name=name,
                         allow_repeat_samples=allow_repeat_samples,
                         embeddings=embeddings)

        # Construct a connectivity graph 
        self.graph_params = graph_params
        self.graph = build_connectivity_graph(normalised=True,
                                              embeddings=embeddings,
                                              n_neighbors=self.graph_params.n_neighbors,
                                              distance_kernel=self.graph_params.distance_kernel)

        # Build Laplacian
        laplacian = scipy.sparse.eye(self.num_samples) - graph_params.diffusion_alpha * self.graph
        self.laplacian_inv = scipy.sparse.linalg.inv(laplacian.tocsc()).todense()

        self.sample_indices = sample_indices
        if bald_score is not None:
            raise NotImplementedError('BALD score ranking not yet implemented')
        self.bald_score = bald_score
        self.class_priors = class_priors
        self.diffusion_normalizing_factor = diffusion_normalizing_factor
        self.validate_class_priors()

    def validate_class_priors(self) -> None:
        """
        Post init function to validate class priors argument
        """
        if self.class_priors is not None:
            assert self.class_priors.ndim == 1
            assert self.class_priors.shape[0] == self.num_classes
            assert np.sum(self.class_priors) == 1

    def get_predicted_label_distribution(self, current_labels: np.ndarray) -> np.ndarray:
        # Normalise current label distributions
        assert isinstance(current_labels, np.ndarray)

        from InnerEyeDataQuality.utils.generic import convert_labels_to_one_hot
        input_label_distribution = convert_labels_to_one_hot(np.argmax(current_labels, axis=1),
                                                             n_classes=self.num_classes)

        # Label diffusion
        start_time = timeit.default_timer()
        lp_batch_size = self.graph_params.diffusion_batch_size
        perm = RandomState(seed=1234).permutation(self.num_samples)
        diffused_labels = np.empty(shape=(self.num_samples, self.num_classes)) * np.nan

        # Run label diffusion for different subsets of query nodes
        assert isinstance(lp_batch_size, int)
        for batch_ids in self.chunks(perm, lp_batch_size):
            diffused_labels[batch_ids, :] = label_diffusion(
                inv_laplacian=self.laplacian_inv,
                labels=input_label_distribution,
                query_batch_ids=batch_ids,
                class_priors=self.class_priors,
                diffusion_normalizing_factor=self.diffusion_normalizing_factor)
        assert not np.isnan(diffused_labels).any()
        logging.info("LP Computation: {0:.3f} seconds".format(timeit.default_timer() - start_time))
        return diffused_labels

    def plot_selected_sample(self, sample_id: int, include_knn: bool = True) -> None:
        if self.sample_indices is None:
            raise ValueError("sample_indices is not set.")
        # Show the closest examples in the dataset
        local_connectivity = self.graph[sample_id, :]
        _m = np.array([sample_id])
        if include_knn:
            _m = np.append(_m, find(local_connectivity)[1], axis=0)
        save_directory = Path(__file__).parent.parent.absolute() / f"experiments/screenshots_graph/{sample_id}"
        plot_cifar10_images(sample_ids=self.sample_indices[_m.tolist()], save_directory=save_directory)

    @staticmethod
    def chunks(lst: List, n: int) -> Generator:
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]


