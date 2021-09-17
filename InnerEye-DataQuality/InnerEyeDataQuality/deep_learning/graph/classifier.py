#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from typing import Tuple

import numpy as np
import scipy
import torch
from sklearn.neighbors import kneighbors_graph

from InnerEyeDataQuality.algorithms.graph import GraphParameters, build_connectivity_graph, label_diffusion
from InnerEyeDataQuality.utils.generic import convert_labels_to_one_hot, find_set_difference_torch


class GraphClassifier:
    """
    Graph based classifier. Builds a graph and runs label diffusion to classify new points.
    """

    def __init__(self,
                 num_samples: int,
                 num_classes: int,
                 labels: np.ndarray,
                 device: torch.device) -> None:
        self.graph = None
        self.device = device
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.graph_params = GraphParameters(n_neighbors=12,
                                            diffusion_alpha=0.95,
                                            cg_solver_max_iter=10,
                                            diffusion_batch_size=None,
                                            distance_kernel="cosine")

        # Convert to one-hot label distribution
        self.labels = np.array(labels) if isinstance(labels, list) else labels
        self.one_hot_labels = convert_labels_to_one_hot(self.labels, n_classes=num_classes)
        assert np.all(self.one_hot_labels.sum(axis=1) == 1.0)
        assert self.labels.shape[0] == num_samples

    def build_graph(self, embeddings: np.ndarray) -> None:
        logging.info("Building a new connectivity graph")
        assert embeddings.shape[0] == self.num_samples

        # Build a connectivity graph and k-nearest neighbours.
        n_neighbors = self.graph_params.n_neighbors
        self.knn = kneighbors_graph(embeddings, n_neighbors, metric=self.graph_params.distance_kernel, n_jobs=-1)
        self.graph = build_connectivity_graph(normalised=True,
                                              embeddings=embeddings,
                                              n_neighbors=n_neighbors,
                                              distance_kernel=self.graph_params.distance_kernel)
        laplacian = scipy.sparse.eye(self.num_samples) - self.graph_params.diffusion_alpha * self.graph  # type: ignore
        self.laplacian_inv = scipy.sparse.linalg.inv(laplacian.tocsc()).todense()

    def fit(self, query_batch_ids: np.ndarray) -> np.ndarray:
        """
        Run label diffusion and identify potential labels for query samples
        """
        diffused_labels = label_diffusion(inv_laplacian=self.laplacian_inv,
                                          labels=self.one_hot_labels,
                                          query_batch_ids=query_batch_ids,
                                          diffusion_normalizing_factor=0.01)
        assert np.all(diffused_labels.shape == (query_batch_ids.size, self.num_classes))
        assert not np.isnan(diffused_labels).any()
        return diffused_labels

    def filter_cases(self,
                     local_ind_keep: torch.Tensor,
                     local_ind_exclude: torch.Tensor,
                     global_ind: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter list of cases to drop based on diffused labels. If labels and diffused labels agree, do not
        exclude the sample.
        :param local_ind_keep: original list of samples to keep
        :param local_ind_exclude: original list of samples to drop
        :param global_ind: list of global indices
        :return: updated list of indices to keep and drop
        """
        # If input indices are empty return an empty tensor
        if torch.numel(local_ind_exclude) == 0:
            return local_ind_keep, local_ind_exclude

        # Check input variable consistency
        num_samples = torch.numel(global_ind)
        assert num_samples == torch.numel(local_ind_exclude) + torch.numel(local_ind_keep)
        global_ind = global_ind.to(local_ind_keep.device)
        all_local_ind = torch.tensor(range(num_samples), device=local_ind_keep.device, dtype=local_ind_keep.dtype)

        # Run graph diffusion to filter out incorrectly picked indices.
        global_ind_exclude = global_ind[local_ind_exclude].cpu().numpy()
        diffused_probs = self.fit(global_ind_exclude)
        graph_pred = np.argmax(diffused_probs, axis=1)
        initial_labels = self.labels[global_ind_exclude]

        # Update the local indices for exclude
        local_ind_exclude_updated = local_ind_exclude[graph_pred != initial_labels]
        local_ind_keep_updated = find_set_difference_torch(all_local_ind, local_ind_exclude_updated)

        return local_ind_keep_updated, local_ind_exclude_updated

    def compute_mingling_index(self, indices: np.ndarray) -> None:
        """
        Computes mingling index of each graph node based on label distribution in local graphs.
        """
        mingling = np.zeros(indices.shape, dtype=np.float)
        for loop_id, _ind in enumerate(indices):
            disagreement = self.labels[self.knn[_ind].indices] != self.labels[_ind]
            mingling[loop_id] = np.sum(disagreement) / float(disagreement.size)
