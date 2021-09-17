#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np
import pytest
import scipy

from scipy.spatial.distance import cosine
from sklearn import datasets
from typing import Tuple

from InnerEyeDataQuality.algorithms.graph import _get_affinity_matrix, build_connectivity_graph, \
    label_diffusion, GraphParameters
from InnerEyeDataQuality.selection.selectors.graph import GraphBasedSelector


def _create_circles_dataset() -> Tuple[np.ndarray, np.ndarray]:
    # Create a toy dataset
    n_samples = 1000
    noisy_circles, labels = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05, random_state=1234)

    return noisy_circles, labels


@pytest.mark.parametrize("distance_kernel", ["euclidean", "cosine"])
def test_affinity_matrix(distance_kernel: str) -> None:
    noisy_circles, _ = _create_circles_dataset()
    n_samples = noisy_circles.shape[0]

    # Check the connectivity of the nodes
    n_neighbors = n_samples // 100
    affinity_matrix = _get_affinity_matrix(embeddings=noisy_circles, n_neighbors=n_neighbors,
                                           distance_kernel=distance_kernel)

    # Assert the type and size of the output graph
    selected_index = 0
    assert isinstance(affinity_matrix, scipy.sparse.csr.csr_matrix)
    assert int(affinity_matrix[selected_index].count_nonzero()) == n_neighbors

    # Check if the neighbours are correct
    neig = affinity_matrix[selected_index]
    noisy_circles_self_removed = np.delete(noisy_circles, (selected_index), axis=0)
    if distance_kernel == "euclidean":
        min_index = np.argmin(np.sum((noisy_circles_self_removed - noisy_circles[selected_index]) ** 2, axis=1)) + 1
        assert min_index in neig.nonzero()[1]
    elif distance_kernel == "cosine":
        _distances = list()
        for _i in range(noisy_circles_self_removed.shape[0]):
            _distances.append(cosine(noisy_circles_self_removed[_i], noisy_circles[selected_index]))
        min_index = np.argmin(_distances) + 1
        assert min_index in neig.nonzero()[1]

    # Check that the diagonal elements are all zero.
    sym_A = (affinity_matrix + affinity_matrix.T) * 0.5
    assert np.all(np.diag(sym_A.todense()) == 0.0)


def test_normalised_symmetric_adjacency_graph() -> None:
    # Create a toy dataset
    noisy_circles, _ = _create_circles_dataset()
    n_samples = noisy_circles.shape[0]

    # Check the connectivity of the nodes
    n_neighbors = n_samples // 100
    Adj = build_connectivity_graph(normalised=True,
                                   embeddings=noisy_circles,
                                   n_neighbors=n_neighbors,
                                   distance_kernel="euclidean")

    # Test semi positive definite properties
    dense_adj = Adj.todense()
    x = np.random.rand(dense_adj.shape[0])
    assert np.matmul(np.matmul(x, dense_adj), x.transpose()) > 0

    # Check that the diagonal elements are all zero.
    assert np.all(np.diag(dense_adj) == 0.0)


def test_graph_diffusion() -> None:
    """
    1) Provide both labelled and unlabelled data points.
    2) Run label propagation, collect diffused labels for the unlabelled data
    3) Verify the match between true and diffused labels
    """

    # Create labelled and unlabelled data points
    noisy_circles, labels = _create_circles_dataset()
    n_samples = noisy_circles.shape[0]
    n_labelled = int(n_samples * 0.75)
    perm = np.random.permutation(n_samples)
    unlabelled_indices = perm[n_labelled:]

    # Build a graph and run label propagation
    graph_param = GraphParameters(
        n_neighbors=20,
        diffusion_alpha=0.99,
        cg_solver_max_iter=10,
        diffusion_batch_size=-1,
        distance_kernel='euclidean')
    graph = build_connectivity_graph(normalised=True,
                                     embeddings=noisy_circles,
                                     n_neighbors=graph_param.n_neighbors,
                                     distance_kernel=graph_param.distance_kernel)
    diffusion_input_labels = np.eye(2)[labels]
    laplacian = scipy.sparse.eye(n_samples) - graph_param.diffusion_alpha * graph
    laplacian_inv = scipy.sparse.linalg.inv(laplacian.tocsc()).todense()
    diffused_labels = label_diffusion(laplacian_inv, diffusion_input_labels, unlabelled_indices)

    # Compare the diffused labels against the true labels
    predicted_classes = np.asarray(np.argmax(diffused_labels, axis=1)).reshape(-1)
    target_classes = labels[unlabelled_indices]
    accuracy = np.sum(predicted_classes == target_classes) / target_classes.shape[0]

    assert np.isclose(accuracy, 1.0, rtol=1e-05, atol=1e-08, equal_nan=False)


def test_validate_class_priors() -> None:
    test_graph_params = GraphParameters(n_neighbors=1,
                                        diffusion_alpha=0.95,
                                        cg_solver_max_iter=10,
                                        diffusion_batch_size=1,
                                        distance_kernel="cosine")
    # Wrong shape
    with pytest.raises(AssertionError):
        GraphBasedSelector(num_classes=3,
                           num_samples=2,
                           embeddings=np.asarray([[3, 0, 1, 3], [1, 0, 1, 3]]),
                           sample_indices=None,
                           class_priors=np.array([0.1, 0.2]),
                           graph_params=test_graph_params)
    # Not summing up to one
    with pytest.raises(AssertionError):
        GraphBasedSelector(num_classes=3,
                           num_samples=2,
                           embeddings=np.asarray([[3, 0, 1, 3], [1, 0, 1, 3]]),
                           sample_indices=None,
                           class_priors=np.array([0.1, 0.2, 0.3]),
                           graph_params=test_graph_params)

    # Correct
    GraphBasedSelector(num_classes=3,
                       num_samples=2,
                       embeddings=np.asarray([[3, 0, 1, 3], [1, 0, 1, 3]]),
                       sample_indices=None,
                       class_priors=np.array([0.1, 0.2, 0.7]),
                       graph_params=test_graph_params)

    GraphBasedSelector(num_classes=3,
                       num_samples=2,
                       embeddings=np.asarray([[3, 0, 1, 3], [1, 0, 1, 3]]),
                       sample_indices=None,
                       class_priors=None,
                       graph_params=test_graph_params)
