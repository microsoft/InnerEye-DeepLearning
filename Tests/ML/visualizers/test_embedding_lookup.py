#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch

from InnerEye.ML.visualizers.embedding_lookup import _find_nearest


def test_find_nearest():
    num_stored = 100
    num_test = 10
    num_neighbors = 4
    similarities = torch.randn(num_test, num_stored)

    sorted_top_indices, sorted_top_similarities = _find_nearest(similarities, num_neighbors)
    assert sorted_top_indices.shape == sorted_top_similarities.shape
    assert sorted_top_indices.shape == (num_test, num_neighbors)

    sorted_similarities, sorted_indices = similarities.sort(-1, descending=True)
    assert torch.allclose(sorted_top_similarities, sorted_similarities[:, :num_neighbors])
    assert torch.allclose(sorted_top_indices, sorted_indices[:, :num_neighbors])
