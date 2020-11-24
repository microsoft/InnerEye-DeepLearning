#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import uuid
from typing import List, Tuple

import torch

from InnerEye.ML.models.architectures.base_model import BaseModel
from InnerEye.ML.visualizers.embedding_lookup import EmbeddingsStore, _find_nearest


def _generate_batch(batch_size: int, input_dim: int) -> Tuple[torch.Tensor, List[str]]:
    input_batch = torch.randn(batch_size, input_dim)
    subject_ids = [str(uuid.uuid4()) for _ in range(batch_size)]
    return input_batch, subject_ids


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


class IdentityModel(BaseModel):
    def __init__(self) -> None:
        super().__init__(input_channels=1, name='IdentityModel')

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        # returns the input as it is
        return x

    def get_all_child_layers(self) -> List[torch.nn.Module]:
        return []


def test_embedding_extraction():
    batch_size = 100
    input_dim = embedding_dim = 10
    num_batches = 3

    mock_dataloader = [_generate_batch(batch_size, input_dim) for _ in range(num_batches)]
    all_inputs = torch.cat([input_batch for input_batch, _ in mock_dataloader])
    all_subject_ids = [sid for _, subject_ids in mock_dataloader for sid in subject_ids]

    model = IdentityModel()

    batched_embeddings_store = EmbeddingsStore(model)
    for input_batch, subject_ids in mock_dataloader:
        batched_embeddings_store.store_embeddings([input_batch], subject_ids)

    full_embeddings_store = EmbeddingsStore(model)
    full_embeddings_store.store_embeddings([all_inputs], all_subject_ids)

    num_stored = num_batches * batch_size
    stored_batched_embeddings = batched_embeddings_store.get_embeddings()
    stored_full_embeddings = full_embeddings_store.get_embeddings()
    assert stored_batched_embeddings.shape == (num_stored, embedding_dim)
    assert stored_full_embeddings.shape == (num_stored, embedding_dim)
    assert torch.allclose(stored_batched_embeddings, stored_full_embeddings)

    assert batched_embeddings_store.subject_ids == all_subject_ids
    assert full_embeddings_store.subject_ids == all_subject_ids


def test_embedding_lookup():
    input_dim = 10
    num_stored_samples = 100
    num_test_samples = 20
    num_neighbors = 5

    all_inputs, all_subject_ids = _generate_batch(num_stored_samples, input_dim)
    test_inputs, test_subject_ids = _generate_batch(num_test_samples, input_dim)

    model = IdentityModel()

    embeddings_store = EmbeddingsStore(model)
    embeddings_store.store_embeddings([all_inputs], all_subject_ids)

    matched_ids, similarities = embeddings_store.lookup([test_inputs], num_neighbors)
    assert matched_ids.shape == (num_test_samples, num_neighbors)
    assert similarities.shape == (num_test_samples, num_neighbors)


# TODO Add tests for embedding serialization
def test_embedding_serialization():
    pass
