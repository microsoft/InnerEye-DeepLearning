#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from pathlib import Path
from torch.nn import CosineSimilarity

from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.ML.visualizers.model_hooks import HookBasedFeatureExtractor

SimilarityFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

DEFAULT_SIMILARITY: SimilarityFunction = CosineSimilarity(dim=-1)


def _get_embedding_extractor(model: Union[DeviceAwareModule, torch.nn.DataParallel]) -> \
        HookBasedFeatureExtractor:
    target_layer = model.get_last_encoder_layer_names()
    return HookBasedFeatureExtractor(model, target_layer)


def _find_nearest(similarities: torch.Tensor, num_neighbors: int) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    # np.argpartition performs a partial sorting in O(N_stored), rather than requiring
    # a full sort that would cost O(N_stored*log(N_stored)).
    # The minus signs select the indices of the highest values instead of lowest.
    top_neighbor_indices = np.argpartition(similarities, -num_neighbors)[:, -num_neighbors:]
    top_similarities = similarities.gather(-1, top_neighbor_indices)

    # Sort K nearest neighbors in O(K*log(K)).
    sorted_indices = top_similarities.argsort(-1, descending=True)
    sorted_top_indices = top_neighbor_indices.gather(-1, sorted_indices)
    sorted_top_similarities = top_similarities.gather(-1, sorted_indices)

    return sorted_top_indices, sorted_top_similarities


class EmbeddingsStore:

    def __init__(self, model: Union[DeviceAwareModule, torch.nn.DataParallel]) -> None:
        self.model = model
        self.extractor = _get_embedding_extractor(model)

        self.embeddings: Optional[torch.Tensor] = None
        self.subject_ids: List[str] = []

    def store_embeddings(self,
                         input_batch: Sequence[torch.Tensor],
                         subject_ids: Sequence[str]) -> None:
        self.extractor.forward(*input_batch)
        self.embeddings = None
        self.subject_ids.extend(subject_ids)

    def get_embeddings(self) -> torch.Tensor:
        if self.embeddings is None:
            self.embeddings = torch.cat(self.extractor.outputs)
        return self.embeddings

    def lookup(self,
               test_input: Sequence[torch.Tensor],
               num_neighbors: int = 5,
               similarity_fn: SimilarityFunction = DEFAULT_SIMILARITY) \
            -> Tuple[np.ndarray, np.ndarray]:
        test_extractor = _get_embedding_extractor(self.model)
        stored_embeddings = self.get_embeddings()

        test_extractor.forward(*test_input)
        test_embeddings: torch.Tensor = test_extractor.outputs[0]
        test_embeddings = test_embeddings.unsqueeze(-3)

        # similarities.shape == (N_test, N_stored)
        similarities = similarity_fn(test_embeddings, stored_embeddings)
        sorted_top_indices, sorted_top_similarities = _find_nearest(similarities, num_neighbors)

        sorted_top_subject_ids = np.take(self.subject_ids, sorted_top_indices)

        return sorted_top_subject_ids, sorted_top_similarities.cpu().numpy()

    def save(self,
             subject_ids: Sequence[str],
             filename: str,
             embeddings_dir: Path) -> None:
        # TODO Save stored embeddings
        raise NotImplementedError

    def load(self,
             filename: str,
             embeddings_dir: Path) -> None:
        # TODO Load embeddings from disk
        raise NotImplementedError
