#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, List, TypeVar

import numpy as np
import torch

from InnerEye.Common.common_util import check_properties_are_not_none
from InnerEye.ML.dataset.scalar_sample import ScalarItem, SequenceDataSource
from InnerEye.ML.utils.sequence_utils import sequences_to_padded_tensor

T = TypeVar('T', SequenceDataSource, ScalarItem)


@dataclass(frozen=True)
class ClassificationItemSequence(Generic[T]):
    """
    A class that holds a sequence of samples for a given patient ID.
    """
    id: str
    items: List[T]

    def __post_init__(self) -> None:
        check_properties_are_not_none(self)

    @staticmethod
    def create_labels_tensor_for_minibatch(sequences: List[ClassificationItemSequence[ScalarItem]],
                                           target_indices: List[int]) -> torch.Tensor:
        """
        Create label tensor for a minibatch training from a list of sequences for the provided
        target indices. If sequences are unequal then they are padded with a NaN value.
        :param sequences: sequences to create label tensor from.
        :param target_indices: label indices for which to extract label for from the provided sequences.
        :return: A label tensor with NaN padding if required.
        """
        return sequences_to_padded_tensor(
            sequences=[seq.get_labels_at_target_indices(target_indices) for seq in sequences],
            padding_value=np.nan
        )

    @staticmethod
    def from_minibatch(minibatch: Dict[str, Any]) -> List[ClassificationItemSequence[ScalarItem]]:
        """
        Creates a list of ClassificationItemSequence from the output of a data loader. The data loader returns a
        dictionary with collated items, this function is effectively the inverse.
        :param minibatch: A dictionary that contains the collated fields of ClassificationItemSequence objects.
        :return: A list of ClassificationItemSequence objects.
        """
        # batched is a de-generate ClassificationItemSequence, with id being a list of strings, and items being
        # a list of lists.
        batched = ClassificationItemSequence(**minibatch)
        return [ClassificationItemSequence(id=sample_id, items=items)
                for (sample_id, items) in zip(batched.id, batched.items)]

    def get_labels_at_target_indices(self, target_indices: List[int]) -> torch.Tensor:
        """
        Gets the label fields for the sequence elements with the given zero-based indices, if they exist
        otherwise fill with NaN.
        """
        target_indices = sorted(target_indices)
        nan = torch.tensor([np.nan], device=self.items[0].label.device)

        def _get_label_or_nan(idx: int) -> torch.Tensor:
            return self.items[idx].label if idx < len(self.items) else nan

        if any(p < 0 for p in target_indices):
            raise ValueError("Argument target_indices cannot contain negative values")

        return torch.stack(list(map(_get_label_or_nan, target_indices)))


ListOfSequences = List[ClassificationItemSequence[SequenceDataSource]]
