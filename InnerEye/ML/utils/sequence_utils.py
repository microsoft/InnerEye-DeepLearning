#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_sequence

from InnerEye.Common.common_util import check_properties_are_not_none
from InnerEye.ML.utils.image_util import NumpyOrTorch


@dataclass(frozen=True)
class MaskedModelOutputAndLabelSequences:
    """
    Dataclass to encapsulate masked model outputs, labels and associated subject ids
    """
    model_outputs: PackedSequence
    labels: PackedSequence
    subject_ids: Optional[Sequence[str]]

    def __post_init__(self) -> None:
        check_properties_are_not_none(self, ignore=["subject_ids"])

        if len(self.model_outputs.data) != len(self.labels.data):
            raise ValueError("model_outputs and labels must have the same length, "
                             f"found {len(self.model_outputs.data)} and {len(self.labels.data)}")

        if not torch.equal(self.model_outputs.batch_sizes, self.labels.batch_sizes):
            raise ValueError("batch_sizes for model_outputs and labels must be equal, "
                             f"found {self.model_outputs.batch_sizes} and {self.labels.batch_sizes}")

        if not torch.equal(self.model_outputs.sorted_indices, self.labels.sorted_indices):
            raise ValueError("sorted_indices for model_outputs and labels must be equal, "
                             f"found {self.model_outputs.sorted_indices} and {self.labels.sorted_indices}")

        if not torch.equal(self.model_outputs.unsorted_indices, self.labels.unsorted_indices):
            raise ValueError("unsorted_indices for model_outputs and labels must be equal, "
                             f"found {self.model_outputs.unsorted_indices} and {self.labels.unsorted_indices}")

        _expected_subjects = self.labels.batch_sizes.max().item()
        if self.subject_ids is not None and len(self.subject_ids) != _expected_subjects:
            raise ValueError(f"expected {_expected_subjects} subject_ids but found {len(self.subject_ids)}")


def sequences_to_padded_tensor(sequences: List[torch.Tensor],
                               padding_value: float = 0.0) -> torch.Tensor:
    """
    Method to convert possibly unequal length sequences to a padded tensor.
    :param sequences: List of Tensors to pad
    :param padding_value: Padding value to use, default is 0.0
    :return: Output tensor with shape B x * where * is the max dimensions from the list of provided tensors.
    And B is the number of tensors in the list of sequences provided.
    """
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)


def map_packed_sequence_data(x: PackedSequence, f: Callable[[torch.Tensor], torch.Tensor]) -> PackedSequence:
    """
    Helper function to apply a map transform to a packed sequence
    """
    _x_data = f(x.data)
    # make sure the function is a map function and maintains the original shape of the data tensor
    if x.data.shape != _x_data.shape:
        raise ValueError("The provided function must be a map function, but it changed the original tensor's shape"
                         f" from {x.data.shape} to {_x_data.shape}")

    return PackedSequence(data=_x_data, batch_sizes=x.batch_sizes,
                          sorted_indices=x.sorted_indices, unsorted_indices=x.unsorted_indices)


def get_masked_model_outputs_and_labels(model_output: torch.Tensor,
                                        labels: NumpyOrTorch,
                                        subject_ids: Optional[Sequence[str]] = None) \
        -> Optional[MaskedModelOutputAndLabelSequences]:
    """
    Helper function to get masked model outputs, labels and their associated subject ids. Masking is performed
    by excluding the NaN model outputs and labels based on a bool mask created using the
    occurrences of NaN in the labels provided.
    :param model_output: The model output tensor to mask.
    :param labels: The label tensor to use for mask, and use for masking.
    :param subject_ids: The associated subject ids.
    :return: None if all labels are required to be masked, otherwise MaskedModelOutputAndLabelSequences
    """
    non_nan_idxs = ~torch.isnan(labels)
    _subject_ids: Optional[List[Any]] = [] if subject_ids is not None else None
    _model_output_tensors, _label_tensors = [], []
    # iterate over each of the sequences to create masked tensors
    for i in range(non_nan_idxs.shape[0]):
        x = non_nan_idxs[i]
        masked_model_output, masked_labels = model_output[i, x], labels[i, x]
        # if all the elements of the sequence are masked, then drop the subject
        if masked_labels.numel() > 0:
            _model_output_tensors.append(masked_model_output)
            _label_tensors.append(masked_labels)
            if _subject_ids is not None:
                assert subject_ids is not None
                _subject_ids.append(subject_ids[i])

    # since it is not possible to create a packed sequence with empty tensors,
    # make sure we have valid tensors to pack, otherwise return None.
    if len(_label_tensors) > 0:
        labels_packed = pack_sequence(_label_tensors, enforce_sorted=False)
        # make sure the subject ids are in the same order as the packed sequences
        if _subject_ids is not None:
            _subject_ids = np.array(_subject_ids)[labels_packed.sorted_indices.cpu()].tolist()
            # If there is only one subject, tolist() returns a string instead of a list.
            if isinstance(_subject_ids, str):
                _subject_ids = [_subject_ids]
        return MaskedModelOutputAndLabelSequences(
            model_outputs=pack_sequence(_model_output_tensors, enforce_sorted=False),
            labels=labels_packed,
            subject_ids=_subject_ids
        )
    else:
        return None
