#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Optional, Sequence

import numpy as np
import pytest
import torch
from torch.nn.utils.rnn import pack_sequence

from InnerEye.ML.utils.sequence_utils import get_masked_model_outputs_and_labels, map_packed_sequence_data, \
    sequences_to_padded_tensor


def test_map_packed_sequence_data() -> None:
    """
    Test to ensure helper function to apply a map transform to a packed sequence returns expected results.
    """
    packed = pack_sequence([torch.tensor([[1.0], [2.0]])], enforce_sorted=False)
    mapped = map_packed_sequence_data(packed, lambda x: x * 2)
    assert torch.equal(mapped.data, torch.tensor([[2.0], [4.0]]))

    with pytest.raises(Exception):
        map_packed_sequence_data(packed, lambda x: x.unsqueeze(dim=0))


def test_get_masked_model_outputs_and_labels() -> None:
    """
    Test to ensure the helper function to get masked model outputs, labels and their associated subject ids
    returns the expected results.
    """

    def _create_masked_and_check_expected(_model_outputs: torch.Tensor,
                                          _labels: torch.Tensor,
                                          _subject_ids: Sequence[str],
                                          _sorted_indices: Optional[torch.Tensor] = None) -> None:
        _masked = get_masked_model_outputs_and_labels(_model_outputs, _labels, _subject_ids)
        assert _masked is not None
        sorted_indices = _masked.labels.sorted_indices if _sorted_indices is None else _sorted_indices
        if sorted_indices is not None:
            _labels = _labels[sorted_indices]
            _model_outputs = _model_outputs[sorted_indices]
            _subject_ids = np.array(_subject_ids)[sorted_indices].tolist()

        _expected_labels = _labels.transpose(dim0=0, dim1=1).flatten()
        _mask = ~torch.isnan(_expected_labels)
        _expected_labels = _expected_labels[_mask]
        _expected_model_outputs = _model_outputs.transpose(dim0=0, dim1=1).flatten()[_mask]
        _expected_subject_ids = _subject_ids

        assert torch.equal(_expected_model_outputs, _masked.model_outputs.data)
        assert torch.equal(_expected_labels, _masked.labels.data)
        assert _expected_subject_ids[:_masked.labels.sorted_indices.shape[0]] == _masked.subject_ids

    # test base case where no masking needs to be applied
    model_outputs = torch.rand((3, 4, 1))
    labels = torch.rand((3, 4, 1)).round()
    subject_ids = ['1', '2', '3']

    _create_masked_and_check_expected(model_outputs, labels, subject_ids)

    # test with unequal length sequences where masking will be performed
    model_outputs = sequences_to_padded_tensor([torch.rand(x + 1, 1) for x in range(3)], padding_value=np.nan)
    labels = sequences_to_padded_tensor([torch.rand(x + 1, 1) for x in range(3)], padding_value=np.nan)

    _create_masked_and_check_expected(model_outputs, labels, subject_ids)

    # test where one sequence is totally removed
    model_outputs[0] = np.nan
    labels[0] = np.nan

    _create_masked_and_check_expected(model_outputs, labels, subject_ids, _sorted_indices=torch.tensor([2, 1, 0]))
