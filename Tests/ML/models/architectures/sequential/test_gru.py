#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from InnerEye.ML.models.architectures.sequential.gru import LayerNormGRU, LayerNormGRUCell


def test_cell_initialisation() -> None:
    model = LayerNormGRU(input_size=2, hidden_size=2, num_layers=3)
    assert len(model.cells) == 3


def test_layer_norm_initialisation() -> None:
    cell = LayerNormGRUCell(input_size=2, hidden_size=2, use_layer_norm=True, dropout=0.50)
    assert isinstance(cell.ln_r, nn.LayerNorm)
    assert cell.dropout == 0.50


def test_gru_forward_pass() -> None:
    num_layers = 2
    batch_size = 10
    hidden_dim = 2
    input_dim = 3
    seq_length = 5

    model = LayerNormGRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
    input_sequence = torch.rand(size=(batch_size, seq_length, input_dim))
    initial_hidden_state = torch.zeros(size=(num_layers, batch_size, hidden_dim))
    output_sequence = model(input_sequence, initial_hidden_state)

    assert output_sequence.size(0) == batch_size
    assert output_sequence.size(1) == seq_length
    assert output_sequence.size(2) == hidden_dim
