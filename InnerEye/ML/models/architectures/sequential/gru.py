#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RNNCellBase

from InnerEye.ML.models.layers.identity import Identity


class LayerNormGRUCell(RNNCellBase):
    """
    Implements GRUCell with layer normalisation and zone-out on top.
    It inherits the base RNN cell whose trainable weight matrices are used.

    References:
    [1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer Normalization." (2016).
    [2] Krueger, David, et al. "Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations." (2016).

    :param input_size: Number of input features to the cell
    :param hidden_size: Number of hidden states in the cell
    :param use_layer_norm: If set to True, layer normalisation is applied to
                           reset, update and new tensors before activation.
    :param dropout: Dropout probability for the hidden states [0,1]
    """

    def __init__(self, input_size: int, hidden_size: int, use_layer_norm: bool = False, dropout: float = 0.0):
        super(LayerNormGRUCell, self).__init__(input_size, hidden_size, bias=False, num_chunks=3)

        self.dropout = dropout
        self.ln_r = nn.LayerNorm(self.hidden_size) if use_layer_norm else Identity()
        self.ln_z = nn.LayerNorm(self.hidden_size) if use_layer_norm else Identity()
        self.ln_n = nn.LayerNorm(self.hidden_size) if use_layer_norm else Identity()

    def forward(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore
        if hx is None:
            hx = input.new_zeros(size=(input.size(0), self.hidden_size), requires_grad=False)

        ih = input.mm(self.weight_ih.t())
        hh = hx.mm(self.weight_hh.t())

        i_r, i_z, i_n = ih.chunk(3, dim=1)
        h_r, h_z, h_n = hh.chunk(3, dim=1)

        # Activations with layer normalisation
        r = torch.sigmoid(self.ln_r(i_r + h_r))
        z = torch.sigmoid(self.ln_z(i_z + h_z))
        n = torch.tanh(self.ln_n(i_n + r * h_n))
        new_h = (torch.tensor(1.0) - z) * n + z * hx

        # Apply zoneout drop-out on hidden states
        if self.dropout > 0.0:
            bernouli_mask = F.dropout(torch.ones_like(new_h), p=self.dropout, training=bool(self.training))
            new_h = bernouli_mask * new_h + (torch.tensor(1.0) - bernouli_mask) * hx

        return new_h


class LayerNormGRU(nn.Module):
    """
    Implements a stacked GRU layers. Differs from torch.nn.GRU implementation by
    the use of layer normalisation and hidden state preserving drop-out techniques
    (zone-out) which are currently not provided in the default implementation.

    https://arxiv.org/pdf/1607.06450.pdf
    https://arxiv.org/pdf/1606.01305.pdf

    :param input_size: Number of input features.
    :param hidden_size: Number of hidden states in GRU, it is used for all layers.
    :param num_layers: Number of stacked GRU layers in the module.
    :param batch_first: If set to true, input tensor should have the batch dimension in the first axis.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 **kwargs: Any):
        super(LayerNormGRU, self).__init__()

        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            LayerNormGRUCell(input_size if i == 0 else hidden_size, hidden_size, **kwargs)
            for i in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:  # type: ignore
        seq_axis = 1
        for i, cell in enumerate(self.cells):  # type: ignore
            y = []
            hidden = hx[i]
            for xc in x.chunk(x.size(seq_axis), dim=seq_axis):
                xc = xc.squeeze(seq_axis)
                hidden = cell(xc, hidden)
                y.append(hidden.unsqueeze(0))
            x = torch.stack(y, dim=seq_axis + 1).squeeze(0)
        return x
