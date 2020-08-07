#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.dataset.sequence_sample import ClassificationItemSequence
from InnerEye.ML.models.architectures.base_model import DeviceAwareModule
from InnerEye.ML.models.architectures.sequential.gru import LayerNormGRU
from InnerEye.ML.models.layers.identity import Identity
from InnerEye.ML.utils.sequence_utils import sequences_to_padded_tensor


class RNNClassifier(DeviceAwareModule[List[ClassificationItemSequence], torch.Tensor]):
    """
    Recurrent neural network (GRU) to perform a binary classification of sequence datasets.
    The class scores that the model outputs are results of a log softmax.
    :param input_dim:   Number of input channels for the GRU layer.
    :param hidden_dim:  Number of hidden states
    :param output_dim:  Number of model output channels
    :param num_rnn_layers: Number of RNN layers to be stacked in the classifier. By default, a single GRU layer is used.
    :param use_layer_norm: If set to True, hidden state activations are normalised at each time step.
    :param target_indices: Output target indices. For many input to one output sequential model,
                           it should be equal to the last index `-1`. If a tensor of indices are provided,
                           it will return sequential model outputs at given time indices.
    :param ref_indices: Optional, if set then the hidden states from these reference indices is concatenated to the
                        hidden state of the target position before computing the class posterior.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 target_indices: List[int],
                 num_rnn_layers: int = 1,
                 use_layer_norm: bool = False,
                 rnn_dropout: float = 0.00,
                 ref_indices: Optional[List[int]] = None) -> None:
        super().__init__()
        self.target_indices = target_indices or [-1]
        self.input_dim = input_dim
        self.ref_indices = ref_indices
        self.hidden_dim = hidden_dim
        # The GRU takes embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.gru: LayerNormGRU = LayerNormGRU(input_size=input_dim,
                                              hidden_size=hidden_dim,
                                              num_layers=num_rnn_layers,
                                              use_layer_norm=use_layer_norm,
                                              dropout=rnn_dropout)

        # The linear layer that maps from hidden state space to class space
        if self.ref_indices is None:
            self.hidden2class = nn.Linear(hidden_dim, output_dim)
        else:
            self.hidden2class = nn.Linear((len(ref_indices) + 1) * hidden_dim, output_dim)  # type: ignore

        # Create a parameter to learn the initial hidden state
        hidden_size = torch.Size([num_rnn_layers, 1, hidden_dim])
        self.h0 = nn.Parameter(torch.zeros(size=hidden_size), requires_grad=True)
        self.initialise_parameters()

    def forward(self, *input_seq: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        input_seq: Input sequence (batch_size x sequence_length x input_dim)
        return: Sequence classification output (batch_size x target_indices x output_dim)
        """
        batch_size, seq_length, _ = input_seq[0].size()
        # GRU forward pass and linear mapping from hidden state to the output space.
        # gru_out of shape [batch_size, seq_length, hidden_dim]
        gru_out: torch.Tensor = self.gru(input_seq[0], self.h0.repeat(1, batch_size, 1))
        # pad the gru output if required to ensure values for each target index
        gru_out = self.pad_gru_output(gru_out)

        if self.ref_indices is None:
            return self.hidden2class(gru_out[:, self.target_indices, :])
        else:
            predictions = []
            for target_index in self.target_indices:
                input_to_classifier = gru_out[:, self.ref_indices + [target_index], :].view(batch_size, -1)
                predictions.append(self.hidden2class(input_to_classifier))
            return torch.stack(predictions, dim=1)

    def pad_gru_output(self, input: torch.Tensor) -> torch.Tensor:
        """
        Pad the GRU output with zeros if required to make sure to make sure there is a value for each target index
        this RNN classifier is initialized with.
        """
        current_sequence_length = len(range(input.shape[0]))
        required_sequence_length = len(range(max(self.target_indices) + 1))
        pad_size = max(required_sequence_length - current_sequence_length, 0)
        return F.pad(input=input, pad=[0, 0, 0, pad_size], mode='constant', value=0)

    def initialise_parameters(self) -> None:
        """
        Initialises the initial hidden state parameter in GRU.
        """
        # Disable type checking here because these parameters are created via setattr, and hence
        # confuse mypy
        nn.init.xavier_normal_(self.h0, gain=nn.init.calculate_gain('tanh'))

    def get_input_tensors(self, sequences: List[ClassificationItemSequence]) -> List[torch.Tensor]:
        """
        Returns the input tensors as a List where the first element corresponds to the non-imaging features.
        """
        seq_flattened = [torch.stack([i.get_all_non_imaging_features() for i in seq.items], dim=0)
                         for seq in sequences]
        return [sequences_to_padded_tensor(seq_flattened)]


class RNNClassifierWithEncoder(RNNClassifier):
    """
    RNN classifier for a combination of imaging and non-imaging features. The images are first encoded using
    an image encoder that is passed to the constructor.
    :param image_encode: torch module to use to encode the image features. For example a ImageEncoder could be
    for a U-Net like encoding of the images.
    :param input_dim:   Number of input channels for the GRU layer i.e. number of non_imaging features + number
    of features at the output of the image encoder (if images/segmentations are used).
    :param hidden_dim:  Number of hidden states
    :param output_dim:  Number of model output channels
    :param num_rnn_layers: Number of RNN layers to be stacked in the classifier. By default, a single GRU layer is used.
    :param use_layer_norm: If set to True, hidden state activations are normalised at each time step.
    :param target_indices: Output target indices. For many input to one output sequential model,
                           it should be equal to the last index `-1`. If a tensor of indices are provided,
                           it will return sequential model outputs at given time indices.
    :param ref_indices: Optional, if set then the hidden state from these reference indices is concatenated to the
                        hidden state of the target position before computing the class posterior.
    :param use_encoder_batch_norm: If True, apply batchNorm to the encoded features at the output of the image_encoder
    module prior to feeding them to the GRU layers. If False, the raw output from the image encoder is fed to the GRU
    layer.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 use_encoder_batch_norm: bool = False,
                 image_encoder: Optional[DeviceAwareModule[ScalarItem, torch.Tensor]] = None,
                 **kwargs: Any) -> None:
        super().__init__(input_dim, hidden_dim, output_dim, **kwargs)
        self.image_encoder = image_encoder
        if self.image_encoder is not None:
            # Adding necessary attributes required by GradCam computation.
            self.imaging_feature_type = image_encoder.imaging_feature_type  # type: ignore
            self.num_non_image_features = image_encoder.num_non_image_features  # type: ignore
            self.last_encoder_layer = ["image_encoder"] + image_encoder.last_encoder_layer  # type: ignore
            self.conv_in_3d = self.image_encoder.conv_in_3d
            self.encode_channels_jointly = False
        self.use_encoder_batch_norm = use_encoder_batch_norm
        self.layer_norm = nn.BatchNorm1d(input_dim) if use_encoder_batch_norm else Identity()

    def forward(self, *input_seq: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        input_seq: Input sequence (batch_size x sequence_length x input_dim)
        return: Sequence classification output (batch_size x target_indices x output_dim)
        """
        batch_size, seq_length = input_seq[0].size()[:2]
        # If we have a model that uses images the input will be a List with 2 elements
        # (imaging_features, non_imaging_features). Else it will be a List with only one
        # element (non_imaging_features).
        non_imaging_seq = input_seq[0] if len(input_seq) == 1 else input_seq[1]
        encoded_seq = []
        if self.image_encoder is not None:
            imaging_seq = input_seq[0]
            for seq in range(seq_length):
                encoded_features = self.image_encoder(imaging_seq[:, seq, :], non_imaging_seq[:, seq, :])
                if self.training and batch_size == 1 and self.use_encoder_batch_norm:
                    # This check is necessary as BatchNorm fails if the
                    # batch_size is equal to 1 (can't compute the variance).
                    logging.warning("BatchNorm will not be applied to the encoded image features as the"
                                    "effective batch size is 1 on this device.")
                else:
                    encoded_features = self.layer_norm(encoded_features)
                encoded_seq.append(encoded_features)
        encoded_input = non_imaging_seq if encoded_seq == [] else torch.stack(encoded_seq, dim=1)
        return super().forward(encoded_input)

    def get_input_tensors(self, sequences: List[ClassificationItemSequence]) -> List[torch.Tensor]:
        """
        Returns the input tensors as a List where the first element corresponds to the non-imaging features.
        The second corresponds to the images loaded as required by the image encoder.
        """
        non_imaging_seq = super().get_input_tensors(sequences)[0]
        if self.image_encoder is not None:
            seq_flattened_imaging = [torch.stack([self.image_encoder.get_input_tensors(item)[0]
                                                  for item in seq.items], dim=0)
                                     for seq in sequences]
            imaging_seq = sequences_to_padded_tensor(seq_flattened_imaging)
            return [imaging_seq, non_imaging_seq]
        else:
            return [non_imaging_seq]
