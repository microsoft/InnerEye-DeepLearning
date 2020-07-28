#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Callable

import torch
import torch.nn.functional as F


class ECELoss(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    Confidence outputs are divided into equally-sized interval bins. In each bin, we compute the confidence gap as:
    bin_gap = l1_norm(avg_confidence_in_bin - accuracy_in_bin)
    A weighted average of the gaps is then returned based on the number of samples in each bin.
    """

    def __init__(self, n_bins: int = 15, activation: Callable = lambda x: F.softmax(x, dim=1)):
        """
        :param n_bins: number of confidence interval bins.
        :param activation: callable function for logit normalisation.
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.activation = activation

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:  # type: ignore
        normalised_logits = self.activation(logits)
        confidences, predictions = torch.max(normalised_logits, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):  # type: ignore
            # Calculated 'confidence - accuracy' in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
