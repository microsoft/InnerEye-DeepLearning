"""
Copyright (c) Microsoft Corporation. All rights reserved.
"""
from typing import Callable, List, Optional

import torch
from torch.optim import LBFGS

from InnerEye.Common.type_annotations import T
from InnerEye.ML.models.losses.ece import ECELoss
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule, E


class ClassificationModelWithTemperature(DeviceAwareModule):
    """
    Torch nn module to wrap a model with temperature scaling.
    model (nn.Module):
        A classification neural network, output of the neural network should be the classification logits.
    """

    def get_input_tensors(self, item: T) -> List[E]:
        return self.model.get_input_tensors(item)

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        super().__init__()
        self.model = model
        self.temperature = torch.nn.Parameter(torch.ones(1, device=device) * 1.0, requires_grad=True)

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        logits = self.model(*x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        return logits / self.temperature

    def set_temperature(self, logits: torch.Tensor, labels: torch.Tensor, forward_criterion: Callable) -> None:
        """
        Tune the temperature of the model (using the validation set).
        """
        self.cuda()
        ece_criterion = ECELoss().cuda()
        logits = logits.cuda()
        labels = labels.cuda()
        # Calculate ECE before temperature scaling
        before_temperature_nll, before_temperature_ece = forward_criterion(logits, labels)
        print('Before temperature - NLL: %.3f ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = LBFGS([self.temperature], lr=0.2, max_iter=500)

        def eval_criterion() -> torch.Tensor:
            scaled = self.temperature_scale(logits)
            # print(scaled, scaled.shape)
            # print(labels, labels.shape)
            nll, ece = forward_criterion(scaled, labels)
            nll.backward()
            return nll

        optimizer.step(eval_criterion)

        after_temperature_nll, after_temperature_ece = forward_criterion(self.temperature_scale(logits), labels)
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
