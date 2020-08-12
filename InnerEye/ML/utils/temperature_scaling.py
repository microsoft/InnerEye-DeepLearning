"""
Copyright (c) Microsoft Corporation. All rights reserved.
"""
from typing import Optional

import torch
from torch.optim import LBFGS

from InnerEye.ML.models.losses.ece import ECELoss


class TemperatureScaleModel(torch.nn.Module):
    """
    Torch nn module to wrap a model with temperature scaling.
    model (nn.Module):
        A classification neural network, output of the neural network should be the classification logits.
    """
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1, device=device) * 1.0, requires_grad=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        return logits / self.temperature

    def set_temperature(self):
        """
        Tune the temperature of the model (using the validation set).
        """
        self.cuda()
        ece_criterion = ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for minibatch in self.validation_data_loader:
                get_scalar_model_inputs_and_labels(self.model_config, model, sample)
                input = minibatch["features"].cuda()
                labels = minibatch["labels"]
                if len(labels.size()) == 0: continue
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(labels)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = LBFGS([self.temperature], lr=0.002, max_iter=50)

        def eval() -> torch.Tensor:
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self
