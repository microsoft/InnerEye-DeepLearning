"""
Copyright (c) Microsoft Corporation. All rights reserved.
"""
from typing import Callable, List, Tuple

import torch
from torch.optim import LBFGS

from InnerEye.Common.type_annotations import T
from InnerEye.ML.deep_learning_config import TemperatureScalingConfig
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule, E


class ModelWithTemperature(DeviceAwareModule):
    """
    Torch nn module to wrap a model with temperature scaling.
    model (nn.Module):
        A classification neural network, output of the neural network should be the classification logits.
    """

    def __init__(self, model: DeviceAwareModule, temperature_scaling_config: TemperatureScalingConfig):
        super().__init__()
        self.model = model
        self.temperature_scaling_config = temperature_scaling_config
        self.temperature = torch.nn.Parameter(torch.ones(1, device=next(model.parameters()).device) * 1.0,
                                              requires_grad=True)

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        logits = self.model(*x)
        return self.temperature_scale(logits)

    def get_input_tensors(self, item: T) -> List[E]:
        _model: DeviceAwareModule = self.model
        return _model.get_input_tensors(item)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, logits: torch.Tensor, labels: torch.Tensor,
                        criterion_fn: Callable[[torch.Tensor, torch.Tensor],
                                               Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """
        Tune the temperature of the model using the provided logits and labels.
        """
        if torch.cuda.is_available():
            logits = logits.cuda()
            labels = labels.cuda()

        # Calculate loss values before scaling
        before_temperature_loss, before_temperature_ece = criterion_fn(logits, labels)
        print('Before temperature scaling - LOSS: {.3f} {.3f}'
              .format(before_temperature_loss.item(), before_temperature_ece.item()))

        # Next: optimize the temperature w.r.t. the provided criterion function
        optimizer = LBFGS([self.temperature], lr=self.temperature_scaling_config.lr,
                          max_iter=self.temperature_scaling_config.max_iter)

        def eval_criterion() -> torch.Tensor:
            loss, ece = criterion_fn(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_criterion)

        after_temperature_loss, after_temperature_ece = criterion_fn(self.temperature_scale(logits), labels)
        print('Optimal temperature: {.3f}'.format(self.temperature.item()))
        print('After temperature scaling - LOSS: {.3f} {.3f}'
              .format(after_temperature_loss.item(), after_temperature_ece.item()))
