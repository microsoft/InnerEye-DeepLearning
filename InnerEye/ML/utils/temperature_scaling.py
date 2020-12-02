#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Callable, List, Optional, Tuple

import torch
from torch.optim import LBFGS  # type: ignore

from InnerEye.Common.type_annotations import T
from InnerEye.ML.deep_learning_config import TemperatureScalingConfig
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule, E


class ModelWithTemperature(DeviceAwareModule):
    """
    Torch module to wrap a model with temperature scaling.
    """

    def __init__(self, model: DeviceAwareModule,
                 temperature_scaling_config: TemperatureScalingConfig):
        super().__init__()
        self.model = model
        self.conv_in_3d = model.conv_in_3d
        self.temperature_scaling_config = temperature_scaling_config

        # assign this parameter to the first model device otherwise use PyTorch default.
        _model_devices = model.get_devices()
        _device = _model_devices[0] if _model_devices else None
        self.temperature = torch.nn.Parameter(torch.ones(1, device=_device), requires_grad=True)

    def forward(self, *x: torch.Tensor) -> torch.Tensor:  # type: ignore
        logits = self.model(*x)
        return self.temperature_scale(logits)

    def get_last_encoder_layer_names(self) -> List[str]:
        _model: DeviceAwareModule = self.model
        return _model.get_last_encoder_layer_names()

    def get_input_tensors(self, item: T) -> List[E]:
        _model: DeviceAwareModule = self.model
        return _model.get_input_tensors(item)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.expand(logits.shape)
        return logits / temperature

    def set_temperature(self,
                        logits: torch.Tensor,
                        labels: torch.Tensor,
                        criterion_fn: Callable[[torch.Tensor, torch.Tensor],
                                               Tuple[torch.Tensor, torch.Tensor]],
                        use_gpu: bool) -> float:
        """
        Tune the temperature of the model using the provided logits and labels.
        :param logits: Logits to use to learn the temperature parameter
        :param labels: Labels to use to learn the temperature parameter
        :param criterion_fn: A criterion function s.t: (logits, labels) => (loss, ECE)
        :param use_gpu: If True then GPU will be used otherwise CPU will be used.
        :return Optimal temperature value
        """
        if use_gpu:
            logits = logits.cuda()
            labels = labels.cuda()

        # Calculate loss values before scaling
        before_temperature_loss, before_temperature_ece = criterion_fn(logits, labels)
        print('Before temperature scaling - LOSS: {:.3f} ECE: {:.3f}'
              .format(before_temperature_loss.item(), before_temperature_ece.item()))

        # Next: optimize the temperature w.r.t. the provided criterion function
        optimizer = LBFGS([self.temperature], lr=self.temperature_scaling_config.lr,
                                      max_iter=self.temperature_scaling_config.max_iter)

        def eval_criterion() -> torch.Tensor:
            # zero the gradients for the next optimization step
            optimizer.zero_grad()
            loss, ece = criterion_fn(self.temperature_scale(logits), labels)
            # TODO antonsc: re-enable logging
            # if logger:
            #     logger.log_to_azure_and_tensorboard("Temp_Scale_LOSS", loss.item())
            #     logger.log_to_azure_and_tensorboard("Temp_Scale_ECE", ece.item())
            loss.backward()
            return loss

        optimizer.step(eval_criterion)  # type: ignore

        after_temperature_loss, after_temperature_ece = criterion_fn(self.temperature_scale(logits), labels)
        print('Optimal temperature: {:.3f}'.format(self.temperature.item()))
        print('After temperature scaling - LOSS: {:.3f} ECE: {:.3f}'
              .format(after_temperature_loss.item(), after_temperature_ece.item()))
        return self.temperature.item()
