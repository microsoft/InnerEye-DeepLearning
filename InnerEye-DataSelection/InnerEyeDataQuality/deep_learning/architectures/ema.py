#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from typing import Dict

import numpy as np
import torch


class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
        step_max (int): Maximum required number of steps to reach specified decay rate from zero.
                        In the initial epochs, decay rate is kept small to keep the teacher model up-to-date.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.99, step_max: int = 150) -> None:
        self.decay_max = decay
        self.step_max = step_max
        self.step_count = int(0)
        self.shadow = {}
        self.original: Dict[str, torch.Tensor] = {}
        self.model = model  # reference to the student model

        # Register model parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

        logging.info("Creating a teacher model (EMA)")

    def update(self) -> None:
        """
        Receives a new set of parameter values and merges them with the previously stored ones.
        """
        self.step_count += int(1)
        decay = self._get_decay_rate(self.step_count)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def _get_decay_rate(self, step: int) -> float:
        """
        Return decay rate for current stored parameters
        """
        if step <= self.step_max:
            ratio = step / self.step_max
            return 0.5 * self.decay_max * (1 - np.cos(ratio * np.pi))
        else:
            return self.decay_max

    @torch.no_grad()
    def inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Runs an inference using the template of student model
        """
        is_train = self.model.training
        self._assign()
        self.model.eval()
        self._switch_hooks(False)
        outputs = self.model.forward(inputs).detach()
        self._switch_hooks(True)
        self._restore()
        self.model.train() if is_train else self.model.eval()

        return outputs

    def _switch_hooks(self, bool_value: bool) -> None:
        for layer in self.model.children():
            if hasattr(layer, "use_hook"):
                layer.use_hook = bool_value  # type: ignore

    def _assign(self) -> None:
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def _restore(self) -> None:
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]

    def save_model(self, save_path: str) -> None:
        """
        Saves EMA model parameters to a checkpoint file.
        """
        self._assign()
        state = {'ema_model': self.model.state_dict(),
                 'ema_params': self.shadow,
                 'step_count': self.step_count}
        torch.save(state, save_path)
        self._restore()

    def restore_from_checkpoint(self, path: str) -> None:
        state = torch.load(path)
        self.shadow = state['ema_params']
        self.step_count = state['step_count']
