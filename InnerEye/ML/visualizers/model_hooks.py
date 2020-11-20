#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List, Optional

import torch
from torch.nn import Module

from InnerEye.ML.utils.temperature_scaling import ModelWithTemperature


class HookBasedFeatureExtractor(Module):

    def __init__(self, model: Module, layer_name: Optional[List[str]]):
        """
        :param model: pytorch model
        :param layer_name: name of direct submodule of modle, or nested module names structure as a list
        layer_name = [_2D_encoder] , layer_name = [_2D_encoder, conv_layers, 0]
        """
        super().__init__()

        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.layer_name = layer_name
        self.model = model
        self.net: Module

        if isinstance(model, torch.nn.DataParallel):
            self.net = model.module  # type: ignore
        else:
            self.net = model

        if isinstance(self.net, ModelWithTemperature):
            self.net = self.net.model

        if layer_name is not None:
            self._verify_layer_name(self.net, layer_name)

    def _verify_layer_name(self, model: Module, layer_name: List[str]) -> None:
        """
        Recursively traverses the model and verifies if the layer name is valid
        :param model: the model
        :param layer_name: hierarchical list of layer names to index within model
        :return:
        """
        submodules = model._modules.keys()  # type: ignore
        submodule = model
        for el in layer_name:
            if el not in submodules:
                raise ValueError("invalid layer name: ", el)

            submodule = submodule._modules[el]  # type: ignore
            submodules = submodule._modules.keys()  # type: ignore

    def forward_hook_fn(self, module: Module, input: Any, output: Any) -> None:
        """
        Registers a forward hook inside module
        :param module:
        :param input:
        :param output:
        :return:
        """

        if isinstance(input, tuple):
            self.inputs.append([input[index].data.clone() for index in range(len(input))])
        else:
            self.inputs.append(input.data.clone())

        if isinstance(output, tuple):
            self.outputs.append([output[index].data.clone() for index in range(len(output))])
        else:
            self.outputs.append(output.data.clone())

    # noinspection PyTypeChecker
    def forward(self, *inputs):  # type: ignore

        if self.layer_name is not None:
            submodule = self.net
            for el in self.layer_name:
                submodule = submodule._modules[el]

            target_layer = submodule
            hook = target_layer.register_forward_hook(self.forward_hook_fn)

        else:
            hook = self.net.register_forward_hook(self.forward_hook_fn)

        self.model(*inputs)
        hook.remove()
