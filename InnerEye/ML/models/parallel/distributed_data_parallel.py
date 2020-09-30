#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Dict, List, Tuple

from torch import device, Tensor
from torch.nn.parallel import DistributedDataParallel

from InnerEye.Common.type_annotations import T
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.ML.utils.device_aware_module import E


class DistributedDataParallelModel(DistributedDataParallel, DeviceAwareModule):
    def get_input_tensors(self, item: T) -> List[E]:
        _module: DeviceAwareModule = self.get_module()
        return _module.get_input_tensors(item)

    def get_module(self) -> DeviceAwareModule:
        module = self.module
        if not isinstance(module, DeviceAwareModule):
            raise ValueError(f"Expecting DeviceAwareModule. Instead found {module}")
        return module

    def get_devices(self) -> List[device]:
        """
        Gets the numeric indices of the CUDA devices that the present object is using.
        :return:
        """
        return [device(x) if isinstance(x, int) else x for x in self.device_ids]


class DistributedDataParallelCriterion(DistributedDataParallel):
    def forward(
            self,  # type: ignore
            inputs: List[Tensor],
            *targets: Tuple[Tensor],
            **kwargs: Dict[str, Any]) -> Tensor:
        """

        :param inputs:
        :param targets:
        :param kwargs:
        :return:
        """
        return self.module(inputs, targets[0], kwargs[0])
