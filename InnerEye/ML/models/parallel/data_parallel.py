#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
from torch.cuda import amp
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.scatter_gather import gather, scatter_kwargs
from typing import Any, Callable, Dict, List, Tuple, Union

from InnerEye.Common.type_annotations import T
from InnerEye.ML.models.architectures.base_model import DeviceAwareModule
from InnerEye.ML.utils.device_aware_module import E


def execute_within_autocast_if_needed(func: Callable[[], T], use_autocast: bool) -> T:
    """
    Runs the given parameterless function, and returns the function result. If the use_autocast
    flag is true, the function is evaluated inside of the torch.cuda.amp.autocast context manager,
    that can automatically cast operations to mixed precision. If the flag is false, the function
    is called as is.
    :param func: The function that should be evaluated
    :param use_autocast: If true, evaluate within the autocast context manager. If false, evaluate as is.
    """
    if use_autocast:
        with amp.autocast():
            return func()
    else:
        return func()


class DataParallelModel(DataParallel, DeviceAwareModule):
    """
    Modifies the DataParallel class by updating the `gather` method. In this child class, Parallel outputs
    are not aggregated into a single tensor but returned as a list of tensors in order to perform loss
    computation in parallel.
    """

    def get_input_tensors(self, item: T) -> List[E]:
        _module: DeviceAwareModule = self.get_module()
        return _module.get_input_tensors(item)

    def get_module(self) -> DeviceAwareModule:
        module = self.module
        if not isinstance(module, DeviceAwareModule):
            raise ValueError(f"Expecting DeviceAwareModule. Instead found {module}")
        return module

    def get_devices(self) -> List[torch.device]:
        """
        Gets the numeric indices of the CUDA devices that the present object is using.
        :return:
        """
        return [torch.device(x) if isinstance(x, int) else x for x in self.device_ids]

    def gather(self, outputs: List[torch.Tensor], output_device: int) -> List[torch.Tensor]:
        """
        This overrides the `gather` method of `DataParallel`, and will NOT actually gather the tensors
        onto one devices. When using a DataParalleModel, a DataParallelCriterion needs to be used as well, which
        in turn would have to do some suitable `gather` operation.
        :param outputs:
        :param output_device:
        :return:
        """
        return outputs


class CriterionWithAutocast(torch.nn.Module):
    """
    A wrapper around a single module, that runs the forward pass in an autocast context manager.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self,  # type: ignore
                *inputs: torch.Tensor,
                **kwargs: Dict[str, Any]) -> torch.Tensor:
        with amp.autocast():
            return self.module(*inputs, **kwargs)


# noinspection PyUnresolvedReferences
class DataParallelCriterion(DataParallel):
    """
    Calculate loss in multiple-GPUs, which balances the memory usage.
    The targets are split across the specified devices by chunking in
    the batch dimension. Please use together with :class:`data_parallel.DataParallelModel`.

    Example::
        >>> net = DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """

    def __init__(self,
                 module: torch.nn.Module,
                 device_ids: List[Union[int, torch.device]],
                 use_mixed_precision: bool):
        super().__init__(module=module, device_ids=device_ids)
        self.use_mixed_precision = use_mixed_precision

    def forward(self,  # type: ignore
                inputs: List[torch.Tensor],
                *targets: Tuple[torch.Tensor],
                **kwargs: Dict[str, Any]) -> torch.Tensor:
        # inputs are expected to be already scattered
        # scattering the targets instead
        if not self.device_ids:
            return self.module(inputs, *targets, **kwargs)
        _targets, _kwargs = scatter_kwargs(targets, kwargs, self.device_ids, dim=self.dim)
        if len(self.device_ids) == 1:
            return self.module(inputs, *_targets[0], **_kwargs[0])
        autocast_if_needed = CriterionWithAutocast(module=self.module) if self.use_mixed_precision else self.module
        replicas = self.replicate(autocast_if_needed, self.device_ids[:len(inputs)])  # type: ignore

        input_tuples: List[Tuple[torch.Tensor, ...]] = [(i, *t) for i, t in zip(inputs, _targets)]
        outputs = torch.nn.parallel.parallel_apply(replicas, input_tuples, _kwargs)

        return gather(outputs, self.output_device, dim=self.dim)
