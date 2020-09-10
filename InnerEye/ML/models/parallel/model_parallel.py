#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from collections import OrderedDict
from typing import Generator, List, Optional

import numpy as np
import torch


def move_to_device(input_tensors: List[torch.Tensor],
                   target_device: Optional[torch.device],
                   non_blocking: bool = False) -> Generator[torch.Tensor]:
    """
    Updates the memory location of tensors stored in a list.
    :param input_tensors: List of torch tensors
    :param target_device: Target device (e.g. cuda:0, cuda:1, etc). If the device is None, the tensors are not moved.
    :param non_blocking: bool
    """
    return (tensor if tensor.device == target_device or target_device is None
            else tensor.to(target_device, non_blocking=non_blocking)
            for tensor in input_tensors)


def get_device_from_parameters(module: torch.nn.Module) -> Optional[torch.device]:
    """
    Reads out the device information from the first of the module's parameters.
    If the module does not have any parameters, return None.
    """
    try:
        first_parameter = next(module.parameters())
        return first_parameter.device
    except StopIteration:
        return None


def group_layers_with_balanced_memory(inputs: List[torch.nn.Module],
                                      num_groups: int,
                                      summary: Optional[OrderedDict]) -> Generator:
    """
    Groups layers in the model in a balanced way as such each group has similar size of memory requirement
    :param inputs: List of input torch modules.
    :param num_groups: Number of groups to be produced.
    :param summary: Model summary of the input layers which is used to retrieve memory requirements.
    """

    class Group:
        def __init__(self) -> None:
            self.length = 0
            self.memory_mbytes = 0.0
            self.indices: List[int] = list()

    num_layers = len(inputs)
    summary_values = [] if summary is None else list(summary.values())

    # Recursive function that collects memory requirement of each layer
    def get_memory(modules: List[torch.nn.Module], module_memory: List[float]) -> None:
        for module in modules:
            submodules = list(module.children())
            has_children = len(submodules) > 0
            if has_children:
                get_memory(submodules, module_memory)
            else:
                layer_summary = summary_values.pop(0)
                module_memory.append(layer_summary.output_memory_megabytes)
                if not (layer_summary.n_params == sum([np.prod(p.size()) for p in module.parameters()])):
                    raise ValueError("The summary does not match with the layer information.")

    def find_available_group(group_id: int,
                             groups: List[Group],
                             mem_to_be_allocated: float,
                             max_mem_allowed: float) -> int:
        """Finds the next available group to store input layer which is represented
        in terms of its memory (`mem_to_be_allocated`). The algorithm assigns the object
        to the current group if it is empty or it has enough capacity to store the layer.
        Otherwise the rest of groups are enquired. If none of the groups accept the incoming layer,
        it is assigned to the group with lowest memory load. This approach find a comprimise between
        sequentiality of layers and memory load balance."""

        num_groups = len(groups)
        available_groups = [True for _ in range(num_groups)]
        group_mems = [groups[g_id].memory_mbytes for g_id in range(num_groups)]
        lowest_mem_group_id = group_mems.index(min(group_mems))

        while (groups[group_id].length > 0) and \
                (mem_to_be_allocated + groups[group_id].memory_mbytes > max_mem_allowed):
            available_groups[group_id] = False
            if not any(available_groups):
                group_id = lowest_mem_group_id
                break
            group_id = (group_id + 1) % num_groups

        return group_id

    # Recursively traverse through the input modules and collect the memory information.
    model_memory = list()
    for layer in inputs:
        layer_memory: List[float] = list()
        get_memory([layer], layer_memory)
        model_memory.append(sum(layer_memory))
    group_max_mem_capacity = sum(model_memory) / float(num_groups)

    # Groups input layers by balancing the memory load of each group/device
    group_id = 0
    groups = [Group() for _ in range(num_groups)]
    for block_id in range(num_layers):
        current_memory_mbytes = model_memory[block_id]
        group_id = find_available_group(group_id, groups, current_memory_mbytes, group_max_mem_capacity)
        groups[group_id].memory_mbytes += current_memory_mbytes
        groups[group_id].length += int(1)
        groups[group_id].indices.append(block_id)

    # Return the groupped layers through a generator
    for group in groups:
        yield [inputs[ii] for ii in range(num_layers) if ii in group.indices]


def is_model_parallel(module: torch.nn.Module) -> bool:
    """
    Checks if the module has been undergoing partitioning across multiple GPUs in the `partition_layers` function.
    Returns True if the model is partitioned across GPUs, and False if no attempts have been recorded.
    """
    return getattr(module, 'is_model_parallel', False)


def partition_layers(layers: List[torch.nn.Module],
                     summary: OrderedDict,
                     target_devices: List[torch.device]) -> None:
    """
    Splits the models into multiple chunks and assigns each sub-model to a particular GPU
    :param layers: The layers to partition
    :param summary: Model architecture summary to use for partitioning
    :param target_devices: The devices to partition layers into
    """

    def update_module_device(input_modules: List[torch.nn.Module], target_device: torch.device) -> None:
        for module in input_modules:
            submodules = list(module.children())
            has_children = len(submodules) > 0
            if has_children:
                update_module_device(submodules, target_device)
            else:
                module.to(target_device)
                # Set a flag that this model uses ModelParallel, that we can later use in testing and to distinguish
                # between DataParallel
                module.is_model_parallel = True

    num_devices = len(target_devices)
    for group, device in zip(group_layers_with_balanced_memory(layers, num_devices, summary), target_devices):
        # Recursively update the device of each submodule
        update_module_device(group, device)
