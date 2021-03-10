#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import copy
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.hooks import RemovableHandle
from torchprof.profile import Profile

from InnerEye.Common.common_util import logging_only_to_file
from InnerEye.Common.fixed_paths import DEFAULT_MODEL_SUMMARIES_DIR_PATH
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.ML.utils.ml_util import RandomStateSnapshot


@dataclass
class LayerSummary:
    """
    Parameters to layer properties such as shapes of input/output tensors and number of parameters
    """
    input_shape: list
    output_shape: list
    n_params: int
    n_trainable_params: int
    device: Optional[torch.device]
    output_memory_megabytes: float = field(init=False)

    def __post_init__(self) -> None:
        self.output_memory_megabytes = ModelSummary.compute_tensor_memory_megabytes(self.output_shape)


class ModelSummary:
    def __init__(self, model: DeviceAwareModule) -> None:
        """
        Class to summarise the detail of neural network including (I) intermediate tensor shapes,
        (II) number of trainable and non-trainable parameters, and (III) total GPU/CPU memory requirements.
        :param model: BaseModel object containing the computation graph.
        """
        # Need a local import here to avoid circular dependency
        from InnerEye.ML.models.architectures.base_model import DeviceAwareModule
        if not isinstance(model, DeviceAwareModule):
            raise ValueError("Input model should be an instance of the DeviceAwareModule class")
        self.use_gpu = model.is_model_on_gpu()
        self.summary: OrderedDict = OrderedDict()
        self.hooks: List[RemovableHandle] = list()

        # Generate a copy to shield the model from torch-profiler hooks.
        self.n_params = 0
        self.n_trainable_params = 0
        self.model = copy.deepcopy(model)

    def generate_summary(self,
                         input_sizes: Optional[Sequence[Tuple]] = None,
                         input_tensors: Optional[List[torch.Tensor]] = None,
                         log_summaries_to_files: bool = False) -> OrderedDict:
        """
        Produces a human readable summary of the model, and prints it via logging.info. The summary is computed by
        doing forward propagation through the model, with tensors of a given size or a given list of tensors.
        :param input_sizes: The list of sizes of the input tensors to the model. These sizes must be specifies
        without the leading batch dimension.
        :param input_tensors: The tensors to use in model forward propagation.
        :param log_summaries_to_files: if True, write the summary to a new file under logs/models instead of stdout
        :return:
        """
        if input_sizes and not input_tensors:
            if not all([isinstance(in_size, tuple) for in_size in input_sizes]):
                raise ValueError("Input size list should contain only tuples")
            input_tensors = [torch.zeros(*(1, *in_size)) for in_size in input_sizes]
        elif input_tensors and not input_sizes:
            pass
        else:
            raise ValueError("You need to specify exactly one of (input_sizes, input_tensors)")
        assert input_tensors is not None  # for mypy
        if log_summaries_to_files:
            self._log_summary_to_file(input_tensors)
        else:
            self._generate_summary(input_tensors)
        return self.summary

    def _log_summary_to_file(self, input_tensors: List[torch.Tensor]) -> None:
        model_log_directory = DEFAULT_MODEL_SUMMARIES_DIR_PATH
        model_log_directory.mkdir(parents=True, exist_ok=True)
        index = 1
        while True:
            log_file_path = model_log_directory / f"model_log{index:03d}.txt"
            if not log_file_path.exists():
                break
            index += 1
        logging.info(f"Writing model summary to: {log_file_path}")
        with logging_only_to_file(log_file_path):
            self._generate_summary(input_tensors)

    @staticmethod
    def _get_sizes_from_list(tensors: Union[List[torch.Tensor], torch.Tensor]) -> List[torch.Size]:
        if isinstance(tensors, (list, tuple)):
            return [t.size() for t in tensors]
        else:
            return list(tensors.size())  # type: ignore

    @staticmethod
    def _get_device(module: torch.nn.Module) -> Optional[torch.device]:
        """Returns the device of module parameters. If the input module has no parameters, returns None"""
        try:
            return next(module.parameters()).device
        except StopIteration:  # The model has no parameters
            return None

    @staticmethod
    def compute_tensor_memory_megabytes(input_size: Union[List[torch.Size], Sequence[Tuple]]) -> float:
        """Returns memory requirement of a tensor by deriving from its size.
        The returned values are in megabytes
        :param input_size: List of input tensor sizes
        """
        # check for the case where the input is not a list of tuples, in which case make it a singleton instance
        # eg: (1,2,3) => [(1,2,3)]
        if not (isinstance(input_size, list) and all([isinstance(x, tuple) for x in input_size])):
            input_size = [input_size]  # type: ignore
        # for each input tensor shape, calculate the sum of their memory requirement
        return sum([np.prod(x) * 4. / (1024 ** 2.) for x in input_size])

    def _register_hook(self, submodule: torch.nn.Module) -> None:
        """
        Adds forward pass hooks to each submodule, module that has no nested modules/layers, in order to
        collect network summary information. Hook handles are stored in a list which are later removed
        outside the scope.
        :param submodule: Children module of the main neural network model.
        """

        def hook(layer: torch.nn.Module, inputs: List[Any], outputs: List[Any]) -> None:
            class_name = str(layer.__class__).split(".")[-1].split("'")[0]
            m_key = "%s-%i" % (class_name, len(self.summary) + 1)
            trainable_params = filter(lambda p: p.requires_grad, layer.parameters())
            input_shape = self._get_sizes_from_list(inputs)
            output_shape = self._get_sizes_from_list(outputs)

            self.summary[m_key] = LayerSummary(
                input_shape=input_shape,
                output_shape=output_shape,
                n_params=sum([np.prod(p.size()) for p in layer.parameters()]),
                n_trainable_params=sum([np.prod(p.size()) for p in trainable_params]),
                device=self._get_device(layer))
            self.n_params += self.summary[m_key].n_params
            self.n_trainable_params += self.summary[m_key].n_trainable_params

        has_no_children = len(list(submodule.modules())) == 1
        if has_no_children:
            self.hooks.append(submodule.register_forward_hook(hook))

    def _generate_summary(self, input_tensors: List[torch.Tensor]) -> None:
        """
        Creates a list of input torch tensors and registers forward pass hooks to the model,
        passes the inputs through the model, and collects model information such num of parameters
        and intermediate tensor size.
        :param input_tensors: A list of tensors which are fed into the torch model.
        """

        def print_summary() -> None:
            logging.info("-------------------------------------------------------------------------------")
            line_new = "{:>20} {:>25} {:>15} {:>15}".format("Layer (type)", "Output Shape", "Param #", "Device")
            logging.info(line_new)
            logging.info("===============================================================================")
            total_output = 0.0
            for layer in self.summary:
                line_new = "{:>20} {:>25} {:>15} {:>15}".format(layer,
                                                                str(self.summary[layer].output_shape),
                                                                "{0:,}".format(self.summary[layer].n_params),
                                                                str(self.summary[layer].device))
                total_output += self.summary[layer].output_memory_megabytes
                logging.info(line_new)

            # Assume 4 bytes/number (float on cuda) - Without mixed precision training and inplace operations
            input_sizes = self._get_sizes_from_list(input_tensors)
            total_input_size = self.compute_tensor_memory_megabytes(input_sizes)
            total_output_size = 2. * total_output  # x2 for gradients

            logging.info("===============================================================================")
            logging.info("Total params: {0:,}".format(self.n_params))
            logging.info("Trainable params: {0:,}".format(self.n_trainable_params))
            logging.info("Input mem size (MB)(Wout mixed-precision): %0.2f" % total_input_size)
            logging.info("Forward/backward pass mem size (MB)(Wout mixed-precision): %0.2f" % total_output_size)
            logging.info("-------------------------------------------------------------------------------")

        # Register the forward-pass hooks, profile the model, and restore its state
        self.model.apply(self._register_hook)
        with Profile(self.model, use_cuda=self.use_gpu) as prof:
            forward_preserve_state(self.model, input_tensors)  # type: ignore

        # Log the model summary: tensor shapes, num of parameters, memory requirement, and forward pass time
        logging.info(self.model)
        logging.info('\n' + prof.display(show_events=False))
        # logging.info('\n' + prof.key_averages().table())
        print_summary()

        # Remove the hooks via handles
        for h in self.hooks:
            h.remove()


def forward_preserve_state(module: DeviceAwareModule, inputs: List[torch.Tensor]) -> torch.Tensor:
    """
    Perform forward pass on input module with given list of torch tensors. The function preserves the random state
    of the backend libraries to avoid reproducibility issues. Additionally, it temporarily sets the model in
    evaluation mode for inference and then restores its previous state.
    :param module: Callable torch module
    :param inputs: List of input torch tensors
    :return output: Output torch tensors
    """
    if not isinstance(inputs, list):
        raise RuntimeError("Inputs object has to be a list of torch tensors")

    if module.is_model_on_gpu():
        inputs = [input_tensor.cuda() for input_tensor in inputs]

    # collect the current state of the model
    is_train = module.training
    module_state = RandomStateSnapshot.snapshot_random_state()

    # set the model in evaluation mode and perform a forward pass
    module.eval()
    with torch.no_grad():
        output = module.forward(*inputs)
    if is_train:
        module.train()

    # restore the seed for torch and numpy
    module_state.restore_random_state()

    return output
