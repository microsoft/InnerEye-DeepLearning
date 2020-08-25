#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Generic, List, TypeVar

import torch

from InnerEye.Common.type_annotations import T
from InnerEye.ML.utils.ml_util import is_gpu_available

E = TypeVar("E")


class DeviceAwareModule(torch.nn.Module, Generic[T, E]):
    """
    Wrapper around base pytorch module class
    that can provide information about its devices
    """

    def get_device_ids(self) -> List[int]:
        """
        :return a list of device ids on which this module
        is deployed.
        """
        return list({x.device.index for x in self.parameters()})

    def get_number_trainable_parameters(self) -> int:
        """
        :return the number of trainable parameters in the module.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def is_model_on_gpu(self) -> bool:
        """
        Checks if the model is cuda activated or not
        :return True if the model is running on the GPU.
        """
        try:
            cuda_activated = next(self.parameters()).is_cuda
        except StopIteration:  # The model has no parameters
            cuda_activated = False

        return True if (cuda_activated and is_gpu_available()) else False

    def get_input_tensors(self, item: T) -> List[E]:
        """
        Extract the input tensors from a data sample as required
        by the forward pass of the module.
        :param item: a data sample
        :return: the correct input tensors for the forward pass
        """
        raise NotImplementedError("get_input_tensor has to be"
                                  "implemented by sub classes.")

    def get_last_encoder_layer_names(self) -> List[str]:
        """
        Return the name of the last encoder layers for GradCam. Default is an empty list.
        """
        return []
