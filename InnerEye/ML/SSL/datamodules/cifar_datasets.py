#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torchvision.datasets import CIFAR10, CIFAR100

OptionalIndexInputAndLabel = Union[Tuple[torch.Tensor, int], Tuple[int, torch.Tensor, int]]


class InnerEyeCIFAR10(CIFAR10):
    """
    Wrapper class around torchvision CIFAR10 class
    """

    def __init__(self, root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 return_index: bool = True,
                 **kwargs: Any) -> None:
        root = root if root is not None else os.getcwd()
        super().__init__(root=root,
                         train=train,
                         transform=transform,
                         **kwargs)
        self.return_index = return_index

    def __getitem__(self, index: int) -> OptionalIndexInputAndLabel:
        img, target = super().__getitem__(index)
        if self.return_index:
            return index, img, target
        else:
            return img, target

    @property
    def num_classes(self) -> int:
        return 10


class InnerEyeCIFAR100(CIFAR100):
    """
    Wrapper class around torchvision CIFAR100 class
    """

    def __init__(self, root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 return_index: bool = True,
                 **kwargs: Any) -> None:

        super().__init__(root=root,
                         train=train,
                         transform=transform,
                         **kwargs)
        self.return_index = return_index

    def __getitem__(self, index: int) -> OptionalIndexInputAndLabel:
        img, target = super().__getitem__(index)
        if self.return_index:
            return index, img, target
        else:
            return img, target

    @property
    def num_classes(self) -> int:
        return 100
