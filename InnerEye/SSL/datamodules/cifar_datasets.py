import os
from typing import Callable, Optional

from torchvision.datasets import CIFAR10, CIFAR100


class InnerEyeCIFAR10(CIFAR10):
    """
    Wrapper class around torchvision CIFAR10 class
    """
    def __init__(self, root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 return_index: bool = True,
                 **kwargs):
        root = root if root is not None else os.getcwd()
        super().__init__(root=root,
                         train=train,
                         transform=transform,
                         **kwargs)
        self.return_index = return_index

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        if self.return_index:
            return index, img, target
        else:
            return img, target

    @property
    def num_classes(self):
        return 10


class InnerEyeCIFAR100(CIFAR100):
    """
    Wrapper class around torchvision CIFAR100 class
    """
    def __init__(self, root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 return_index: bool = True,
                 **kwargs):

        super().__init__(root=root,
                         train=train,
                         transform=transform,
                         **kwargs)
        self.return_index = return_index

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        if self.return_index:
            return index, img, target
        else:
            return img, target

    @property
    def num_classes(self):
        return 100
