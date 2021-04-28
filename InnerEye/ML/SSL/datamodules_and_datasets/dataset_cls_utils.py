#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Tuple, Union

import torch

OptionalIndexInputAndLabel = Union[Tuple[torch.Tensor, int], Tuple[int, torch.Tensor, int]]


class InnerEyeDataClassBaseWithReturnIndex:
    """
    Class to be use with double inheritance with a VisionDataset.
    Overloads the __getitem__ function so that we can optionally also return
    the index within the dataset.
    """

    def __init__(self, root: str, return_index: bool, **kwargs):
        self.return_index = return_index
        super().__init__(root=root, **kwargs)

    def __getitem__(self, index: int) -> OptionalIndexInputAndLabel:
        img, target = super().__getitem__(index)
        if self.return_index:
            return index, img, target
        else:
            return img, target

    @property
    def num_classes(self) -> int:
        raise NotImplementedError
