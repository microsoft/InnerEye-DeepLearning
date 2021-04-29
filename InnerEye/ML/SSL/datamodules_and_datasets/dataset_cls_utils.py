#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Tuple, Union

import torch

OptionalIndexInputAndLabel = Union[Tuple[torch.Tensor, int], Tuple[int, torch.Tensor, int]]


class InnerEyeDataClassBaseWithReturnIndex:
    """
    Class to be use with double inheritance with a VisionDataset.
    Overloads the __getitem__ function so that we can optionally also return
    the index within the dataset.
    """

    def __init__(self, root: str, return_index: bool, **kwargs: Any) -> None:
        self.return_index = return_index
        super().__init__(root=root, **kwargs)  # type: ignore

    def __getitem__(self, index: int) -> Any:
        item = super().__getitem__(index)  # type: ignore
        if self.return_index:
            return (index, *item)
        else:
            return item

    @property
    def num_classes(self) -> int:
        raise NotImplementedError
