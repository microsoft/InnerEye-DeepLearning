#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch

from typing import List

from InnerEye.ML.models.architectures.classification.bit import BiTResNetV2
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule


class BiTModel(DeviceAwareModule[ScalarItem, torch.Tensor]):
    def __init__(self):
        super().__init__()
        self.model = BiTResNetV2()

    def forward(self, x):
        return self.model(x)

    def get_input_tensors(self, item: ScalarItem) -> List[torch.Tensor]:
        return [item.images]
