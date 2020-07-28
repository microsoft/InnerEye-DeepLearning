#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List, Tuple

import torch

from InnerEye.ML.utils.supervised_criterion import SupervisedLearningCriterion


class MixtureLoss(SupervisedLearningCriterion):

    def __init__(self, components: List[Tuple[float, SupervisedLearningCriterion]]):
        """
        Loss function defined as a weighted mixture (interpolation) of other loss functions.
        :param components: a non-empty list of weights and loss function instances.
        """
        super().__init__()
        if not components:
            raise ValueError("At least one (weight, loss_function) pair must be supplied.")
        self.components = components

    def forward_minibatch(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Wrapper for mixture loss function implemented in PyTorch. Arguments should be suitable for the
        component loss functions, typically:
        :param output: Class logits (unnormalised), e.g. in 3D : BxCxWxHxD  or in 1D BxC
        :param target: Target labels encoded in one-hot representation, e.g. in 3D BxCxWxHxD or in 1D BxC
        """
        result = None
        for (weight, loss_function) in self.components:
            loss = weight * loss_function(output, target, **kwargs)
            if result is None:
                result = loss
            else:
                result = result + loss
        assert result is not None
        torch.cuda.empty_cache()
        return result
