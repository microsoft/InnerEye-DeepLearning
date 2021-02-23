#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import abc
from typing import Any, Dict, List, Optional, TypeVar

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils.rnn import PackedSequence

from InnerEye.ML.utils.sequence_utils import map_packed_sequence_data

T = TypeVar('T', torch.Tensor, PackedSequence)


class SupervisedLearningCriterion(torch.nn.Module, abc.ABC):
    """
    Base class for criterion functions used for supervised learning, with the ability to
    smooth labels if required.
    """

    def __init__(self, smoothing_eps: float = 0.0, is_binary_classification: bool = False):
        super().__init__()
        if not (0.0 <= smoothing_eps <= 1.0):
            raise ValueError(f"Expected 0.0 <= smoothing_eps <= 1.0 found {smoothing_eps}")
        self.smoothing_eps = smoothing_eps
        self.is_binary_classification = is_binary_classification

    def forward(self, *input: T, **kwargs: Any) -> Any:
        def _smooth_target(target: torch.Tensor) -> torch.Tensor:
            if self.is_binary_classification or len(target.shape) <= 2:
                _num_classes = 2
            else:
                # Get the number of classes from the class dimension, otherwise assume binary problem
                _num_classes = target.shape[min(1, len(target.shape))]
            # Smooth the one-hot target: 1.0 becomes 1.0-eps, 0.0 becomes eps / (nClasses - 1)
            # noinspection PyTypeChecker
            return target * (1.0 - self.smoothing_eps) + \
                   (1.0 - target) * self.smoothing_eps / (_num_classes - 1.0)  # type: ignore

        _input: List[T] = list(input)
        if self.smoothing_eps > 0.0:
            if isinstance(_input[1], PackedSequence):
                _input[1] = map_packed_sequence_data(_input[1], _smooth_target)
            else:
                _input[1] = _smooth_target(_input[1])

        return self.forward_minibatch(*_input, **kwargs)

    @abc.abstractmethod
    def forward_minibatch(self, output: Any, target: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("forward must be implemented by sub classes")


class BinaryCrossEntropyWithLogitsLoss(SupervisedLearningCriterion):
    """A wrapper function for torch.nn.BCEWithLogitsLoss to enable label smoothing"""

    def __init__(self, num_classes: int,
                 class_counts: Optional[Dict[float, float]] = None,
                 num_train_samples: Optional[int] = None,
                 **kwargs: Any):
        super().__init__(is_binary_classification=True, **kwargs)
        if class_counts and not num_train_samples:
            raise ValueError("Need to specify the num_train_samples with class_counts")
        self._positive_class_weights = None
        self._class_counts = class_counts
        self._num_train_samples = num_train_samples
        self.num_classes = num_classes
        if class_counts:
            self._positive_class_weights = self.get_positive_class_weights()
            if torch.cuda.is_available():
                self._positive_class_weights = self._positive_class_weights.cuda()
        self._loss_fn = BCEWithLogitsLoss(pos_weight=self._positive_class_weights)

    def get_positive_class_weights(self) -> torch.Tensor:
        """
        Returns the weights of the positive class only from the list of
        dictionaries containing the counts for all classes for each
        target position.
        :return: a list of weights to use for the positive class for each target position.
        """
        assert self._class_counts is not None
        weights = [(self._num_train_samples - value) / value if value != 0 else 1.0 for (key, value) in
                   sorted(self._class_counts.items())]  # Uses the first number on the tuple to compare
        return torch.tensor(weights, dtype=torch.float32)

    def forward_minibatch(self, output: T, target: T, **kwargs: Any) -> Any:
        if isinstance(target, PackedSequence) and isinstance(output, PackedSequence):
            return self._loss_fn(output.data.view(-1, 1), target.data.view(-1, 1))
        else:
            return self._loss_fn(output, target)
