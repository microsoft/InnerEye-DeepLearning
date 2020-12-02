#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch

from InnerEye.ML.config import SegmentationLoss
from InnerEye.ML.models.losses.mixture import MixtureLoss
from InnerEye.ML.utils.model_util import create_segmentation_loss_component
from Tests.ML.configs.DummyModel import DummyModel


def test_single_element() -> None:
    config = DummyModel()
    element = create_segmentation_loss_component(config, SegmentationLoss.CrossEntropy, power=None)
    mixture = MixtureLoss([(1.0, element)])
    target = torch.tensor([[[0, 0, 1], [1, 1, 0]]], dtype=torch.float32)
    logits = torch.tensor([[[-1e9, -1e9, 0], [0, 0, 0]]], dtype=torch.float32)
    # Extract class indices
    element_loss = element(logits, target)
    mixture_loss = mixture(logits, target)
    assert torch.isclose(element_loss, mixture_loss)


def test_two_elements() -> None:
    config = DummyModel()
    element1 = create_segmentation_loss_component(config, SegmentationLoss.CrossEntropy, power=None)
    element2 = create_segmentation_loss_component(config, SegmentationLoss.SoftDice, power=None)
    weight1, weight2 = 0.3, 0.7
    mixture = MixtureLoss([(weight1, element1), (weight2, element2)])
    target = torch.tensor([[[0, 0, 1], [1, 1, 0]]], dtype=torch.float32)
    logits = torch.tensor([[[-1e9, -1e9, 0], [0, 0, 0]]], dtype=torch.float32)
    # Extract class indices
    element1_loss = element1(logits, target)
    element2_loss = element2(logits, target)
    mixture_loss = mixture(logits, target)
    assert torch.isclose(weight1 * element1_loss + weight2 * element2_loss, mixture_loss)
